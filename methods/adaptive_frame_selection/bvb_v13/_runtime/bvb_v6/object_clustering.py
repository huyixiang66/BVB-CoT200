"""Experimental object-focused grouping. Offline; never imports API clients.

Unknown geometry is not negative evidence. Clusters remain instance hypotheses.
All thresholds are dimensionless heuristics, not calibrated probabilities.
"""
from itertools import combinations
import numpy as np
from scipy.spatial import cKDTree

PARAMETERS = dict(threshold=.468, appearance_weight=.75, min_points=12,
                  component_radius_depth=.08, geometry_radius_scale=.18,
                  minimum_frame_pairs=2, supported_edge_fraction=.5,
                  distinct_box_iou=.05, distinct_box_score=.5,
                  reliable_quality=.55, far_gap_scale=.75, far_center_scale=1.5,
                  far_pair_fraction=.7)


def roi_surface(index, c2w, intrinsics, cloud, image_size, box, params=PARAMETERS):
    if box is None:
        return None
    low, high = np.asarray(box, float).reshape(2, 2)
    low = np.maximum(low, 0); high = np.minimum(high, image_size)
    if np.any(high-low < 2):
        return None
    nearby = sorted(np.unique(cloud['frame_ids']), key=lambda f: (abs(int(f)-index), f))
    sources = [int(f) for f in nearby[:3] if abs(int(f)-index) <= 12]
    points = cloud['points'][np.isin(cloud['frame_ids'], sources)]
    xyz = (points-c2w[:3, 3]) @ c2w[:3, :3]
    mask = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 1e-4)
    points, xyz = points[mask], xyz[mask]
    uvh = xyz @ intrinsics.T
    uv = uvh[:, :2] / uvh[:, 2:3]
    mask = ((uv >= low) & (uv <= high)).all(axis=1)
    points, xyz, uv = points[mask], xyz[mask], uv[mask]
    # Grid is local to the object box, so a small remote object is not discarded
    # solely because it occupies few full-image cells.
    cells = np.minimum(23, ((uv-low)/(high-low)*24).astype(int))
    front = {}
    for i in np.argsort(xyz[:, 2]):
        front.setdefault(tuple(cells[i]), i)
    kept = list(front.values())
    if len(kept) < params['min_points']:
        return None
    points, xyz, uv = points[kept], xyz[kept], uv[kept]
    # A spatial component is a background rejection heuristic, not segmentation.
    depth = float(np.median(xyz[:, 2]))
    parent = list(range(len(points)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for a,b in cKDTree(points).query_pairs(max(depth*params['component_radius_depth'],1e-6)):
        parent[root(a)] = root(b)
    parts = {}
    for i in range(len(points)):
        parts.setdefault(root(i), []).append(i)
    center_weight = np.exp(-2*np.sum(((uv-(low+high)/2)/(high-low))**2, axis=1))
    component = max(parts.values(), key=lambda ids: (float(center_weight[ids].sum()), len(ids)))
    if len(component) < params['min_points']:
        return None
    fraction = len(component)/len(points)
    surface = points[component]
    normalized_uv = (uv[component]-low)/(high-low)
    coverage = float(np.prod(np.ptp(normalized_uv,axis=0)))
    clipped = bool(np.any(low <= 1) or np.any(high >= np.asarray(image_size)-1))
    quality = min(1.,len(surface)/40)*fraction*min(1.,coverage/.35)*(.7 if clipped else 1.)
    return dict(anchor=np.median(surface,axis=0), samples=surface,
                low=np.quantile(surface,.1,axis=0), high=np.quantile(surface,.9,axis=0),
                tree=cKDTree(surface), quality=float(quality), points=len(surface),
                component_fraction=fraction, box_coverage=coverage, clipped=clipped,
                depth=float(np.median(xyz[component,2])), sources=sources)


def compare_roi(a,b,scale,semantic,params=PARAMETERS):
    if a is None or b is None:
        return dict(score=float(semantic), geometry=None, reliability=0.,
                    geometry_status='unknown', far=False)
    center = float(np.linalg.norm(a['anchor']-b['anchor'])/scale)
    extent = max(float(np.linalg.norm(a['high']-a['low'])),
                 float(np.linalg.norm(b['high']-b['low'])))/scale
    da=a['tree'].query(b['samples'])[0]/scale
    db=b['tree'].query(a['samples'])[0]/scale
    # Different visible faces need not coincide. Use a robust nearest-surface
    # distance plus an extent-aware center distance, not exact overlap alone.
    near = float(max(np.quantile(da,.25),np.quantile(db,.25)))
    overlap = float(min(np.mean(da <= params['geometry_radius_scale']),
                        np.mean(db <= params['geometry_radius_scale'])))
    g = float(.5*np.exp(-.5*(center/(.3+.5*extent))**2)+.5*np.exp(-.5*(near/.25)**2))
    reliability = float(np.sqrt(a['quality']*b['quality']))
    fused = g+params['appearance_weight']*semantic*(1-g)
    score = reliability*fused+(1-reliability)*semantic
    gap = float(np.linalg.norm(np.maximum(0,np.maximum(a['low']-b['high'],b['low']-a['high'])))/scale)
    far = (min(a['quality'],b['quality']) >= params['reliable_quality'] and
           gap > params['far_gap_scale'] and center > params['far_center_scale'])
    return dict(score=float(score),geometry=g,reliability=reliability,center_distance=center,
                box_gap=gap,nearest_surface=near,surface_overlap=overlap,
                geometry_status='available',far=bool(far))


def box_iou(a,b):
    a=np.asarray(a);b=np.asarray(b)
    inter=float(np.prod(np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]))))
    union=float(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-inter)
    return inter/max(union,1e-9)


def simultaneous_conflict(a,b,params=PARAMETERS):
    for x in a:
        for y in b:
            if (x['source_frame_index']==y['source_frame_index'] and x.get('box_xyxy') and y.get('box_xyxy')
                and min(x.get('dino_score') or 0,y.get('dino_score') or 0)>=params['distinct_box_score']
                and box_iou(x['box_xyxy'],y['box_xyxy'])<params['distinct_box_iou']):
                return True
    return False


def supported_pair_score(pairs,minimum=2):
    """Strongest minimum score from independent source frames on both sides."""
    ranked=sorted((p for p in pairs if p.get('valid',True)),key=lambda p:p['score'],reverse=True)
    if minimum==1:
        return (ranked[0]['score'],1) if ranked else (0.,0)
    for i,p in enumerate(ranked):
        if any(p['frames'][0]!=q['frames'][0] and p['frames'][1]!=q['frames'][1] for q in ranked[:i]):
            return p['score'],2
    return (ranked[0]['score'],1) if ranked else (0.,0)


def cluster(peaks,evidence,mode='robust',params=PARAMETERS):
    groups=[(p,) for p in sorted(peaks)];history=[]
    def assessment(a,b):
        pairs=[evidence[tuple(sorted((x,y)))] for x in a for y in b]
        scores=[p['score'] for p in pairs]
        if any(p['cannot_link'] for p in pairs):return False,float(np.mean(scores)),'cannot_link'
        if mode=='complete':return min(scores)>=params['threshold'],min(scores),'complete_link'
        supported=sum(p['score']>=params['threshold'] and p['independent_frame_pairs']>=params['minimum_frame_pairs'] for p in pairs)
        mean=float(np.mean(scores))
        allowed=(mean>=params['threshold'] and supported/len(pairs)>=params['supported_edge_fraction']
                 and supported>=min(2,len(pairs)))
        return allowed,mean,'mean_and_multiple_support'
    while True:
        options=[]
        for i,j in combinations(range(len(groups)),2):
            ok,score,reason=assessment(groups[i],groups[j])
            if ok:options.append((score,groups[i],groups[j],i,j,reason))
        if not options:break
        score,a,b,i,j,reason=max(options,key=lambda x:(x[0],x[1],x[2]))
        history.append(dict(left=list(a),right=list(b),score=score,rule=reason))
        groups=[g for k,g in enumerate(groups) if k not in (i,j)]+[tuple(sorted(a+b))]
        groups.sort()
    blocked=[]
    for a,b in combinations(groups,2):
        _,score,reason=assessment(a,b);blocked.append(dict(left=list(a),right=list(b),score=score,reason=reason))
    return [list(g) for g in groups],dict(merges=history,remaining=blocked)
