"""Offline single-link experiment over frozen frame-pair evidence.

V12 connected policy; detection, poses and appearance features remain unchanged.
Surface distance means the existing robust 3-D nearest-surface distance.
"""
import math

PARAMETERS=dict(threshold=.468,center_weight=.2,surface_weight=.8,
                appearance_weight=.75,minimum_frame_pairs=1,
                preserve_cannot_link=True)

def reweight(pair,params=PARAMETERS):
    p=dict(pair)
    if p['geometry'] is None:
        p['score']=float(p['semantic'])
        return p
    surface=math.exp(-.5*(p['nearest_surface']/.25)**2)
    # Frozen geometry was exactly .5*center + .5*surface. Recover the
    # original center kernel without rerunning or changing the point cloud.
    center=2*p['geometry']-surface
    if not -1e-10<=center<=1+1e-10:raise ValueError('Unexpected frozen geometry formula')
    center=min(1.,max(0.,center))
    geometry=params['center_weight']*center+params['surface_weight']*surface
    appearance=p['semantic'];reliability=p['reliability']
    p.update(center_kernel=center,surface_kernel=surface,geometry=geometry,
             score=reliability*(geometry+params['appearance_weight']*appearance*(1-geometry))+(1-reliability)*appearance)
    return p

def edges_from_pairs(pairs):
    out={}
    for old in pairs:
        frames=[reweight(p) for p in old['frame_pairs']]
        valid=[p for p in frames if p.get('valid',True)]
        best=max(valid,key=lambda p:p['score']) if valid else None
        out[tuple(sorted(old['peaks']))]=dict(peaks=old['peaks'],score=best['score'] if best else 0.,
            valid=bool(valid),cannot_link=old['cannot_link'],conflict_reason=old['conflict_reason'],
            witness=best,old_score=old['score'],old_independent_pairs=old['independent_frame_pairs'])
    return out

def connected_groups(peaks,edges,protect=True,threshold=.468):
    """Descending single-link union, optionally respecting component conflicts.

    A-B and B-C suffice even if A-C has low similarity. An explicit A-C
    cannot-link still prevents joining the two components in protected mode.
    """
    parents={p:p for p in peaks};members={p:{p} for p in peaks}
    def root(p):
        while parents[p]!=p:
            parents[p]=parents[parents[p]];p=parents[p]
        return p
    eligible=sorted((e for e in edges.values() if e['valid'] and not e['cannot_link'] and e['score']>=threshold),
                    key=lambda e:(-e['score'],tuple(e['peaks'])))
    history=[];blocked=[]
    for edge in eligible:
        a,b=map(root,edge['peaks'])
        if a==b:continue
        conflicts=[list(sorted((x,y))) for x in sorted(members[a]) for y in sorted(members[b])
                   if edges[tuple(sorted((x,y)))]['cannot_link']]
        if protect and conflicts:
            blocked.append(dict(edge=edge['peaks'],score=edge['score'],conflicts=conflicts));continue
        history.append(dict(left=sorted(members[a]),right=sorted(members[b]),edge=edge['peaks'],
                            score=edge['score'],witness=edge.get('witness')))
        parents[b]=a;members[a]|=members.pop(b)
    groups=sorted([sorted(v) for v in members.values()])
    violations=[e['peaks'] for e in edges.values() if e['cannot_link'] and root(e['peaks'][0])==root(e['peaks'][1])]
    return groups,dict(merges=history,blocked=blocked,internal_conflicts=violations,
                       eligible_edge_count=len(eligible),protect_cannot_link=protect)
