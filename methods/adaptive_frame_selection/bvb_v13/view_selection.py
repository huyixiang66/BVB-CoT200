"""Deterministic envelope-based selection of existing, envelope-fitted renders.

No model, image interpretation, answer keys or metric answers enter this module.
Visibility is a conservative projected-box proxy, NOT verified mesh visibility.
"""
import itertools
import math
import numpy as np
from bvb_v10.pipeline import read, save, sha

POLICY = 'envelope-complementarity-v1'
LIMITATION = ('包络投影的遮挡估计，不等同真实网格或语义可见性；'
              '最佳仅指现有候选及固定评分规则，不能保证重建准确。')


def configure(out, mode=None):
    exp = read(out/'experiment.json')
    current = exp.get('view_selection_mode', 'fixed')
    mode = mode or current
    if mode not in {'astra', 'fixed'}:
        raise ValueError('view_selection must be astra or fixed')
    selection = out/'view-selection/selection.json'
    if selection.exists():
        recorded = read(selection).get('selection_mode', 'astra')
        if recorded != mode:
            raise ValueError('View-selection mode differs from saved results; use a fresh output directory')
    if mode != current and ((out/'qa').exists() or (out/'view-selection').exists()):
        raise ValueError('View-selection/QA receipts exist; use a fresh output directory to change mode')
    if exp.get('view_selection_mode') != mode:
        exp['view_selection_mode'] = mode
        save(out/'experiment.json', exp)
    return mode


def basis(angles):
    """Blender XYZ Euler: local camera axes in world coordinates."""
    x, y, z = map(math.radians, angles)
    cx, cy, cz = math.cos(x), math.cos(y), math.cos(z)
    sx, sy, sz = math.sin(x), math.sin(y), math.sin(z)
    rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    return rz @ ry @ rx


def camera(view, points):
    # Explicit geometry metadata is emitted by the current renderer. The fallback
    # exactly matches the frozen renderer for existing V11 candidate manifests.
    angles = view.get('camera_euler_degrees')
    if angles is None:
        vid = view['id']
        if vid == 'top': angles = [0,0,0]
        elif vid.startswith('oblique-'): angles = [20,-20,int(vid.split('-')[1])]
        elif vid.startswith('elevated-'): angles = [45,0,int(vid.split('-')[1])]
        else: raise ValueError('Unknown candidate camera: '+vid)
    rotation = basis(angles)
    local = points @ rotation
    lo, hi = local.min(axis=0), local.max(axis=0)
    center = (lo+hi)/2
    location = view.get('camera_location', (rotation @ np.array([center[0],center[1],hi[2]+30])).tolist())
    scale = float(view.get('ortho_scale', max(hi[:2]-lo[:2])*1.15))
    if not math.isfinite(scale) or scale <= 0: raise ValueError('Degenerate scene envelope')
    return rotation, np.array(location), scale


def score_candidates(geometry, views, categories, *, walls_hidden=False, occlusion=None):
    points = np.array([v for o in geometry for v in o['world_bbox']], dtype=float)
    if points.size == 0 or not np.isfinite(points).all(): raise ValueError('Invalid scene envelope')
    hidden_by_view = {v['view_id']:{o['name'] for o in v.get('hidden_objects', [])}
                      for v in (occlusion or {}).get('views', [])}
    keys = {}
    for obj in geometry:
        if obj['role'] == 'target' and obj['category'] in categories:
            # Missing IDs in legacy inspection: treat each mesh as a diagnostic
            # unit; never reinterpret its count as physical-instance truth.
            keys[obj['name']] = (obj['category'], str(obj.get('instance_id') or obj['name']))
    if set(k[0] for k in keys.values()) != set(categories): raise ValueError('Missing target category envelope')
    units = sorted(set(keys.values()))
    index = {key:i for i,key in enumerate(units)}
    scores = []
    for view in sorted(views, key=lambda v:v['id']):
        rotation, location, scale = camera(view, points)
        boxes = []
        hidden = hidden_by_view.get(view['id'], set())
        for obj in geometry:
            if obj.get('hide_render') or obj['name'] in hidden: continue
            if walls_hidden and obj['role'] == 'structure' and obj['category'] == 'wall': continue
            local = (np.array(obj['world_bbox'])-location) @ rotation
            lo, hi = local.min(axis=0), local.max(axis=0)
            rect = np.r_[lo[:2]/scale+.5, hi[:2]/scale+.5]
            boxes.append((obj, rect, hi[2]))
        fractions = [[] for _ in units]
        sizes = [0. for _ in units]
        for obj, rect, depth in boxes:
            if obj['name'] not in keys: continue
            unit = keys[obj['name']]; i = index[unit]
            width, height = rect[2:]-rect[:2]
            area = float(max(0,width)*max(0,height))
            if area <= 1e-12:
                fractions[i].append((0.,1e-12)); continue
            # Deterministic 12x12 grid on each projected part envelope. Parts of
            # the same physical instance cannot occlude each other in this proxy.
            t = (np.arange(12)+.5)/12
            x,y = np.meshgrid(rect[0]+t*width, rect[1]+t*height)
            visible = (x>=0)&(x<=1)&(y>=0)&(y<=1)
            for other, blocker, front in boxes:
                if other['name'] == obj['name'] or keys.get(other['name']) == unit: continue
                if front <= depth+1e-7: continue
                visible &= ~((x>=blocker[0])&(x<=blocker[2])&(y>=blocker[1])&(y<=blocker[3]))
            fractions[i].append((float(visible.mean()),area))
            sizes[i] += area
        visibility = [sum(v*a for v,a in rows)/sum(a for _,a in rows) if rows else 0. for rows in fractions]
        # Equal instance weighting; large objects must not mask missing small ones.
        quality = [v*min(1.,math.sqrt(a/.01)) for v,a in zip(visibility,sizes)]
        scores.append(dict(id=view['id'],visibility=visibility,quality=quality,
                           projected_area=sizes,camera_direction=(-rotation[:,2]).tolist(),
                           camera_location=location.tolist(),ortho_scale=scale))
    return units, scores


def choose(scores, task):
    count = 1 if task == 'rel_dir' else 3
    best = None
    for group in itertools.combinations(scores, count):
        ids = [v['id'] for v in group]
        if count > 1:
            azimuths = {int(v.rsplit('-',1)[1]) for v in ids if v != 'top'}
            if len(azimuths) < 2: continue
        if task == 'route' and 'top' not in ids: continue
        union = np.max([v['quality'] for v in group],axis=0)
        mean_quality = float(np.mean([v['quality'] for v in group]))
        pairs = list(itertools.combinations(group,2))
        diversity = sum((1.-float(np.dot(a['camera_direction'],b['camera_direction'])))/2 for a,b in pairs)/max(1,len(pairs))
        # First cover every target in at least one view; then make the selected
        # views individually useful and geometrically complementary.
        objective = 4*float(union.min())+2*float(union.mean())+.5*mean_quality+.25*diversity
        if best is None or objective > best[0]+1e-12: best = (objective,group)
    if best is None: raise ValueError('Insufficient compatible candidate views')
    return best


def ensure_selection(out):
    if configure(out) != 'fixed': return
    manifest = read(out/'views/views.json')
    source_files = ['input.json','question-public.json','geometry-inspection.json','views/views.json']
    if (out/'scene.blend').exists(): source_files.append('scene.blend')
    if (out/'occlusion-report.json').exists(): source_files.append('occlusion-report.json')
    provenance = {name:sha(out/name) for name in source_files}
    for v in manifest['views']:
        if sha(out/'views'/v['image']) != v['sha256']: raise ValueError('Candidate image hash mismatch')
    destination = out/'view-selection/selection.json'
    if destination.exists():
        old = read(destination)
        if old.get('policy') != POLICY or old.get('source_sha256') != provenance:
            raise ValueError('Fixed selection sources/policy changed; use a fresh output directory')
        return old
    from .driver import kind
    question = read(out/'input.json')['question']; task = kind(question)
    if task == 'route' and manifest.get('qa_walls_hidden') is not False:
        raise ValueError('Route selection must retain walls')
    categories = question['objects']
    units, scores = score_candidates(read(out/'geometry-inspection.json'),manifest['views'],categories,
        walls_hidden=manifest.get('qa_walls_hidden',False),
        occlusion=read(out/'occlusion-report.json') if 'occlusion-report.json' in source_files else None)
    objective, selected = choose(scores,task)
    ids = [v['id'] for v in selected]
    coverage = []
    for category in categories:
        indices = [i for i,key in enumerate(units) if key[0] == category]
        coverage.append(dict(category=category,visible_in=[v['id'] for v in selected if all(v['quality'][i]>=.15 for i in indices)],
                             uncertainty=LIMITATION))
    result = dict(selection_mode='fixed',policy=POLICY,model_calls=0,input_tokens=0,output_tokens=0,
        view_id=ids[0],view_ids=ids,coverage=coverage,reason='按包络投影覆盖、可见面积及视角互补性进行确定性选择。',
        limitations=LIMITATION,source_sha256=provenance)
    # Numerical envelopes, part/instance identities and scores are diagnostics
    # only, never supplied to final QA (especially counting).
    save(out/'view-selection/geometry-scores.json',dict(policy=POLICY,objective=objective,
        units=units,candidates=scores,source_sha256=provenance,limitations=LIMITATION))
    save(destination,result)
    return result
