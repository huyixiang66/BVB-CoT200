"""Pose diversity, thumbnail sharpness and temporal bridges; no detector required."""
import math
from .selection import choose_frames


def thumbnail_sharpness(path):
    import cv2
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f'Cannot read candidate frame: {path}')
    image = cv2.resize(image, (160, 120))
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def choose_light_frames(duration, rows, reference_fps=.125):
    # Reuse the established input checks and the uniform fallback.
    fallback, _ = choose_frames(duration, rows, reference_fps)
    if not rows:
        return fallback, {'mode': 'uniform', 'selected_count': len(fallback)}
    rows = sorted(rows, key=lambda r: r['timestamp_s'])
    positions = [[float(r['c2w_blender'][a][3]) for a in range(3)] for r in rows]
    directions = [[-float(r['c2w_blender'][a][2]) for a in range(3)] for r in rows]
    for d in directions:
        norm = math.sqrt(sum(x*x for x in d))
        if norm < 1e-8:
            raise ValueError('Invalid camera direction')
        d[:] = [x/norm for x in d]
    span = max(max(p[a] for p in positions)-min(p[a] for p in positions) for a in range(3))
    distance_scale = max(span/4, 1e-8)
    sharp = [float(r.get('sharpness', 1.)) for r in rows]
    if any(not math.isfinite(x) or x < 0 for x in sharp):
        raise ValueError('Invalid sharpness')
    sharp_scale = max(sorted(sharp)[int(.9*(len(sharp)-1))], 1e-8)
    # Blender shared_pose is Z-up and looks along local -Z.
    quality = [(0.25+0.75*min(s/sharp_scale, 1)) *
               (1-.75*abs(d[2])**2) for s,d in zip(sharp,directions)]

    def separation(i,j):
        dist = math.sqrt(sum((a-b)**2 for a,b in zip(positions[i],positions[j])))
        angle = math.acos(max(-1.,min(1.,sum(a*b for a,b in zip(directions[i],directions[j])))))
        return max(dist/distance_scale, angle/math.radians(50))

    selected = [max(range(len(rows)), key=lambda i: (quality[i], -i))]
    reasons = {selected[0]: 'clear_overview'}
    reference_count = max(1, math.ceil(duration*reference_fps))
    while len(selected) < len(rows):
        remaining = [i for i in range(len(rows)) if i not in reasons]
        def gain(i):
            novelty = min(1., min(separation(i,j) for j in selected))
            return novelty*quality[i]/(1+(len(selected)/reference_count)**2)
        best = max(remaining, key=lambda i: (gain(i), -i))
        if gain(best) < .25:
            break
        selected.append(best); reasons[best] = 'new_position_or_direction'
    # Add a bridge only if a large time/pose jump admits an actual intermediate
    # view closer to BOTH endpoints. Do not infer doorways from poses alone.
    for left,right in zip(sorted(selected), sorted(selected)[1:]):
        if rows[right]['timestamp_s']-rows[left]['timestamp_s'] < 4:
            continue
        gap = separation(left,right)
        if gap < 1.5:
            continue
        candidates = [i for i in range(left+1,right) if quality[i] >= .35 and
                      max(separation(left,i), separation(i,right)) < .85*gap]
        if candidates:
            best = max(candidates, key=lambda i: quality[i]/(1+max(separation(left,i), separation(i,right))))
            selected.append(best); reasons[best] = 'temporal_pose_bridge'
    selected.sort()
    return [rows[i]['timestamp_s'] for i in selected], {
        'mode': 'light_pose_quality', 'reference_count': reference_count,
        'candidate_count': len(rows), 'selected_count': len(selected), 'soft_budget': True,
        'frames': [{'timestamp_s': rows[i]['timestamp_s'], 'reason': reasons[i],
                    'quality': quality[i], 'sharpness': sharp[i]} for i in selected],
        'warning': 'Pose diversity does not establish room coverage or reference completeness.'}
