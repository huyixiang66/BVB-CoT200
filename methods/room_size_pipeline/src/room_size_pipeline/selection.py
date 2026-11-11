"""Temporal coverage with optional camera/surface diversity and a soft image cost."""
import hashlib
import math
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def choose_frames(duration, pose_rows=(), reference_fps=.125):
    if not math.isfinite(duration) or duration <= 0 or not math.isfinite(reference_fps) or reference_fps <= 0:
        raise ValueError('Invalid video duration or reference FPS')
    reference_count = max(1, math.ceil(duration * reference_fps))
    if not pose_rows:
        times = [i / reference_fps for i in range(reference_count)
                 if i / reference_fps < duration]
        return times, {'mode': 'uniform', 'reference_count': reference_count,
                       'selected_count': len(times), 'soft_budget': True}
    rows = sorted(pose_rows, key=lambda x: x['timestamp_s'])
    for row in rows:
        m=row['c2w_blender']
        if (len(m)!=4 or any(len(r)!=4 for r in m) or
                any(not math.isfinite(float(v)) for r in m for v in r) or
                any(abs(float(a)-b)>1e-6 for a,b in zip(m[3],[0,0,0,1]))):
            raise ValueError('Invalid c2w matrix')
    times = [float(x['timestamp_s']) for x in rows]
    if len(set(times)) != len(times) or any(not 0 <= t < duration for t in times):
        raise ValueError('Pose timestamps must be unique and inside the video')
    positions = [[float(row['c2w_blender'][i][3]) for i in range(3)] for row in rows]
    span = max(max(p[i] for p in positions) - min(p[i] for p in positions) for i in range(3))
    cell = max(span / 8, 1e-8)  # Relative units; never assumes metric LingBot depth.
    features = []
    weights = {}
    for row, pos in zip(rows, positions):
        m = row['c2w_blender']
        direction = [-float(m[i][2]) for i in range(3)]
        az = int((math.atan2(direction[1], direction[0]) + math.pi) / (math.pi / 4)) % 8
        el = round(math.atan2(direction[2], math.hypot(*direction[:2])) / (math.pi / 6))
        place = tuple(math.floor(v / cell) for v in pos)
        f = {('time', int(row['timestamp_s'] * reference_fps)): 1.,
             ('view', place, az, el): .8}
        # Optional cached visible surface-cell IDs; pose-only mode does not claim visibility.
        for surface in row.get('visible_cells', []):
            f[('surface', str(surface))] = .2
        features.append(set(f)); weights.update(f)
    selected, covered = [], set()
    remaining = set(range(len(rows)))
    while remaining:
        cost = 1 + (len(selected) / reference_count) ** 4
        def utility(i):
            return sum(weights[k] for k in features[i] - covered) / cost
        best = max(remaining, key=lambda i: (utility(i), -i))
        if selected and utility(best) < .55:
            break
        selected.append(best); covered.update(features[best]); remaining.remove(best)
    return sorted(times[i] for i in selected), {
        'mode': 'surface_and_camera' if any(x.get('visible_cells') for x in rows) else 'camera_diversity',
        'reference_count': reference_count, 'selected_count': len(selected), 'soft_budget': True,
        'weighted_coverage': sum(weights[k] for k in covered) / sum(weights.values()),
        'warning': 'Coarse geometric/temporal diversity is not semantic room completeness.'}

