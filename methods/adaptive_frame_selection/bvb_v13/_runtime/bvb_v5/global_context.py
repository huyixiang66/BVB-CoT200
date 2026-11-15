"""Select explicit scene-context frames for questions without object targets."""
import numpy as np


def select_context(candidates, c2w, budget=16, temporal_bins=8):
    if len(candidates) != len(c2w) or not candidates:
        raise ValueError('Candidate/pose alignment required')
    budget = min(budget, len(candidates))
    sharp = np.asarray([r['sharpness'] for r in candidates], float)
    quality = .5 + .5 * sharp / (sharp + max(float(np.median(sharp)), 1.))
    centers = np.asarray(c2w)[:, :3, 3]
    forward = np.array(c2w[:, :3, 2], dtype=float, copy=True)
    forward /= np.maximum(np.linalg.norm(forward, axis=1, keepdims=True), 1e-9)
    scale = max(float(np.linalg.norm(np.ptp(centers, axis=0))), 1e-6)
    times = np.asarray([r['timestamp_s'] for r in candidates])
    edges = np.linspace(times[0], times[-1] + 1e-6, min(temporal_bins, budget)+1)
    selected, reasons = [], {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        eligible = np.flatnonzero((times >= lo) & (times < hi))
        if len(eligible):
            k = int(eligible[np.argmax(quality[eligible])])
            selected.append(k); reasons[k] = 'sharpest frame in temporal coverage bin'
    while len(selected) < budget:
        translation = np.linalg.norm(centers[:, None]-centers[selected], axis=2)/scale
        angle = np.arccos(np.clip(forward @ forward[selected].T, -1, 1))/np.pi
        distance = np.min(.5*translation + .5*angle, axis=1)*quality
        gap = np.min(abs(times[:, None]-times[selected]), axis=1)
        distance[gap < .5] = -1
        distance[selected] = -np.inf
        k = int(np.argmax(distance))
        selected.append(k); reasons[k] = 'marginal camera position/orientation coverage, quality weighted'
    return [{'source_frame_index': candidates[k]['source_frame_index'],
             'timestamp_s': candidates[k]['timestamp_s'], 'reason': reasons[k],
             'status': 'global_scene_context', 'frame': k}
            for k in sorted(selected, key=lambda i: times[i])]
