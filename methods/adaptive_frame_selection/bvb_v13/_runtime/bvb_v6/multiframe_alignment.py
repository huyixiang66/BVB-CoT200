"""Experimental multi-frame alignment of already-merged V6 object groups.

This does not call a model or change its input scene. Detection boxes, model
geometry, and camera poses are hypotheses; fitting is not a correctness check.
"""
import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def transformed(points, pivot, x):
    c, s = math.cos(x[3]), math.sin(x[3])
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
    return (points-pivot) @ rotation.T + pivot + x[:3]


def projection(points, c2w, intrinsics):
    """Project world vertices into 640x480 input pixels, Blender -Z forward."""
    camera = (np.asarray(points)-np.asarray(c2w)[:3, 3]) @ np.asarray(c2w)[:3, :3]
    good = -camera[:, 2] > .03
    if np.count_nonzero(good) < max(4, len(camera)*.8):
        return None
    z = -camera[good, 2]
    k = np.asarray(intrinsics)
    xy = np.stack([k[0, 0]*camera[good, 0]/z+k[0, 2],
                   k[1, 2]-k[1, 1]*camera[good, 1]/z], axis=1)
    return np.array([xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()])


def edge_residual(predicted, box):
    if predicted is None:
        return np.full(4, 2.)
    box = np.asarray(box)
    residual = (predicted-box)/np.array([640., 480., 640., 480.])
    # A clipped frame boundary is a one-sided constraint, not an exact edge.
    if box[0] <= 2: residual[0] = max(0., residual[0])
    if box[1] <= 2: residual[1] = max(0., residual[1])
    if box[2] >= 638: residual[2] = min(0., residual[2])
    if box[3] >= 478: residual[3] = min(0., residual[3])
    return residual


def observations(prepared, group, category, cached):
    folder = Path(prepared)/group
    frames = read(folder/'input-frames.json')
    poses = read(folder/'pose-reference.json')['rows']
    rows = []
    for frame, pose in zip(frames, poses, strict=True):
        if frame['source_frame_index'] != pose['source_frame_index']:
            raise ValueError('Frame/pose mismatch in '+group)
        boxes = [o for o in frame.get('observations', [])
                 if o.get('object') == category and o.get('box_xyxy')]
        source = 'original_input_detection'
        if not boxes:
            boxes = [o for o in cached.get(frame['source_frame_index'], [])
                     if o.get('object') == category and o.get('box_xyxy')]
            source = 'optional_cached_detection'
        if not boxes:
            continue
        best = max(boxes, key=lambda x: x.get('dino_score', x.get('score')) or 0)
        rows.append({'source_frame_index': frame['source_frame_index'],
                     'timestamp_s': frame['timestamp_s'], 'box': best['box_xyxy'],
                     'evidence_source': source, 'dino_score': best.get('dino_score', best.get('score')),
                     'c2w': pose['c2w_blender'], 'k': pose['intrinsics_640x480']})
    return rows


def fit_group(target, points, rows):
    gid = target['group_id']
    pivot = (points.min(axis=0)+points.max(axis=0))/2
    valid = []
    for row in rows:
        if projection(points, row['c2w'], row['k']) is None:
            row['split'] = 'excluded_initial_projection_invalid'
        else:
            valid.append(row)
    if len(valid) < 3:
        return {'group_id': gid, 'category': target['category'], 'status': 'insufficient',
                'reason': 'Need at least two fit frames and one held-out frame with valid projections.', 'rows': rows}
    training, holdout = valid[:-1], valid[-1]

    def residual(x):
        moved = transformed(points, pivot, x)
        evidence = [edge_residual(projection(moved, r['c2w'], r['k']), r['box']) for r in training]
        return np.concatenate([*evidence, .06*x/np.array([.75, .75, .35, math.radians(20)])])

    fit = least_squares(residual, np.zeros(4), bounds=([-.75, -.75, -.35, -math.radians(20)],
                                                        [.75, .75, .35, math.radians(20)]),
                        loss='soft_l1', f_scale=.08, max_nfev=250)
    x = fit.x
    for row in rows:
        before = projection(points, row['c2w'], row['k'])
        after = projection(transformed(points, pivot, x), row['c2w'], row['k'])
        row.update(split='holdout' if row is holdout else 'fit' if row in training else 'excluded_initial_projection_invalid',
                   before_projected_box=before.tolist() if before is not None else None,
                   after_projected_box=after.tolist() if after is not None else None,
                   before_edge_error=float(np.mean(np.abs(edge_residual(before, row['box'])))),
                   after_edge_error=float(np.mean(np.abs(edge_residual(after, row['box'])))))
    return {'group_id': gid, 'category': target['category'], 'status': 'fitted' if fit.success else 'not_converged',
            'pivot': pivot.tolist(), 'translation': x[:3].tolist(), 'yaw_degrees': math.degrees(x[3]),
            'training_frames': len(training), 'held_out_frames': 1, 'success': bool(fit.success),
            'function_evaluations': int(fit.nfev), 'rows': rows}


def blender_call(blender, blend, script, merged, out, log):
    command = [str(blender), '--background', '--threads', '1', '--disable-autoexec', str(blend),
               '--python-exit-code', '1', '--python', str(script), '--', str(merged), str(out)]
    with Path(log).open('w', encoding='utf-8') as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True, help='Original V6 per-question prepared input directory')
    parser.add_argument('--merged', type=Path, required=True, help='Existing build output merged/ directory')
    parser.add_argument('--out', type=Path, required=True, help='New output directory; source files are untouched')
    parser.add_argument('--blender', default=shutil.which('blender'), help='Blender executable')
    parser.add_argument('--detections', type=Path, help='Optional detections.jsonl from the SAME video and frame indexing')
    args = parser.parse_args(argv)
    if not args.blender:
        parser.error('Provide --blender or put Blender on PATH')
    prepared, merged, out = args.prepared.resolve(), args.merged.resolve(), args.out.resolve()
    for path in (prepared/'manifest.json', merged/'merge.json', merged/'merged-scene.blend'):
        if not path.is_file():
            parser.error('Missing input: '+str(path))
    if out.exists():
        parser.error('Output already exists: '+str(out))
    out.mkdir(parents=True)
    cached = {}
    if args.detections:
        for line in args.detections.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            cached[row['source_frame_index']] = row['detections']
    blend = merged/'merged-scene.blend'
    blender_call(args.blender, blend, ROOT/'runtime/multiframe_export.py', merged, out, out/'export.log')
    targets = read(out/'targets.json')
    vertices = np.load(out/'target-vertices.npz')
    groups = [fit_group(t, vertices[t['group_id']],
                        observations(prepared, t['group_id'], t['category'], cached)) for t in targets]
    groups.extend(read(out/'skipped.json'))
    result = {'groups': groups, 'method': 'multi-frame robust target-box reprojection; one rigid correction per group',
              'limits': 'Estimated boxes, poses and geometry are not ground truth; held-out nearby frames can be correlated.',
              'constraints': {'translation_xy': .75, 'translation_z': .35, 'yaw_degrees': 20},
              'source_blend': str(blend), 'prepared': str(prepared), 'qa_run': False, 'api_calls': 0}
    write(out/'alignment.json', result)
    blender_call(args.blender, blend, ROOT/'runtime/multiframe_render.py', merged, out, out/'render.log')
    print('Experimental alignment saved:', out)
    print('Groups:', {status: sum(g['status'] == status for g in groups)
                      for status in sorted({g['status'] for g in groups})})


if __name__ == '__main__':
    main()
