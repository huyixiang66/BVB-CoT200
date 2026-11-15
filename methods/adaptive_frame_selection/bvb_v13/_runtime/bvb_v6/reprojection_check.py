"""Diagnostic projection of reconstructed target centers onto input boxes.

This tests consistency with provisional detector boxes and estimated poses; it
does not establish object identity, visibility, metric accuracy, or QA success.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def project(center, c2w, intrinsics):
    camera = np.linalg.solve(np.asarray(c2w, dtype=float), np.array([*center, 1.]))
    if camera[2] >= -1e-6:
        return None
    k = np.asarray(intrinsics, dtype=float)
    return [float(k[0, 0] * camera[0] / -camera[2] + k[0, 2]),
            float(k[1, 2] - k[1, 1] * camera[1] / -camera[2])]


def check_group(folder):
    folder = Path(folder)
    frames = read(folder/'input-frames.json')
    poses = read(folder/'pose-reference.json')['rows']
    spec = read(folder/'assets/scene-spec.json')
    audit = read(folder/'assets/question-audit.json')
    labels = {t['id']: t['label'] for t in audit['targets']}
    targets = [o for o in spec['objects'] if o['id'] in labels]
    if len(frames) != len(poses):
        raise ValueError('Frames and poses have different lengths')
    rows = []
    for frame, pose in zip(frames, poses, strict=True):
        if frame['source_frame_index'] != pose['source_frame_index']:
            raise ValueError('Frame and pose identifiers differ')
        for observation in frame['observations']:
            box = observation.get('box_xyxy')
            if not box:
                continue
            candidates = []
            for target in targets:
                if labels[target['id']] != observation['object']:
                    continue
                xy = project(target['p'], pose['c2w_blender'], pose['intrinsics_640x480'])
                inside = xy is not None and box[0] <= xy[0] <= box[2] and box[1] <= xy[1] <= box[3]
                candidates.append({'target_id': target['id'], 'center_xy': xy, 'inside_box': inside})
            rows.append({'source_frame_index': frame['source_frame_index'], 'timestamp_s': frame['timestamp_s'],
                         'object': observation['object'], 'box_xyxy': box, 'candidates': candidates,
                         'any_center_inside': any(t['inside_box'] for t in candidates)})
    return {'group': folder.name, 'observations_with_box': len(rows),
            'any_center_inside': sum(r['any_center_inside'] for r in rows), 'rows': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    manifest = read(args.prepared/'manifest.json')
    results = []
    for group in manifest['groups']:
        folder = args.prepared/group['folder']
        if (folder/'assets/scene-spec.json').exists():
            results.append(check_group(folder))
        else:
            results.append({'group': group['group_id'], 'status': 'missing_reconstruction'})
    args.out.write_text(json.dumps({'groups': results, 'diagnostic_only': True}, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Checked', len(results), 'groups; diagnostic only:', args.out)


if __name__ == '__main__':
    main()
