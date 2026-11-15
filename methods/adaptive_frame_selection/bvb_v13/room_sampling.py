"""Room-only 1-fps selection: position guards, angular coverage, quality ties.

Coverage means camera position/optical-axis coverage, not measured floor visibility.
All selected observations belong to ONE whole-scene evidence group.
"""
import copy
import math

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

MODE = 'room_position_coverage_1fps'


def select_room_frames(question, frames, pose_points, position_ratio=.25,
                       angle_degrees=45.):
    if not question['question_type'].startswith('room_size_estimation'):
        raise ValueError('Room sampling is only applicable to room_size_estimation')
    if not 0 < position_ratio <= 1 or not 0 < angle_degrees < 180:
        raise ValueError('Invalid position ratio or angular coverage radius')
    rows = sorted(copy.deepcopy(frames), key=lambda r: r['timestamp_s'])
    if not rows:
        raise ValueError('Empty room candidate pool')
    if len({r['source_frame_index'] for r in rows}) != len(rows):
        raise ValueError('Duplicate candidate frame IDs')
    if not np.allclose([r['timestamp_s'] for r in rows], np.arange(len(rows)), atol=1e-6):
        raise ValueError('Room candidates must be the full 1-fps pool starting at zero')
    poses = {p['frame']: p for p in pose_points}
    valid, excluded = [], []
    for i, row in enumerate(rows):
        p = poses.get(row['source_frame_index'])
        if (p is None or abs(p['time']-row['timestamp_s']) > 1e-6 or
            np.shape(p['position']) != (3,) or np.shape(p['forward']) != (3,) or
            not np.isfinite(p['position']).all() or not np.isfinite(p['forward']).all() or
            np.linalg.norm(p['forward']) < 1e-9):
            excluded.append(row['source_frame_index'])
        else:
            valid.append(i)
        row.update(selected_for_group=False, observations=[], evidence_role='full_scene')
    if not valid:
        raise ValueError('No valid exactly aligned room poses')
    positions = np.array([poses[rows[i]['source_frame_index']]['position'] for i in valid], dtype=float)
    directions = np.array([poses[rows[i]['source_frame_index']]['forward'] for i in valid], dtype=float)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    span = float(np.linalg.norm(np.quantile(positions, .95, axis=0)-np.quantile(positions, .05, axis=0)))
    threshold = max(span*position_ratio, 1e-6)
    labels = (fcluster(linkage(pdist(positions), method='complete'), threshold, criterion='distance')
              if len(valid) > 1 else np.ones(1, dtype=int))
    groups = []
    # Renumber position groups chronologically, independent of scipy label numbering.
    partitions = sorted((np.flatnonzero(labels == label).tolist() for label in set(labels)),
                        key=lambda ids: valid[ids[0]])
    for gid, members in enumerate(partitions, 1):
        forward = directions[members]
        covers = forward @ forward.T >= math.cos(math.radians(angle_degrees))-1e-12
        uncovered = set(range(len(members)))
        selected = []
        while uncovered:
            def rank(k):
                sharp = float(rows[valid[members[k]]].get('sharpness', 0) or 0)
                sharp = sharp if math.isfinite(sharp) else 0
                return (sum(bool(covers[k, u]) for u in uncovered), sharp, -valid[members[k]])
            k = max((k for k in range(len(members)) if k not in selected), key=rank)
            selected.append(k)
            uncovered.difference_update(np.flatnonzero(covers[k]).tolist())
        for k in selected:
            rows[valid[members[k]]]['selected_for_group'] = True
        for k in members:
            rows[valid[k]]['position_group'] = gid
        groups.append(dict(id=gid,
            candidate_source_frame_indices=[rows[valid[k]]['source_frame_index'] for k in members],
            selected_source_frame_indices=[rows[valid[members[k]]]['source_frame_index'] for k in selected]))
    selected = [r for r in rows if r['selected_for_group']]
    report = dict(mode=MODE, sampling_fps=1, position_diameter_ratio=position_ratio,
                  position_diameter_native_units=threshold, direction_radius_degrees=angle_degrees,
                  position_group_count=len(groups), represented_position_groups=len(groups),
                  candidate_count=len(rows), valid_pose_count=len(valid), selected_count=len(selected),
                  excluded_pose_frame_indices=excluded, position_groups=groups,
                  selected_source_frame_indices=[r['source_frame_index'] for r in selected],
                  coverage_proxy='Within each complete-link position group, every valid candidate optical axis is within the angular radius of a selected frame. Maximize newly covered candidates, break ties by image sharpness then timestamp. No physical surface visibility guarantee.',
                  group_count=1, frame_cap=None)
    group = dict(group_id='room-whole-scene-position-coverage', object='全场景', objects=[], peak_ids=[],
                 evidence_role='full_scene', candidate_frames=rows, selected=selected,
                 status='room_position_coverage_input', frame_cap=None, sampling_fps=1)
    return group, report


def prepare(question_path, sampling_path, poses_path, out,
            position_ratio=.25, angle_degrees=45.):
    """Reusable room-only entry point over extracted 1-fps frames + LingBot cache."""
    import hashlib
    import json
    from pathlib import Path
    from PIL import Image
    from scipy.ndimage import laplace
    question_path, sampling_path, poses_path, out = map(Path, (question_path, sampling_path, poses_path, out))
    if out.exists():
        raise FileExistsError('Use a fresh room sampling output')
    question = json.loads(question_path.read_text())
    sampling = json.loads(sampling_path.read_text())
    pose_sampling = json.loads((poses_path/'sampling.json').read_text())
    if sampling.get('sampling_fps') != 1:
        raise ValueError('Room sampling requires a 1-fps candidate pool')
    if sampling.get('source_sha256') != pose_sampling.get('video_sha256') or not sampling.get('source_sha256'):
        raise ValueError('Room frames and LingBot poses must identify the same video')
    from bvb_v6.reconstruction import shared_pose
    matrices, _, _ = shared_pose(poses_path, None)
    points = [dict(frame=r['source_frame_index'],time=r['timestamp_s'],
                   position=m[:3,3].tolist(),forward=(-m[:3,2]).tolist())
              for r,m in zip(pose_sampling['frames'],matrices,strict=True)]
    frames = copy.deepcopy(sampling['frames'])
    if len(frames) != math.ceil(sampling['duration_s']):
        raise ValueError('Incomplete 1-fps candidate pool')
    for row in frames:
        path = (sampling_path.parent/row['image']).resolve()
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['frame_sha256']:
            raise ValueError('Room frame hash mismatch: '+str(path))
        with Image.open(path) as im:
            row['sharpness'] = float(laplace(np.asarray(im.convert('L'),dtype=float)).var())
        row['image'] = str(path)
    group, report = select_room_frames(question, frames, points, position_ratio, angle_degrees)
    report.update(sampling_source=str(sampling_path.resolve()),pose_source=str(poses_path.resolve()))
    out.mkdir(parents=True)
    (out/'selection.json').write_text(json.dumps(dict(question=question,
        arms={MODE:[group]},parameters=report,new_model_calls=0),ensure_ascii=False,indent=2)+'\n')
    return report


if __name__=='__main__':
    import argparse
    import json
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['question','sampling','poses','out']:
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--position-ratio',type=float,default=.25)
    parser.add_argument('--angle-degrees',type=float,default=45.)
    args=parser.parse_args()
    print(json.dumps(prepare(args.question,args.sampling,args.poses,args.out,
                            args.position_ratio,args.angle_degrees),ensure_ascii=False))
