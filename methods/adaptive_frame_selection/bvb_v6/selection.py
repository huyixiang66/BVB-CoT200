"""Portable adapter for the upstream 0.75 geometry/appearance peak preset."""
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import copy
import hashlib
import json
import re

import numpy as np

from .cluster.group_regions import (describe_view, geometric_affinity,
                                   semantic_affinity, frame_rows, read, write)
from .cluster.group_peak_regions import group_peaks
from .reconstruction import pick, shared_pose

PRESET = Path(__file__).parent / 'preset/geometry-appearance-075.json'


def score_pair(g, s, parameters):
    return g + parameters['appearance_weight'] * s * (1-g) if g >= parameters['geometry_gate'] else g


def cluster_rows(rows, pairs, parameters):
    if not rows:
        return [], {}
    data = {'arms': {'geometry': [{'selected': list(rows.values())}]},
            'pairs': [{**p, 'tuned': score_pair(p['geometry'], p['semantic'], parameters)} for p in pairs]}
    return group_peaks(data, 'tuned', parameters['threshold'], parameters['minimum_pairs'])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    parameters = read(args.preset)['parameters']
    if not (0 <= parameters['appearance_weight'] <= 1 and 0 <= parameters['geometry_gate'] <= 1
            and 0 <= parameters['threshold'] <= 1 and parameters['minimum_pairs'] in (1, 2)):
        raise ValueError('Invalid clustering preset')
    if args.frame_cap < 4:
        raise ValueError('frame-cap must be at least four')
    if args.out.exists():
        raise FileExistsError(args.out)
    peaks = read(args.v3 / 'peaks.json')
    questions = read(args.v3 / 'questions.json')['questions']
    if args.qid:
        questions = [q for q in questions if str(q['id']) in args.qid]
        if {str(q['id']) for q in questions} != set(args.qid):
            raise ValueError('Requested question not found')
    candidates_list = read(args.cache / 'candidates.json')
    candidates = {r['source_frame_index']: r for r in candidates_list}
    samples = read(args.poses / 'sampling.json')['frames']
    poses = dict(np.load(args.poses / 'camera-poses.npz'))
    cloud = dict(np.load(args.poses / 'sparse-geometry-corrected.npz'))
    # Match upstream: one median depth scale over the entire frozen peak pool.
    all_rows = frame_rows(peaks, candidates)
    image_size = np.asarray(args.image_size)
    meta = args.poses / 'geometry-info.json'
    intrinsic_size = read(meta)['image_size_wh'] if meta.exists() else [518, 392]
    resize = np.diag([*np.divide(image_size, intrinsic_size), 1.])
    geometry = {}
    for fid, row in all_rows.items():
        i = row['frame']
        if (samples[i]['source_frame_index'] != fid or samples[i]['sha256'] != row['frame_sha256']
                or sha(args.cache / row['image']) != row['frame_sha256']):
            raise ValueError(f'Frame/pose provenance mismatch at {fid}')
        geometry[fid] = describe_view(i, poses['c2w'][i], resize @ poses['intrinsics'][i], cloud, image_size)
    depths = [g['depth'] for g in geometry.values() if g]
    if not depths or not np.isfinite(depths).all():
        raise ValueError('No usable central-view depth; cannot calibrate geometric affinity')
    scale = float(np.median(depths))
    features_npz = np.load(args.appearance / 'features.npz')
    features = dict(zip(features_npz['ids'], features_npz['cls']))
    vectors = {}
    for r in read(args.appearance / 'rois.json')['rows']:
        fid = r['source_frame_index']
        if r['source_sha256'] != candidates[fid]['frame_sha256']:
            raise ValueError('Appearance/frame hash mismatch')
        v = features[r['id']]
        vectors[(fid, r['object'], tuple(r['box_xyxy']))] = v / max(np.linalg.norm(v), 1e-12)
    output = {'version': 'bvb_v6', 'preset': read(args.preset), 'depth_scale': scale,
              'scope': 'per-category', 'questions': {}, 'missing_roi_observations': [],
              'frame_cap': args.frame_cap, 'qa_run': False, 'new_model_calls': False,
              'upstream_commit': '9a9c0a0b5e9342b2ca7d82cfa747d884d48dd232',
              'inputs': {k: str(getattr(args, k).resolve()) for k in ['v3', 'cache', 'poses', 'appearance']}}
    world, _, _ = shared_pose(args.poses, args.reference_cameras)
    for question in questions:
        # Persist public question fields only, never the answer/GT fields.
        question = {k: copy.deepcopy(v) for k, v in question.items()
                    if k in ('id', 'scene_name', 'question_type', 'question', 'options', 'roles', 'objects', 'official_clip')}
        groups, audits, missing = [], {}, []
        for category in question['objects']:
            subset = [p for p in peaks if p['object'] == category]
            rows = frame_rows(subset, candidates)
            if not rows:
                missing.append(category)
                continue
            descriptors = defaultdict(lambda: defaultdict(list))
            for fid, row in rows.items():
                for hint in row['observations']:
                    if hint['box_xyxy'] is None:
                        continue
                    key = (fid, category, tuple(hint['box_xyxy']))
                    if key in vectors:
                        descriptors[fid][category].append(vectors[key])
                    else:
                        output['missing_roi_observations'].append({'frame': fid, 'object': category, 'peak': hint['peak_id']})
            pairs = []
            for a, b in combinations(sorted(rows), 2):
                g, distance, overlap = geometric_affinity(geometry[a], geometry[b], scale)
                pairs.append({'frames': [a, b], 'geometry': g,
                              'semantic': semantic_affinity(descriptors[a], descriptors[b]),
                              'depth_normalized_distance': distance, 'surface_overlap': overlap})
            clustered, evidence = cluster_rows(rows, pairs, parameters)
            audits[category] = {'pairs': pairs, 'peak_pair_scores': evidence}
            for n, group in enumerate(clustered, 1):
                group['group_id'] = re.sub(r'[^A-Za-z0-9_-]+', '-', category).strip('-') + f'-{n:02d}'
                group['candidate_frames'] = copy.deepcopy(group['selected'])
                group['selected'] = sorted(pick(group['selected'], candidates, world, args.frame_cap),
                                           key=lambda r: r['source_frame_index'])
                group['status'] = 'region_hypothesis_not_verified_instance'
                group['minimum_four_satisfied'] = len(group['selected']) >= 4
                groups.append(group)
        output['questions'][str(question['id'])] = {'question': question, 'arms': {'v6': groups},
            'missing_objects': missing, 'evidence': audits,
            'context_only_requires_explicit_frames': not bool(question['objects']),
            'summary': {'groups': len(groups), 'image_slots': sum(len(g['selected']) for g in groups),
                        'unique_frames': len({r['source_frame_index'] for g in groups for r in g['selected']})}}
    if output['missing_roi_observations'] and not args.allow_missing_appearance:
        raise ValueError(f"{len(output['missing_roi_observations'])} missing ROI features. Rebuild the V5 expanded-crop cache or explicitly allow missing appearance.")
    args.out.mkdir(parents=True)
    write(args.out / 'selection.json', output)
    for qid, record in output['questions'].items():
        print(qid, record['summary'], 'missing:', record['missing_objects'])
    return output
