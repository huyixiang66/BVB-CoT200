"""Conservative question-level requirements for reviewed object clusters."""
import re
import argparse
import hashlib
import json
from pathlib import Path


def classify_question(question):
    """Return an auditable starting policy; unrecognized questions keep V6 behavior.

    A single target requires one admitted cluster per question-relevant category
    after semantic review and, when necessary, Astra comparison. Explicit per-question overrides remain necessary
    when multiple instances of a category serve different roles.
    """
    kind = str(question.get('question_type') or question.get('task') or question.get('type') or '').lower()
    text = str(question.get('question') or question.get('ori_question') or '').lower()
    evidence = []

    # User policy: counting and relative-distance comparisons retain multiple
    # clusters; all other VSI/ReVSI types select one cluster per object category.
    if kind.startswith(('object_counting', 'object_rel_distance')):
        return _policy('multiple', [], kind)
    if (kind.startswith(('object_rel_direction', 'room_size_estimation')) or
            kind in {'object_abs_distance', 'object_size_estimation', 'route_planning',
                     'obj_appearance_order', 'object_appearance_order'}):
        if kind in {'object_abs_distance', 'object_size_estimation'}:
            evidence.append('metric_scale')
        if kind.startswith('room_size_estimation'):
            evidence.extend(['room_boundary', 'metric_scale'])
        if kind == 'route_planning':
            evidence.append('navigable_layout')
        if kind in {'obj_appearance_order', 'object_appearance_order'}:
            evidence.append('appearance_time')
        return _policy('single', evidence, kind)

    if any(term in kind for term in ('displacement', 'path length', 'speed', 'acceleration',
                                      'ego-centric orientation', 'trajectory', 'pose estimation')):
        return _policy('supplemental', ['trajectory', 'timestamps'], kind)
    if '3d video grounding' in kind:
        return _policy('single', ['camera_pose_at_query', 'camera_coordinates', 'metric_scale'], kind)
    if 'dimensional measurement' in kind:
        evidence.append('metric_scale')
        if re.search(r'\b(camera|photographer|cameraman)\b', text):
            evidence.append('camera_pose_at_query')
        if re.search(r'\b(two|three|both|multiple|all)\s+\w+(?:\s+\w+)?s\b', text):
            return _policy('unknown', evidence, 'plural referents require question-level review')
        return _policy('single', evidence, kind)
    if kind == 'spatial relation':
        # This task includes selection and comparisons; retain all plausible groups.
        return _policy('multiple', [], kind)

    if 'camera motion' in kind:
        return _policy('supplemental', ['camera_trajectory', 'timestamps'], kind)
    if 'instance motion' in kind:
        return _policy('supplemental', ['object_trajectory', 'timestamps'], kind)
    if 'interactive motion' in kind:
        return _policy('supplemental', ['multi_object_tracking', 'timestamps'], kind)
    if 'motion understanding' in kind:
        return _policy('supplemental', ['motion_evidence', 'timestamps'], kind)
    if 'cross-video' in kind or 'cross video' in kind:
        return _policy('supplemental', ['cross_video_alignment', 'multiple_time_states'], kind)
    if 'prediction' in kind:
        return _policy('supplemental', ['motion_history', 'timestamps', 'prediction_assumptions'], kind)
    if 'planning' in kind:
        return _policy('multiple', ['navigable_layout', 'check_dynamic_conditions'], kind)
    if 'camera-scene spatial relationship' in kind:
        return _policy('multiple', ['camera_pose_at_query', 'scene_layout'], kind)
    if 'camera-instance spatial relationship' in kind:
        return _policy('single', ['camera_pose_at_query'], kind)
    if 'scene-scene spatial relationship' in kind or 'instance-scene spatial relationship' in kind:
        return _policy('multiple', ['scene_layout'], kind)
    if 'instance-instance spatial relationship' in kind:
        return _policy('multiple', [], kind)
    if 'instance/scene attribute' in kind:
        if re.search(r'\b(how many|number of|count)\b', text):
            return _policy('multiple', [], 'attribute counting')
        return _policy('unknown', [], 'attribute questions require question-level review')
    return _policy('unknown', [], 'unrecognized question type')


def _policy(category, evidence, reason):
    return {'category': category, 'additional_evidence': evidence,
            'reason': reason, 'source': 'question_type_and_text', 'provisional': True}


def question_policy(question):
    """Respect an explicit structured override when supplied upstream."""
    override = question.get('retention_policy')
    if override is None:
        return classify_question(question)
    if not isinstance(override, dict) or override.get('category') not in {
            'single', 'multiple', 'supplemental', 'unknown'}:
        raise ValueError('Invalid retention_policy category')
    evidence = override.get('additional_evidence', [])
    if not isinstance(evidence, list) or not all(isinstance(x, str) for x in evidence):
        raise ValueError('retention_policy additional_evidence must be a list of strings')
    return {'category': override['category'], 'additional_evidence': evidence,
            'reason': override.get('reason', 'explicit question override'),
            'source': 'explicit_override', 'provisional': False}


def main():
    parser = argparse.ArgumentParser(description='Audit provisional retention policies without model calls')
    parser.add_argument('--input', required=True, type=Path, help='JSONL question bank or V6 selection JSON')
    parser.add_argument('--scene', help='Only include questions from this scene')
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.input.suffix == '.parquet':
        import pandas as pd
        frame = pd.read_parquet(args.input)
        questions = [{'id': row['ID'], 'scene': row['Video'], 'task': row['Task'], 'question': row['Question'],
                      'time_start': row.get('time_start'), 'time_end': row.get('time_end')}
                     for row in frame.to_dict('records')]
    elif args.input.suffix == '.jsonl':
        questions = [json.loads(line) for line in args.input.read_text(encoding='utf-8').splitlines() if line.strip()]
    else:
        data = json.loads(args.input.read_text(encoding='utf-8'))
        questions = data if isinstance(data, list) else data['questions']
        if isinstance(questions, dict):
            questions = [dict(row.get('question', row), id=row.get('question', row).get('id', key))
                         for key, row in questions.items()]
    records = []
    for q in questions:
        if args.scene and str(q.get('scene_name', q.get('scene', ''))) != args.scene:
            continue
        scene = q.get('scene_name', q.get('scene'))
        identity = json.dumps([scene,q.get('id'),q.get('question',q.get('ori_question')),
                               q.get('time_start'),q.get('time_end')],ensure_ascii=False,default=str)
        records.append({'question_key': hashlib.sha256(identity.encode('utf-8')).hexdigest(),
                        'id': q.get('id'), 'scene': scene,
                        'type': q.get('question_type', q.get('task', q.get('type'))),
                        'question': q.get('question', q.get('ori_question')),
                        'policy': question_policy(q)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'questions': records}, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    counts = {category: sum(r['policy']['category'] == category for r in records)
              for category in ('single', 'multiple', 'supplemental', 'unknown')}
    print(json.dumps({'question_count': len(records), 'categories': counts}, ensure_ascii=False))


if __name__ == '__main__':
    main()
