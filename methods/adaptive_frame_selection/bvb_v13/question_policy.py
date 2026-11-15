"""Deterministic seven-task retention policy; zero model requests."""
import hashlib
import json
from collections import Counter
from pathlib import Path

REVISION = 'v12-fixed-question-policy-1'


def policy_for(question):
    kind = question.get('question_type', '')
    evidence = []
    category = 'single'
    if kind.startswith('object_counting'):
        category = 'multiple'
    elif kind in {'object_rel_distance_closest', 'object_rel_distance_farthest'}:
        category = 'multiple'
    elif kind.startswith('object_rel_direction'):
        evidence = ['standing_facing_target_roles']
    elif kind == 'object_abs_distance':
        evidence = ['metric_scale', 'between_objects_structure']
    elif kind == 'object_size_estimation':
        evidence = ['metric_scale']
    elif kind == 'route_planning':
        evidence = ['navigable_layout']
    elif kind.startswith('room_size_estimation'):
        evidence = ['whole_scene', 'room_boundary', 'metric_scale']
    else:
        raise ValueError('No fixed V12 question policy for '+repr(kind))
    # Distinct roles can refer to separate instances of the same category.
    # Preserve the hypotheses here; identity is still decided by semantic review.
    roles = question.get('roles') or {}
    labels = [v.strip().casefold() for v in roles.values() if isinstance(v, str) and v.strip()]
    repeated = any(n > 1 for n in Counter(labels).values())
    if repeated and not kind.startswith('room_size_estimation'):
        category = 'multiple'
    return dict(id=str(question['id']), category=category,
                reason='Fixed question-type rule'+('; repeated category across explicit roles' if repeated else ''),
                confidence='high', additional_evidence=evidence,
                decision_source='fixed_question_type_v12', revision=REVISION,
                evidence_group='whole_scene' if kind.startswith('room_size_estimation') else 'object',
                model_calls=0)


def decide_policies(selection, folder):
    """Persist a local decision receipt compatible with the existing auditor."""
    public = [{k:v for k,v in row['question'].items()
               if k in {'id','scene_name','question_type','question','options','roles','objects'}}
              for row in selection['questions'].values()]
    results = [policy_for(q) for q in public]
    if len({r['id'] for r in results}) != len(results):
        raise ValueError('Duplicate question IDs')
    fingerprint = hashlib.sha256(json.dumps(dict(revision=REVISION,questions=public),sort_keys=True).encode()).hexdigest()
    folder = Path(folder);path = folder/'result.json'
    record = dict(questions=results, request_hash=fingerprint, revision=REVISION,
                  decision_source='fixed_question_type_v12',model_calls=0,model=None,
                  input_tokens=0,output_tokens=0)
    if path.exists():
        if json.loads(path.read_text()) != record:
            raise ValueError('Question policy changed; use a fresh V12 output')
        return results
    if (folder/'request.json').exists():
        raise ValueError('Historical model policy receipt present; use a fresh V12 output')
    folder.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    return results
