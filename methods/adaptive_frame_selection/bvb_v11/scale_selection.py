import json,time
from pathlib import Path
from types import SimpleNamespace
from bvb_v10.pipeline import read,save,sha
from .common import parse,state
from . import abs_distance
run=SimpleNamespace(read=read,save=save,sha=sha,parse=parse,state=state)
BASE=None
def frame_score(f):
    obs = f.get('observations') or [f]
    return max((o.get('dino_score') or 0 for o in obs), default=0)

def audit(sid, allow_endpoints=False):
    """Reuse target audit; new Astra pass picks one wide third object per question."""
    from subscription_direct_client import request, image_item
    packet = BASE/'packets'/sid; out = BASE/'scenes'/sid
    catalog = run.read(packet/'reference-catalog.json')
    questions = [q for q in run.read(BASE/'questions.public.json') if q['scene_name'] == sid]
    stable_only=run.read(BASE/'run-config.json').get('scale_reference_policy')=='stable_dimensions_only'
    run.state(out, 'selecting-scale-reference')
    choice_path = out/'reference-selection/selection.json'
    if not choice_path.exists():
        folder = choice_path.parent
        if (folder/'request.json').exists():
            ledger = run.read(folder/'request.json')
            if ledger['status'] == 'started':
                raise RuntimeError('Potentially live reference selection; inspect before retry')
            if ledger['status'] != 'completed' or (folder/'invalid-output.json').exists():
                folder.rename(out/('reference-selection-failed-'+str(time.time_ns())))
        prompt = '''For EACH supplied absolute-distance question independently choose
ONE supplied cluster of a THIRD physical object as a common-sense size reference.
Inspect the CLEAN representative image and detection coordinates of every candidate.
Prefer a broad/wide object with an identifiable complete horizontal extent and a
relatively stable common-sense width/length/diameter. Avoid tiny, deformable, heavily
occluded, strongly variable-size or ambiguous objects when a better reference exists.
Choose from this scene's real clusters only. No hallucinated ruler, no hardcoded
category preference. Exclude both question objects AND aliases or parts of those same
physical objects (sofa/couch, desk/table etc.). A different label alone is insufficient.
Do not infer any distance answer. Reference dimensions are uncertain common-sense
estimates, never calibrated measurements. Explain which horizontal edge is estimated;
do not use height or a screen's diagonal as a horizontal width. More cluster frames
will be supplied to reconstruction. The same reference may be reused for questions
where it is distinct from both targets. Choose your best candidate even if all are
weak, and explicitly describe uncertainty. No ground truth or prior answers supplied.
Return JSON {"choices":[{"question_id":str,"candidate_id":str,
"distinct_from_question_objects":true,"dimension":"width|length|diameter",
"dimension_description":"Chinese specific edge and physical subtype",
"size_m":number,"size_range_m":[lower,upper],"reason":"Chinese",
"uncertainty":"Chinese"}]}. Exactly one entry per question.
'''+ '\nPUBLIC QUESTIONS: '+json.dumps(questions, ensure_ascii=False)
        if allow_endpoints:
            prompt = '''For EACH absolute-distance question choose ONE evidenced
physical object as a common-sense size reference. You may freely choose the FIRST
question object itself, the SECOND question object itself, or a THIRD scene object.
Prefer a broad, rigid object with a complete identifiable horizontal extent and a
stable common-sense width/length/diameter. Inspect all supplied clean images and
coordinate hints. Avoid variable-size, ambiguous or deformable references where
possible. No fixed category preference. No ground truth or previous answers.
For FIRST/SECOND, select only the candidate explicitly bound to that question ID
and endpoint role. This is the exact already admitted target cluster; do not select
a different same-category instance. For THIRD, select a scene candidate distinct
from both endpoints, excluding aliases and parts of the same physical object.
The reconstruction uses a top view. Overlying non-target objects can be hidden at
render time, but cannot manufacture missing reference geometry. Prefer a horizontal
dimension that can be identified after removing overhead occluders. Do not use height
or a screen diagonal as width. The physical size and range are common-sense estimates,
NOT measured calibration. Explain the edge and subtype. Do not answer the distance.
Return JSON {"choices":[{"question_id":str,"candidate_id":str,
"reference_role":"first|second|third","distinct_from_question_objects":boolean,
"dimension":"width|length|diameter","dimension_description":"Chinese",
"size_m":number,"size_range_m":[lower,upper],"reason":"Chinese",
"uncertainty":"Chinese"}]}. For THIRD distinct_from_question_objects must be true;
for an endpoint it is false. Exactly one choice per question.
'''+ '\nPUBLIC QUESTIONS: '+json.dumps(questions, ensure_ascii=False)
        content = [dict(type='input_text', text=prompt)]
        if stable_only:
            prompt='''For EACH supplied question select ONE real object with a
STABLE physical dimension as a common-sense scale reference. Dimensional stability
is the eligibility criterion. Do not prefer an object because it is broad, wide,
large, near a target or convenient to render. A broad object with variable product
sizes is unsuitable. Identify its visible subtype and explain why the specified
dimension varies little across plausible instances of that subtype. Do not assume
all monitors, sofas, tables, sinks or arbitrary containers have one standard size.
You may select the FIRST or SECOND question object itself, using only the endpoint
candidate bound to that exact question ID and role, or a THIRD object distinct from
both endpoints and their aliases/parts. Inspect the clean images and coordinates.
Choose length, width, height or diameter, whichever is genuinely stable and clearly
defined. No preference for a horizontal dimension: QA will use three complementary
views. Explain its exact endpoints and plausible physical size range. Do not invent
a narrow uncertainty range to qualify an unstable object. Priors are estimates,
not calibration; no GT or prior answers are supplied. Do not answer the distance.
If no candidate has a defensibly stable dimension, say so with candidate_id=null
and dimension_stability="unavailable"; never silently choose a variable-size ruler.
Return JSON {"choices":[{"question_id":str,"candidate_id":str,
"reference_role":"first|second|third","distinct_from_question_objects":boolean,
"dimension":"width|length|height|diameter","dimension_description":"Chinese",
"dimension_stability":"stable","stability_reason":"Chinese subtype-specific rationale",
"size_m":number,"size_range_m":[lower,upper],"reason":"Chinese","uncertainty":"Chinese"}]}.
Exactly one entry per public question. For THIRD distinct_from_question_objects=true;
for an endpoint it is false.
'''+ '\nPUBLIC QUESTIONS: '+json.dumps(questions,ensure_ascii=False)
            if run.read(BASE/'run-config.json').get('scale_policy')=='width-largest':
                from .abs_distance_width import PROMPT
                prompt=PROMPT+'\nPUBLIC QUESTIONS: '+json.dumps(questions,ensure_ascii=False)
            content=[dict(type='input_text',text=prompt)]
        for c in catalog:
            r = c['representative']; p = Path(r['path']); assert run.sha(p) == r['sha256']
            content += [dict(type='input_text', text=json.dumps(dict(candidate_id=c['candidate_id'],
                category=c['category'], endpoint_question_id=c.get('endpoint_question_id'),
                endpoint_role=c.get('endpoint_role'), timestamp_s=r['timestamp_s'], observations=r['observations']), ensure_ascii=False)), image_item(p)]
        response = request(out/'reference-selection', content, model='gpt-6-astra', effort='high')
        try:
            result = run.parse(response['answer'])
            abs_distance.validate_choices(result, questions, catalog, allow_endpoints, stable_only)
        except (ValueError, TypeError, KeyError) as exc:
            run.save(folder/'invalid-output.json', dict(error=str(exc)))
            raise
        run.save(choice_path, result)
    choices = abs_distance.validate_choices(run.read(choice_path), questions, catalog, allow_endpoints, stable_only)
    width_only=run.read(BASE/'run-config.json').get('scale_policy')=='width-largest'
    if width_only:
        from .abs_distance_width import validate_choices
        validate_choices(run.read(choice_path),questions,catalog)
    byid = {c['candidate_id']: c for c in catalog}
    target = run.read(packet/'target-selection.json')
    for choice in choices:
        qid = str(choice['question_id']); c = byid[choice['candidate_id']]
        row = target['questions'][qid]
        public = next(q for q in questions if str(q['id']) == qid)
        role = choice.get('reference_role', 'third')
        row['question']['objects'] = list(public['objects'])
        if role == 'third':
            row['question']['objects'].append(c['category'])
            row['question'].setdefault('instance_limits', {})[c['category']] = 1
            row['arms']['semantic_selected'].append(c['group'])
        run.save(BASE/'references'/(qid+'.json'), dict(public_question=public, category=c['category'],
            reference_role=role,
            group_id=c['group']['group_id'], choice=choice, candidate_source=c['source_selection'],
            candidate_source_sha256=c['source_sha256'], selection_sha256=run.sha(choice_path),
            reference_frames=len(c['group']['selected']), metric_scale_calibrated=False, scale_reference_policy='stable_width_only' if width_only else 'stable_dimensions_only'))
    target['inputs']['cache'] = str(packet/'cache')
    target['inputs']['poses'] = run.read(BASE/'run-config.json')['poses'][sid]
    run.save(out/'input/review/reviewed-selection.json', target)
    run.save(packet/'selection-remote.json', target)
    run.state(out, 'completed', reference_questions=len(choices))
