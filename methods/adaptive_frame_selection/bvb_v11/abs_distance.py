"""Absolute distance with endpoint-or-third-object scale and cleared top views."""
import json
import math
from pathlib import Path
from bvb_v10.pipeline import read, save, sha
from . import reliability

REVISION = '20261112-abs-distance-flexible-reference-clear-top'


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def validate_choices(result, questions, catalog, allow_endpoints=False, stable_only=False):
    rows = result.get('choices', [])
    expected = {str(q['id']): q for q in questions}
    if len(rows) != len(expected) or {str(r.get('question_id')) for r in rows} != set(expected):
        raise ValueError('Reference selection must cover each public question exactly once')
    candidates = {c['candidate_id']: c for c in catalog}
    for r in rows:
        q = expected[str(r['question_id'])]
        c = candidates.get(r.get('candidate_id'))
        if c is None: raise ValueError('Reference must be a supplied cluster')
        role = r.get('reference_role', 'third')
        if allow_endpoints and role in {'first', 'second'}:
            if c.get('endpoint_question_id') != str(q['id']) or c.get('endpoint_role') != role:
                raise ValueError('Endpoint reference must use the same admitted question cluster')
            if c['category'] != q['roles'][role]: raise ValueError('Wrong endpoint category')
        else:
            if role != 'third' or c['category'] in q['objects'] or c.get('endpoint_question_id'):
                raise ValueError('Third reference must be distinct from both endpoints')
            if r.get('distinct_from_question_objects') is not True:
                raise ValueError('Third reference may alias a question object')
        dimensions={'width','length','diameter','height'} if stable_only else {'width','length','diameter'}
        if r.get('dimension') not in dimensions:
            raise ValueError('Reference dimension must be horizontal and visible from above')
        if stable_only and (r.get('dimension_stability') != 'stable' or not str(r.get('stability_reason','')).strip()):
            raise ValueError('Only a dimensionally stable reference is eligible')
        lo, hi = r.get('size_range_m', [None, None])
        if not all(positive(v) for v in [lo, hi, r.get('size_m')]) or not lo <= r['size_m'] <= hi:
            raise ValueError('Invalid common-sense size prior')
        for k in ['reason', 'dimension_description', 'uncertainty']:
            if not isinstance(r.get(k), str) or not r[k].strip():
                raise ValueError('Missing reference rationale: '+k)
    return rows


def prepare(source, out, poses, replace_poses=False):
    """Include a third cluster only when selected; public QA still has two endpoints."""
    ref = reliability.prepare(source, out, poses, replace_poses)
    info = read(source/'scale-reference.json')
    public = info['public_question']
    d = read(out/'input.json')
    if public['question_type'] != 'object_abs_distance':
        raise ValueError('Wrong task for absolute-distance adapter')
    d['public_question'] = public
    d['scale_reference'] = info['choice']
    d['scale_reference_category'] = info['category']
    d['scale_reference_group_id'] = info['group_id']
    save(out/'input.json', d)
    save(out/'question-public.json', public)
    save(out/'scale-reference.json', info)
    prompt = out/'system-prompt.txt'
    prompt.write_text(prompt.read_text()+'''

ABSOLUTE-DISTANCE SCALE-REFERENCE OVERRIDE:
The scale reference may be the FIRST or SECOND question object itself, or a THIRD
physical object. Its role is explicitly supplied below. If it is an endpoint, use
that SAME physical instance and existing cluster; do not create a duplicate reference.
Only if the role is THIRD does the internal object list add one reference category.
That third object is authorized even without direct support/contact with endpoints.
Reconstruct the reference recognizably from its supplied cluster and original images.
The distance question still concerns ONLY the original two endpoints.
Keep the current V10_2 joint scene, shared native pose frame, soft anchors, walls,
direct-support/contact scope and all other reconstruction rules. Respect the final
question-specific instance caps. Do not duplicate the selected reference instance.
Independently reject mislabeled or aliased candidates; never invent a ruler/object.
Preserve image-supported relative dimensions and positions. The common-sense size
prior is approximate, NOT calibrated metric data: do not rescale or force geometry
to that value. Do not add rulers, labels, answer lines or numeric geometry.
The final QA uses an orthographic TOP view. Preserve true vertical positions; do
not flatten the scene or move an object for visibility. Do not solve the question.
'''+ '\nPUBLIC TWO-OBJECT QUESTION = '+json.dumps(public, ensure_ascii=False)
        + '\nSCALE REFERENCE ROLE = '+info.get('reference_role', 'third')
        + '\nSCALE REFERENCE CATEGORY = '+json.dumps(info['category'])
        + '\nREFERENCE CLUSTER = '+info['group_id'])
    exp = read(out/'experiment.json')
    exp.update(revision=REVISION, qa_task='object_abs_distance', qa_view_count=1,
               qa_view_policy='fixed_orthographic_top', metric_scale_calibrated=False,
               third_object_reference=info.get('reference_role', 'third') == 'third',
               reference_role=info.get('reference_role', 'third'), remove_overhead_occluders=True,
               input_sha256=sha(out/'input.json'),
               prompt_sha256=sha(prompt), scale_reference_sha256=sha(out/'scale-reference.json'))
    save(out/'experiment.json', exp)
    return ref


def validate_answer(a, category):
    value = a.get('answer')
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('Distance must be finite and nonnegative')
    if a.get('unit') != 'm' or type(a.get('sufficient_evidence')) is not bool:
        raise ValueError('Invalid distance unit or evidence assessment')
    if a.get('reference_category') != category:
        raise ValueError('QA used a different reference')
    if type(a.get('reference_visible')) is not bool:
        raise ValueError('Missing reference visibility assessment')
    if not a['reference_visible'] and a['sufficient_evidence']:
        raise ValueError('Hidden reference cannot establish sufficient scale evidence')
    for k in ['reason', 'scale_reasoning', 'vertical_limitations', 'limitations']:
        if not isinstance(a.get(k), str) or not a[k].strip():
            raise ValueError('Missing QA explanation: '+k)


def qa(out):
    from subscription_direct_client import request, image_item
    from bvb_v11.common import parse, state
    d = read(out/'input.json'); q = read(out/'question-public.json')
    if not read(out/'coverage-validation.json')['valid']:
        raise ValueError('Both endpoints and the scale reference must pass coverage')
    exp = read(out/'experiment.json')
    if sha(out/'scale-reference.json') != exp['scale_reference_sha256']:
        raise ValueError('Scale reference changed after preparation')
    views = read(out/'views/views.json')
    if exp.get('remove_overhead_occluders'):
        if not views.get('overhead_occlusion_removed'):
            raise ValueError('New strategy requires the overhead visibility pass before QA')
        occlusion = read(out/'occlusion-report.json')
        top_check = next(v for v in occlusion['views'] if v['view_id'] == 'top')
        if top_check.get('remaining_removable_blockers') != 0 or top_check.get('targets_removed'):
            raise ValueError('Incomplete or unsafe overhead visibility pass')
    v = next(v for v in views['views'] if v['id'] == 'top')
    p = out/'views'/v['image']
    assert sha(p) == v['sha256']
    save(out/'view-selection/selection.json', dict(view_id='top', view_ids=['top'],
         policy='fixed_orthographic_top', model_selection=False, reason='用户指定 abs dist 使用俯视图'))
    state(out, 'qa')
    if (out/'qa/answer.json').exists():
        return
    info = read(out/'scale-reference.json')
    # Only selection priors are exposed, never ground truth, mesh coordinates or old answers.
    prior = {k: info['choice'][k] for k in ['dimension', 'dimension_description', 'size_m', 'size_range_m', 'reason', 'uncertainty']}
    prompt = '''Estimate the shortest straight-line 3D distance between the closest
SURFACE points of the TWO public question objects, in meters, using this ONE
orthographic TOP reconstruction and the supplied common-sense size prior.
The reference's explicit role says whether it is endpoint FIRST, endpoint SECOND,
or a THIRD object. An endpoint reference is the same physical object, not an extra
copy. A third reference is not a distance endpoint. Use the two public endpoints
exactly as asked. Match categories using the color legend. Use the
reference's specified HORIZONTAL dimension as an approximate ruler: compare its
rendered extent to the nearest-surface gap. Explain the visual ratio and scale
assumption. Do not substitute center-to-center distance, walking distance, object
height, image width or Blender units. There is no calibrated metric scale.
The prior is an estimate with uncertainty, not a measured physical dimension.
Check the reference is identifiable and its specified edge visible. If occluded,
ambiguous or visibly incompatible with its prior, disclose it, set sufficient_evidence
false, and still provide a best estimate. Do not invent a missing reference.
Top projection can hide vertical separation and overlaps: projected XY gap alone
does not establish the true 3D distance, and XY overlap does not prove contact.
Report vertical ambiguity honestly and lower confidence/evidence accordingly.
Walls and detected higher non-target occluders are hidden for visibility only;
the objects have NOT physically moved or disappeared, and original geometry is
unchanged. White means remaining evidenced floor/direct supports. Do not infer a
missing support, contact, changed height or free passage from this cutaway display.
Use only this scene, question, legend and prior.
Return JSON {"answer":number,"unit":"m","reason":"Chinese",
"reference_category":str,"reference_visible":boolean,"scale_reasoning":"Chinese",
"vertical_limitations":"Chinese","sufficient_evidence":boolean,
"confidence":"low|medium|high","limitations":"Chinese"}.
'''+ '\nPUBLIC QUESTION: '+json.dumps(q, ensure_ascii=False)
    prompt += '\nCATEGORY COLORS: '+json.dumps(d['colors'])
    prompt += '\nSCALE REFERENCE ROLE: '+info.get('reference_role', 'third')+'\nSCALE REFERENCE: '+info['category']+'\nAPPROXIMATE PRIOR: '+json.dumps(prior, ensure_ascii=False)
    a = parse(request(out/'qa', [dict(type='input_text', text=prompt), image_item(p)],
                      model='gpt-6-astra', effort='high')['answer'])
    validate_answer(a, info['category'])
    a.update(view_ids=['top'], revision=REVISION, metric_scale_calibrated=False)
    save(out/'qa/answer.json', a)
