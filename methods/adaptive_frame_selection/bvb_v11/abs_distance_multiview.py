"""Stable scale prior plus current V10_2 three-view selection and joint metric QA."""
import json
from bvb_v10.pipeline import read,save,sha
from . import abs_distance,reliability

REVISION='20261112-strict-semantic-stable-scale-threeview'


def prepare(source,out,poses):
    ref=abs_distance.prepare(source,out,poses)
    p=out/'system-prompt.txt'
    text=p.read_text().replace('The final QA uses an orthographic TOP view.',
        'Final QA selects THREE complementary reconstructed views, including different azimuths and useful elevations.')
    text+='\nReference selection prioritizes dimension stability only, not size or width. Its specified dimension may be vertical. Preserve full three-dimensional shape, vertical placement and relative geometry; do not flatten to a top view.\n'
    p.write_text(text)
    exp=read(out/'experiment.json');exp.update(revision=REVISION,qa_view_count=3,
        qa_view_policy='astra_three_complementary',scale_reference_policy='stable_dimensions_only',
        semantic_recovery_policy='reject_and_uncertain_all_tasks',prompt_sha256=sha(p))
    save(out/'experiment.json',exp)
    return ref


def qa(out):
    from subscription_direct_client import request,image_item
    from bvb_v11.common import parse,state
    d=read(out/'input.json');q=read(out/'question-public.json');info=read(out/'scale-reference.json');exp=read(out/'experiment.json')
    assert read(out/'coverage-validation.json')['valid']
    assert sha(out/'scale-reference.json')==exp['scale_reference_sha256']
    manifest=read(out/'views/views.json');assert manifest['overhead_occlusion_removed']
    byid={v['id']:v for v in manifest['views']}
    prior={k:info['choice'][k] for k in ['dimension','dimension_description','size_m','size_range_m','stability_reason','uncertainty']}
    context='PUBLIC QUESTION: '+json.dumps(q,ensure_ascii=False)+'\nCATEGORY RGBA COLORS: '+json.dumps(d['colors'])
    context+='\nGEOMETRY CATEGORIES REQUIRING COVERAGE: '+json.dumps(d['question']['objects'])
    context+='\nSCALE REFERENCE: '+json.dumps(dict(category=info['category'],role=info['reference_role'],prior=prior),ensure_ascii=False)
    context+='\nAll images show the SAME geometry and physical positions. White is architecture/support/contact. Walls and higher non-target occluders may be hidden independently per view. This cutaway display does not change physical height, support, contact or paths. All question targets remain. Scale is uncalibrated.'
    if not (out/'view-selection/selection.json').exists():
        prompt='''Select EXACTLY THREE DISTINCT complementary supplied views, with
at least two non-top views of different azimuths, following the usual V10_2 view
selection policy. Top is optional. Prioritize joint visibility of BOTH distance
endpoints, their nearest surfaces, height relationships, and the scale reference's
EXACT specified dimension. Avoid redundant views and severe foreshortening. Prefer
a view where that complete dimension is visible; use complementary views to resolve
occlusion and vertical separation. Do not solve the question or estimate distance.
Return JSON {"view_ids":[id,id,id],"reason":"Chinese","limitations":"Chinese",
"coverage":[{"category":str,"visible_in":[selected_view_id],"uncertainty":str}]}.
Include every category in GEOMETRY CATEGORIES REQUIRING COVERAGE exactly once,
including a third reference if any. Do not add scene context or support categories.
'''+context
        content=[dict(type='input_text',text=prompt)]
        for v in byid.values():
            p=out/'views'/v['image'];assert sha(p)==v['sha256']
            content.extend([dict(type='input_text',text='VIEW '+v['id']),image_item(p)])
        selection=parse(request(out/'view-selection',content,model='gpt-6-astra',effort='high')['answer'])
        reliability.validate_views(selection,byid,d['question']['objects'])
        selection['view_id']=selection['view_ids'][0];save(out/'view-selection/selection.json',selection)
    selected=read(out/'view-selection/selection.json');reliability.validate_views(selected,byid,d['question']['objects'])
    state(out,'qa')
    if (out/'qa/answer.json').exists():return
    prompt='''Estimate the shortest straight-line THREE-DIMENSIONAL distance between
the closest SURFACE points of the two public question objects, in meters. Integrate
these THREE complementary views jointly, not independent voting. Use the specified
stable physical dimension of the reference as an approximate scale prior. It may be
endpoint FIRST/SECOND itself or a separate THIRD object; never add a third endpoint.
Identify the same instances across images. Use complementary views to assess height,
depth, orientation and occluded nearest surfaces. Do not silently equate a top-view
XY gap with the true 3D distance or assume equal height. Avoid center-to-center or
walking distance. Account for foreshortening: compare lengths within a suitable view;
different views have independently fitted orthographic scales, so raw pixels across
different images are not interchangeable. The prior is uncertain and not calibration.
If the specified reference dimension is not visible in any selected image, say so
and set sufficient_evidence=false; do not invent measured edges or use an unrelated
object as its replacement. Return a best finite nonnegative estimate with honest
uncertainty when evidence is insufficient. Do not infer missing physical supports or
changed height from cutaway display. Use only the supplied views, question, legend
and prior. No model coordinates, old answers or GT are provided.
Return JSON {"answer":number,"unit":"m","reason":"Chinese",
"reference_category":str,"reference_visible":boolean,"reference_visible_in":[view_id],
"scale_reasoning":"Chinese","vertical_limitations":"Chinese height assessment and remaining uncertainty",
"cross_view_consistency":"Chinese","sufficient_evidence":boolean,
"confidence":"low|medium|high","limitations":"Chinese"}.
'''+context
    content=[dict(type='input_text',text=prompt)]
    for vid in selected['view_ids']:
        v=byid[vid];p=out/'views'/v['image'];assert sha(p)==v['sha256']
        content.extend([dict(type='input_text',text='VIEW '+vid),image_item(p)])
    a=parse(request(out/'qa',content,model='gpt-6-astra',effort='high')['answer'])
    abs_distance.validate_answer(a,info['category'])
    if not set(a.get('reference_visible_in',[]))<=set(selected['view_ids']):raise ValueError('Reference visibility uses unselected view')
    if bool(a.get('reference_visible_in'))!=a['reference_visible']:raise ValueError('Reference visibility inconsistency')
    if not a.get('cross_view_consistency'):raise ValueError('Missing joint-view assessment')
    a.update(view_ids=selected['view_ids'],revision=REVISION,metric_scale_calibrated=False)
    save(out/'qa/answer.json',a)
