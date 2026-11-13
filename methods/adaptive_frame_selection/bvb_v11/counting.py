"""Counting adapter: same reconstruction, three views, no count hints or GT."""
import json
from . import reliability
from bvb_v10.pipeline import read, save, sha

TYPES={'object_counting_single','object_counting_multiple'}

def prepare(source,out,poses,replace_poses=False,counting_mode=False,allow_all_rejected=False):
    ref=reliability.prepare(source,out,poses,replace_poses)
    d=read(out/'input.json');assert d['question']['question_type'] in TYPES
    assert all(v is None for v in d['instance_limits'].values())
    from .counting_mode import enabled
    active=enabled(d['question'],counting_mode)
    d['counting_mode']=active
    zero=read(source/'counting-zero-evidence.json') if (source/'counting-zero-evidence.json').exists() else []
    if zero and not allow_all_rejected:raise ValueError('Zero evidence requires counting rejection policy')
    d['counting_allow_all_rejected']=allow_all_rejected
    d['counting_zero_evidence_categories']=zero
    save(out/'input.json',d)
    extra='''
COUNTING TASK: Reconstruct ALL visually supported physical instances of every listed
question category across the supplied frames. A single-category question can have many
instances. Preserve distinct objects even when they look similar. Deduplicate repeat
views of the same object using image context and the shared trajectory; neither one
peak nor one cluster implies one physical object. Use separate instance_id values for
different physical objects; all mesh parts of one object share its instance_id.
Do not infer or fabricate a required number from the question. Do not remove an object
just because it is not among the peak's primary targets. Document unresolved visibility.
'''
    p=out/'system-prompt.txt';p.write_text(p.read_text()+extra)
    if allow_all_rejected:
        p.write_text(p.read_text()+'''\nCOUNTING ZERO-EVIDENCE POLICY (priority): A question does not prove an object exists.
Explicitly rejected clusters are not rescued for minimum coverage. Categories listed
below had no reliable candidates after full-video 1fps recovery and may remain absent.
Never fabricate them to satisfy the earlier minimum-one coverage instruction. If other
supplied images genuinely show them, retain those image-supported instances.
ZERO-EVIDENCE CATEGORIES = '''+json.dumps(zero)+'\n')
    if active:
        p.write_text(p.read_text()+'''\nCOUNTING DISPLAY MODE: Build targets in the category colors specified above.
After construction the runtime will recolor distinct nearby physical instances to
different colors. Keep every mesh part of one object under its same instance_id.
Do not split an object or invent instances just to obtain more colors.\n''')
    exp=read(out/'experiment.json');exp.update(qa_task='object_counting',qa_declared_counts_provided=False,
        counting_mode=active,instance_color_mode='nearest-neighbour' if active else 'category',
        input_sha256=sha(out/'input.json'),prompt_sha256=sha(p));save(out/'experiment.json',exp)
    return ref

def validate_answer(a,categories):
    if type(a.get('answer')) is not int or a['answer']<0:raise ValueError('Count must be a nonnegative integer')
    rows=a.get('counts_by_category',[])
    if len(rows)!=len(categories) or {r.get('category') for r in rows}!=set(categories):raise ValueError('Count every requested category')
    if any(type(r.get('count')) is not int or r['count']<0 for r in rows):raise ValueError('Invalid per-category count')
    if sum(r['count'] for r in rows)!=a['answer']:raise ValueError('Total differs from category counts')
    if type(a.get('sufficient_evidence')) is not bool:raise ValueError('Missing evidence assessment')

def qa(out):
    from subscription_direct_client import request,image_item
    from bvb_v11.common import parse,state
    d=read(out/'input.json');q=d['question'];assert q['question_type'] in TYPES
    assert read(out/'coverage-validation.json')['valid']
    views=read(out/'views/views.json')['views'];byid={v['id']:v for v in views}
    # Deliberately do not send coverage-validation.instance counts or scene code.
    context=json.dumps(q,ensure_ascii=False)+'\nCategory RGBA colors (0..1): '+json.dumps(d['colors'])+'''
Same category has the same color. Different physical objects of one category may
share a color. White geometry is floor/architecture/support/contact, not an extra
target. Walls/doorway boundaries are hidden for visibility, not removed from the
stored scene. Three views show the SAME scene, not three separate collections.
'''
    if d.get('counting_mode'):
        legend=read(out/'qa-color-legend.json')
        context=json.dumps(q,ensure_ascii=False)+'\nCategory color palettes (RGBA 0..1): '+json.dumps(legend)+'''
Different nearby instances of the same category have different colors. All mesh parts
of one physical object keep the same color across views. Non-neighbour instances may
reuse a color, so neither number of colors nor number of mesh parts equals the count.
Use category palettes for identity, and spatial boundaries for physical instances.
White is floor/architecture/support/contact. Walls are hidden for visibility.
The three views show the SAME scene. No instance inventory or count is provided.
'''
    context+='\nHigher non-target occluders may be hidden separately per view. All question target instances remain; hiding a support/context object does not remove or add any target instance.'
    if not (out/'view-selection/selection.json').exists():
        prompt='''Select EXACTLY THREE DISTINCT complementary views for counting ALL
instances of the requested categories. At least two non-top views must have different
azimuths. Prefer visible separation, low occlusion, coverage of shelves and heights.
Do not answer the count. Return JSON {"view_ids":[id,id,id],"reason":"Chinese",
"limitations":"Chinese","coverage":[{"category":str,"visible_in":[view_id],
"uncertainty":str}]}. Include every category, using only chosen view IDs.\n'''+context
        content=[dict(type='input_text',text=prompt)]
        for v in views:
            p=out/'views'/v['image'];assert sha(p)==v['sha256']
            content.extend([dict(type='input_text',text='VIEW '+v['id']),image_item(p)])
        a=parse(request(out/'view-selection',content,model='gpt-6-astra',effort='high')['answer'])
        reliability.validate_views(a,byid,q['objects']);a['view_id']=a['view_ids'][0]
        save(out/'view-selection/selection.json',a)
    chosen=read(out/'view-selection/selection.json');reliability.validate_views(chosen,byid,q['objects'])
    state(out,'qa')
    if not (out/'qa/answer.json').exists():
        prompt='''Answer the counting question using ONLY these THREE rendered views,
the public question and color legend. Integrate all views jointly. Track the same
object across views without counting it repeatedly. Count physical objects, not mesh
parts, repeated views, supports, reflections, or shadows. Distinct same-category
objects must all be counted. For multiple requested categories report their TOTAL.
Explain each category's count using spatial descriptions of visible objects; disclose
occlusion or uncertain identity. The reconstruction itself may omit or duplicate
objects: do not assume a hidden exact inventory. Give your best nonnegative integer.
Return JSON {"answer":integer,"counts_by_category":[{"category":str,"count":integer,
"evidence":"Chinese spatial description"}],"reason":"Chinese","confidence":
"low|medium|high","sufficient_evidence":boolean,"limitations":"Chinese",
"cross_view_consistency":"Chinese"}. Total must equal the sum of category counts.\n'''+context
        content=[dict(type='input_text',text=prompt)]
        for vid in chosen['view_ids']:
            v=byid[vid];p=out/'views'/v['image'];assert sha(p)==v['sha256']
            content.extend([dict(type='input_text',text='VIEW '+vid),image_item(p)])
        a=parse(request(out/'qa',content,model='gpt-6-astra',effort='high')['answer'])
        validate_answer(a,q['objects']);a.update(view_ids=chosen['view_ids'],revision=reliability.REVISION,declared_counts_provided=False)
        save(out/'qa/answer.json',a)
