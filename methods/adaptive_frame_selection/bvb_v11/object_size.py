"""Current V10_2 reconstruction + three-view numeric object-size QA (centimeters)."""
import json
import math
from . import reliability
from bvb_v10.pipeline import read, save, sha

EXTRA = '''
OBJECT SIZE TASK: preserve the target's complete shape and relative proportions,
including any genuine dimension-bearing parts visible in the input. Reconstruct
only the question target in detail, all evidenced walls/openings/floors, and direct
supports/contacts as white blocks. Keep image-supported relative size cues.
Do not add a standard door, ruler or unrelated reference object merely to create a
scale. LingBot geometry has uncalibrated global scale; Blender units are not measured
meters. Do not encode an estimated answer as text, ruler or forced target dimension.
Candidate labels may be false positives: independently verify target identity from
the original frames. Follow question-specific instance limits and explain ambiguities.
'''


def prepare(source, out, poses, replace_poses=False):
    ref = reliability.prepare(source, out, poses, replace_poses)
    q = read(out/'question-public.json')
    if q['question_type'] != 'object_size_estimation': raise ValueError('Wrong question type')
    prompt = out/'system-prompt.txt'
    prompt.write_text(prompt.read_text()+EXTRA)
    exp = read(out/'experiment.json')
    exp.update(qa_task='object_size_estimation', qa_unit='cm', metric_scale_calibrated=False,
               prompt_sha256=sha(prompt), qa_view_count=3)
    save(out/'experiment.json', exp)
    return ref


def validate_answer(a):
    value = a.get('answer')
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('Size must be a finite nonnegative numeric estimate')
    if a.get('unit') != 'cm': raise ValueError('Size QA must return centimeters')
    if type(a.get('sufficient_evidence')) is not bool: raise ValueError('Missing evidence assessment')
    if not isinstance(a.get('reason'), str) or not a['reason'].strip(): raise ValueError('Missing rationale')


def qa(out):
    from subscription_direct_client import request, image_item
    from bvb_v11.common import parse, state
    d = read(out/'input.json'); q = d['question']
    if q['question_type'] != 'object_size_estimation': raise ValueError('Wrong question type')
    if not read(out/'coverage-validation.json')['valid']: raise ValueError('Missing target geometry')
    views = read(out/'views/views.json')['views']; byid = {v['id']:v for v in views}
    context = json.dumps(q, ensure_ascii=False)+'\nTarget RGBA colors: '+json.dumps(d['colors'])
    context += '\nAll views show the same reconstruction. White geometry represents evidenced floors and direct supports/contacts. Walls are hidden for QA visibility only. LingBot poses and the model have no calibrated metric scale.'
    context += ' Higher non-target occluders may also be hidden per view. This cutaway does not change target shape, size, height or physical support/contact; targets remain present.'
    if not (out/'view-selection/selection.json').exists():
        prompt = 'Select EXACTLY THREE DISTINCT complementary views that reveal the complete target and its longest dimension, including relevant supports and size cues. Use at least two non-top views with different azimuths. Avoid foreshortening and occlusion. Do not answer the size question yet. Return JSON {"view_ids":[id,id,id],"reason":"Chinese","limitations":"Chinese","coverage":[{"category":str,"visible_in":[selected_view_id],"uncertainty":str}]}. Report visibility for every question category, and only these categories.\n'+context
        content = [dict(type='input_text',text=prompt)]
        for v in views:
            p = out/'views'/v['image']; assert sha(p)==v['sha256']
            content.extend([dict(type='input_text',text='VIEW '+v['id']), image_item(p)])
        a = parse(request(out/'view-selection',content,model='gpt-6-astra',effort='high')['answer'])
        reliability.validate_views(a,byid,q['objects'])
        a['view_id'] = a['view_ids'][0]; save(out/'view-selection/selection.json',a)
    chosen = read(out/'view-selection/selection.json')
    reliability.validate_views(chosen,byid,q['objects']); state(out,'qa')
    if not (out/'qa/answer.json').exists():
        prompt = '''Estimate the length of the target's LONGEST dimension (length,
width, or height), in CENTIMETERS, using ONLY the three supplied reconstructed views,
the public question and category colors. Integrate the views jointly; do not vote
independently. Match the same physical target across views. Account for foreshortening
and distinguish visible parts from additional instances. Use shape/proportion and
recognizable size priors where necessary, but disclose assumptions. There is NO
metric calibration: never treat Blender units, the image dimensions, or a supposed
standard door as measured centimeters. Do not invent a ruler or measured reference.
If the intended instance or scale is ambiguous, explain the ambiguity and return
your best nonnegative finite numeric estimate with sufficient_evidence=false.
Return JSON {"answer":number,"unit":"cm","reason":"Chinese concise explanation",
"sufficient_evidence":boolean,"confidence":"low|medium|high","limitations":"Chinese",
"longest_dimension":"length|width|height|uncertain","scale_basis":"Chinese assumptions",
"cross_view_consistency":"Chinese concise explanation"}.\n'''+context
        content = [dict(type='input_text',text=prompt)]
        for vid in chosen['view_ids']:
            v = byid[vid]; p = out/'views'/v['image']; assert sha(p)==v['sha256']
            content.extend([dict(type='input_text',text='VIEW '+vid),image_item(p)])
        a = parse(request(out/'qa',content,model='gpt-6-astra',effort='high')['answer'])
        validate_answer(a)
        a.update(view_ids=chosen['view_ids'],revision=reliability.REVISION,metric_scale_calibrated=False)
        save(out/'qa/answer.json',a)
