"""Width-only scale selection for the two-second structural-context experiment."""
import copy
import json
import time
from pathlib import Path
from bvb_v10.pipeline import read,save,sha
from . import abs_distance,abs_distance_context as context

REVISION='20261112-abs-distance-between-context-2s-width-prefer-large'
PROMPT='''For EACH supplied absolute-distance question select ONE real supplied
object cluster with a STABLE physical WIDTH as a common-sense scale reference.
WIDTH IS THE ONLY ELIGIBLE DIMENSION. Do not use height, depth, front-to-back
length, diameter, radius, circumference, diagonal or an arbitrary longest edge.
Do not relabel a wheel diameter, chair seat height or another excluded dimension
as width. For the identified physical subtype, use the natural left-to-right
width across its front, specifying the actual physical part and both endpoints.
First restrict eligibility to rigid objects whose physical width varies little
across plausible products of the visually supported subtype. WITHIN THESE ELIGIBLE
CANDIDATES, choose the LARGEST PHYSICAL WIDTH POSSIBLE. Prefer a broad, large
reference with clearly identifiable left and right edges, so it provides a long
visual ruler. Compare actual physical widths, not apparent pixel size, camera
proximity, object height, volume or overall longest dimension. Do not pick a small
reference when a larger, comparably stable and clearly evidenced width is available.
Keep dimensional stability and correct identity as requirements: do not invent a
stable width for a large variable-size object. If the largest candidate's width is
ambiguous or cannot be identified, choose the largest reliable eligible alternative.
Inspect the clean representative images and detection hints. You may select the
FIRST or SECOND question object itself, using the exact candidate bound to that
question and role, or a THIRD physical object distinct from both endpoints and
their aliases/parts. Do not introduce a duplicate endpoint reference.
Give an honest common-sense width estimate and plausible interval, never measured
calibration or a falsely narrow interval. No ground truth, prior answer or prior
reference choice is supplied. Do not answer the distance question.
If no supplied candidate has a defensibly stable width, set candidate_id=null and
dimension_stability="unavailable"; do not fall back to a different dimension.
Return JSON {"choices":[{"question_id":str,"candidate_id":str|null,
"reference_role":"first|second|third","distinct_from_question_objects":boolean,
"dimension":"width","dimension_description":"Chinese exact left/right edges",
"dimension_stability":"stable|unavailable","stability_reason":"Chinese",
"size_m":number,"size_range_m":[lower,upper],"reason":"Chinese",
"uncertainty":"Chinese"}]}. Exactly one entry for each supplied question.
For THIRD distinct_from_question_objects=true; for an endpoint it is false.
'''


def validate_choices(result,questions,catalog):
    rows=abs_distance.validate_choices(result,questions,catalog,allow_endpoints=True,stable_only=True)
    if any(r['dimension']!='width' for r in rows):
        raise ValueError('Width-only references cannot use height, length or diameter')
    return rows


def audit(base,sid):
    from subscription_direct_client import request,image_item
    from bvb_v11.common import parse,state
    packet=base/'packets'/sid;out=base/'scenes'/sid
    catalog=read(packet/'reference-catalog.json')
    rerun=set(read(base/'run-config.json')['rerun_question_ids'])
    questions=[q for q in read(base/'questions.public.json') if q['scene_name']==sid and str(q['id']) in rerun]
    if not questions:raise ValueError('Reference selection is only authorized for non-width questions')
    folder=out/'reference-selection';path=folder/'selection.json'
    state(out,'selecting-width-reference')
    if not path.exists():
        ledger=folder/'request.json'
        if ledger.exists():
            if read(ledger)['status']=='started':raise RuntimeError('Potentially live width-reference request')
            folder.rename(out/('reference-selection-failed-'+str(time.time_ns())))
        content=[dict(type='input_text',text=PROMPT+'\nPUBLIC QUESTIONS: '+json.dumps(questions,ensure_ascii=False))]
        for c in catalog:
            r=c['representative'];p=Path(r['path']);assert sha(p)==r['sha256']
            content.extend([dict(type='input_text',text=json.dumps(dict(candidate_id=c['candidate_id'],category=c['category'],
                endpoint_question_id=c.get('endpoint_question_id'),endpoint_role=c.get('endpoint_role'),
                timestamp_s=r['timestamp_s'],observations=r['observations']),ensure_ascii=False)),image_item(p)])
        response=request(folder,content,model='gpt-6-astra',effort='high')
        try:
            result=parse(response['answer']);validate_choices(result,questions,catalog)
        except (ValueError,TypeError,KeyError) as exc:
            save(folder/'invalid-output.json',dict(error=str(exc)));raise
        save(path,result)
    choices=validate_choices(read(path),questions,catalog);byid={c['candidate_id']:c for c in catalog}
    target=read(packet/'target-selection.json');refs={}
    for choice in choices:
        qid=str(choice['question_id']);c=byid[choice['candidate_id']];row=target['questions'][qid]
        public=next(q for q in questions if str(q['id'])==qid);role=choice['reference_role']
        row['question']['objects']=list(public['objects'])
        if role=='third':
            row['question']['objects'].append(c['category']);row['question'].setdefault('instance_limits',{})[c['category']]=1
            row['arms']['semantic_selected'].append(copy.deepcopy(c['group']))
        refs[qid]=dict(public_question=public,category=c['category'],reference_role=role,group_id=c['group']['group_id'],
            choice=choice,candidate_source=c['source_selection'],candidate_source_sha256=c['source_sha256'],
            selection_sha256=sha(path),reference_frames=len(c['group']['selected']),metric_scale_calibrated=False,
            scale_reference_policy='stable_width_only',reference_size_preference='largest_reliable_stable_width')
    cfg=read(base/'run-config.json');target['inputs'].update(cache=str(packet/'cache'),poses=cfg['poses'][sid])
    selection=out/'input/review/reviewed-selection.json';save(selection,target)
    save(packet/'selection-remote.json',target)
    for qid,ref in refs.items():
        p=base/'references'/(qid+'.json');save(p,ref)
        cp=base/'contexts'/(qid+'.json');plan=read(cp)
        plan.update(source_selection_sha256=sha(selection),source_reference_sha256=sha(p));save(cp,plan)
    state(out,'completed',reference_questions=len(choices),width_only=True)


def prepare(source,out,poses,context_plan):
    info=read(source/'scale-reference.json');assert info['choice']['dimension']=='width'
    context.REVISION=REVISION
    result=context.prepare(source,out,poses,context_plan)
    p=out/'system-prompt.txt'
    text=p.read_text().replace(
        'Reference selection prioritizes dimension stability only, not size or width. Its specified dimension may be vertical.',
        'The scale reference uses ONLY its specified stable physical WIDTH, across the identified front from left to right. Do not substitute height, depth, length, diameter or diagonal.')
    p.write_text(text)
    exp=read(out/'experiment.json');exp.update(scale_reference_policy='stable_width_only',reference_dimension='width',prompt_sha256=sha(p))
    save(out/'experiment.json',exp)
    return result


def qa(out):
    assert read(out/'scale-reference.json')['choice']['dimension']=='width'
    context.REVISION=REVISION
    context.qa(out)
    a=read(out/'qa/answer.json');a['reference_dimension']='width';save(out/'qa/answer.json',a)
