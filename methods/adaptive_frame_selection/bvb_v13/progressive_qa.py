"""Answer with one rendered view first; expand evidence only on explicit uncertainty."""
import hashlib,json
from pathlib import Path
import numpy as np
from bvb_v10.pipeline import read,save,sha
from .model_images import prepare as prepare_image

POLICY='progressive-answer-only-half-resolution-v3'

class InsufficientEvidence(RuntimeError):
    """All available views were checked without a supported answer."""
    def __init__(self,result):
        super().__init__('No sufficient evidence after all available views')
        self.result=result

def completion_stage(out):
    p=Path(out)/'qa/answer.json'
    return 'insufficient-evidence' if p.exists() and read(p).get('status')=='insufficient_evidence' else 'completed'


def order_views(out,views,task):
    """Geometry-only greedy coverage/complementarity; no answer/GT inputs."""
    byid={v['id']:v for v in views}
    if not byid or len(byid)!=len(views):raise ValueError('Need distinct available QA views')
    preferred=[];p=out/'view-selection/selection.json'
    if p.exists():
        d=read(p);preferred=d.get('view_ids') or [d['view_id']]
    rank={v:i for i,v in enumerate(dict.fromkeys(preferred+list(byid)))}
    p=out/'view-selection/geometry-scores.json'
    scores={s['id']:s for s in read(p)['candidates']} if p.exists() else {}
    ordered=[]
    if task=='route':
        if 'top' not in byid:raise ValueError('Route needs an available top view')
        ordered=['top']
    while len(ordered)<len(byid):
        remaining=[v for v in byid if v not in ordered]
        def objective(vid):
            if vid not in scores:return (-1.,-rank[vid])
            s=scores[vid];q=np.asarray(s['quality'],float)
            union=np.max([q]+[scores[x]['quality'] for x in ordered if x in scores],axis=0)
            diversity=min([(1.-float(np.dot(s['camera_direction'],scores[x]['camera_direction'])))/2 for x in ordered if x in scores],default=0.)
            return (4*float(union.min())+2*float(union.mean())+.5*float(q.mean())+.25*diversity,-rank[vid])
        ordered.append(max(remaining,key=objective))
    return [byid[vid] for vid in ordered]

def answer(out,prompt,validator,*,views=None,task=None,request_fn=None,question=None,original_images=None):
    from subscription_direct_client import request,image_item
    from .answer_text import build as answer_prompt,parse as parse_answer,question_for,MORE
    out=Path(out)
    if (out/'qa/answer.json').exists():
        saved=read(out/'qa/answer.json')
        if saved.get('status')=='insufficient_evidence':raise InsufficientEvidence(saved)
        return saved
    if (out/'qa/request.json').exists():raise ValueError('Legacy QA receipt exists; use a fresh QA directory')
    if task is None:
        from .driver import kind
        task=kind(read(out/'question-public.json'))
    if views is None:
        views=[dict(id=v['id'],path=str(out/'views'/v['image']),sha256=v['sha256']) for v in read(out/'views/views.json')['views']]
    for v in views:
        if sha(v['path'])!=v['sha256']:raise ValueError('QA candidate image changed')
    ordered=order_views(out,views,task)
    originals=original_images or []
    if originals and task!='object_size':raise ValueError('Original QA evidence is only enabled for object size')
    for f in originals:
        if sha(f['path'])!=f['sha256']:raise ValueError('Original QA image changed')
    question=question_for(out,question)
    upstream_prompt_sha=hashlib.sha256(prompt.encode()).hexdigest()
    prompt=answer_prompt(out,question,task)
    plan=dict(policy=POLICY,task=task,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        upstream_prompt_sha256=upstream_prompt_sha,
        candidates=[dict(id=v['id'],sha256=v['sha256']) for v in ordered],image_scale=.5,
        escalation='NEED_MORE_VIEWS',uncertain_output='NEED_MORE_VIEWS',
        answer_output='bare option letter or number',
        transport='independent cumulative-view requests; previous answers are not inputs')
    planfile=out/'qa/progressive-plan.json'
    if originals:
        from .size_originals import POLICY as original_policy
        plan.update(original_policy=original_policy,original_candidates=originals,original_image_scale=1,
                    evidence_schedule='first render + first original; then alternate additional originals and renders')
    if planfile.exists() and read(planfile)!=plan:raise ValueError('Progressive QA inputs changed; use a fresh QA directory')
    save(planfile,plan);steps=[];sent=[];invoke=request_fn or request
    batches=[[v] for v in ordered]
    if originals:
        batches=[[ordered[0],originals[0]]]
        for i in range(1,max(len(ordered),len(originals))):
            if i<len(originals):batches.append([originals[i]])
            if i<len(ordered):batches.append([ordered[i]])
    for n,batch in enumerate(batches,1):
        for v in batch:
            sent.append(dict(v) if v.get('kind')=='original' else dict(id=v['id'],**prepare_image(v['path'],out/'model-images')))
        ids=[x['id'] for x in sent];remaining=len(batches)-n
        content=[dict(type='input_text',text=prompt+'\nSupplied view IDs: '+json.dumps(ids)+'; remaining unseen candidates: '+str(remaining))]
        for image in sent:
            label=('ORIGINAL VIDEO FRAME '+image['id']+' timestamp_s='+str(image['timestamp_s'])) if image.get('kind')=='original' else ('RECONSTRUCTED VIEW ' if originals else 'VIEW ')+image['id']
            content.extend([dict(type='input_text',text=label),image_item(image['path'])])
        folder=out/'qa'/f'step-{n:02d}';save(folder/'image-inputs.json',dict(images=sent))
        raw=invoke(folder,content,model='gpt-6-astra',effort='high')['answer']
        result=parse_answer(raw,question,task)
        more=result.get('need_more_views') is True
        short=raw.strip()==MORE
        # Preserve raw model text/usage in request.json; never publish a provisional answer.
        step_result={'need_more_views':True} if more else result
        save(folder/'answer.json',dict(step_result,view_ids=ids))
        steps.append(dict(step=n,view_ids=list(ids),needs_more=more,request_path=str(folder/'request.json'),
                          short_response=short,uncertain_full_response=bool(more and not short)))
        if more and remaining:continue
        if more:
            final=dict(status='insufficient_evidence',answer=None,need_more_views=True,
                confused=True,sufficient_evidence=False,confidence='low',
                limitations='所有可用视角仍不足以可靠回答；需要额外证据。',
                view_ids=ids,qa_policy=POLICY,image_scale=.5,qa_steps=n,
                candidate_views=len(ordered),evidence_exhausted=True,stopping_reason='views_exhausted')
            if originals:final.update(original_frame_ids=[x['id'] for x in sent if x.get('kind')=='original'],original_image_scale=1)
            save(out/'qa/progressive-summary.json',dict(plan=plan,steps=steps,stopping_reason=final['stopping_reason']))
            save(out/'qa/answer.json',final)
            raise InsufficientEvidence(final)
        # Answer types/options were validated above. Legacy rationale schemas are not model outputs.
        final=dict(result,view_ids=ids,qa_policy=POLICY,image_scale=.5,qa_steps=n,
            candidate_views=len(ordered),evidence_exhausted=False,stopping_reason='sufficient_evidence')
        if originals:final.update(original_frame_ids=[x['id'] for x in sent if x.get('kind')=='original'],original_image_scale=1)
        save(out/'qa/progressive-summary.json',dict(plan=plan,steps=steps,stopping_reason=final['stopping_reason']))
        (out/'qa/answer.txt').write_text(result['plain_answer']+'\n')
        return final
    raise RuntimeError('No QA candidate')
