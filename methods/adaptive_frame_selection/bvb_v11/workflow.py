"""Portable orchestration of reviewed caches, recovery, references and task adapters."""
import copy
import fcntl
import hashlib
import re
import subprocess
import sys
from pathlib import Path
from bvb_v10.pipeline import ROOT,read,save,sha
from . import driver,semantic_recovery,abs_distance_context,scale_selection
from .common import state

def catalog(base,sid,selection,extra=None):
    data=read(selection);cache=base/'packets'/sid/'cache'
    frames={r['source_frame_index']:r for r in read(cache/'candidates.json')}
    entries=[];seen=set()
    def add(group,origin,qid=None,role=None):
        if group.get('semantic_review',{}).get('status')!='accept':return
        key=(group['object'],group['group_id'],qid,role)
        if key in seen:return
        seen.add(key)
        for f in group['selected']:
            r=frames[f['source_frame_index']]
            if sha(cache/r['frame'])!=f['frame_sha256']:raise ValueError('Reference cache hash mismatch')
        rep=max(group['selected'],key=scale_selection.frame_score)
        path=cache/frames[rep['source_frame_index']]['frame']
        cid='reference-'+hashlib.sha256(repr(key).encode()).hexdigest()[:16]
        admitted=copy.deepcopy(group)
        admitted['semantic_review']['admission']=dict(selected=True,mode='semantic_accept')
        entries.append(dict(candidate_id=cid,category=group['object'],group=admitted,
            representative=dict(path=str(path),sha256=rep['frame_sha256'],timestamp_s=rep['timestamp_s'],observations=rep.get('observations',[])),
            source_selection=str(origin),source_sha256=sha(origin),endpoint_question_id=qid,endpoint_role=role))
    sources=[(selection,data)]+([(extra,read(extra))] if extra else [])
    for origin,d in sources:
        for row in d['questions'].values():
            if row['question']['scene_name']!=sid:raise ValueError('Reference candidates must be from this scene')
            for g in row['arms'].get('semantic_accepted',row['arms'].get('semantic_selected',[])):add(g,origin)
    for qid,row in data['questions'].items():
        if driver.kind(row['question'])!='abs_dist':continue
        for role in ['first','second']:
            for g in row['arms']['semantic_selected']:
                if g['object']==row['question']['roles'][role]:add(g,selection,str(qid),role)
    save(base/'packets'/sid/'reference-catalog.json',entries)

def run(selection,cache,poses,video,out,blender,reference_selection=None,scale_policy='stable'):
    """One scene and one task family per run. Resume only identical source inputs."""
    sys.path.insert(0,str(ROOT/'scripts'))
    selection,cache,poses,video,out=[p.resolve() for p in [selection,cache,poses,video,out]]
    data=read(selection)
    public=[{k:v for k,v in r['question'].items() if k in {'id','scene_name','question','question_type','options','objects','roles'}} for r in data['questions'].values()]
    scenes={q['scene_name'] for q in public};tasks={driver.kind(q) for q in public}
    if len(scenes)!=1 or len(tasks)!=1 or 'room_size' in tasks:raise ValueError('Use one scene/task family per run; use room for room-size')
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+',str(x)) for q in public for x in [q['id'],q['scene_name']]):raise ValueError('Unsafe scene or question ID')
    for q in public:data['questions'][str(q['id'])]['question']=copy.deepcopy(q)
    sid=next(iter(scenes));task=next(iter(tasks));out.mkdir(parents=True,exist_ok=True)
    with (out/'workflow.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity=dict(selection_sha256=sha(selection),cache=str(cache),poses=str(poses),video_sha256=sha(video),
            scale_policy=scale_policy,reference_sha256=sha(reference_selection) if reference_selection else None)
        if (out/'workflow-input.json').exists() and read(out/'workflow-input.json')!=identity:raise ValueError('Inputs changed; use a fresh output directory')
        save(out/'workflow-input.json',identity)
        cfg=dict(poses={sid:str(poses)},videos={sid:str(video)},scenes=[sid],counting_mode=task=='counting',counting_allow_all_rejected=task=='counting',
            scale_reference_policy='stable_dimensions_only',scale_policy=scale_policy,missing_object_video_fallback=dict(enabled=False))
        save(out/'run-config.json',cfg);save(out/'questions.public.json',public)
        packet=out/'packets'/sid;packet.mkdir(parents=True,exist_ok=True)
        if not (packet/'cache').exists():(packet/'cache').symlink_to(cache,target_is_directory=True)
        source=out/'scenes'/sid/'input/review/reviewed-selection.json'
        if not source.exists():save(source,data)
        references_ready=task=='abs_dist' and all((out/'references'/(str(q['id'])+'.json')).exists() for q in public)
        recovered=source if references_ready else semantic_recovery.ensure_question_selection(out,str(public[0]['id']))
        if task!='abs_dist' and recovered!=source:
            save(source,read(recovered));recovered=source
        if task=='abs_dist' and not references_ready:
            save(packet/'target-selection.json',read(recovered));catalog(out,sid,recovered,reference_selection)
            scale_selection.BASE=out;scale_selection.audit(sid,allow_endpoints=True);recovered=source
        failures=[]
        for q in public:
            qid=str(q['id']);destination=out/'questions'/qid
            try:
                if (destination/'qa/answer.json').exists():continue
                current=recovered;row=read(current)['questions'][qid];zero=[]
                if task=='counting' and row.get('coverage_missing_categories'):
                    from .candidate_recovery import recover_question
                    current=recover_question(out,qid,row['coverage_missing_categories']);row=read(current)['questions'][qid]
                    zero=row.get('counting_zero_evidence_categories',[])
                    if not row['arms']['semantic_selected']:
                        from .counting import validate_answer
                        answer=dict(answer=0,counts_by_category=[dict(category=c,count=0) for c in q['objects']],sufficient_evidence=False,
                            confidence='low',limitations='No supported instances after completed full-video recovery; not proof of true absence.',
                            result_source='full_video_fallback_no_supported_instances',view_ids=[],declared_counts_provided=False)
                        validate_answer(answer,q['objects']);save(destination/'question-public.json',q);save(destination/'qa/answer.json',answer);state(destination,'completed');continue
                prepared=out/'prepared'/qid
                if not (prepared/'manifest.json').exists():
                    subprocess.run([sys.executable,'-m','bvb_v6','prepare','--selection',str(current),'--qid',qid,'--cache',str(packet/'cache'),'--poses',str(poses),'--out',str(prepared)],cwd=ROOT,check=True)
                    save(prepared/'semantic-source.json',dict(path=str(current),sha256=sha(current)))
                if zero:save(prepared/'counting-zero-evidence.json',zero)
                context=None
                if task=='abs_dist':
                    save(prepared/'scale-reference.json',read(out/'references'/(qid+'.json')))
                    context=out/'contexts'/(qid+'.json')
                    if not context.exists():save(context,abs_distance_context.plan(q,row['arms']['semantic_selected'],packet/'cache',poses,interval_seconds=2.))
                if not (destination/'input.json').exists():driver.prepare(prepared,destination,poses,context=context)
                driver.run(destination,blender)
            except Exception as exc:
                state(destination,'failed',error=str(exc));failures.append(qid)
        state(out,'completed' if not failures else 'needs-attention',failed_questions=failures)
        if failures:raise RuntimeError('Failed question IDs: '+', '.join(failures))
