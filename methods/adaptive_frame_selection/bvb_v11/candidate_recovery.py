"""Shared per-video/category 1fps recovery, followed by clean-image identity review.
Only public questions and exact cached frames enter inference. A proposal is never
admitted merely because the search model emitted it.
"""
import copy,fcntl,hashlib,json,os,shlex,shutil,subprocess,tarfile
from pathlib import Path
from bvb_v10.pipeline import ROOT,read,save,sha
from .video_fallback import locate
from .retrieval_policy import FULL_VIDEO_RETRIEVAL_ENABLED
from .common import video_path

def canonical_frames(base,sid):
    packet=base/'packets'/sid;cache=packet/'cache';out=base/'fallback'/sid
    out.mkdir(parents=True,exist_ok=True)
    with (base/'fallback/packet-detach.lock').open('a') as detach_lock:
        fcntl.flock(detach_lock,fcntl.LOCK_EX)
        for shared in [base/'packets',packet]:
            if shared.is_symlink():
                detached=shared.with_name(shared.name+'-local-copy')
                if not detached.exists():shutil.copytree(shared,detached,copy_function=os.link)
                shared.unlink();detached.rename(shared)
    with (out/'frames.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if cache.is_symlink():
            detached=out/'detached-cache'
            if not detached.exists():shutil.copytree(cache,detached,copy_function=os.link)
            cache.unlink();detached.rename(cache)
        rows=read(cache/'candidates.json')
        uniform=[r for r in rows if abs(r['timestamp_s']-round(r['timestamp_s']))<1e-6]
        assert uniform and uniform[0]['timestamp_s']==0
        assert all(abs(b['timestamp_s']-a['timestamp_s']-1)<1e-6 for a,b in zip(uniform,uniform[1:]))
        missing=[r['frame'] for r in uniform if not (cache/r['frame']).exists()]
        if missing:
            sources=list((ROOT/'results').glob('*/packets/'+sid+'/cache'))+list((ROOT/'results').glob('*/fallback/'+sid+'/cache'))
            sources+=list((ROOT/'results').glob('*/rerun*/fallback-recovery/'+sid+'/cache'))
            for r in uniform:
                dest=cache/r['frame']
                if dest.exists():continue
                src=next((p/r['frame'] for p in sources if (p/r['frame']).exists() and sha(p/r['frame'])==r['frame_sha256']),None)
                if src:
                    dest.parent.mkdir(parents=True,exist_ok=True);os.link(src,dest)
            missing=[r['frame'] for r in uniform if not (cache/r['frame']).exists()]
        if missing:raise FileNotFoundError('Supply exact canonical cached frames before recovery: '+', '.join(missing[:8]))
        poses=Path(read(base/'run-config.json')['poses'][sid]);samples={x['source_frame_index']:x for x in read(poses/'sampling.json')['frames']}
        for r in uniform:assert sha(cache/r['frame'])==r['frame_sha256']==samples[r['source_frame_index']]['sha256']
        video=video_path(base,sid)
        duration=float(json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','json',str(video)],text=True))['format']['duration'])
        assert 0<=duration-uniform[-1]['timestamp_s']<=1.1
        manifest=dict(video_sha256=sha(video),fps=1,duration_s=duration,source='exact cached source frames; validated against LingBot sampling hashes',frames=[dict(timestamp_s=r['timestamp_s'],source_frame_index=r['source_frame_index'],path=str((cache/r['frame']).resolve()),sha256=r['frame_sha256']) for r in uniform])
        save(out/'frames.json',manifest)
    return rows,manifest

def proposals_to_groups(proposals,category,rows,cache):
    bytime={r['timestamp_s']:r for r in rows};groups=[]
    result=next(r for r in proposals['categories'] if r['category']==category)
    slug=hashlib.sha256(category.encode()).hexdigest()[:10]
    for n,inst in enumerate(result['instances'],1):
        if not inst['evidence']:continue
        gid=f'{slug}-video-fallback-{n:02d}';selected={}
        for i,e in enumerate(inst['evidence']):
            c=bytime[e['timestamp_s']];box=[v*(640 if j%2==0 else 480) for j,v in enumerate(e['box_xyxy_normalized'])]
            obs=dict(object=category,peak_id=f'{gid}-e{i+1:02d}',box_xyxy=box,dino_score=None,status='astra_full_video_proposal')
            selected[c['source_frame_index']]=dict(c,image=c['frame'],frame=c['index'],object=category,box_xyxy=box,observations=[obs],status='astra_full_video_proposal')
        times=[e['timestamp_s'] for e in inst['evidence']]
        near=sorted((r for r in rows if float(r['timestamp_s']).is_integer() and (cache/r['frame']).exists()),key=lambda r:(min(abs(r['timestamp_s']-t) for t in times),r['timestamp_s']))
        for c in near:
            if len(selected)>=4:break
            if c['source_frame_index'] not in selected:
                selected[c['source_frame_index']]=dict(c,image=c['frame'],frame=c['index'],object=category,observations=[dict(object=category,box_xyxy=None,dino_score=None,status='context_fallback')],status='context_fallback')
        ordered=sorted(selected.values(),key=lambda r:r['timestamp_s']);assert len(ordered)>=4
        groups.append(dict(group_id=gid,object=category,objects=[category],peak_ids=[o['peak_id'] for r in ordered for o in r['observations'] if 'peak_id' in o],selected=ordered,candidate_frames=ordered,minimum_four_satisfied=True,source='astra_full_video_1fps_recovery',description=inst['description']))
    return groups

def search_category(base,sid,category):
    if not FULL_VIDEO_RETRIEVAL_ENABLED:raise RuntimeError('Full-video retrieval disabled; missing evidence must not become a zero count')
    from bvb_v6 import semantic_review as sr
    from bvb_v6.api_provider import resolve
    root=base/'fallback'/sid/hashlib.sha256(category.encode()).hexdigest()[:16];root.mkdir(parents=True,exist_ok=True)
    allow_empty=read(base/'run-config.json').get('counting_allow_all_rejected',False)
    strict=base/'semantic-recovery'/sid/'accepted-by-category.json'
    if strict.exists() and category in read(strict):
        accepted=read(strict)[category]
        if accepted or allow_empty:return accepted
        raise RuntimeError('Astra full-video recovery has no positively confirmed '+category)
    with (root/'search.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (root/'accepted.json').exists():return read(root/'accepted.json')
        rows,manifest=canonical_frames(base,sid);cache=base/'packets'/sid/'cache'
        save(root/'locator/frames.json',manifest)
        q=dict(id='fallback-identity',question=f'Locate every distinct {category} visible in this video. Distinguish repeat views from different instances. Do not guess an unseen object.',question_type='object_localization',objects=[category],roles={})
        save(root/'status.json',dict(stage='locating',category=category,frames=len(manifest['frames'])))
        proposals=locate(video_path(base,sid),q,[category],root/'locator',enabled=True)
        groups=proposals_to_groups(proposals,category,rows,cache)
        if not groups:
            if not allow_empty:raise RuntimeError('Full-video search found no supported '+category)
            save(root/'accepted.json',[])
            save(root/'status.json',dict(stage='completed-no-evidence',category=category,accepted_groups=0,
                source_manifest_sha256=sha(root/'locator/frames.json'),reason='Full-video locator returned no supported instances'))
            return []
        source=root/'selection.json';review=root/'review';poses=read(base/'run-config.json')['poses'][sid]
        save(source,dict(inputs=dict(cache=str(cache),poses=poses),questions={'fallback-identity':dict(question=q,arms=dict(roi_robust=groups),summary={})}))
        if not (review/'manifest.json').exists():sr.prepare(source,review)
        config=resolve('subscription_direct')
        for gid in read(review/'manifest.json')['jobs']:
            job=review/'jobs'/gid
            if not (job/'review.json').exists():sr.request_job(job,config.key,max_tokens=None,provider_config=config)
        if any(read(review/'jobs'/gid/'request.json')['status'] not in {'reviewed','no_evidence'} for gid in read(review/'manifest.json')['jobs']):
            raise RuntimeError('Fallback review requests incomplete; not a zero result')
        result=sr.export(review);accepted=result['questions']['fallback-identity']['arms']['semantic_accepted']
        if not accepted:
            if not allow_empty:raise RuntimeError('No '+category+' proposal passed clean-image review')
            if any(read(review/'jobs'/gid/'request.json')['status'] not in {'reviewed','no_evidence'} for gid in read(review/'manifest.json')['jobs']):
                raise RuntimeError('Incomplete fallback review must not become zero')
        for g in accepted:g['semantic_review']['admission']=dict(selected=True,mode='semantic_accept',object=category,candidate_count=len(accepted))
        save(root/'accepted.json',accepted);save(root/'status.json',dict(stage='completed' if accepted else 'completed-no-evidence',category=category,accepted_groups=len(accepted),
            source_manifest_sha256=sha(root/'locator/frames.json')))
        return accepted

def recover_question(base,qid,missing):
    if not FULL_VIDEO_RETRIEVAL_ENABLED:raise RuntimeError('Full-video retrieval disabled; unresolved categories: '+', '.join(missing))
    cfg=read(base/'run-config.json')
    if not cfg.get('missing_object_video_fallback',{}).get('enabled'):raise RuntimeError('Missing-candidate video fallback disabled')
    q=next(q for q in read(base/'questions.public.json') if str(q['id'])==str(qid));sid=q['scene_name']
    source=base/'scenes'/sid/'input/review/reviewed-selection.json'
    dest=base/'fallback/questions'/str(qid)/'reviewed-selection.json'
    data=read(dest) if dest.exists() else copy.deepcopy(read(source));row=data['questions'][str(qid)]
    selected=[g for g in row['arms']['semantic_selected'] if g['semantic_review']['status']=='accept']
    row['arms']['semantic_selected']=selected
    for category in missing:
        new=search_category(base,sid,category)
        known={g['group_id'] for g in selected};selected.extend(copy.deepcopy(g) for g in new if g['group_id'] not in known)
    covered={o for g in selected for o in g.get('objects',[g['object']])}
    absent=sorted(set(q['objects'])-covered)
    row.update(coverage_missing_categories=absent,coverage_complete=not absent,missing_objects=absent,coverage_blockers={o:'no_candidates' for o in absent})
    row['summary']['semantic_selected']=dict(groups=len(selected),image_slots=sum(len(g['selected']) for g in selected))
    row['fallback_recovery']=dict(method='shared full-video 1fps identity search + clean-image review',categories=missing)
    if cfg.get('counting_allow_all_rejected'):
        row['counting_zero_evidence_categories']=absent
        # Zero is now permitted only after completed full-video recovery. The
        # frozen region preparer must not apply its old minimum-one admission gate.
        data['semantic_review']['coverage_required']=False
    save(dest,data)
    return dest
