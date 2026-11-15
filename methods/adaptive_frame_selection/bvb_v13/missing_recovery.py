"""Only missing targets trigger 0.5fps localization, followed by semantic review.

Existing positive groups are preserved. Negative/uncertain candidates do not
trigger a search if the same category already has an admitted positive group.
"""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
from bvb_v10.pipeline import read, save, sha
from .retrieval_policy import (MISSING_OBJECT_RECOVERY_ENABLED,
    MISSING_OBJECT_RECOVERY_FPS as FPS, MISSING_OBJECT_POLICY as POLICY, admitted_only)
from .common import video_path
from .question_policy import policy_for


def missing_categories(row):
    positive={g['object'] for g in row.get('arms',{}).get('semantic_selected',[])
              if g.get('semantic_review',{}).get('status')=='accept'}
    return sorted(set(row['question'].get('objects',[]))-positive)


def choose_for_question(question, category, groups, root, phase):
    """Retain multiple instances where allowed; compare only confirmed candidates."""
    if any(g['semantic_review']['status']!='accept' for g in groups):
        raise ValueError('Unreviewed fallback proposals cannot be admitted')
    if len(groups)<=1 or policy_for(question)['category']=='multiple':return copy.deepcopy(groups)
    from subscription_direct_client import request,image_item
    from .common import parse
    from bvb_v6.coverage import PROMPT,validate_result
    ids=[g['semantic_review']['review_id'] for g in groups]
    identity=dict(question={k:v for k,v in question.items() if k in ['id','question','question_type','objects','roles','options']},
                  category=category,candidates=ids,policy=POLICY)
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:20]
    folder=root/'instance-comparison'/key;result=folder/'result.json'
    if not result.exists():
        content=[dict(type='input_text',text=PROMPT+'\nOnly already positively reviewed candidates are supplied. Return JSON with selected_id, confidence, reason, uncertainty and ranking [{candidate_id,rank,reason}].\n'+json.dumps(identity,ensure_ascii=False))]
        for group in groups:
            review=Path(group['semantic_review']['review_file']).parent
            entry=read(review/'input.json');gid=group['semantic_review']['review_id']
            content.append(dict(type='input_text',text='CANDIDATE '+gid))
            for e in entry['evidence']:
                content.append(dict(type='input_text',text=json.dumps(dict(timestamp_s=e['timestamp_s'],box_xyxy=e['box_xyxy']))))
                for kind in ['full','crop']:
                    image=review/e[kind]
                    if sha(image)!=e[kind+'_sha256']:raise ValueError('Recovery comparison image changed')
                    content.append(image_item(image))
        phase('missing-object-comparison')
        answer=parse(request(folder,content,model='gpt-6-astra',effort='high')['answer'])
        validate_result(answer,ids);save(result,answer)
    answer=read(result);validate_result(answer,ids)
    return [copy.deepcopy(next(g for g in groups if g['semantic_review']['review_id']==answer['selected_id']))]


def recover_selection(base,sid,source,destination,phase=None):
    """Batch missing categories once per scene; share search and review receipts."""
    phase=phase or (lambda name:None)
    base,source,destination=map(Path,(base,source,destination))
    data=admitted_only(read(source))
    missing={qid:missing_categories(row) for qid,row in data['questions'].items()}
    categories=sorted({c for values in missing.values() for c in values})
    enabled=MISSING_OBJECT_RECOVERY_ENABLED and read(base/'run-config.json').get('missing_object_video_fallback',{}).get('enabled',True)
    root=base/'missing-object-recovery-0p5fps'/sid;root.mkdir(parents=True,exist_ok=True)
    with (root/'recovery.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        recovered={c:[] for c in categories};batch=None
        if enabled and categories:
            from .candidate_recovery import canonical_frames,proposals_to_groups
            from .video_fallback import locate
            from bvb_v6 import semantic_review as sr
            from bvb_v6.api_provider import resolve
            _,manifest=canonical_frames(base,sid,fps=FPS)
            # Proposals and supplemental group frames both stay on the 0.5fps grid.
            selected_ids={f['source_frame_index'] for f in manifest['frames']}
            cache=base/'packets'/sid/'cache'
            rows=[r for r in read(cache/'candidates.json') if r['source_frame_index'] in selected_ids]
            key=hashlib.sha256(json.dumps(dict(policy=POLICY,categories=categories,manifest=manifest),sort_keys=True).encode()).hexdigest()[:20]
            batch=root/'batches'/key;batch.mkdir(parents=True,exist_ok=True)
            save(batch/'locator/frames.json',manifest)
            if (batch/'accepted.json').exists():
                recovered=read(batch/'accepted.json')
                if any(g['semantic_review']['status']!='accept' for gs in recovered.values() for g in gs):raise ValueError('Invalid cached recovery admission')
            else:
                question=dict(id='missing-target-identity',question_type='object_localization',
                    question='Locate all genuinely visible instances of the requested missing categories. Do not assume they exist. Distinguish repeated views from separate physical instances.',objects=categories,roles={})
                phase('missing-object-search')
                proposals=locate(video_path(base,sid),question,categories,batch/'locator',enabled=True,fps=FPS,missing_only=True)
                groups=[g for c in categories for g in proposals_to_groups(proposals,c,rows,cache,sampling_fps=FPS)]
                if groups:
                    proposal_source=batch/'proposals-selection.json';review=batch/'review'
                    save(proposal_source,dict(inputs=dict(cache=str(cache),poses=read(base/'run-config.json')['poses'][sid]),
                        questions={'recovery':dict(question=question,arms=dict(roi_robust=groups),summary={})}))
                    if not (review/'manifest.json').exists():sr.prepare(proposal_source,review)
                    connection=resolve('subscription_direct');phase('missing-object-review')
                    for gid in read(review/'manifest.json')['jobs']:
                        job=review/'jobs'/gid
                        if not (job/'review.json').exists():sr.request_job(job,connection.key,max_tokens=None,provider_config=connection)
                        if read(job/'request.json')['status'] not in {'reviewed','no_evidence'}:
                            raise RuntimeError('Missing-object review incomplete; never interpret as absence')
                    reviewed=sr.export(review)['questions']['recovery']['arms']['semantic_accepted']
                    for group in reviewed:
                        if group['semantic_review']['status']!='accept':continue
                        group['semantic_review']['admission']=dict(selected=True,mode='semantic_accept',reason=POLICY)
                        recovered[group['object']].append(group)
                save(batch/'accepted.json',recovered)
        for qid,row in data['questions'].items():
            if enabled:
                for category in missing[qid]:
                    groups=choose_for_question(row['question'],category,recovered[category],root,phase)
                    known={g['group_id'] for g in row['arms']['semantic_selected']}
                    row['arms']['semantic_selected'].extend(g for g in groups if g['group_id'] not in known)
            remaining=missing_categories(row)
            row.update(coverage_missing_categories=remaining,missing_objects=remaining,coverage_complete=not remaining,
                coverage_blockers={c:'not_confirmed_after_half_fps_search' if enabled else 'missing_search_disabled' for c in remaining},
                semantic_recovery=dict(policy=POLICY,enabled=enabled,fps=FPS,trigger='missing_positive_category_only',
                    triggered_categories=missing[qid],search_performed=bool(enabled and missing[qid]),
                    completed=bool(enabled and missing[qid]),unresolved_categories=remaining))
            row['summary']['semantic_selected']=dict(groups=len(row['arms']['semantic_selected']),image_slots=sum(len(g['selected']) for g in row['arms']['semantic_selected']))
        data['semantic_review']['recovery_policy']=POLICY
        save(destination,data)
        save(root/'receipt.json',dict(policy=POLICY,enabled=enabled,fps=FPS,source_sha256=sha(source),selection_sha256=sha(destination),
            searched_categories=categories if enabled else [],batch=str(batch) if batch else None,
            unresolved={qid:r['coverage_missing_categories'] for qid,r in data['questions'].items() if r['coverage_missing_categories']},
            no_evidence_is_not_zero_count=True))
    return destination
