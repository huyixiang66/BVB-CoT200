"""All-task semantic recovery: reject/uncertain -> Astra search -> positive review."""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from bvb_v10.pipeline import ROOT, METHOD, read, save, sha

from .common import video_path

POLICY = 'all-task-reject-uncertain-astra-recovery-v1'


def archive_failed_request(folder, success):
    """A bounded worker retry may retry a terminal failure, never a live request."""
    ledger=folder/'request.json'
    if not ledger.exists() or (folder/success).exists():return
    status=read(ledger)['status']
    if status in {'started','completed','reviewed','no_evidence'}:return
    archive=folder/('failed-attempt-'+str(time.time_ns()));archive.mkdir()
    for name in ['request.json','response.txt']:
        if (folder/name).exists():(folder/name).rename(archive/name)


def reviewed_groups(row):
    groups = []
    for key in ['semantic_selected', 'semantic_accepted', 'roi_robust']:
        groups.extend(row.get('arms', {}).get(key, []))
    for key in ['semantic_excluded', 'coverage_excluded']:
        groups.extend(row.get(key, []))
    return [g for g in groups if 'semantic_review' in g]


def triggered_categories(row):
    categories = set(row['question']['objects'])
    bad = {g['object'] for g in reviewed_groups(row)
           if g['semantic_review'].get('status') in {'reject', 'uncertain'}}
    return sorted(categories & (bad | set(row.get('coverage_missing_categories', []))))


def apply(data, recovered):
    """Replace affected category evidence; never relabel an old negative verdict."""
    result = copy.deepcopy(data)
    for row in result['questions'].values():
        affected = triggered_categories(row)
        selected = row['arms']['semantic_selected']
        if affected:
            row['pre_recovery_selected'] = copy.deepcopy(selected)
            selected = [g for g in selected if g['object'] not in affected]
            for category in affected:
                groups = recovered[category]
                if any(g['semantic_review']['status'] != 'accept' for g in groups):
                    raise ValueError('Recovery may admit only positively reviewed proposals')
                selected.extend(copy.deepcopy(groups))
        if any(g['semantic_review']['status'] != 'accept' for g in selected):
            raise ValueError('Nonpositive candidate remains after recovery')
        row['arms']['semantic_selected'] = selected
        missing = sorted(set(row['question']['objects'])-{g['object'] for g in selected})
        row.update(coverage_missing_categories=missing, missing_objects=missing,
            coverage_complete=not missing, coverage_blockers={c:'no_candidates' for c in missing},
            semantic_recovery=dict(policy=POLICY, triggered_categories=affected,
                                   completed=True, unresolved_categories=missing))
        row.setdefault('summary', {})['semantic_selected'] = dict(groups=len(selected), image_slots=sum(len(g['selected']) for g in selected))
    result.setdefault('semantic_review', {}).update(required=True, selection_arm='semantic_selected',
        coverage_required=True, recovery_policy=POLICY)
    return result


def recover_selection(base, sid, source, destination):
    """One shared full-video search per scene, plus independent clean identity audits."""
    from .candidate_recovery import canonical_frames, proposals_to_groups
    from .video_fallback import locate
    sys.path.insert(0, str(METHOD))
    from bvb_v6 import semantic_review as sr
    from bvb_v6.api_provider import resolve
    root = base/'semantic-recovery'/sid; root.mkdir(parents=True, exist_ok=True)
    with (root/'recovery.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        fingerprint = sha(source)
        receipt = root/'receipt.json'
        if receipt.exists():
            record = read(receipt)
            if record['source_sha256'] != fingerprint: raise ValueError('Recovery source changed; use fresh directory')
            if sha(destination) != record['selection_sha256']: raise ValueError('Recovered selection changed')
            return destination
        data = read(source)
        categories = sorted({c for row in data['questions'].values() for c in triggered_categories(row)})
        recovered = {c:[] for c in categories}
        if categories:
            save(root/'status.json', dict(stage='full-video-search', categories=categories))
            rows, manifest = canonical_frames(base, sid)
            save(root/'locator/frames.json', manifest)
            confusions = {c:[] for c in categories}
            for row in data['questions'].values():
                for g in reviewed_groups(row):
                    if g['object'] in confusions and g['semantic_review']['status'] in {'reject', 'uncertain'}:
                        for e in g['semantic_review'].get('evidence', []):
                            text = {k:e.get(k) for k in ['observed_object','reason']}
                            if text not in confusions[g['object']]: confusions[g['object']].append(text)
            q = dict(id='shared-semantic-recovery', question_type='object_localization',
                question='Find actual instances of the requested categories throughout the full video. Prior detector candidates included these confusions; do not repeat them or assume the requested object exists. '+json.dumps(confusions, ensure_ascii=False),
                objects=categories, roles={})
            archive_failed_request(root/'locator/locate-request','proposals.json')
            proposals = locate(video_path(base,sid), q, categories, root/'locator', enabled=True)
            groups = [g for c in categories for g in proposals_to_groups(proposals,c,rows,base/'packets'/sid/'cache')]
            if groups:
                selection = root/'proposals-selection.json'; review = root/'review'
                save(selection, dict(inputs=dict(cache=str(base/'packets'/sid/'cache'), poses=read(base/'run-config.json')['poses'][sid]),
                    questions={'recovery':dict(question=q, arms=dict(roi_robust=groups), summary={})}))
                if not (review/'manifest.json').exists(): sr.prepare(selection, review)
                config = resolve('subscription_direct')
                jobs = [review/'jobs'/gid for gid in read(review/'manifest.json')['jobs']]
                save(root/'status.json', dict(stage='clean-identity-review', categories=categories, jobs=len(jobs)))
                def review_job(job):
                    archive_failed_request(job,'review.json')
                    if not (job/'review.json').exists(): sr.request_job(job, config.key, max_tokens=None, provider_config=config)
                    if read(job/'request.json')['status'] not in {'reviewed','no_evidence'}:
                        raise RuntimeError('Incomplete recovery review: '+str(job))
                with ThreadPoolExecutor(max_workers=2) as pool: list(pool.map(review_job,jobs))
                audited = sr.export(review)['questions']['recovery']['arms']['semantic_accepted']
                for g in audited:
                    if g['semantic_review']['status'] != 'accept': continue
                    g['semantic_review']['admission'] = dict(selected=True, mode='semantic_accept', reason=POLICY)
                    recovered[g['object']].append(g)
            save(root/'accepted-by-category.json', recovered)
        result = apply(data, recovered)
        save(destination, result)
        record = dict(policy=POLICY, source_sha256=fingerprint, selection_sha256=sha(destination),
            triggered_categories=categories, accepted_groups={c:len(g) for c,g in recovered.items()},
            unresolved_questions={qid:r['coverage_missing_categories'] for qid,r in result['questions'].items() if r['coverage_missing_categories']})
        save(receipt, record); save(root/'status.json', dict(stage='completed', **record))
        return destination


def ensure_question_selection(base, qid):
    """Common batch preflight, independent of question type or old fallback switch."""
    q = next(q for q in read(base/'questions.public.json') if str(q['id']) == str(qid))
    source = base/'scenes'/q['scene_name']/'input/review/reviewed-selection.json'
    data = read(source); row = data['questions'][str(qid)]
    if row.get('semantic_recovery', {}).get('completed'): return source
    if not any(triggered_categories(r) for r in data['questions'].values()): return source
    dest = base/'semantic-recovery'/q['scene_name']/'reviewed-selection.json'
    return recover_selection(base, q['scene_name'], source, dest)


def ensure_prepared(source, poses):
    """Standalone V10_2 safety net for every task, including manually prepared inputs."""
    manifest = read(source/'manifest.json'); groups = manifest['groups']
    reviews=[read(source/g['folder']/'semantic-review.json') for g in groups]
    bad = [g for g,r in zip(groups,reviews) if r['status'] in {'reject','uncertain'}]
    audit_rows=[]
    receipt=source/'semantic-source.json'
    if receipt.exists():
        origin=read(receipt);selection=Path(origin['path'])
        if sha(selection)!=origin['sha256']:raise ValueError('Prepared semantic source changed')
        row=read(selection)['questions'][str(manifest['question']['id'])]
        if row.get('semantic_recovery',{}).get('completed') and not bad:return source
        audit_rows.append(row)
    else:
        # Standalone prepared inputs still retain provenance of discarded audits.
        # Examine those too: a retained positive group does not cancel a reject.
        for r in reviews:
            if not r.get('review_file'):continue
            selection=Path(r['review_file']).parents[2]/'reviewed-selection.json'
            if selection.exists():
                row=read(selection).get('questions',{}).get(str(manifest['question']['id']))
                if row is not None:audit_rows.append(row)
    excluded=[g for row in audit_rows for g in reviewed_groups(row)
              if g['semantic_review']['status'] in {'reject','uncertain'} and g['object'] in manifest['question']['objects']]
    if not bad and not excluded:return source
    sys.path.insert(0,str(METHOD))
    from bvb_v6.reconstruction import prepare
    q = manifest['question']; sid=q['scene_name']; qid=str(q['id'])
    base = source.parent/(source.name+'-semantic-recovery'); base.mkdir(parents=True, exist_ok=True)
    converted=[]; cache=None
    for g in groups:
        frames=read(source/g['folder']/'input-frames.json')
        if cache is None:
            cache=next((p for p in Path(frames[0]['path']).parents if (p/'candidates.json').exists()),None)
        review=read(source/g['folder']/'semantic-review.json')
        selected=[dict(source_frame_index=f['source_frame_index'],timestamp_s=f['timestamp_s'],frame_sha256=f['sha256'],observations=f['observations'],object=g['objects'][0]) for f in frames]
        converted.append(dict(group_id=g['group_id'],object=g['objects'][0],objects=g['objects'],selected=selected,semantic_review=review))
    if cache is None: raise ValueError('Astra recovery requires the canonical frame cache')
    packet=base/'packets'/sid/'cache'
    if not packet.exists(): shutil.copytree(cache,packet,copy_function=__import__('os').link)
    save(base/'run-config.json',dict(poses={sid:str(poses)},videos={sid:(read(cache/'config.json') if (cache/'config.json').exists() else {}).get('video')})); save(base/'questions.public.json',[q])
    selection=base/'source.json'; save(selection,dict(questions={qid:dict(question=q,arms=dict(semantic_selected=converted),semantic_excluded=excluded,summary={})},semantic_review={}))
    recovered=recover_selection(base,sid,selection,base/'reviewed-selection.json');row=read(recovered)['questions'][qid]
    if not row['coverage_complete']: raise RuntimeError('Astra recovery has no confirmed evidence for '+', '.join(row['coverage_missing_categories']))
    dest=base/'prepared'/qid
    if not (dest/'manifest.json').exists():prepare(dest,q,row['arms']['semantic_selected'],None,manifest['frame_cap'],None,cache=packet,poses=poses)
    for name in ['scale-reference.json','counting-zero-evidence.json']:
        if (source/name).exists(): shutil.copy2(source/name,dest/name)
    return dest
