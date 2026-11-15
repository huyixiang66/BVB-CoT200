"""Fixed local question policy + Astra semantic audit and instance comparison."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
METHOD=ROOT
sys.path.insert(0,str(METHOD))
from bvb_v6 import semantic_review as sr, coverage
from bvb_v6.api_provider import resolve

OUT=ROOT/'outputs/v12-audit'
REVIEW=OUT/'input/review'

def status(stage,**extra):
    sr.write(OUT/'status.json',dict(stage=stage,updated_unix=time.time(),**extra))

def decide_policies(connection, selection):
    from bvb_v13.question_policy import decide_policies as fixed
    return fixed(selection, OUT/'question-policy')

def main():
    source=OUT/'input/selection.json';selection=sr.read(source)
    if not (OUT/'input/selection-original.json').exists():
        sr.write(OUT/'input/selection-original.json',selection)
    # Always use the original public questions when checking the policy request hash.
    original=sr.read(OUT/'input/selection-original.json')
    status('fixed-question-policy')
    policies=decide_policies(None,original)
    connection=resolve('subscription_direct')
    for row in policies:
        selection['questions'][row['id']]['question']['retention_policy']={
            'category':row['category'],'reason':row['reason'],'additional_evidence':row['additional_evidence'],
            'decision_source':'fixed_question_type_v12','confidence':row['confidence'],
            'decision_file':str(OUT/'question-policy/result.json')}
    sr.write(source,selection)
    manifest=sr.read(REVIEW/'manifest.json')
    manifest.update(selection=str(source.resolve()),selection_sha256=sr.sha(source))
    sr.write(REVIEW/'manifest.json',manifest)
    pending=[gid for gid in manifest['jobs'] if not (REVIEW/'jobs'/gid/'request.json').exists()]
    done=len(manifest['jobs'])-len(pending)
    status('semantic-review',completed=done,total=len(manifest['jobs']))
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures={executor.submit(sr.request_job,REVIEW/'jobs'/gid,connection.key,None,connection):gid for gid in pending}
        for future in as_completed(futures):
            record=future.result();done+=1
            print('REVIEW',futures[future],record['status'],done,'/',len(manifest['jobs']),flush=True)
            status('semantic-review',completed=done,total=len(manifest['jobs']))
    failures=[gid for gid in manifest['jobs'] if sr.read(REVIEW/'jobs'/gid/'request.json')['status'] not in ('reviewed','no_evidence')]
    if failures:raise RuntimeError('Review requests need attention: '+','.join(failures))
    sr.export(REVIEW)
    status('question-specific-comparison')
    if not (REVIEW/'coverage/manifest.json').exists():coverage.prepare(REVIEW)
    count=len(sr.read(REVIEW/'coverage/manifest.json')['comparisons'])
    if count:coverage.submit(REVIEW,connection.key,limit=count,max_tokens=None,provider_config=connection)
    result=sr.export(REVIEW)
    for row in result['questions'].values():
        blocked=row.get('coverage_blockers',{})
        assert all(v=='no_candidates' for v in blocked.values()),blocked
        if row['retention_policy']['category']=='single':
            for obj in row['question']['objects']:
                kept=[g for g in row['arms']['semantic_selected'] if g['object']==obj]
                available=[g for g in row['arms']['roi_robust'] if g['object']==obj]
                assert len(kept)==(1 if available else 0)
    status('completed',questions=len(result['questions']),unique_reviews=len(manifest['jobs']),comparisons=count,
        missing={qid:r.get('coverage_missing_categories',[]) for qid,r in result['questions'].items() if r.get('coverage_missing_categories')})
    print('COMPLETED',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        status('failed',error_type=type(exc).__name__,error=str(exc));raise
