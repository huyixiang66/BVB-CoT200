"""Question policy + clean-image semantic audit + one-cluster comparison via OAuth."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
METHOD=ROOT
sys.path.insert(0,str(METHOD))
from bvb_v6 import semantic_review as sr, coverage
from bvb_v6.api_provider import resolve, create_client

OUT=ROOT/'outputs/v10-audit'
REVIEW=OUT/'input/review'
PROMPT='''Determine the candidate-cluster retention policy separately for each supplied question.
Use only the public question, options and object roles; do not answer the question.
Choose multiple if distinct same-category instances may be needed to answer correctly,
especially counting and closest/farthest comparisons. Otherwise choose single when
each relevant category has one intended referent and competing candidate clusters
should be compared to retain one. When the referents are ambiguous, conservatively
choose multiple and explain. This is a question-specific decision, not a hard-coded
question-type lookup. Single means one cluster PER OBJECT CATEGORY, not one cluster
for the whole question. Single does not remove room/route/temporal supporting evidence.
Give concise Chinese reasons and describe any required supplemental evidence.
Return every supplied question ID once. Do not use GT or infer the number of real objects.'''
SCHEMA={'type':'object','properties':{'questions':{'type':'array','items':{
    'type':'object','properties':{'id':{'type':'string'},
    'category':{'type':'string','enum':['single','multiple']},
    'reason':{'type':'string'},'confidence':{'type':'string','enum':['low','medium','high']},
    'additional_evidence':{'type':'array','items':{'type':'string'}}},
    'required':['id','category','reason','confidence','additional_evidence'],'additionalProperties':False}}},
    'required':['questions'],'additionalProperties':False}

def status(stage,**extra):
    sr.write(OUT/'status.json',dict(stage=stage,updated_unix=time.time(),**extra))

def decide_policies(connection,selection):
    folder=OUT/'question-policy';folder.mkdir(exist_ok=True)
    questions=[r['question'] for r in selection['questions'].values()]
    payload=json.dumps(questions,ensure_ascii=False)
    fingerprint=sr.digest(dict(prompt=PROMPT,schema=SCHEMA,questions=questions,model=sr.MODEL))
    result_path=folder/'result.json'
    if result_path.exists():
        result=sr.read(result_path);assert result['request_hash']==fingerprint
        return result['questions']
    if (folder/'request.json').exists():raise RuntimeError('Policy request already attempted; inspect its ledger before retrying')
    sr.write(folder/'input.json',dict(questions=questions,prompt=PROMPT,schema=SCHEMA))
    record=dict(status='started',request_hash=fingerprint,model=sr.MODEL,reasoning_effort='high',**connection.metadata())
    sr.write(folder/'request.json',record)
    try:
        response=create_client(connection).chat.completions.create(model=sr.MODEL,reasoning_effort='high',store=False,
            messages=[{'role':'system','content':PROMPT},{'role':'user','content':payload}],
            response_format={'type':'json_schema','json_schema':{'name':'question_cluster_retention','strict':True,'schema':SCHEMA}})
        choice=response.choices[0];text=choice.message.content or ''
        (folder/'response.txt').write_text(text)
        record.update(response_id=response.id,returned_model=response.model,
            usage=response.usage.model_dump() if response.usage else None)
        assert choice.finish_reason=='stop' and not choice.message.refusal
        result=json.loads(text);rows=result['questions']
        assert len(rows)==len(questions) and {r['id'] for r in rows}==set(selection['questions'])
        for r in rows:assert r['category'] in ('single','multiple') and r['reason'].strip()
        sr.write(result_path,dict(questions=rows,request_hash=fingerprint))
        record['status']='completed'
    except Exception as exc:
        record.update(status='failed',error_type=type(exc).__name__)
        raise
    finally:sr.write(folder/'request.json',record)
    return rows

def main():
    source=OUT/'input/selection.json';selection=sr.read(source)
    if not (OUT/'input/selection-original.json').exists():
        sr.write(OUT/'input/selection-original.json',selection)
    # Always use the original public questions when checking the policy request hash.
    original=sr.read(OUT/'input/selection-original.json')
    connection=resolve('subscription_direct')
    status('question-policy')
    policies=decide_policies(connection,original)
    for row in policies:
        selection['questions'][row['id']]['question']['retention_policy']={
            'category':row['category'],'reason':row['reason'],'additional_evidence':row['additional_evidence'],
            'decision_source':'astra_question_audit','confidence':row['confidence'],
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
