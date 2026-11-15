"""One original+crop pair per cluster, with exact-input identity-audit reuse."""
import fcntl,hashlib,json,os,shutil
from pathlib import Path
from bvb_v10.pipeline import read,save,sha

POLICY='one-representative-identity-v1'

def select_one(group,candidates):
    if not candidates:return []
    ranks={}
    for frame in group['candidate_frames']:
        for obs in frame.get('observations',[]):
            if not obs.get('box_xyxy') or obs.get('object')!=group['object']:continue
            a,b,c,d=obs['box_xyxy'];area=max(0,c-a)*max(0,d-b)
            key=(obs.get('peak_id'),frame['source_frame_index'])
            ranks[key]=(float(obs.get('dino_score') or 0),area,float(frame.get('sharpness') or 0),-frame['source_frame_index'])
    return [max(candidates,key=lambda c:ranks.get((c['peak_id'],c['source_frame_index']),(0,0,0,-c['source_frame_index'])))]

def request_with_cache(invoke,folder,key,max_tokens=8192,provider_config=None):
    from bvb_v6 import semantic_review as sr
    folder=Path(folder)
    if (folder/'request.json').exists():return invoke(folder,key,max_tokens,provider_config)
    job=read(folder/'input.json');sr.verify_input(job)
    # Validate the actual clean files before looking up a cached identity judgment.
    for e in job['evidence']:
        for part in ['full','crop']:
            if sha(folder/e[part])!=e[part+'_sha256']:raise ValueError('Audit image changed')
    provider=provider_config.provider if provider_config else 'openai'
    identity=dict(request_hash=job['request_hash'],provider=provider,policy=POLICY,account=os.environ.get('BVB_DIRECT_ACCOUNT','default'))
    cid=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    cache=Path(os.environ.get('BVB_LIGHT_REVIEW_CACHE',str(Path(__file__).resolve().parents[1]/'.cache/bvb_v13/semantic')))
    cache.mkdir(parents=True,exist_ok=True);entry=cache/cid
    with (cache/(cid+'.lock')).open('a') as mutex:
        fcntl.flock(mutex,fcntl.LOCK_EX)
        if (entry/'cache.json').exists():
            meta=read(entry/'cache.json')
            if meta['identity']!=identity or any(sha(entry/name)!=digest for name,digest in meta['files'].items()):raise ValueError('Semantic cache changed')
            verdict=read(entry/'review.json')
            if verdict['request_hash']!=job['request_hash']:raise ValueError('Wrong cached review identity')
            sr.validate_reply(dict(evidence=verdict['evidence']),[e['evidence_id'] for e in job['evidence']])
            save(folder/'review.json',verdict)
            if (entry/'response.txt').exists():shutil.copy2(entry/'response.txt',folder/'response.txt')
            source=read(entry/'request.json')
            record=dict(source,status='reviewed',usage=None,cache_hit=True,cache_path=str(entry),cache_policy=POLICY,elapsed_s=0,source_usage=source.get('usage'),source_response_id=source.get('response_id'))
            record.pop('response_id',None);save(folder/'request.json',record);return record
        result=invoke(folder,key,max_tokens,provider_config)
        if result['status']=='reviewed' and (folder/'review.json').exists():
            entry.mkdir(exist_ok=True);files={}
            for name in ['review.json','request.json','response.txt']:
                if (folder/name).exists():shutil.copy2(folder/name,entry/name);files[name]=sha(entry/name)
            save(entry/'cache.json',dict(identity=identity,files=files))
        return result
