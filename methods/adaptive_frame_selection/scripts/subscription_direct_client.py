"""Direct ChatGPT-plan Responses transport. Never falls back to paid API keys."""
import base64,fcntl,hashlib,json,os,sys,tempfile,time
from pathlib import Path
import requests
from openai import OpenAI
ROOT=Path(__file__).resolve().parents[1]
API='https://api.openai.com/v1'
STATE=ROOT/'.local/siwc-direct'


def account_name(account=None):
    account=account or os.environ.get('BVB_DIRECT_ACCOUNT','default')
    if account not in ('default','account4'):
        raise ValueError('BVB_DIRECT_ACCOUNT must be default or account4')
    return account


def account_state(account=None):
    return STATE if account_name(account)=='default' else ROOT/'.local/siwc-direct-account4'


def atomic_private(path,data):
    fd,tmp=tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:json.dump(data,f)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def access_token(account=None):
    account=account_name(account);state=account_state(account)
    path=state/'credentials.json'
    if not path.exists():raise RuntimeError('Authorize first with scripts/test_subscription_direct.py --account '+account)
    with (state/'refresh.lock').open('a') as lock:
        os.chmod(state/'refresh.lock',0o600);fcntl.flock(lock,fcntl.LOCK_EX)
        old=json.loads(path.read_text())
        if 'chatgpt.tokens.use.direct' not in old.get('scopes',[]):raise RuntimeError('Direct plan usage not granted')
        if time.time()<old['saved_at']+old['expires_in']-120:return old['access_token']
        r=requests.post('https://auth.openai.com/api/accounts/oauth/token',data={
            'grant_type':'refresh_token','client_id':old['client_id'],'refresh_token':old['refresh_token'],
            'resource':API},timeout=40)
        if r.status_code!=200:raise RuntimeError('OAuth refresh HTTP '+str(r.status_code)+'; no inference sent')
        new=r.json()
        if not new.get('access_token') or not new.get('refresh_token'):raise RuntimeError('Incomplete OAuth refresh')
        if new.get('id_token'):
            sys.path.insert(0,str(ROOT/'.tools/siwc-python'))
            import jwt
            signing=jwt.PyJWKClient('https://auth.openai.com/.well-known/jwks.json').get_signing_key_from_jwt(new['id_token'])
            claims=jwt.decode(new['id_token'],signing.key,algorithms=['RS256'],audience=old['client_id'],issuer='https://auth.openai.com',options={'require':['exp','iss','aud','sub']})
            if claims['sub']!=old['subject']:raise RuntimeError('Refreshed identity mismatch')
        scopes=new['scope'].split() if 'scope' in new else old['scopes']
        record={**old,**{k:new[k] for k in ['access_token','refresh_token','id_token','token_type','expires_in'] if k in new},'scopes':scopes,'saved_at':time.time()}
        atomic_private(path,record)
        if 'chatgpt.tokens.use.direct' not in scopes:raise RuntimeError('Direct plan permission lost')
        return record['access_token']


def image_item(path):
    p=Path(path);mime='image/png' if p.suffix.lower()=='.png' else 'image/jpeg'
    return {'type':'input_image','detail':'high','image_url':'data:'+mime+';base64,'+base64.b64encode(p.read_bytes()).decode()}


def request(folder,content,*,model='gpt-6-astra',effort='high',instructions=None,account=None):
    account=account_name(account)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    body={'model':model,'input':[{'role':'user','content':content}],'reasoning':{'effort':effort},'store':False,'stream':True}
    if instructions:body['instructions']=instructions
    fingerprint=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()
    path=folder/'request.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old.get('account','default')!=account:raise RuntimeError('Account changed; use new attempt folder')
        if old['fingerprint']!=fingerprint:raise RuntimeError('Inputs changed; use new attempt folder')
        if old['status']=='completed':return old
        raise RuntimeError('Prior attempt not completed; inspect before retry')
    token=access_token(account)
    catalog=requests.get(API+'/models',headers={'Authorization':'Bearer '+token},timeout=40)
    if catalog.status_code!=200:raise RuntimeError('Catalog HTTP '+str(catalog.status_code))
    if model not in [r.get('slug') for r in catalog.json().get('models',[])]:raise RuntimeError('Requested model unavailable')
    record={'provider':'chatgpt_oauth_direct' if account=='default' else 'subscription_direct_4','account':account,'model_requested':model,'effort':effort,'fingerprint':fingerprint,'status':'started','usage':None,'started_unix':time.time(),'answer':''}
    with path.open('x') as f:json.dump(record,f,indent=2)
    (folder/'input-text.json').write_text(json.dumps({'instructions':instructions,'text_blocks':[x['text'] for x in content if x['type']=='input_text'],'image_count':sum(x['type']=='input_image' for x in content)},ensure_ascii=False,indent=2))
    start=time.monotonic()
    try:
        with OpenAI(api_key=token,base_url=API,max_retries=0,timeout=900) as client:
            with client.responses.create(**body) as stream:
                for event in stream:
                    if event.type=='response.output_text.delta':record['answer']+=event.delta
                    elif event.type=='response.completed':
                        record.update(status='completed',usage=event.response.usage.model_dump() if event.response.usage else None,
                                      response_id=event.response.id,returned_model=event.response.model)
                    elif event.type in ('response.failed','response.incomplete','error'):
                        record['status']=event.type
                        err=getattr(getattr(event,'response',None),'error',None)
                        record['error_code']=getattr(err,'code',None) or getattr(event,'code',None)
            if record['status']=='started':record['status']='stream_ended_without_completion'
    except Exception as exc:
        record.update(status='failed',error_type=type(exc).__name__,http_status=getattr(exc,'status_code',None))
    finally:
        record['elapsed_s']=round(time.monotonic()-start,2)
        path.write_text(json.dumps(record,ensure_ascii=False,indent=2))
        (folder/'response.txt').write_text(record['answer'])
    print(json.dumps({k:record.get(k) for k in ['provider','status','elapsed_s','usage']},ensure_ascii=False),flush=True)
    if record['status']!='completed':raise RuntimeError('Inference not completed; see '+str(path))
    return record
