"""One text-only ChatGPT-plan Responses request after explicit browser OAuth."""
import base64
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
import time
from urllib.parse import parse_qs, urlencode, urlsplit
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.tools/siwc-python'))
import jwt
import requests
from openai import OpenAI

AUTH='https://auth.openai.com'
API='https://api.openai.com/v1'
PRIVATE=ROOT/'.local/siwc-direct'
MODEL='gpt-6-astra'
QUESTION='1＋1等于几？只回答数字。'


def write_private(path,data):
    fd,tmp=tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:
            json.dump(data,f,ensure_ascii=False,indent=2)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def main():
    PRIVATE.mkdir(parents=True,exist_ok=True,mode=0o700)
    host=PRIVATE/'host.json'
    if not host.exists():write_private(host,{'id':'urn:uuid:'+str(uuid.uuid4())})
    host_id=json.loads(host.read_text())['id']
    regfile=PRIVATE/'registration.json'
    registration=json.loads(regfile.read_text()) if regfile.exists() else {}
    oldfile=PRIVATE/'credentials.json'
    previous=json.loads(oldfile.read_text()) if oldfile.exists() else {}
    state,nonce,verifier=(secrets.token_urlsafe(32) for _ in range(3))
    challenge=base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip('=')
    callback={}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def reply(self,status,body):
            self.send_response(status)
            self.send_header('Content-Type','text/html; charset=utf-8')
            self.send_header('Cache-Control','no-store')
            self.send_header('Referrer-Policy','no-referrer')
            self.end_headers();self.wfile.write(body.encode())
        def do_GET(self):
            parts=urlsplit(self.path)
            if parts.path=='/':
                self.reply(200,'<h2>BVB subscription direct test</h2><p>Authorize BVB Direct Test to use your ChatGPT plan. Then one text-only question will run: 1 + 1. No project files or images are sent.</p><a href="/login">Continue with ChatGPT</a>')
            elif parts.path=='/login':
                self.send_response(302);self.send_header('Location',authorize_url);self.send_header('Cache-Control','no-store');self.end_headers()
            elif parts.path=='/auth/callback':
                values=parse_qs(parts.query)
                if not secrets.compare_digest(values.get('state',[''])[0],state):
                    self.reply(400,'Invalid login state.');return
                if callback:
                    self.reply(409,'Callback already received.');return
                callback.update({k:v[0] for k,v in values.items()})
                self.reply(200,'<h2>Authorization received</h2><p>You may return to the chat. BVB will verify the authorization and report the test result.</p>')
            else:self.reply(404,'Not found')
    server=HTTPServer(('127.0.0.1',0),Handler)
    redirect=f'http://127.0.0.1:{server.server_port}/auth/callback'
    params={'client_id':registration.get('client_id','dynamic_agent_client'),'ext_agent_host_id':host_id,
            'response_type':'code','redirect_uri':redirect,'scope':'openid profile email offline_access resource.invoke chatgpt.tokens.use.direct',
            'resource':API,'state':state,'nonce':nonce,'code_challenge_method':'S256','code_challenge':challenge}
    if not registration:params['agent_name_hint']='BVB Direct Test'
    authorize_url=AUTH+'/api/accounts/authorize?'+urlencode(params)
    print('Continue with ChatGPT: http://127.0.0.1:'+str(server.server_port)+'/',flush=True)
    subprocess.run(['open','-a','Safari','http://127.0.0.1:'+str(server.server_port)+'/'],check=False)
    server.timeout=1
    deadline=time.monotonic()+900
    try:
        while not callback and time.monotonic()<deadline:server.handle_request()
    finally:server.server_close()
    if not callback:raise RuntimeError('oauth_callback_timeout')
    if callback.get('error'):raise RuntimeError('oauth_authorization_declined_or_failed')
    issued=callback.get('client_id') or registration.get('client_id')
    if not issued or issued=='dynamic_agent_client':raise RuntimeError('missing_issued_client_id')
    if registration and registration['client_id']!=issued:raise RuntimeError('client_id_mismatch')
    # Persist the issued registration even if exchange must be retried with fresh OAuth.
    write_private(regfile,{'client_id':issued,'ext_agent_host_id':host_id})
    response=requests.post(AUTH+'/api/accounts/oauth/token',data={'grant_type':'authorization_code','client_id':issued,
        'code':callback['code'],'code_verifier':verifier,'redirect_uri':redirect,'resource':API},timeout=30)
    if response.status_code!=200:raise RuntimeError('oauth_exchange_http_'+str(response.status_code))
    tokens=response.json()
    signing=jwt.PyJWKClient(AUTH+'/.well-known/jwks.json').get_signing_key_from_jwt(tokens['id_token'])
    claims=jwt.decode(tokens['id_token'],signing.key,algorithms=['RS256'],audience=issued,issuer=AUTH,
                      options={'require':['exp','iss','aud','sub','nonce']})
    if not secrets.compare_digest(claims['nonce'],nonce):raise RuntimeError('nonce_mismatch')
    if previous and previous['client_id']==issued and previous['subject']!=claims['sub']:raise RuntimeError('account_identity_mismatch')
    scopes=tokens.get('scope','').split()
    credentials={k:tokens.get(k) for k in ['access_token','refresh_token','id_token','token_type','expires_in']}
    credentials.update(subject=claims['sub'],client_id=issued,issuer=AUTH,ext_agent_host_id=host_id,scopes=scopes,saved_at=time.time())
    write_private(oldfile,credentials)
    if 'chatgpt.tokens.use.direct' not in scopes:raise RuntimeError('chatgpt_plan_usage_not_granted')
    print('OAuth identity verified; ChatGPT plan usage granted.',flush=True)
    token=credentials['access_token']
    catalog=requests.get(API+'/models',headers={'Authorization':'Bearer '+token},timeout=30)
    if catalog.status_code!=200:raise RuntimeError('model_catalog_http_'+str(catalog.status_code))
    names=[m.get('slug') for m in catalog.json().get('models',[])]
    if MODEL not in names:raise RuntimeError('requested_model_not_in_account_catalog')
    client=OpenAI(api_key=token,base_url=API,max_retries=0,timeout=120)
    report={'provider':'chatgpt_oauth_direct','model_requested':MODEL,'question':QUESTION,'status':'started','answer':''}
    out=ROOT/'results/subscription-direct-20261106';out.mkdir(parents=True,exist_ok=True)
    target=out/(str(time.time_ns())+'.json')
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    start=time.monotonic()
    try:
        with client.responses.create(model=MODEL,input=[{'role':'user','content':QUESTION}],
                                     reasoning={'effort':'low'},store=False,stream=True) as stream:
            for event in stream:
                if event.type=='response.output_text.delta':report['answer']+=event.delta
                elif event.type=='response.completed':
                    report.update(status='completed',usage=event.response.usage.model_dump() if event.response.usage else None,
                                  returned_model=event.response.model,response_id=event.response.id)
                elif event.type in ('response.failed','response.incomplete','error'):
                    report['status']=event.type
                    error=getattr(getattr(event,'response',None),'error',None)
                    report['error_code']=getattr(error,'code',None) or getattr(event,'code',None)
        if report['status']=='started':report['status']='stream_ended_without_completion'
    except Exception as exc:
        report.update(status='failed',error_type=type(exc).__name__,http_status=getattr(exc,'status_code',None),error_code=getattr(exc,'code',None))
    finally:
        client.close()
        report['elapsed_s']=round(time.monotonic()-start,2)
        target.write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
        print('Report: '+str(target),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        # Never print exception bodies/URLs containing OAuth codes or tokens.
        safe=str(exc) if type(exc) is RuntimeError else type(exc).__name__
        print('Stopped: '+safe,flush=True)
        sys.exit(1)
