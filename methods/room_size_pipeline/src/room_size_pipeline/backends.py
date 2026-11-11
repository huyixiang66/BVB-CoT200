"""Optional bridge to the existing signed-in subscription client, no credentials bundled."""
import importlib
import json
import sys
import time
from pathlib import Path


class SubscriptionBackend:
    def __init__(self, client_dir, profile, expected_email, model='gpt-6-astra', effort='high', pro_confirmed=False):
        if not pro_confirmed or not expected_email or not profile:
            raise ValueError('Configure the expected account and explicitly confirmed Pro profile')
        if model!='gpt-6-astra' or effort!='high':raise ValueError('This pipeline requires Astra High')
        sys.path.insert(0,str(Path(client_dir).resolve()))
        self.call=importlib.import_module('call');self.auth=importlib.import_module('auth')
        self.profile,self.model,self.effort=profile,model,effort
        self.expected_email=expected_email

    def __call__(self, request):
        import jwt
        import requests
        a=self.auth;p=a.profile_path(self.profile)/'credentials.json'
        d=json.loads(p.read_text(encoding='utf8'))
        key=jwt.PyJWKClient(a.AUTH+'/.well-known/jwks.json').get_signing_key_from_jwt(d['id_token'])
        claims=jwt.decode(d['id_token'],key.key,algorithms=['RS256'],audience=d['client_id'],issuer=a.AUTH,options={'verify_exp':False})
        if claims.get('email')!=self.expected_email or claims.get('sub')!=d['subject']:
            raise ValueError('Subscription identity mismatch')
        if time.time()>d['saved_at']+d['expires_in']-300:
            response=requests.post(a.AUTH+'/api/accounts/oauth/token',data={'grant_type':'refresh_token',
                'client_id':d['client_id'],'refresh_token':d['refresh_token'],'resource':a.API},timeout=30)
            if response.status_code!=200: raise RuntimeError('Subscription refresh failed')
            for k,v in response.json().items():
                if k in ['access_token','refresh_token','id_token','token_type','expires_in'] and v is not None:d[k]=v
            d['saved_at']=time.time();a.write_private(p,d)
        # Existing infer checks the model catalog and disables automatic retries.
        make_content=self.call.make_content
        try:
            if 'content' in request:
                content=[]
                for part in request['content']:
                    if part['type']=='text':content.append({'type':'input_text','text':part['text']})
                    else:
                        path=Path(request['images'][part['index']]['path'])
                        content.extend(x for x in make_content('',[path]) if x['type']!='input_text')
                self.call.make_content=lambda prompt,images:content
                result=self.call.infer('',model=self.model,effort=self.effort,profile=self.profile,instructions=request['prompt'])
            else:
                result=self.call.infer(request['prompt'],[Path(x['path']) for x in request['images']],
                                      model=self.model,effort=self.effort,profile=self.profile)
        finally:self.call.make_content=make_content
        if result.get('status')=='completed' and result.get('returned_model')!=self.model:
            result['status']='model_mismatch'
        result['account_check']={'identity_verified':True,'plan':'user-confirmed Pro','remaining_quota':'not exposed'}
        return result
