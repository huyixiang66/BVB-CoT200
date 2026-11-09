"""Adapt existing V8 Chat requests to direct OAuth Responses streams."""
from types import SimpleNamespace
from openai import OpenAI
from openai.types.chat import ChatCompletion


def convert_request(kwargs):
    forbidden={'max_completion_tokens','max_tokens','max_output_tokens'} & kwargs.keys()
    if forbidden:raise ValueError('Subscription direct does not support output token caps')
    instructions=[];items=[]
    for message in kwargs['messages']:
        role=message['role'];content=message['content']
        if role in ('system','developer'):
            if not isinstance(content,str):raise ValueError('Instructions must be text')
            instructions.append(content);continue
        if role!='user':raise ValueError('This adapter accepts independent user requests only')
        blocks=[{'type':'text','text':content}] if isinstance(content,str) else content
        converted=[]
        for block in blocks:
            if block['type']=='text':converted.append({'type':'input_text','text':block['text']})
            elif block['type']=='image_url':
                src=block['image_url']
                converted.append({'type':'input_image','image_url':src['url'],'detail':src.get('detail','high')})
            else:raise ValueError('Unsupported content type')
        items.append({'role':'user','content':converted})
    body={'model':kwargs['model'],'input':items,'reasoning':{'effort':kwargs.get('reasoning_effort','high')},'store':False,'stream':True}
    if instructions:body['instructions']='\n\n'.join(instructions)
    fmt=kwargs.get('response_format')
    if fmt:
        if fmt['type']!='json_schema':raise ValueError('Only JSON schema format supported')
        body['text']={'format':{'type':'json_schema',**fmt['json_schema']}}
    return body


class DirectClient:
    def __init__(self,connection,timeout=900):
        self.connection=connection;self.timeout=timeout
        self.chat=SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self,**kwargs):
        body=convert_request(kwargs);answer='';completed=None;refusal=None
        with OpenAI(api_key=self.connection.key,base_url='https://api.openai.com/v1',max_retries=0,timeout=self.timeout) as client:
            with client.responses.create(**body) as events:
                for event in events:
                    if event.type=='response.output_text.delta':answer+=event.delta
                    elif event.type=='response.refusal.delta':refusal=(refusal or '')+event.delta
                    elif event.type=='response.completed':completed=event.response
                    elif event.type in ('response.failed','response.incomplete','error'):
                        err=getattr(getattr(event,'response',None),'error',None)
                        raise RuntimeError('Direct subscription stream '+event.type+' '+str(getattr(err,'code',None)))
        if completed is None:raise RuntimeError('Direct subscription stream has no completion event')
        u=completed.usage;usage=None
        if u:
            raw=u.model_dump();usage={'prompt_tokens':raw['input_tokens'],'completion_tokens':raw['output_tokens'],
                 'total_tokens':raw['total_tokens'],'prompt_tokens_details':{'cached_tokens':raw.get('input_tokens_details',{}).get('cached_tokens',0)},
                 'completion_tokens_details':{'reasoning_tokens':raw.get('output_tokens_details',{}).get('reasoning_tokens',0)}}
        return ChatCompletion.model_validate({'id':completed.id,'object':'chat.completion','created':int(completed.created_at),
             'model':completed.model,'choices':[{'index':0,'finish_reason':'stop','message':{'role':'assistant','content':answer,'refusal':refusal}}],
             'usage':usage})
