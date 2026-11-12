"""Execute an already prepared request; no model call until explicitly invoked."""
import argparse
import ast
import json
from pathlib import Path
from room_size_pipeline.backends import SubscriptionBackend
from room_size_pipeline.pipeline import invoke
from .handoff import digest


def execute(request_path, output_dir, backend, kind):
    request_path=Path(request_path);request=json.loads(request_path.read_text(encoding='utf8'))
    if request['model']!='gpt-6-astra' or request['effort']!='high':raise ValueError('Expected Astra High')
    paths=[]
    for image in request['images']:
        path=Path(image['path'])
        if digest(path)!=image['sha256']:raise ValueError('Request image changed')
        paths.append(path)
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    result=invoke(out,kind,request['prompt'],paths,backend,content=request.get('content'))
    answer=result.get('answer','').strip()
    if answer.startswith('```'):answer=answer.split('\n',1)[1].rsplit('```',1)[0].strip()
    if kind=='reconstruct':
        ast.parse(answer)
        artifact=out/kind/'generated.py';artifact.write_text(answer,encoding='utf8')
    else:
        data=json.loads(answer)
        if kind=='review' and not isinstance(data.get('target_confirmed'),bool):raise ValueError('Invalid review')
        if kind=='qa' and (not isinstance(data.get('answer'),(float,int)) or isinstance(data['answer'],bool) or data['answer']<0):raise ValueError('Invalid QA answer')
        artifact=out/kind/('decision.json' if kind=='review' else 'answer.json')
        artifact.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
    return artifact


def execute_batch(entries, backend):
    """Run a caller-prepared stage list; keep failed attempts and continue other cases."""
    results=[]
    for entry in entries:
        try:
            artifact=execute(entry['request'],entry['output'],backend,entry['kind'])
            results.append({'id':entry['id'],'kind':entry['kind'],'status':'completed','artifact':str(artifact)})
        except Exception as exc:
            results.append({'id':entry['id'],'kind':entry['kind'],'status':'failed','error_type':type(exc).__name__})
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',choices=['review','reconstruct','qa'],required=True)
    p.add_argument('--request',required=True);p.add_argument('--output',required=True)
    p.add_argument('--subscription-client',required=True);p.add_argument('--profile',required=True)
    p.add_argument('--expected-email',required=True);p.add_argument('--pro-confirmed',action='store_true')
    a=p.parse_args();backend=SubscriptionBackend(a.subscription_client,a.profile,a.expected_email,pro_confirmed=a.pro_confirmed)
    print(execute(a.request,a.output,backend,a.kind))


if __name__=='__main__':main()
