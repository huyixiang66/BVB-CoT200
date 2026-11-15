"""Per-request inference receipts. No credentials or image payloads are persisted."""
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import uuid


def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2)); os.replace(tmp, path)


def install(job, experiment):
    import subscription_direct_client as direct
    from bvb_v6.direct_subscription import DirectClient, convert_request
    root = Path(job)/'cost-calls'; root.mkdir(exist_ok=True)
    phase = {'name': 'preflight'}
    def set_phase(name): phase['name'] = name
    @contextlib.contextmanager
    def slot():
        folder = Path(experiment)/'api-slots'; folder.mkdir(exist_ok=True)
        while True:
            for n in range(6):
                f = (folder/str(n)).open('a')
                try: fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError: f.close(); continue
                try: yield
                finally: f.close()
                return
            time.sleep(.2)
    def measure(body, invoke, logical_path=None):
        stamp = time.time(); key = str(time.time_ns())+'-'+uuid.uuid4().hex[:8]
        content = [c for x in body.get('input', []) for c in x.get('content', [])]
        texts = [x.get('text', '') for x in content if x.get('type') == 'input_text']
        row = dict(id=key, phase=phase['name'], model=body.get('model'), started_unix=stamp,
            status='waiting', logical_path=str(logical_path) if logical_path else None,
            image_count=sum(c.get('type') == 'input_image' for c in content),
            input_text_chars=len(body.get('instructions', ''))+sum(map(len, texts)),
            request_sha256=hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest(),
            usage=None, usage_known=False, output_text_chars=None)
        path = root/(key+'.json'); write(path, row)
        try:
            with slot():
                row.update(status='started', inference_started_unix=time.time(), queue_s=time.time()-stamp)
                write(path, row)
                result = invoke()
                if isinstance(result, dict):
                    usage=result.get('usage'); answer=result.get('answer', '')
                    row.update(response_id=result.get('response_id'), returned_model=result.get('returned_model'))
                else:
                    usage=result.usage.model_dump() if result.usage else None
                    answer=result.choices[0].message.content or ''
                    row.update(response_id=result.id, returned_model=result.model)
                row.update(status='completed', usage=usage, usage_known=usage is not None,
                           output_text_chars=len(answer))
                return result
        except Exception as exc:
            row.update(status='failed', error_type=type(exc).__name__)
            # Direct transport persists usage even when its postcondition raises.
            p=Path(logical_path)/'request.json' if logical_path else None
            if p and p.exists():
                receipt=json.loads(p.read_text()); row.update(usage=receipt.get('usage'),
                    usage_known=receipt.get('usage') is not None, transport_status=receipt.get('status'))
            raise
        finally:
            row.update(ended_unix=time.time(), elapsed_s=time.time()-stamp)
            row['inference_s']=max(0, row['ended_unix']-row.get('inference_started_unix',row['ended_unix']))
            write(path,row)
    original = direct.request
    def request(folder, content, **kwargs):
        # A completed receipt is a resume, not a new inference call.
        if (Path(folder)/'request.json').exists(): return original(folder,content,**kwargs)
        body=dict(model=kwargs.get('model','gpt-6-astra'), input=[dict(role='user',content=content)],
                  instructions=kwargs.get('instructions') or '')
        return measure(body, lambda: original(folder,content,**kwargs), folder)
    direct.request = request
    old_create = DirectClient.create
    def create(self, **kwargs):
        return measure(convert_request(kwargs), lambda: old_create(self,**kwargs))
    DirectClient.create = create
    return set_phase


def tokens(usage):
    if usage is None: return None
    i=usage.get('input_tokens',usage.get('prompt_tokens',0))
    o=usage.get('output_tokens',usage.get('completion_tokens',0))
    c=(usage.get('input_tokens_details') or usage.get('prompt_tokens_details') or {}).get('cached_tokens',0) or 0
    r=(usage.get('output_tokens_details') or usage.get('completion_tokens_details') or {}).get('reasoning_tokens',0) or 0
    return dict(input=i,cached_input=c,uncached_input=i-c,output=o,reasoning=r,total=i+o)
