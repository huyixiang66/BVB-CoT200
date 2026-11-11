import hashlib
import json
import subprocess
import sys
from pathlib import Path
from . import prompts
from .contracts import validate_regions
from .selection import sha256
from . import v10_adapter

STAGES = ['select','estimate','reconstruct','render','qa']


def write(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8');temp.replace(path)


def read(path): return json.loads(Path(path).read_text(encoding='utf8'))


def parse_json(answer):
    text=answer.strip()
    if text.startswith('```'):text=text.split('\n',1)[1].rsplit('```',1)[0]
    return json.loads(text)


def invoke(out, stage, prompt, image_paths, backend, validator=None, content=None):
    d=out/stage;d.mkdir(exist_ok=True)
    request={'stage':stage,'model':'gpt-6-astra','effort':'high','prompt':prompt,'images':[{'path':str(Path(p).resolve()),'sha256':sha256(p)} for p in image_paths]}
    if content is not None:request['content']=content
    previous=d/'request.json';resultpath=d/'result.json'
    if previous.exists() and read(previous)!=request:
        raise ValueError('Stage inputs changed; use a new run directory')
    if resultpath.exists():
        result=read(resultpath)
        if result.get('status')!='completed':raise RuntimeError('Previous interrupted/failed request; no automatic retry')
    else:
        if backend is None:raise ValueError('An explicit inference backend is required')
        write(previous,request);write(resultpath,{'status':'started','usage':None})
        try:result=backend(request)
        except Exception as exc:
            write(resultpath,{'status':'failed','error_type':type(exc).__name__,'usage':None})
            raise RuntimeError('Inference failed; attempt preserved, usage unknown') from None
        write(resultpath,result)
    if result.get('status')!='completed':raise RuntimeError('Inference did not complete; partial answer preserved')
    if validator:
        value=parse_json(result['answer']);validator(value);write(d/'data.json',value);return value
    return result


def run(config, backend=None, until='qa', selector=None, renderer=None, preparer=None):
    if until not in STAGES:raise ValueError('Unknown stage')
    allowed={'video','out','reference_fps','ffprobe','blender','qa_mode','question',
             'v10_root','poses','frames_root','detections'}
    if set(config)-allowed:raise ValueError('Unexpected config keys; scoring data must stay outside pipeline')
    config=dict(config);config.setdefault('question',prompts.QA_BASE)
    if not isinstance(config['question'],str) or not config['question'].strip():raise ValueError('Empty question')
    out=Path(config['out']).resolve();out.mkdir(parents=True,exist_ok=True)
    source_hash=hashlib.sha256(b''.join(p.read_bytes() for p in sorted(Path(__file__).parent.glob('*.py')))).hexdigest()
    pose_files=['camera-poses.npz','sampling.json','sparse-geometry-corrected.npz']
    if (Path(config['poses'])/'geometry-info.json').exists():pose_files.append('geometry-info.json')
    frozen={'config':config,'video_sha256':sha256(config['video']),'source_sha256':source_hash,
            'upstream_hashes':v10_adapter.source_hashes(config['v10_root']),
            'pose_hashes':{n:sha256(Path(config['poses'])/n) for n in pose_files},
            'detections_sha256':sha256(config['detections']) if config.get('detections') else None}
    lock=out/'run.json'
    if lock.exists() and read(lock)!=frozen:raise ValueError('Run inputs/code changed; use new output directory')
    if not lock.exists():write(lock,frozen)
    try:
        selected=out/'selected.json'
        if not selected.exists():write(selected,(selector or v10_adapter.select)(config,out/'frames'))
        manifest=read(selected);images=[f['file'] for f in manifest['frames']]
        for f in manifest['frames']:
            if sha256(f['file'])!=f['sha256']:raise ValueError('Selected image was modified')
        if until=='select':return manifest
        context='QUESTION: '+config['question']+'\nOriginal frame timestamps (seconds): '+json.dumps([f['timestamp_s'] for f in manifest['frames']])
        if config.get('detections'):
            labels=sorted({d['object'] for line in Path(config['detections']).read_text(encoding='utf8').splitlines()
                           for d in json.loads(line)['detections']})
            context+='\nCached detector category labels (provisional, NOT proof): '+json.dumps(labels,ensure_ascii=False)
            context+='\nUse the matching exact detector label for references when semantically correct; do not invent or rename instances to match detections.'
        regions=invoke(out,'estimate',prompts.REGIONS+'\n'+context,images,backend,validate_regions)
        if until=='estimate':return regions
        prepared=out/'v10_input/prepared.json'
        if not prepared.exists():(preparer or v10_adapter.prepare)(config,out,manifest,regions)
        prep=read(prepared)
        result=invoke(out,'reconstruct',prep['prompt'],prep['images'],backend,content=prep['content'])
        code=v10_adapter.python_answer(result['answer'])
        (out/'reconstruct/generated.py').write_text(code,encoding='utf8')
        if until=='reconstruct':return {'generated_python':str(out/'reconstruct/generated.py')}
        rendered=out/'render';rendered.mkdir(exist_ok=True)
        if not (rendered/'complete.json').exists():
            (renderer or v10_adapter.render)(config,out)
            required=['top.png','scene.blend','reference.json','geometry.json','joint-original.blend']
            write(rendered/'complete.json',{n:sha256(rendered/n) for n in required})
        for name,digest in read(rendered/'complete.json').items():
            if sha256(rendered/name)!=digest:raise ValueError('Rendered artifact modified')
        ref=read(rendered/'reference.json')
        if not ref.get('accepted'):raise ValueError('No unambiguous visible reference; QA blocked')
        if until=='render':return ref
        result=invoke(out,'qa',prompts.qa_prompt(ref['text'],config.get('qa_mode','upper'),config['question']),[rendered/'top.png'],backend)
        write(out/'status.json',{'status':'completed','stage':'qa'})
        return result
    except Exception as exc:
        write(out/'status.json',{'status':'blocked','error_type':type(exc).__name__,'message':str(exc)})
        raise
