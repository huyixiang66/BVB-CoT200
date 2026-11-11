"""Room-size adapter around the upstream V10_2 code, not a replacement modeler."""
import ast
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from .selection import choose_frames, sha256


def load(root):
    root=Path(root).resolve()
    sys.path.insert(0,str(root))
    modules={name:importlib.import_module(name) for name in
             ['bvb_v10.prompt','bvb_v10_2.pipeline','bvb_v10_2.reference',
              'bvb_v6.reconstruction','bvb_v6.runtime.compact_pose_input']}
    for module in modules.values():
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise ValueError('A different upstream checkout is already imported; start a fresh process')
    return modules


def source_hashes(root):
    root=Path(root)
    files=[p for name in ['bvb_v6','bvb_v10','bvb_v10_2'] for p in (root/name).rglob('*')
           if p.suffix in ['.py','.txt','.json'] and '__pycache__' not in p.parts]
    if not all((root/name).is_dir() for name in ['bvb_v6','bvb_v10','bvb_v10_2']):
        raise ValueError('Supply a complete upstream methods/adaptive_frame_selection checkout')
    return {p.relative_to(root).as_posix():sha256(p) for p in sorted(files)}


def select(config,out):
    """Copy cached original frames exactly: no image/pose rebinding or LingBot rerun."""
    modules=load(config['v10_root']);poses=Path(config['poses']);frames_root=Path(config['frames_root']).resolve()
    samples=json.loads((poses/'sampling.json').read_text(encoding='utf8'))['frames']
    cams,intr,sources=modules['bvb_v6.reconstruction'].shared_pose(poses,None)
    if len(cams)!=len(samples) or len(intr)!=len(samples):raise ValueError('Pose count mismatch')
    duration=float(json.loads(subprocess.check_output([config.get('ffprobe','ffprobe'),'-v','error',
                   '-show_format','-of','json',str(config['video'])],text=True))['format']['duration'])
    candidates=[{'timestamp_s':s['timestamp_s'],'c2w_blender':cams[i].tolist()} for i,s in enumerate(samples)]
    times,report=choose_frames(duration,candidates,config.get('reference_fps',.125))
    chosen=set(times);out.mkdir(parents=True,exist_ok=True);frames=[]
    for i,s in enumerate(samples):
        if s['timestamp_s'] not in chosen:continue
        # Upstream pose/frames entries intentionally symlink to the shared original cache.
        if Path(s['filename']).name!=s['filename']:
            raise ValueError('Sampling filename must be a basename')
        source=(frames_root/s['filename']).resolve()
        if sha256(source)!=s['sha256']:
            raise ValueError('Original frame does not match LingBot sampling hash')
        dest=out/f'{i:06d}.jpg';shutil.copy2(source,dest)
        frames.append({'timestamp_s':s['timestamp_s'],'source_frame_index':s['source_frame_index'],
                       'sample_index':i,'file':str(dest.resolve()),'sha256':sha256(dest),
                       'c2w_blender':cams[i].tolist(),'intrinsics_640x480':intr[i].tolist()})
    return {'frames':frames,'selection':report,'shared_pose_sources':sources,'duration_s':duration,
            'video_sha256':sha256(config['video']),
            'source_binding':'Cached JPEG hashes verified against sampling.json; video/cache association supplied by operator.'}


def prepare(config,out,manifest,spec):
    from .pipeline import write
    from .prompts import HARD_DIMENSIONS
    modules=load(config['v10_root']);dest=out/'v10_input';dest.mkdir(exist_ok=True)
    # No per-question category list: the original-frame estimates supply visible scale objects.
    objects=sorted({r['label'] for r in spec['references']})
    colors={o:[.5,.5,.5,1] for o in objects};limits={o:None for o in objects}
    detections={}
    if config.get('detections'):
        for line in Path(config['detections']).read_text(encoding='utf8').splitlines():
            row=json.loads(line);detections[row['source_frame_index']]=row['detections']
    frames=[]
    for original in manifest['frames']:
        f=dict(original);f['path']=f['file'];f['groups']=['room_context']
        f['observations']=[{'object':d['object'],'box_xyxy':d['box_xyxy'],'dino_score':d['score']}
                           for d in detections.get(f['source_frame_index'],[]) if d['object'] in objects]
        frames.append(f)
    refs={'source_hashes':manifest['shared_pose_sources'],
          'rows':[{k:f[k] for k in ['timestamp_s','source_frame_index','c2w_blender','intrinsics_640x480']} for f in frames]}
    transport=modules['bvb_v6.runtime.compact_pose_input'].build_transport(refs)
    for f,c in zip(frames,transport['rows'],strict=True):f['compact_camera']=c
    question={'id':'room','scene_name':'input','question_type':'room_size_estimation',
              'question':config['question'],'objects':objects}
    data={'version':'bvb_v10_2_room_adapter','question':question,'frames':frames,
          'groups':[{'group_id':'room_context','primary_objects':objects,'timestamps_s':[f['timestamp_s'] for f in frames]}],
          'shared_pose_sources':manifest['shared_pose_sources'],'instance_limits':limits,'colors':colors}
    write(dest/'input.json',data);write(dest/'pose-reference.json',refs)
    reference=modules['bvb_v10_2.reference'].build(dest,Path(config['poses']))
    prompt=modules['bvb_v10.prompt'].build(question,limits,colors,out/'render',transport['explanation'])
    prompt+=modules['bvb_v10_2.pipeline'].EXTRA+'\n'+HARD_DIMENSIONS
    prompt+='\nIMMUTABLE REGION DIMENSIONS AND ORIGINAL-FRAME REFERENCE ESTIMATES:\n'+json.dumps(spec,ensure_ascii=False)
    (dest/'system-prompt.txt').write_text(prompt,encoding='utf8')
    images=[];content=[]
    def text(value):content.append({'type':'text','text':value})
    def image(path):images.append(Path(path));content.append({'type':'image','index':len(images)-1})
    text('Question and evidence grouping: '+json.dumps({k:data[k] for k in ['question','groups','instance_limits']},ensure_ascii=False))
    for f in frames:
        text(json.dumps({k:f[k] for k in ['timestamp_s','groups','compact_camera','observations']},ensure_ascii=False));image(f['file'])
    text('SOFT surface anchor hypotheses, not measured dimensions: '+json.dumps(reference['anchors'],ensure_ascii=False))
    for name in ['reference-top.png','reference-oblique.png']:
        text(name+' — empty-space trajectory and soft surface evidence, NOT reconstructed geometry');image(dest/name)
    write(dest/'prepared.json',{'prompt':prompt,'images':[str(p.resolve()) for p in images],'content':content})
    return prompt,images,content


def python_answer(answer):
    code=answer.strip()
    if code.startswith('```'):code=code.split('\n',1)[1].rsplit('```',1)[0]
    tree=ast.parse(code)
    if not any(isinstance(n,ast.Import) and any(a.name=='bpy' for a in n.names) or
               isinstance(n,ast.ImportFrom) and n.module=='bpy' for n in ast.walk(tree)):
        raise ValueError('Expected upstream Blender Python, not a JSON layout')
    return code


def render(config,out):
    """Generated upstream Python runs only in the same Linux isolation pattern as experiments."""
    if not sys.platform.startswith('linux') or not shutil.which('bwrap'):
        raise RuntimeError('Run reconstruction on the Linux server with bubblewrap; no unsandboxed fallback')
    dest=out/'render';dest.mkdir(exist_ok=True)
    sandbox=['nice','-n','15','bwrap','--die-with-parent','--unshare-net','--ro-bind','/','/',
             '--tmpfs','/tmp','--ro-bind',str(out),str(out),
             '--ro-bind',str(Path(__file__).parent),str(Path(__file__).parent),
             '--bind',str(dest),str(dest),'--dev','/dev','--proc','/proc',
             '--chdir',str(dest),'--setenv','HOME','/tmp','--setenv','PYTHONDONTWRITEBYTECODE','1']
    blender=config.get('blender','blender')
    subprocess.run(sandbox+[blender,'-b','-t','2','--python-exit-code','1','--python',str(out/'reconstruct/generated.py')],check=True)
    if not (dest/'scene.blend').is_file():raise ValueError('V10_2 script did not save scene.blend')
    shutil.copy2(dest/'scene.blend',dest/'joint-original.blend')
    # A second Blender process validates the saved result, not model-mutated Python globals.
    subprocess.run(sandbox+[blender,'-b',str(dest/'joint-original.blend'),'-t','2','--python-exit-code','1',
                   '--python',str(Path(__file__).parent/'blender_scene.py'),'--',str(out)],check=True)
