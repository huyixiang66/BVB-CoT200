"""Whole-scene room sampling -> compact scene parameters -> existing geometry QA."""
import json,os,subprocess,sys
from pathlib import Path
from bvb_v10.pipeline import read,save,sha
from .common import state
from .compact_inference import infer,SceneSpecError

CHECKER='''import bpy,json,sys
from pathlib import Path
from mathutils import Vector
out=Path(sys.argv[sys.argv.index('--')+1]).resolve()
objects=[o for o in bpy.context.scene.objects if o.type=='MESH']
floors=[o for o in objects if o.get('bvb_role')=='floor']
assert floors, 'Room needs floor geometry'
for o in floors:
    points=[o.matrix_world @ Vector(v) for v in o.bound_box]
    spans=[max(p[i] for p in points)-min(p[i] for p in points) for i in range(3)]
    assert spans[0]>0 and spans[1]>0 and spans[2]<min(spans[:2]), 'Floor must be horizontal and nondegenerate'
q=json.loads((out/'question-public.json').read_text())
q['qa_id']=str(q['id']);q['question_type']='room_size_estimation';q['targets']=[]
q['room']={'floor':[o.name for o in floors],'ceiling':[],'walls':[o.name for o in objects if o.get('bvb_role')=='wall'],'remove_walls':False}
(out/'selector-question.json').write_text(json.dumps(q))
(out/'room-validated.json').write_text(json.dumps({'floors':[o.name for o in floors],'mesh_count':len(objects)}))
'''

def prepare(selection,poses,video,out):
    from bvb_v6.reconstruction import shared_pose
    from bvb_v6.runtime.compact_pose_input import build_transport
    selection,poses,video,out=map(lambda p:Path(p).resolve(),(selection,poses,video,out))
    identity=dict(selection_sha256=sha(selection),sampling_sha256=sha(poses/'sampling.json'),video_sha256=sha(video))
    if (out/'room-build-input.json').exists():
        if read(out/'room-build-input.json')!=identity:raise ValueError('Room input changed; use fresh output')
        return
    if out.exists():raise FileExistsError('Use fresh room-build output')
    data=read(selection);q=data['question'];groups=[g for gs in data['arms'].values() for g in gs]
    if not q['question_type'].startswith('room_size_estimation') or len(groups)!=1:raise ValueError('Room needs one whole-scene group')
    if groups[0].get('sampling_fps')!=1:raise ValueError('Room requires position-aware 1fps sampling')
    cams,intr,sources=shared_pose(poses,None);sampling=read(poses/'sampling.json')
    if sampling['video_sha256']!=identity['video_sha256']:raise ValueError('Video/pose mismatch')
    lookup={r['source_frame_index']:(i,r) for i,r in enumerate(sampling['frames'])};frames=[]
    for r in groups[0]['selected']:
        path=Path(r['image']);path=path if path.is_absolute() else selection.parent/path
        i,canonical=lookup[r['source_frame_index']]
        if sha(path)!=r['frame_sha256'] or canonical['sha256']!=r['frame_sha256'] or abs(canonical['timestamp_s']-r['timestamp_s'])>1e-6:raise ValueError('Room frame/pose mismatch')
        frames.append(dict(path=str(path),sha256=r['frame_sha256'],timestamp_s=r['timestamp_s'],source_frame_index=r['source_frame_index'],groups=[groups[0]['group_id']],observations=[],c2w_blender=cams[i].tolist(),intrinsics_640x480=intr[i].tolist()))
    transport=build_transport(dict(rows=frames))
    for f,c in zip(frames,transport['rows'],strict=True):f['compact_camera']=c
    public={k:q[k] for k in ['id','scene_name','question','question_type','objects','roles','options'] if k in q}
    packet=dict(question=public,frames=frames,groups=[dict(group_id=groups[0]['group_id'],evidence_role='full_scene')],room_mode=True,colors={},instance_limits={},shared_pose_sources=sources)
    save(out/'input.json',packet);save(out/'pose-reference-vlm.json',transport);save(out/'question-public.json',public)
    save(out/'experiment.json',dict(version='bvb_v13',task_adapter='room_size',reconstruction_output='compact-scene-v2',input_sha256=sha(out/'input.json')))
    save(out/'room-build-input.json',identity)

def execute(out,blender):
    if sys.platform!='darwin':raise RuntimeError('Use the existing isolated macOS Blender backend')
    for name in ['home','tmp']:(out/name).mkdir(exist_ok=True)
    sandbox='(version 1)\n(allow default)\n(deny network*)\n(deny file-read* (subpath '+json.dumps(str(Path.home()))+'))\n(allow file-read* (subpath '+json.dumps(str(out))+'))\n(deny file-write*)\n(allow file-write* (subpath '+json.dumps(str(out))+') (subpath "/private/tmp") (subpath "/tmp") (literal "/dev/null"))\n'
    (out/'execution.sb').write_text(sandbox);(out/'validate-room.py').write_text(CHECKER)
    cmd=['/usr/bin/sandbox-exec','-f',str(out/'execution.sb'),blender,'--background','--disable-autoexec','--python-exit-code','1','--factory-startup','--python',str(out/'generated.py'),'--python',str(out/'validate-room.py'),'--',str(out)]
    with (out/'build.log').open('w') as log:
        result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=600,env=dict(PATH='/usr/bin:/bin:/usr/sbin:/sbin',HOME=str(out/'home'),TMPDIR=str(out/'tmp')))
    if result.returncode:
        (out/'repair-feedback.txt').write_text((out/'build.log').read_text()[-5000:]);raise RuntimeError('Room scene validation failed')

def run(selection,poses,video,out,blender,selector_blender=None):
    from .room import run as qa
    out=Path(out).resolve();prepare(selection,poses,video,out)
    if (out/'qa/answer.json').exists():
        from .progressive_qa import completion_stage
        state(out,completion_stage(out));return
    existing=[int(p.name.split('-')[1]) for p in out.glob('attempt-*') if p.is_dir() and p.name.split('-')[1].isdigit()]
    start=max(existing,default=1)
    for n in range(start,4):
        if (out/'room-validated.json').exists():break
        state(out,'reconstructing',attempt=n)
        try:infer(out,f'attempt-{n:02d}')
        except SceneSpecError:
            if n==3:raise
            continue
        try:execute(out,blender)
        except RuntimeError:
            if n==3:raise
    if not (out/'room-validated.json').exists():raise RuntimeError('Room needs validated geometry before QA')
    qa(out/'scene.blend',Path(video),out/'selector-question.json',out,selector_blender or blender)
    answer=read(out/'qa/answer.json');answer.update(source_scene_reused=False,reconstruction_output='compact-scene-v2');save(out/'qa/answer.json',answer)
