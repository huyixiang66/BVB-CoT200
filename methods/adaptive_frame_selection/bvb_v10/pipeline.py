"""Reusable joint reconstruction, separate from immutable V8 experiments."""
import ast,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
from .prompt import build as make_prompt

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
METHOD=ROOT
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2))

def prepare(source,out):
    source=source.resolve();out=out.resolve()
    if out.exists():raise FileExistsError('Use a fresh V10 directory: '+str(out))
    m=read(source/'manifest.json')
    if not m.get('semantic_review_required'):raise ValueError('V10 requires Astra-reviewed groups; no silent audit bypass')
    if m.get('object_size_reference_mode'):raise ValueError('Scale-reference mode is outside V10 minimal-support scope')
    if not m['groups']:raise ValueError('No observed groups; global/temporal tasks need explicit supplemental evidence')
    sys.path.insert(0,str(METHOD))
    from bvb_v6.semantic_review import reconstruction_gate
    for g in m['groups']:reconstruction_gate(source/g['folder'])
    question={k:m['question'][k] for k in ['id','scene_name','question','question_type','options','objects','roles'] if k in m['question']}
    limits=m['final_instance_policy']['limits'];palette=m['colors'];frames={};groups=[];sources=None;encoding=None
    # Validate the complete input before writing the new directory.
    for g in m['groups']:
        folder=source/g['folder'];rr=read(folder/'input-frames.json');refs=read(folder/'pose-reference.json');compact=read(folder/'pose-reference-vlm.json')
        if sources is not None and sources!=refs['source_hashes']:raise ValueError('Incompatible shared pose frames')
        if encoding is not None and encoding!=compact['explanation']:raise ValueError('Incompatible compact camera encodings')
        sources=refs['source_hashes'];encoding=compact['explanation']
        groups.append(dict(group_id=g['group_id'],primary_objects=g['objects'],timestamps_s=[r['timestamp_s'] for r in rr]))
        for f,p,co in zip(rr,refs['rows'],compact['rows'],strict=True):
            if (f['source_frame_index'],f['timestamp_s'])!=(p['source_frame_index'],p['timestamp_s']):raise ValueError('Frame/pose mismatch')
            if sha(Path(f['path']))!=f['sha256']:raise ValueError('Image changed')
            fid=f['source_frame_index']
            if fid in frames:
                if frames[fid]['sha256']!=f['sha256'] or frames[fid]['c2w_blender']!=p['c2w_blender'] or frames[fid]['intrinsics_640x480']!=p['intrinsics_640x480']:raise ValueError('Duplicate frame has conflicting evidence')
            else:
                frames[fid]=dict(path=f['path'],sha256=f['sha256'],timestamp_s=f['timestamp_s'],source_frame_index=fid,
                    groups=[],observations=[],compact_camera=co,c2w_blender=p['c2w_blender'],intrinsics_640x480=p['intrinsics_640x480'])
            frames[fid]['groups'].append(g['group_id']);frames[fid]['observations'].extend(f['observations'])
    out.mkdir(parents=True);(out/'frames').mkdir()
    rows=sorted(frames.values(),key=lambda f:f['timestamp_s'])
    for f in rows:
        dest=out/'frames'/f'frame-{f["source_frame_index"]:08d}.jpg';shutil.copy2(f['path'],dest);f['path']=str(dest)
    data=dict(version='bvb_v10',question=question,groups=groups,frames=rows,shared_pose_sources=sources,
              instance_limits=limits,colors=palette,scope='targets_direct_support_or_contact_all_evidenced_architecture',architecture_inventory_required=True)
    save(out/'input.json',data);save(out/'question-public.json',question)
    save(out/'pose-reference.json',dict(rows=[{k:f[k] for k in ['timestamp_s','source_frame_index','c2w_blender','intrinsics_640x480']} for f in rows]))
    (out/'system-prompt.txt').write_text(make_prompt(question,limits,palette,out,encoding))
    save(out/'experiment.json',dict(version='bvb_v10',source_prepared=str(source),source_manifest_sha256=sha(source/'manifest.json'),
      frame_count=len(rows),group_count=len(groups),model='gpt-6-astra',reasoning='high',provider='subscription_direct',
      scope=data['scope'],joint_generation=True,previous_geometry_provided=False,ground_truth_provided=False,
      geometry_alignment='shared-pose-conditioned generation, not numerical geometric registration',
      prompt_sha256=sha(out/'system-prompt.txt'),input_sha256=sha(out/'input.json'),
      frontend_lock_sha256=sha(HERE/'method-lock.json')))
    print(json.dumps(dict(stage='prepared',out=str(out),groups=len(groups),frames=len(rows),api_calls=0)))

def infer(out,attempt):
    out=out.resolve();d=read(out/'input.json');exp=read(out/'experiment.json')
    if sha(out/'input.json')!=exp['input_sha256'] or sha(out/'system-prompt.txt')!=exp['prompt_sha256']:raise ValueError('Frozen inputs changed')
    sys.path.insert(0,str(ROOT/'scripts'))
    from subscription_direct_client import request,image_item
    content=[dict(type='input_text',text='Question and evidence grouping: '+json.dumps({k:d[k] for k in ['question','groups','instance_limits']},ensure_ascii=False))]
    for f in d['frames']:
        if sha(Path(f['path']))!=f['sha256']:raise ValueError('Input frame changed')
        content+=[dict(type='input_text',text=json.dumps({k:f[k] for k in ['timestamp_s','groups','compact_camera','observations']},separators=(',',':'))),image_item(f['path'])]
    if attempt!='attempt-01':
        content.append(dict(type='input_text',text='Fix the reported execution/schema error; return the full script.\n'+(out/'repair-feedback.txt').read_text()+'\n'+(out/'generated.py').read_text()))
    result=request(out/attempt,content,instructions=(out/'system-prompt.txt').read_text(),model='gpt-6-astra',effort='high')
    code=result['answer'].strip()
    if code.startswith('```'):code=code.split('\n',1)[1].rsplit('```',1)[0]
    ast.parse(code);(out/(attempt+'.py')).write_text(code);(out/'generated.py').write_text(code)

def execute(out,blender,render=False):
    out=out.resolve()
    if sys.platform!='darwin' or not Path('/usr/bin/sandbox-exec').exists():
        raise RuntimeError('This build adapter requires macOS sandbox-exec; use an isolated Blender runner on other hosts')
    for name in ['home','tmp']:(out/name).mkdir(exist_ok=True)
    sandbox='(version 1)\n(allow default)\n(deny network*)\n(deny file-read* (subpath '+json.dumps(str(Path.home()))+'))\n(allow file-read* (subpath '+json.dumps(str(out))+'))\n(deny file-write*)\n(allow file-write* (subpath '+json.dumps(str(out))+') (subpath "/private/tmp") (subpath "/tmp") (literal "/dev/null"))\n'
    (out/'execution.sb').write_text(sandbox)
    env=dict(PATH='/usr/bin:/bin:/usr/sbin:/sbin',HOME=str(out/'home'),TMPDIR=str(out/'tmp'))
    cmd=['/usr/bin/sandbox-exec','-f',str(out/'execution.sb'),blender,'--background','--disable-autoexec','--python-exit-code','1']
    if render:
        shutil.copy2(HERE/'runtime/render.py',out/'render.py');shutil.copy2(HERE/'contract.py',out/'v10_contract.py')
        cmd += [str(out/'scene.blend'),'--python',str(out/'render.py'),'--',str(out)]
    else:
        # Validation runs in the same Blender process before results are accepted.
        shutil.copy2(HERE/'runtime/validate.py',out/'validate.py');shutil.copy2(HERE/'contract.py',out/'v10_contract.py')
        cmd += ['--factory-startup','--python',str(out/'generated.py'),'--python',str(out/'validate.py'),'--',str(out)]
    name='render' if render else 'build'
    with (out/(name+'.log')).open('w') as log:r=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
    if r.returncode:
        tail=(out/(name+'.log')).read_text()[-6000:];(out/'repair-feedback.txt').write_text(tail)
        raise RuntimeError(name+' failed; inspect log; no automatic new model call')
    print(name+' completed: '+str(out))
