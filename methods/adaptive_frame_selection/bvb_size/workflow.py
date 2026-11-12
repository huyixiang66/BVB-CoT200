"""Stage orchestration for prepared object-size evidence; explicit render and review gates."""
import json
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path
from .handoff import prepare_review, prepare_joint, prepare_reconstruction, prepare_qa, write
from .inference import execute


def prepare_case(evidence, base_prompt, work, backend, runtime_root):
    """Review and generate Blender code; executes only when caller supplies backend."""
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    review=work/'review-request.json';prepare_review(evidence,review)
    decision=execute(review,work/'inference',backend,'review')
    joint=work/'joint'
    if not (joint/'input.json').exists():
        prepare_joint(evidence,decision,base_prompt,joint,runtime_root)
    recon=work/'reconstruct-request.json';prepare_reconstruction(joint/'input.json',recon)
    code=execute(recon,work/'inference',backend,'reconstruct')
    return {'status':'awaiting_server_build_and_visual_review','code':str(code),'joint':str(joint/'input.json')}


def build_server_case(case_dir, virtual_dir, blender, validator_dir):
    """Build saved generated Python inside the verified Linux bwrap adapter."""
    if not sys.platform.startswith('linux') or not shutil.which('bwrap'):
        raise RuntimeError('Linux bubblewrap is required; no unsandboxed fallback')
    case=Path(case_dir).resolve();joint=case/'joint';code=case/'inference/reconstruct/generated.py'
    if not code.is_file() or not (joint/'input.json').is_file():raise ValueError('Missing prepared joint or generated code')
    validator=Path(validator_dir)
    for name in ('validate.py','v10_contract.py'):
        source=validator/name
        if not source.is_file():raise ValueError('Missing upstream validation file: '+name)
        shutil.copy2(source,joint/name)
    shutil.copy2(code,joint/'generated.py')
    cmd=['nice','-n','15',sys.executable,str(Path(__file__).with_name('linux_build.py')),
         '--out',str(joint),'--virtual',virtual_dir,'--blender',str(Path(blender).resolve())]
    subprocess.run(cmd,check=True,timeout=660)
    scene=joint/'scene.blend'
    if not scene.is_file():raise RuntimeError('Build completed without scene.blend')
    return {'status':'built_needs_view_review','scene':str(scene),'scene_sha256':hashlib.sha256(scene.read_bytes()).hexdigest()}


def render_server_views(scene, target, reference, out, blender, target_spec=None):
    """Run trusted camera code on a scene in a network-isolated CPU-only sandbox."""
    if not sys.platform.startswith('linux') or not shutil.which('bwrap'):
        raise RuntimeError('Linux bubblewrap is required; no unsandboxed fallback')
    scene=Path(scene).resolve();out=Path(out).resolve();binary=Path(blender).resolve()
    if not scene.is_file() or not binary.is_file():raise ValueError('Missing scene or Blender')
    if out.exists():raise ValueError('View output already exists; inspect existing result')
    out.mkdir(parents=True)
    package=Path(__file__).resolve().parent
    cmd=['nice','-n','15','bwrap','--unshare-all','--die-with-parent','--new-session']
    for name in ('/usr','/bin','/sbin','/lib','/lib64','/etc'):
        if Path(name).exists():cmd+=['--ro-bind',name,name]
    cmd+=['--proc','/proc','--dev','/dev','--tmpfs','/tmp',
          '--ro-bind',str(binary.parent),str(binary.parent),'--ro-bind',str(package),'/opt/bvb_size',
          '--ro-bind',str(scene),'/input/scene.blend','--bind',str(out),'/output',
          '--chdir','/output','--clearenv','--setenv','HOME','/tmp',
          '--setenv','PATH','/usr/bin:/bin','--setenv','OMP_NUM_THREADS','2',
          '--setenv','OPENBLAS_NUM_THREADS','2',str(binary),'-b','/input/scene.blend',
          '--disable-autoexec','-t','2','--python-exit-code','1']
    if reference:
        cmd+=['--python','/opt/bvb_size/in_situ.py','--','--blend','/input/scene.blend',
              '--target',target,'--reference',reference,'--out','/output/views']
        manifest=out/'views/result.json'
    else:
        if target_spec is None:raise ValueError('Target-only rendering requires a reviewed complete mesh specification')
        spec=json.loads(Path(target_spec).read_text(encoding='utf8'))
        if not spec.get('target',{}).get('verified') or spec['target'].get('category')!=target:
            raise ValueError('Target mesh identity must be reviewed')
        spec['candidates']=[]
        (out/'target-spec.json').write_text(json.dumps(spec,ensure_ascii=False),encoding='utf8')
        cmd+=['--python','/opt/bvb_size/render_size.py','--','/output/target-spec.json','/output/views']
        manifest=out/'views/manifest.json'
    subprocess.run(cmd,check=True,timeout=660)
    if not manifest.is_file():raise RuntimeError('Renderer did not produce its manifest')
    if not reference:
        from .handoff import target_views
        target_views(manifest,manifest.parent/'result.json');manifest=manifest.parent/'result.json'
    views=json.loads(manifest.read_text(encoding='utf8'))
    if views.get('mode') not in ('single','pair','target_only'):
        return {'status':'unresolved','views':str(manifest)}
    return {'status':'awaiting_visual_review','views':str(manifest)}


def record_visual_review(views, out, usable, notes):
    """Record a human visual decision bound to actual image bytes, never auto-approve."""
    views=Path(views);data=json.loads(views.read_text(encoding='utf8'))
    if not isinstance(usable,bool) or not notes.strip():raise ValueError('Explicit decision and notes required')
    if data.get('mode') not in ('single','pair','target_only') and usable:
        raise ValueError('Unresolved views cannot be approved')
    hashes={}
    for name in data.get('images',[]):
        if Path(name).name!=name:raise ValueError('Unsafe image path')
        hashes[name]=hashlib.sha256((views.parent/name).read_bytes()).hexdigest()
    if usable and not hashes:raise ValueError('No rendered image')
    record={'usable':usable,'notes':notes,'views_sha256':hashlib.sha256(views.read_bytes()).hexdigest(),
            'image_sha256':hashes,'review_type':'visual_inspection'}
    write(out,record);return record


def prepare_reviewed_qa(joint, views, visual_review, work, backend):
    """Use actual rendering plus all original frames only after explicit visual review."""
    review=json.loads(Path(visual_review).read_text(encoding='utf8'))
    if review.get('usable') is not True:raise ValueError('Views not visually approved')
    if review.get('views_sha256') != hashlib.sha256(Path(views).read_bytes()).hexdigest():
        raise ValueError('Visual approval does not match views manifest')
    manifest=json.loads(Path(views).read_text(encoding='utf8'))
    hashes=review.get('image_sha256',{})
    for name in manifest.get('images',[]):
        image=Path(views).parent/name
        if hashes.get(name)!=hashlib.sha256(image.read_bytes()).hexdigest():
            raise ValueError('Visual approval does not match rendered image')
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    req=work/'qa-request.json';prepare_qa(joint,views,req,include_frames=True)
    return execute(req,work/'inference',backend,'qa')


def batch(items, handler):
    """Process prepared cases sequentially without retrying a failed model request."""
    results=[]
    for item in items:
        try: results.append({'id':item['id'],'result':handler(item),'status':'completed'})
        except Exception as exc: results.append({'id':item['id'],'status':'failed','error_type':type(exc).__name__,'reason':str(exc)})
    return results
