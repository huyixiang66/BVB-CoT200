"""Two visual geometry refinements against fixed LingBot reference cameras.

This is render/compare/Astra correction, not ICP or a new LingBot inference.
Every round is immutable and uses the identical source images and camera matrices.
"""
import ast,json,shutil,sys,time
from pathlib import Path
from .pipeline import ROOT,read,save,sha,prepare,infer,execute

REFINEMENT = '''
FIXED-LINGBOT-POSE ALIGNMENT ITERATION:
The previous generated scene has been rendered through the EXACT supplied LingBot
camera extrinsics and intrinsics. Each ORIGINAL is followed by its CURRENT RENDER
at the SAME timestamp and camera. Compare all pairs jointly, then correct the
previous bpy geometry: positions, yaw, scale, floor height, support contact and
wall extents/openings where the source images provide evidence. Correct any
target which is outside the rendered view but clearly visible in the original.
Rendered walls can occlude a target; distinguish this from wrong target position.
Do not move the reference cameras, invent a new trajectory, independently transform
groups, or transform the whole coordinate system to hide residual errors. Use the
full-precision matrices in the pair metadata as fixed constraints. Preserve good
geometry and stable instance IDs; make evidence-based corrections rather than
redesigning the scene. Do not change target identity just to match a noisy box.
DINO projected-box IoU is a noisy diagnostic, NOT a verified segmentation, objective
score, or ground truth. Do not optimize boxes at the expense of visual evidence.
The previous script is supplied below. Return a COMPLETE updated bpy script under
the same STRICT SCENE SCOPE and OUTPUT CONTRACT. Do not answer the question or use
an expected answer to move objects. No additional furniture beyond direct supports or
direct physical contacts to targets. Independently review ALL evidenced walls, partitions,
openings and floors; add omitted architecture even when not near a target. Do not remove
walls to improve target visibility. Update the architecture inventory and its evidence.
If no correction is justified, retain geometry.
'''

def pairs(previous,data):
    v=read(previous/'validation.json')
    if not v['camera_alignment_verified'] or not v['geometry_unchanged']:
        raise ValueError('Previous fixed-camera render validation did not pass')
    if sha(previous/'scene.blend')!=v['original_scene_sha256']:
        raise ValueError('Previous scene changed after rendering')
    if len(v['cameras'])!=len(data['frames']):raise ValueError('Incomplete frame pairs')
    result=[]
    for f,c in zip(data['frames'],v['cameras'],strict=True):
        if (f['source_frame_index'],f['timestamp_s'])!=(c['source_frame_index'],c['timestamp_s']):
            raise ValueError('Render does not match source frame')
        if sha(Path(f['path']))!=f['sha256']:raise ValueError('Original image changed')
        rendered=previous/c['render']
        if not rendered.is_file():raise ValueError('Missing reference-camera render')
        result.append((f,rendered))
    return result

def refine(previous,out,index):
    previous=previous.resolve();out=out.resolve()
    if out.exists():raise FileExistsError('Refinement output already exists')
    d=read(previous/'input.json');pair_list=pairs(previous,d)
    out.mkdir(parents=True)
    # Original images remain in round 0; checksums and camera arrays never change.
    save(out/'input.json',d)
    for name in ['question-public.json','pose-reference.json']:
        shutil.copy2(previous/name,out/name)
    original_prompt=(previous/'system-prompt.txt').read_text().split('\nFIXED-LINGBOT-POSE ALIGNMENT ITERATION:')[0]
    prompt=original_prompt.replace(str(previous/'scene.blend'),str(out/'scene.blend'))+REFINEMENT
    (out/'system-prompt.txt').write_text(prompt)
    exp=read(previous/'experiment.json');exp.update(round=index,previous_round=str(previous),
        refinement='fixed LingBot camera render/compare/Astra geometry correction',
        reference_cameras_modified=False,ground_truth_provided=False,
        prompt_sha256=sha(out/'system-prompt.txt'),input_sha256=sha(out/'input.json'))
    save(out/'experiment.json',exp)
    sys.path.insert(0,str(ROOT/'scripts'))
    from subscription_direct_client import request,image_item
    content=[dict(type='input_text',text='PUBLIC QUESTION AND GROUPS: '+json.dumps(
        {k:d[k] for k in ['question','groups','instance_limits']},ensure_ascii=False))]
    feedback=[]
    for f,rendered in pair_list:
        metadata={k:f[k] for k in ['timestamp_s','source_frame_index','groups','c2w_blender','intrinsics_640x480','observations']}
        content.extend([dict(type='input_text',text='ORIGINAL; FIXED CAMERA: '+json.dumps(metadata,separators=(',',':'))),
                        image_item(f['path']),dict(type='input_text',text='CURRENT RENDER at exactly the preceding camera:'),image_item(rendered)])
        feedback.append(dict(source_frame_index=f['source_frame_index'],original_sha256=f['sha256'],render=str(rendered),render_sha256=sha(rendered)))
    diagnostics=read(previous/'validation.json')['box_comparisons']
    content.append(dict(type='input_text',text='NOISY BOX DIAGNOSTICS (not GT): '+json.dumps(diagnostics,separators=(',',':'))))
    content.append(dict(type='input_text',text='PREVIOUS COMPLETE SCRIPT:\n'+(previous/'generated.py').read_text()))
    save(out/'feedback-manifest.json',dict(previous_scene_sha256=sha(previous/'scene.blend'),pairs=feedback))
    result=request(out/'attempt-01',content,instructions=prompt,model='gpt-6-astra',effort='high')
    code=result['answer'].strip()
    if code.startswith('```'):code=code.split('\n',1)[1].rsplit('```',1)[0]
    ast.parse(code);(out/'attempt-01.py').write_text(code);(out/'generated.py').write_text(code)

def report(root):
    rounds=[]
    for folder in sorted(root.glob('round-*')):
        row=dict(round=folder.name)
        if (folder/'attempt-01/request.json').exists():
            r=read(folder/'attempt-01/request.json');u=r.get('usage') or {}
            row.update(status=r['status'],elapsed_s=r.get('elapsed_s'),usage=u)
        if (folder/'validation.json').exists():
            v=read(folder/'validation.json');groups={}
            for b in v['box_comparisons']:groups.setdefault(b['category'],[]).append(b['iou'])
            row.update(rendered=True,scope=v['scope_validation'],box_iou_mean={k:sum(a)/len(a) for k,a in groups.items()})
        rounds.append(row)
    save(root/'progress.json',dict(rounds=rounds,note='Box IoU is diagnostic only; no geometric accuracy claim or QA scoring.'))

def run(prepared,out,blender):
    out=out.resolve()
    if out.exists():raise FileExistsError('Use a fresh iteration experiment directory')
    out.mkdir(parents=True)
    from . import __version__
    save(out/'protocol.json',dict(version='bvb_v10_'+__version__,rounds=[0,1,2],fixed_cameras=True,
         method='Astra visual reprojection correction guided by fixed LingBot poses, not ICP',
         scope='question targets, direct support/contact white cuboids, all evidenced white walls/opening boundaries/floor',
         ground_truth_used=False,regenerate_lingbot=False))
    try:
        for index in range(3):
            folder=out/f'round-{index}'
            print('START_ROUND',index,flush=True)
            if index==0:prepare(prepared,folder);infer(folder,'attempt-01')
            else:refine(out/f'round-{index-1}',folder,index)
            report(out)
            execute(folder,blender);execute(folder,blender,True)
            frozen=read(folder/'experiment.json')
            if sha(folder/'input.json')!=frozen['input_sha256']:
                raise ValueError('Frozen evidence modified by scene execution')
            if sha(folder/'pose-reference.json')!=sha(out/'round-0/pose-reference.json'):
                raise ValueError('Reference cameras changed across rounds')
            report(out);print('COMPLETED_ROUND',index,flush=True)
    finally:report(out)
