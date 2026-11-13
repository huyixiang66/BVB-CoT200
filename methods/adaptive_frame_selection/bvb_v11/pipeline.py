"""V10-compatible soft-reference experiment; no geometry clamping or local-scene stitching."""
import ast,sys,time,json
from pathlib import Path
from bvb_v10.pipeline import ROOT,prepare as prepare_v10,read,save,sha,execute
from .reference import build
EXTRA='''

V10_2 SOFT SPATIAL REFERENCE EXPERIMENT (these instructions take priority):
The additional diagrams show an EMPTY reference space: the complete LingBot camera path,
selected frame camera positions and headings, and provisional object surface anchors.
The camera path is NOT a room boundary. Stars/boxes are NOT verified objects or ground truth.
Anchors come from DINO ROI-filtered sparse LingBot points; background, wrong detections,
partial visible surfaces and pose drift can shift them. Their ranges are observed surfaces,
NOT full object bounding boxes, exact centers or confidence intervals. Confidence is heuristic.
Use them as soft evidence to reconcile disconnected views, not a mandatory placement.
You retain freedom to choose coherent object positions, orientation, size, walls and layout
from the ORIGINAL video images. You may move, reject or reinterpret an anchor when image
or cross-view evidence contradicts it. Missing anchors mean unknown depth, not missing objects.
There is NO forced per-region coordinate transform, no post-generation scene registration,
no clamping of output coordinates to anchors and no geometric refinement iteration.
Prefer the supplied common frame and approximate scale for interpretable camera comparisons,
but do not force walls/floors into a skewed layout merely to fit noisy trajectory/anchor evidence.
All V10 target/context/architecture scope and output contracts still apply.
Never create trajectory, camera-frustum, anchor, uncertainty-box or diagram-label meshes
in the reconstructed room. They are input evidence only. Do not answer the QA question.

Cluster labels and bounding boxes may be false positives, even after prior review; independently determine from the original images and the wording of the question whether each candidate really is an instance of the queried object category.
Do not reconstruct a candidate merely because its cluster is labeled with that category: omit it if the images show a different object, and explicitly record ambiguity rather than treating an uncertain identity as confirmed.
'''
def replace_pose_reference(out, poses):
    """Rebind identical selected images to a new full-video pose estimate."""
    import numpy as np
    from bvb_v10.pipeline import METHOD
    sys.path.insert(0,str(METHOD))
    from bvb_v6.reconstruction import shared_pose
    from bvb_v6.runtime.compact_pose_input import build_transport
    from bvb_v10.prompt import build as make_prompt
    d=read(out/'input.json');previous=d['shared_pose_sources']
    cams,intr,sources=shared_pose(poses,None)
    samples=read(poses/'sampling.json')['frames'];indices={r['source_frame_index']:i for i,r in enumerate(samples)}
    for f in d['frames']:
        i=indices[f['source_frame_index']];s=samples[i]
        assert abs(s['timestamp_s']-f['timestamp_s'])<1e-6
        assert s['sha256']==f['sha256']==sha(Path(f['path'])), 'Changed image, not a pose-only comparison'
        f['c2w_blender']=cams[i].tolist();f['intrinsics_640x480']=intr[i].tolist()
    refs=dict(source_hashes=sources,rows=[{k:f[k] for k in ['timestamp_s','source_frame_index','c2w_blender','intrinsics_640x480']} for f in d['frames']])
    transport=build_transport(refs)
    for f,c in zip(d['frames'],transport['rows'],strict=True):f['compact_camera']=c
    d['shared_pose_sources']=sources;save(out/'input.json',d);save(out/'pose-reference.json',refs)
    save(out/'pose-reference-vlm.json',transport)
    (out/'system-prompt.txt').write_text(make_prompt(d['question'],d['instance_limits'],d['colors'],out,transport['explanation']))
    exp=read(out/'experiment.json');exp.update(input_sha256=sha(out/'input.json'),prompt_sha256=sha(out/'system-prompt.txt'),
        pose_replacement=dict(previous_sources=previous,new_sources=sources,frames_and_reviewed_groups_unchanged=True))
    save(out/'experiment.json',exp)

def prepare(source,out,poses,replace_poses=False):
    from .semantic_recovery import ensure_prepared
    source=ensure_prepared(source,poses)
    prepare_v10(source,out)
    if replace_poses:replace_pose_reference(out,poses)
    ref=build(out,poses)
    prompt=(out/'system-prompt.txt').read_text()+EXTRA
    (out/'system-prompt.txt').write_text(prompt)
    exp=read(out/'experiment.json');exp.update(version='bvb_v11',prompt_sha256=sha(out/'system-prompt.txt'),reference_sha256=sha(out/'spatial-reference.json'),reference_images={f:sha(out/f) for f in ['reference-top.png','reference-oblique.png']},soft_anchors=True,full_trajectory_reference=True,forced_registration=False,geometry_clamping=False,round=0)
    save(out/'experiment.json',exp)
    return ref

def prepare_reference_images(out, experiment, attempt, max_side=1024):
    """Keep display originals; record the exact smaller reference images sent to VLM."""
    from PIL import Image
    folder=out/attempt/'reference-images'
    folder.mkdir(parents=True,exist_ok=True)
    records=[]
    for name,expected_hash in experiment['reference_images'].items():
        source=out/name
        assert sha(source)==expected_hash
        destination=folder/name
        with Image.open(source) as im:
            original_size=list(im.size)
            im.thumbnail((max_side,max_side),Image.Resampling.LANCZOS)
            im.save(destination,format='PNG')
            sent_size=list(im.size)
        records.append(dict(name=name,path=str(destination),source_sha256=expected_hash,
                            sha256=sha(destination),original_size=original_size,sent_size=sent_size))
    save(out/attempt/'reference-image-inputs.json',dict(max_side=max_side,images=records))
    return records

def infer(out,attempt='attempt-01'):
    d=read(out/'input.json');exp=read(out/'experiment.json');ref=read(out/'spatial-reference.json')
    assert sha(out/'input.json')==exp['input_sha256'] and sha(out/'system-prompt.txt')==exp['prompt_sha256']
    assert sha(out/'spatial-reference.json')==exp['reference_sha256']
    sys.path.insert(0,str(ROOT/'scripts'))
    from subscription_direct_client import request,image_item
    content=[dict(type='input_text',text='Question and evidence grouping: '+json.dumps({k:d[k] for k in ['question','groups','instance_limits']},ensure_ascii=False))]
    # Exactly the V10 per-frame originals and pose encodings, then two extra geometry references.
    for f in d['frames']:
        assert sha(Path(f['path']))==f['sha256']
        content+=[dict(type='input_text',text=json.dumps({k:f[k] for k in ['timestamp_s','groups','compact_camera','observations']},separators=(',',':'))),image_item(f['path'])]
    compact=json.loads(json.dumps(ref['anchors']),parse_float=lambda x:round(float(x),4))
    content.append(dict(type='input_text',text='SOFT references only. Full trajectory is visualized below; no walls or object geometry are supplied. Anchor hypotheses: '+json.dumps(compact,separators=(',',':'))))
    for item in prepare_reference_images(out,exp,attempt):
        content.extend([dict(type='input_text',text=item['name']+' — empty reference space, not a reconstruction'),image_item(item['path'])])
    if attempt!='attempt-01' and (out/'repair-feedback.txt').exists() and (out/'generated.py').exists():content.append(dict(type='input_text',text='Correct only the execution/schema errors below and return a complete script.\n'+(out/'repair-feedback.txt').read_text()+'\n'+(out/'generated.py').read_text()))
    result=request(out/attempt,content,instructions=(out/'system-prompt.txt').read_text(),model='gpt-6-astra',effort='high')
    code=result['answer'].strip()
    if code.startswith('```'):code=code.split('\n',1)[1].rsplit('```',1)[0]
    ast.parse(code);(out/(attempt+'.py')).write_text(code);(out/'generated.py').write_text(code)

def run(out,blender=None):
    from .common import BLENDER
    from .driver import run as execute_workflow
    return execute_workflow(out,blender or BLENDER)
