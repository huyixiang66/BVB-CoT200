"""Route planning adapter for V10_2: architectural context and three walled views."""
import json,re,shutil,sys
from pathlib import Path
from bvb_v10.pipeline import ROOT,METHOD,read,save,sha
from . import reliability,pipeline

def route_steps(question):
    steps=re.findall(r'(?:^|\s)(\d+)\.\s*(.*?)(?=\s+\d+\.\s|You have reached|$)',question)
    if not steps:raise ValueError('No numbered route steps')
    # Public Q6383 repeats printed step 2. Preserve that text, but index actions
    # by their unambiguous order of appearance rather than silently losing one.
    rows=[dict(step=i,printed_step=int(n),instruction=t.strip()) for i,(n,t) in enumerate(steps,1)]
    return rows

def prepare(source,out,poses,replace_poses=False):
    reliability.prepare(source,out,poses,replace_poses)
    d=read(out/'input.json');q=d['question'];assert q['question_type']=='route_planning'
    packet=out.parent.parent/'packets'/q['scene_name'];layout=read(packet/'layout-frames.json')['frames']
    sys.path.insert(0,str(METHOD))
    from bvb_v6.reconstruction import shared_pose
    from bvb_v6.runtime.compact_pose_input import build_transport
    cams,intr,sources=shared_pose(poses,None)
    if sources!=d['shared_pose_sources']:raise ValueError('Layout and target poses use different references')
    samples=read(poses/'sampling.json')['frames'];index={r['source_frame_index']:i for i,r in enumerate(samples)}
    frames={f['source_frame_index']:f for f in d['frames']};added=0
    for r in layout:
        fid=r['source_frame_index'];i=index[fid];p=packet/'cache'/r['frame']
        assert sha(p)==r['frame_sha256']==samples[i]['sha256']
        assert abs(r['timestamp_s']-samples[i]['timestamp_s'])<1e-6
        if fid in frames:frames[fid]['groups'].append('uniform-layout-context');continue
        dest=out/'frames'/f'frame-{fid:08d}.jpg';shutil.copy2(p,dest);added+=1
        frames[fid]=dict(path=str(dest),sha256=sha(dest),timestamp_s=r['timestamp_s'],source_frame_index=fid,groups=['uniform-layout-context'],observations=[],c2w_blender=cams[i].tolist(),intrinsics_640x480=intr[i].tolist())
    d['frames']=sorted(frames.values(),key=lambda f:f['timestamp_s'])
    d['groups'].append(dict(group_id='uniform-layout-context',primary_objects=[],timestamps_s=[r['timestamp_s'] for r in layout],purpose='architecture context, not semantic object acceptance'))
    refs=dict(source_hashes=sources,rows=[{k:f[k] for k in ['timestamp_s','source_frame_index','c2w_blender','intrinsics_640x480']} for f in d['frames']]);transport=build_transport(refs)
    for f,c in zip(d['frames'],transport['rows'],strict=True):f['compact_camera']=c
    d['route_steps']=route_steps(q['question']);save(out/'input.json',d);save(out/'pose-reference.json',refs);save(out/'pose-reference-vlm.json',transport)
    from bvb_v10.prompt import build
    extra='''
ROUTE PLANNING: preserve all evidenced walls, floors, partitions, door openings and
their connectivity. Uniform layout frames are context only, NOT accepted detections.
Question roles include start, initial facing object, destination and intermediate
waypoints; reconstruct all of them when supported. Preserve orientation of the start
object when the question specifies its front/back. Inspect additional frames for
question objects even when no detector box is present. Never add arrows, paths, labels
or geometry encoding an answer. Respect the supplied question-specific instance caps;
the counting instruction to keep unlimited instances does NOT override these caps.
Keep relevant direct supports/contacts as white blocks and omit unrelated furnishings
under the existing V10 scope. Do not imply omitted furniture proves free passage.
No ceiling/roof. All walls remain in final QA renders. Do not solve the question.
'''
    p=out/'system-prompt.txt';p.write_text(build(q,d['instance_limits'],d['colors'],out,transport['explanation'])+pipeline.EXTRA+reliability.EXTRA+extra+'\nCATEGORY_COLOR_MAP='+json.dumps(d['colors']))
    from .reference import build as build_reference
    ref=build_reference(out,poses)
    exp=read(out/'experiment.json');exp.update(qa_task='route_planning',qa_walls_hidden=False,layout_uniform_frames=len(layout),layout_additional_unique_frames=added,frame_count=len(d['frames']),group_count=len(d['groups']),input_sha256=sha(out/'input.json'),prompt_sha256=sha(p),reference_sha256=sha(out/'spatial-reference.json'),reference_images={n:sha(out/n) for n in ['reference-top.png','reference-oblique.png']})
    save(out/'experiment.json',exp);return ref

def validate_answer(a,q):
    options=dict(o.split('. ',1) for o in q['options'])
    if a.get('option') not in options or a.get('answer','').strip()!=options[a['option']].strip():raise ValueError('Option and exact answer differ')
    if type(a.get('sufficient_evidence')) is not bool:raise ValueError('Missing evidence assessment')
    steps=route_steps(q['question']);blank=[r['step'] for r in steps if '[please fill in]' in r['instruction']]
    rows=a.get('turns',[])
    if [r.get('step') for r in rows]!=blank:raise ValueError('Missing or reordered route turns')
    expected=[x.strip().lower() for x in options[a['option']].split(',')]
    if [r.get('action','').strip().lower() for r in rows]!=expected:raise ValueError('Turn sequence differs from chosen option')
    if any(x not in {'turn left','turn right','turn back'} for x in expected):raise ValueError('Invalid turn action')

def normalize_selection(a,byid,categories):
    """Keep route/context diagnostics without treating them as target categories."""
    a=dict(a);rows=a['coverage'];targets={};context=[]
    for r in rows:
        label=r.get('category','');canonical=label
        if label not in categories and ':' in label:
            suffix=label.split(':',1)[1].strip()
            if suffix in categories:canonical=suffix
        if canonical not in categories:context.append(r);continue
        r=dict(r,category=canonical)
        if canonical in targets:
            old=targets[canonical];old['visible_in']=list(dict.fromkeys(old['visible_in']+r['visible_in']))
            old['uncertainty']='; '.join(filter(None,[old.get('uncertainty'),r.get('uncertainty')]))
        else:targets[canonical]=r
    a['coverage']=list(targets.values())
    a['context_coverage']=a.get('context_coverage',[])+context
    reliability.validate_views(a,byid,categories)
    if 'top' not in a['view_ids']:raise ValueError('Route selection must include top view')
    if any(not set(r.get('visible_in',[]))<=set(a['view_ids']) for r in a['context_coverage']):raise ValueError('Context refers to unselected view')
    return a

def qa(out):
    from subscription_direct_client import request,image_item
    from bvb_v11.common import parse,state
    d=read(out/'input.json');q=d['question'];assert q['question_type']=='route_planning'
    assert read(out/'coverage-validation.json')['valid']
    manifest=read(out/'views/views.json')
    if manifest.get('qa_walls_hidden') is not False:raise ValueError('Route planning requires walls retained in QA images')
    byid={v['id']:v for v in manifest['views']}
    context=json.dumps(q,ensure_ascii=False)+'\nCategory RGBA colors: '+json.dumps(d['colors'])+'\nWhite geometry represents architecture and direct supports/contacts. Walls and door openings are retained. Unrelated furniture may be omitted: absence is not proof of navigable free space. All views show the SAME scene.\nInstructions in order: step is the unique ordinal position; printed_step preserves the source numbering, which can contain typos. Use step in your output without changing action order. '+json.dumps(route_steps(q['question']))
    context+='\nHigher non-target occluders may be hidden per view to expose targets. They still physically exist: this is not a new passage or changed support/contact. Walls and question targets are protected.'
    if not (out/'view-selection/selection.json').exists():
        prompt='Select EXACTLY THREE DISTINCT complementary views for this route-planning question, including TOP and two non-top views with different azimuths. Show start orientation, all waypoints, destination, floor connectivity and door openings. Do not answer yet. Return JSON {"view_ids":[id,id,id],"reason":"Chinese","limitations":"Chinese","coverage":[{"category":str,"visible_in":[view_id],"uncertainty":str}]}. Include every category; use only selected IDs.\n'+context
        content=[dict(type='input_text',text=prompt)]
        for v in byid.values():
            p=out/'views'/v['image'];assert sha(p)==v['sha256'];content.extend([dict(type='input_text',text='VIEW '+v['id']),image_item(p)])
        a=normalize_selection(parse(request(out/'view-selection',content,model='gpt-6-astra',effort='high')['answer']),byid,q['objects'])
        if 'top' not in a['view_ids']:raise ValueError('Route selection must include top view')
        a['view_id']=a['view_ids'][0];save(out/'view-selection/selection.json',a)
    chosen=normalize_selection(read(out/'view-selection/selection.json'),byid,q['objects'])
    if 'top' not in chosen['view_ids']:raise ValueError('Route selection must include top view')
    state(out,'qa')
    if not (out/'qa/answer.json').exists():
        prompt='''Answer using ONLY the three reconstructed views, public instructions/options
and color legend. Follow the numbered actions in order. Establish the robot's initial
position (including front/back qualifiers) and initial facing, then update heading
after every turn. Left/right are relative to the robot's CURRENT heading, not the
image axes. Turn back reverses heading. Ground-plane navigation cannot pass through
walls; respect visible doorway openings. Do not invent a replacement route or ignore
specified waypoints. Reconcile the same landmarks across all views. If reconstruction
is ambiguous/occluded/inconsistent, disclose it and still choose your best option.
Return JSON {"option":letter,"answer":exact_option_text_without_letter,
"turns":[{"step":integer,"action":"Turn Left|Turn Right|Turn Back",
"from_landmark":str,"toward_landmark":str,"evidence":"short Chinese justification"}],
"reason":"Chinese concise route explanation","confidence":"low|medium|high",
"sufficient_evidence":boolean,"limitations":"Chinese"}. Include one turns entry per
[please fill in], in instruction order, matching the selected option's turn sequence.\n'''+context
        content=[dict(type='input_text',text=prompt)]
        for vid in chosen['view_ids']:
            v=byid[vid];p=out/'views'/v['image'];assert sha(p)==v['sha256'];content.extend([dict(type='input_text',text='VIEW '+vid),image_item(p)])
        a=parse(request(out/'qa',content,model='gpt-6-astra',effort='high')['answer']);validate_answer(a,q)
        a.update(view_ids=chosen['view_ids'],revision=reliability.REVISION,qa_walls_hidden=False);save(out/'qa/answer.json',a)
