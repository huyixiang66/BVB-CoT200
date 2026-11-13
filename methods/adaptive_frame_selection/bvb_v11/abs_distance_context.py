"""Abs-distance-only temporal bridge evidence; architecture, never a new target."""
import math
import shutil
import sys
from pathlib import Path
import numpy as np
from scipy.cluster.hierarchy import linkage,fcluster
from bvb_v10.pipeline import METHOD,read,save,sha
from . import abs_distance_multiview as multi,reference

REVISION='v11-abs-distance-between-objects-context-2s'
GROUP='between-objects-structure-context'
INTERVAL_SECONDS=2.
POSITION_DIAMETER_RATIO=.25


def appearance_times(question,groups):
    result={}
    for role in ['first','second']:
        category=question['roles'][role]
        result[role]=sorted({float(f['timestamp_s']) for g in groups if g['object']==category
            for f in g['selected'] if any(o.get('box_xyxy') is not None and o.get('object',category)==category
                                        for o in f.get('observations',[]))})
        if not result[role]:raise ValueError('No localized endpoint evidence: '+role)
    return result


def bridge_intervals(times):
    """Use consecutive opposite-endpoint sightings, including return sweeps."""
    events={}
    for role,values in times.items():
        for t in values:events.setdefault(float(t),set()).add(role)
    ordered=sorted(events);intervals=[]
    for left,right in zip(ordered,ordered[1:]):
        if any(a!=b for a in events[left] for b in events[right]):intervals.append([left,right])
    if not intervals and ordered:intervals=[[ordered[0],ordered[-1]]]
    merged=[]
    for left,right in intervals:
        if merged and left<=merged[-1][1]:merged[-1][1]=max(right,merged[-1][1])
        else:merged.append([left,right])
    return merged


def select_rows(rows,cameras,times,interval_seconds=INTERVAL_SECONDS):
    """Near-uniform time samples plus uncovered camera-position representatives."""
    if not math.isfinite(interval_seconds) or interval_seconds<=0:raise ValueError('Invalid temporal spacing')
    intervals=bridge_intervals(times)
    indices=[i for i,r in enumerate(rows) if any(a<=r['timestamp_s']<=b for a,b in intervals)]
    if not indices:raise ValueError('No cached bridge frames inside endpoint intervals')
    stamps=np.array([r['timestamp_s'] for r in rows],float)
    selected={};reasons={}
    def add(i,why):
        selected[i]=True;reasons.setdefault(i,[]).append(why)
    uniform=[]
    for left,right in intervals:
        local=[i for i in indices if left<=stamps[i]<=right]
        if not local:continue
        count=max(1,math.ceil((right-left)/interval_seconds))
        for t in np.linspace(left,right,count+1):
            i=min(local,key=lambda j:(abs(stamps[j]-t),-float(rows[j].get('sharpness',0) or 0),j))
            add(i,'uniform temporal bridge');uniform.append(i)
    positions=np.asarray(cameras)[indices,:3,3]
    span=float(np.linalg.norm(np.quantile(positions,.95,axis=0)-np.quantile(positions,.05,axis=0)))
    threshold=max(span*POSITION_DIAMETER_RATIO,1e-6)
    labels=fcluster(linkage(positions,method='complete'),threshold,criterion='distance') if len(indices)>1 else np.ones(1,dtype=int)
    groups={int(g):[indices[k] for k in range(len(indices)) if labels[k]==g] for g in sorted(set(labels))}
    guards=[]
    for gid,members in groups.items():
        if any(i in selected for i in members):continue
        center=np.median(np.asarray(cameras)[members,:3,3],axis=0)
        distance={i:float(np.linalg.norm(np.asarray(cameras)[i,:3,3]-center)) for i in members}
        nearest=min(distance.values());near=[i for i in members if distance[i]<=nearest+.1*threshold]
        i=max(near,key=lambda j:(float(rows[j].get('sharpness',0) or 0),-j))
        add(i,'unrepresented camera-position group');guards.append(i)
    chosen=sorted(selected,key=lambda i:stamps[i])
    gaps=[]
    for left,right in intervals:
        coverage=sorted({left,right,*[float(stamps[i]) for i in chosen if left<=stamps[i]<=right]})
        gaps.extend(b-a for a,b in zip(coverage,coverage[1:]))
    report=dict(policy=REVISION,endpoint_appearance_times=times,bridge_intervals_s=intervals,
        interval_seconds=interval_seconds,candidate_frames=len(indices),selected_count=len(chosen),
        uniform_count=len(set(uniform)),position_guard_count=len(guards),position_group_count=len(groups),
        represented_position_groups=len(groups),position_diameter_native_units=threshold,
        max_temporal_gap_s=max(gaps,default=0.),selected_source_frame_indices=[rows[i]['source_frame_index'] for i in chosen],
        reasons={str(rows[i]['source_frame_index']):reasons[i] for i in chosen},
        position_groups=[dict(id=g,candidate_count=len(m),selected_source_frame_indices=[rows[i]['source_frame_index'] for i in m if i in selected]) for g,m in groups.items()],
        limitations='Temporal sweeps and camera positions are coverage proxies, not proof of a direct spatial corridor, complete floor/wall visibility, room boundaries, or metric distance. No GT, old answer, or prior reconstruction is used.')
    return chosen,report


def plan(question,groups,cache,poses,interval_seconds=INTERVAL_SECONDS):
    if question['question_type']!='object_abs_distance':raise ValueError('Bridge context is abs-distance only')
    sys.path.insert(0,str(METHOD))
    from bvb_v6.reconstruction import shared_pose
    samples=read(poses/'sampling.json')['frames'];sample_index={r['source_frame_index']:i for i,r in enumerate(samples)}
    cams,intr,sources=shared_pose(poses,None)
    # The preceding run recovered exact 1fps originals for every scene. Never re-encode.
    times=appearance_times(question,groups);endpoint_times=set(times['first']+times['second'])
    rows=[r for r in read(cache/'candidates.json') if abs(r['timestamp_s']-round(r['timestamp_s']))<1e-6 or r['timestamp_s'] in endpoint_times]
    for r in rows:
        s=samples[sample_index[r['source_frame_index']]]
        if r['frame_sha256']!=s['sha256'] or abs(r['timestamp_s']-s['timestamp_s'])>1e-6:raise ValueError('Cache/pose mismatch')
        if sha(cache/r['frame'])!=r['frame_sha256']:raise ValueError('Missing or changed canonical context frame')
    matrices=np.array([cams[sample_index[r['source_frame_index']]] for r in rows])
    chosen,report=select_rows(rows,matrices,times,interval_seconds=interval_seconds)
    frames=[]
    for i in chosen:
        r=rows[i];j=sample_index[r['source_frame_index']]
        frames.append(dict(path=str(cache/r['frame']),sha256=r['frame_sha256'],timestamp_s=r['timestamp_s'],
            source_frame_index=r['source_frame_index'],c2w_blender=cams[j].tolist(),intrinsics_640x480=intr[j].tolist()))
    return dict(public_question=question,group_id=GROUP,scope='floor_wall_opening_only',frames=frames,selection=report,shared_pose_sources=sources)


def prepare(source,out,poses,context_plan):
    multi.prepare(source,out,poses)
    d=read(out/'input.json');exp=read(out/'experiment.json');context=read(context_plan)
    if d['public_question']!=context['public_question']:raise ValueError('Context bound to wrong question')
    if d['shared_pose_sources']!=context['shared_pose_sources']:raise ValueError('Mixed context pose frame')
    save(out/'between-context.json',context)
    original_frames={f['source_frame_index'] for f in d['frames']};byid={f['source_frame_index']:f for f in d['frames']}
    for raw in context['frames']:
        fid=raw['source_frame_index']
        if sha(raw['path'])!=raw['sha256']:raise ValueError('Changed context image')
        if fid not in byid:
            dest=out/'frames'/f'frame-{fid:08d}.jpg';shutil.copy2(raw['path'],dest)
            f=dict(raw,path=str(dest),groups=[],observations=[]);d['frames'].append(f);byid[fid]=f
        f=byid[fid]
        if f['sha256']!=raw['sha256'] or f['c2w_blender']!=raw['c2w_blender']:raise ValueError('Conflicting duplicate frame')
        f['groups'].append(GROUP)
        f['observations'].append(dict(object='scene context',box_xyxy=None,status='between_objects_structure_only'))
    d['frames'].sort(key=lambda f:f['timestamp_s'])
    d['groups'].append(dict(group_id=GROUP,primary_objects=[],timestamps_s=[f['timestamp_s'] for f in context['frames']],
        evidence_role='structure_only',allowed_geometry=['floor','wall','opening'],not_a_detected_object_cluster=True))
    d['between_object_context']=dict(group_id=GROUP,scope='floor_wall_opening_only',selection_sha256=sha(out/'between-context.json'))
    from bvb_v6.runtime.compact_pose_input import build_transport
    refs=dict(source_hashes=d['shared_pose_sources'],rows=[{k:f[k] for k in ['timestamp_s','source_frame_index','c2w_blender','intrinsics_640x480']} for f in d['frames']])
    transport=build_transport(refs)
    for f,c in zip(d['frames'],transport['rows'],strict=True):f['compact_camera']=c
    save(out/'pose-reference.json',refs);save(out/'pose-reference-vlm.json',transport);save(out/'input.json',d)
    ref=reference.build(out,poses)
    prompt=out/'system-prompt.txt'
    prompt.write_text(prompt.read_text()+'''

ABS-DISTANCE BETWEEN-OBJECT STRUCTURAL EVIDENCE:
The additional evidence group between-objects-structure-context is NOT a detected
object, a third/fourth target, or a verified physical corridor. It consists of sparse
near-uniform frames along temporal sweeps between observations of the TWO original
distance endpoints, with additional camera-position representatives. Use it ONLY to
recover image-supported floors, walls, partitions, openings and their continuity
between the endpoints. Build those structures as white blocks under the existing
architecture contract. Preserve gaps/openings and uncertainty; never fill an unseen
region merely because a video interval or camera path passes through it. A detour in
the walkthrough is not the shortest spatial distance and not a room boundary.
Do not introduce unrelated furniture or extra object categories from context-only
frames. The two target evidence groups and the optional reference group remain the
identity/shape evidence for those objects. A frame shared with such a group retains
that group's object evidence. Do not invent a scale anchor for the structure-only
group. Integrate everything in the SAME shared pose frame; do not reconstruct an
independent room per group, rescale parts separately, or alter the reference prior.
The QA question still concerns closest SURFACE-to-SURFACE straight-line 3D distance,
not a walking path. This additional context does not provide a measured answer.
''')
    exp.update(revision=REVISION,frame_count=len(d['frames']),group_count=len(d['groups']),
        context_frame_count=len(context['frames']),context_added_unique_frames=len(set(byid)-original_frames),
        context_sha256=sha(out/'between-context.json'),context_scope='floor_wall_opening_only',
        input_sha256=sha(out/'input.json'),prompt_sha256=sha(prompt),reference_sha256=sha(out/'spatial-reference.json'),
        reference_images={n:sha(out/n) for n in ['reference-top.png','reference-oblique.png']})
    save(out/'experiment.json',exp)
    return ref


def qa(out):
    multi.qa(out)
    answer=read(out/'qa/answer.json');answer.update(revision=REVISION,between_object_context=True)
    save(out/'qa/answer.json',answer)
