"""Deterministic fixed-budget object-conditioned nonuniform sampling."""
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np

def capped(values):
    values=np.asarray(values,dtype=float);positive=values[values>0]
    scale=float(np.quantile(positive,.95)) if len(positive) else 1.
    return np.clip(values/max(scale,1e-12),0,1)

def select(question,candidates,detections,flow,budget=25,context_fraction=.20):
    n=len(candidates)
    if budget<1:raise ValueError('Frame budget must be positive')
    requested_budget=budget;budget=min(budget,n);objects=question['objects'];times=np.array([r['timestamp_s'] for r in candidates])
    if not n:raise ValueError('Empty video candidates')
    sharp=np.array([r['sharpness'] for r in candidates]);median=max(float(np.median(sharp)),1.)
    quality=.5+.5*sharp/(sharp+median)
    global_motion=capped([r['global_motion_since_previous_candidate'] for r in flow])
    presence={o:np.array([max((d['score'] for d in frame['detections'] if d['object']==o),default=0.) for frame in detections]) for o in objects}
    scores={};missing=[]
    for obj in objects:
        roi=capped([r['object_roi_motion_per_second'].get(obj,0.) for r in flow])
        # Nonzero static floor keeps a clear, stationary object eligible.
        score=presence[obj]*(.4+.6*(.75*roi+.25*global_motion))*quality
        if score.sum()>0:scores[obj]=score/score.sum()
        else:missing.append(obj)
    if scores:
        object_density=np.mean(list(scores.values()),axis=0)
    else:
        object_density=(.4+.6*global_motion)*quality;object_density/=object_density.sum()
    fraction=max(context_fraction,.5 if not objects else (.3 if question['question_type']=='object_counting' else context_fraction))
    fraction=float(np.clip(fraction,0,1));density=(1-fraction)*object_density+fraction/n;density/=density.sum()
    # Weighted cumulative time spreads points where relevant visual change is concentrated.
    coordinate=np.cumsum(density)-density/2
    selected={};reserve=min(budget,max(1,int(round(budget*fraction))))
    def add(i,reason):
        i=int(i)
        if len(selected)<budget or i in selected:selected.setdefault(i,[]).append(reason)
    for i in np.linspace(0,n-1,reserve).round().astype(int):add(i,'global_temporal_coverage')
    for obj in objects:
        if obj not in scores:continue
        if question['question_type']=='obj_appearance_order':
            first=int(np.flatnonzero(presence[obj]>0)[0]);add(first,'first_detected:'+obj)
            if first>0:add(first-1,'before_first_detected:'+obj)
        elif not any(presence[obj][i]>0 for i in selected):
            add(np.argmax(scores[obj]),'object_coverage:'+obj)
    # Counting: retain separate visibility episodes when budget permits. These are NOT tracked identities.
    if question['question_type']=='object_counting':
        for obj in objects:
            visible=np.flatnonzero(presence[obj]>0)
            episodes=np.split(visible,np.flatnonzero(np.diff(times[visible])>3.)+1) if len(visible) else []
            for episode in sorted(episodes,key=lambda e:float(scores[obj][e].max()),reverse=True):
                if not any(i in selected for i in episode):add(episode[np.argmax(scores[obj][episode])],'visibility_episode:'+obj)
    while len(selected)<budget:
        nearest=np.min(np.abs(coordinate[:,None]-coordinate[list(selected)][None,:]),axis=1)
        utility=nearest*quality;utility[list(selected)]=-1
        add(np.argmax(utility),'object_motion_weighted_temporal_coverage')
    indices=sorted(selected);uniform=np.linspace(0,n-1,budget).round().astype(int).tolist()
    assert len(set(indices))==len(set(uniform))==budget
    coverage={obj:any(presence[obj][i]>0 for i in indices) for obj in objects}
    selected_rows=[{**candidates[i],'selection_reasons':selected[i],'importance_density':float(density[i]),
                   'detected_relevant_objects':[o for o in objects if presence[o][i]>0],
                   'detections':[d for d in detections[i]['detections'] if d['object'] in objects],
                   'global_motion_normalized':float(global_motion[i])} for i in indices]
    gaps=np.diff(times[indices])
    return {'question':question,'requested_budget':requested_budget,'selected_count':len(indices),'candidate_count':n,
            'context_fraction':fraction,'selected_indices':indices,'uniform_indices':uniform,'selected_frames':selected_rows,
            'density':density.tolist(),'objects_not_detected':missing,'selected_object_coverage':coverage,
            'detected_but_not_covered':[o for o in scores if not coverage[o]],
            'sampling_gap_seconds':{'min':float(gaps.min()) if len(gaps) else None,'max':float(gaps.max()) if len(gaps) else None},
            'fallback':not bool(scores),'ground_truth_used':False,'qa_accuracy_evaluated':False,
            'note':'Optical flow measures image change, including camera motion. Detection episodes are not unique object instances. First detection is not guaranteed true first appearance.'}

def run(cache,questions,out,budget=25,context_fraction=.20):
    cache=Path(cache);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    candidates=json.loads((cache/'candidates.json').read_text());config=json.loads((cache/'config.json').read_text())
    lookup={r['source_frame_index']:r for r in map(json.loads,(cache/'detections.jsonl').read_text().splitlines())}
    detections=[lookup[r['source_frame_index']] for r in candidates];flow=json.loads((cache/'flow.json').read_text())['rows']
    assert len(candidates)==len(flow) and all(c['source_frame_index']==f['source_frame_index'] for c,f in zip(candidates,flow))
    public=json.loads(Path(questions).read_text());expected=set(public['objects']);assert expected<=set(config['objects'])
    manifest={'budget':budget,'candidate_fps':config['candidate_fps'],'video_sha256':config['video_sha256'],
              'cache':str(cache.resolve()),'questions':[],'algorithm':'Object-balanced capped ROI/global flow density + blur quality + global anchors + weighted cumulative-time coverage',
              'cache_fingerprints':{name:hashlib.sha256((cache/name).read_bytes()).hexdigest() for name in ['config.json','detections.jsonl','flow.json']}}
    for q in public['questions']:
        result=select(q,candidates,detections,flow,budget,context_fraction)
        name=f'question-{q["id"]}.json';(out/name).write_text(json.dumps(result,indent=2)+'\n')
        manifest['questions'].append({'id':q['id'],'question_type':q['question_type'],'objects':q['objects'],'selection':name,'missing':result['objects_not_detected'],'gaps':result['sampling_gap_seconds']})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',required=True);p.add_argument('--questions',required=True);p.add_argument('--out',required=True);p.add_argument('--budget',type=int,default=25);p.add_argument('--context-fraction',type=float,default=.2);a=p.parse_args()
    m=run(a.cache,a.questions,a.out,a.budget,a.context_fraction);print('SELECTION_OK',len(m['questions']),'questions')
