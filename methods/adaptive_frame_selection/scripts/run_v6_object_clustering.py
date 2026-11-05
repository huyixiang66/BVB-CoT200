"""Recluster frozen question inputs; no detector, LLM, or reconstruction."""
from pathlib import Path
import argparse,copy,hashlib,json,sys
from itertools import combinations,product
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from bvb_v6.object_clustering import (PARAMETERS,roi_surface,compare_roi,
    simultaneous_conflict,supported_pair_score,cluster)
from bvb_v6.cluster.group_regions import frame_rows,format_group
from bvb_v6.reconstruction import pick,shared_pose

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--minimum-frame-pairs',type=int,choices=[1,2],default=2)
    ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    params={**PARAMETERS,'minimum_frame_pairs':args.minimum_frame_pairs}
    if args.out.exists():raise FileExistsError(args.out)
    old=read(args.source);paths={k:Path(v) for k,v in old['inputs'].items()}
    candidates={r['source_frame_index']:r for r in read(paths['cache']/'candidates.json')}
    peaks=read(paths['v3']/'peaks.json');wanted={o for q in old['questions'].values() for o in q['question']['objects']}
    poses=dict(np.load(paths['poses']/'camera-poses.npz'));cloud=dict(np.load(paths['poses']/'sparse-geometry-corrected.npz'))
    samples=read(paths['poses']/'sampling.json')['frames'];size=[640,480]
    intrinsic_size=read(paths['poses']/'geometry-info.json')['image_size_wh']
    resize=np.diag([*np.divide(size,intrinsic_size),1.])
    world,_,_=shared_pose(paths['poses'],None)
    feature_data=np.load(paths['appearance']/'features.npz');features=dict(zip(feature_data['ids'],feature_data['cls']))
    vectors={}
    for r in read(paths['appearance']/'rois.json')['rows']:
        assert r['source_sha256']==candidates[r['source_frame_index']]['frame_sha256']
        v=features[r['id']];vectors[(r['source_frame_index'],r['object'],tuple(r['box_xyxy']))]=v/max(np.linalg.norm(v),1e-12)
    descriptors={};verified=set();categories={};groupings={};scale=old['depth_scale']
    for category in sorted(wanted):
        ps=[p for p in peaks if p['object']==category];obs={};rows=frame_rows(ps,candidates)
        for p in ps:
            obs[p['peak_id']]=[]
            for r in p['selected']:
                fid=r['source_frame_index'];box=r.get('box_xyxy');c=candidates[fid];i=c['index']
                key=(fid,category,tuple(box) if box else None)
                if fid not in verified:
                    assert samples[i]['source_frame_index']==fid and samples[i]['sha256']==c['frame_sha256']
                    assert sha(paths['cache']/c['frame'])==c['frame_sha256'];verified.add(fid)
                if key not in descriptors:
                    if box and key not in vectors:raise ValueError(f'Missing appearance for {key}')
                    descriptors[key]=roi_surface(i,poses['c2w'][i],resize@poses['intrinsics'][i],cloud,size,box)
                obs[p['peak_id']].append({**r,'key':key})
        evidence={};pairs_out=[]
        for a,b in combinations(sorted(obs),2):
            pairs=[]
            for x,y in product(obs[a],obs[b]):
                va,vb=vectors.get(x['key']),vectors.get(y['key'])
                semantic=float(np.clip(va@vb,0,1)) if va is not None and vb is not None else 0.
                pair=compare_roi(descriptors[x['key']],descriptors[y['key']],scale,semantic)
                pair.update(frames=[x['source_frame_index'],y['source_frame_index']],semantic=semantic,
                            valid=va is not None and vb is not None)
                pairs.append(pair)
            score,count=supported_pair_score(pairs,minimum=params['minimum_frame_pairs'])
            reliable=[p for p in pairs if p['reliability']>=PARAMETERS['reliable_quality']]
            far=(len(reliable)>=4 and sum(p['far'] for p in reliable)/len(reliable)>=PARAMETERS['far_pair_fraction'])
            coexist=simultaneous_conflict(obs[a],obs[b])
            e=dict(peaks=[a,b],score=score,independent_frame_pairs=count,cannot_link=bool(far or coexist),
                   conflict_reason='simultaneous_distinct_boxes' if coexist else 'repeated_reliable_spatial_separation' if far else None,
                   frame_pairs=pairs)
            evidence[(a,b)]=e;pairs_out.append(e)
        groupings[category]={};histories={}
        for mode in ['complete','robust']:
            grouped,history=cluster(obs,evidence,mode,params=params);groupings[category][mode]=grouped;histories[mode]=history
        descriptor_records=[]
        for pid,rr in obs.items():
            for r in rr:
                d=descriptors[r['key']]
                descriptor_records.append(dict(peak=pid,frame=r['source_frame_index'],box=r.get('box_xyxy'),
                    geometry=None if d is None else {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in d.items() if k not in ['tree','samples']}))
        categories[category]=dict(peaks=list(obs),pairs=pairs_out,descriptors=descriptor_records,histories=histories)
        print(category,'peaks',len(obs),'ROI complete',len(groupings[category]['complete']),'robust',len(groupings[category]['robust']),flush=True)
    output=copy.deepcopy(old);output['version']='bvb_v6_object_roi_experiment';output['preset']=dict(name=f'roi-reliability-support-{args.minimum_frame_pairs}',parameters=params)
    output['baseline_sha256']=sha(args.source);output['new_model_calls']=False;output['reconstruction_run']=False
    output['provenance']={str(p):sha(p) for p in [Path(__file__),ROOT/'bvb_v6/object_clustering.py',paths['poses']/'camera-poses.npz',paths['poses']/'sparse-geometry-corrected.npz',paths['appearance']/'features.npz',paths['v3']/'peaks.json']}
    summaries=[]
    for qid,q in output['questions'].items():
        q.pop('evidence',None);q['arms']={'baseline':old['questions'][qid]['arms']['v6']}
        for mode in ['complete','robust']:
            groups=[]
            for category in q['question']['objects']:
                ps=[p for p in peaks if p['object']==category];rows=frame_rows(ps,candidates)
                for i,members in enumerate(groupings[category][mode],1):
                    ids=sorted({r['source_frame_index'] for p in ps if p['peak_id'] in members for r in p['selected']})
                    group=format_group(category.replace(' ','-')+f'-{i:02d}',ids,rows,set(members))
                    group['candidate_frames']=copy.deepcopy(group['selected'])
                    group['selected']=sorted(pick(group['selected'],candidates,world,8),key=lambda r:r['source_frame_index'])
                    group['status']='instance_hypothesis_not_verified';group['minimum_four_satisfied']=len(group['selected'])>=4
                    groups.append(group)
            q['arms']['roi_'+mode]=groups
        q['summary']={k:dict(groups=len(gs),image_slots=sum(len(g['selected']) for g in gs)) for k,gs in q['arms'].items()}
        summaries.append(dict(qid=qid,**q['summary']))
    write(args.out/'selection.json',output);write(args.out/'evidence.json',categories)
    write(args.out/'comparison.json',dict(questions=summaries,parameters=params,verified_unique_frames=len(verified),new_model_calls=False,reconstruction_run=False))
    print(json.dumps(summaries,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
