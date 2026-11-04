"""Reuse V3 candidates; conservative object association and pose-aware selection.

Candidate groups are hypotheses, not ground-truth instances. No model inference.
"""
import argparse, hashlib, json, math, time
from pathlib import Path
import cv2
import numpy as np
from bvb_v3.adaptive_peaks import iou


def read(p):return json.loads(Path(p).read_text())
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def unit(x):return x/max(float(np.linalg.norm(x)),1e-9)


def pose_distance(a,b,scale):
    """No reward for pure rotation: position baseline gates angular novelty."""
    baseline=float(np.linalg.norm(np.asarray(a['camera_center'])-b['camera_center']))/max(scale,1e-6)
    angle=float(np.degrees(np.arccos(np.clip(np.dot(a['view_direction'],b['view_direction']),-1,1))))
    novelty=.7*min(angle/25,1)*min(baseline/.08,1)+.3*min(baseline/.30,1)
    return {'angle_deg':angle,'baseline_over_depth':baseline,'novelty':novelty}


def choose(candidates,scale,minimum=4,maximum=8,threshold=.45,required_frame_sets=None):
    assert len({r['source_frame_index'] for r in candidates})>=minimum
    evidence=[r for r in candidates if r['status']!='context_fallback']
    pool=evidence if len(evidence)>=minimum else candidates
    required=set().union(*(set(s) for s in (required_frame_sets or [])))
    selected=[r for r in candidates if r['source_frame_index'] in required]
    assert len(selected)==len(required)
    reasons={r['source_frame_index']:'reserve four observations for same-frame distinct-instance evidence' for r in selected}
    if not selected:
        selected=[max(pool,key=lambda r:(r['quality_score'],-r['source_frame_index']))]
        reasons={selected[0]['source_frame_index']:'best usable observation across all peaks for this question object'}
    maximum=max(maximum,len(selected))
    def novelty(r):return min(pose_distance(r,s,scale)['novelty'] for s in selected)
    while len(selected)<minimum:
        remaining=[r for r in pool if r['source_frame_index'] not in reasons]
        # If insufficient evidence, exhaust all evidence before context fillers.
        preferred=[r for r in remaining if r['status']!='context_fallback'] or remaining
        r=max(preferred,key=lambda r:(r['quality_score']*(.35+.65*novelty(r)),-r['source_frame_index']))
        reasons[r['source_frame_index']]='minimum four: best remaining quality / pose complement' if r['status']!='context_fallback' else 'minimum four: unverified context fallback'
        selected.append(r)
    decisions=[]
    bestq=max(r['quality_score'] for r in evidence) if evidence else 1
    while len(selected)<maximum:
        eligible=[]
        for r in evidence:
            if r['source_frame_index'] in reasons:continue
            n=novelty(r);closest_angle=min(pose_distance(r,s,scale)['angle_deg'] for s in selected)
            if n>=threshold and r['quality_score']>=.55*bestq and closest_angle<=65:eligible.append((r['quality_score']*n,r,n))
        if not eligible:break
        _,r,n=max(eligible,key=lambda x:(x[0],-x[1]['source_frame_index']));decisions.append({'frame':r['source_frame_index'],'marginal_pose_novelty':n});reasons[r['source_frame_index']]='additional geometric coverage above threshold';selected.append(r)
    for r in candidates:
        r['selected']=r['source_frame_index'] in reasons
        r['selection_reason']=reasons.get(r['source_frame_index'],'redundant view, insufficient quality/overlap, or group budget reached')
        r['nearest_selected_pose_novelty']=min(pose_distance(r,s,scale)['novelty'] for s in selected)
    return sorted(selected,key=lambda r:r['source_frame_index']),decisions


def main():
    ap=argparse.ArgumentParser()
    for k in ['v3','cache','lingbot','out']:ap.add_argument('--'+k,type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False);start=time.perf_counter();cv2.setNumThreads(4)
    peaks=read(a.v3/'peaks.json');qs=read(a.v3/'questions.json');cand=read(a.cache/'candidates.json');sample=read(a.lingbot/'sampling.json');pose=np.load(a.lingbot/'camera-poses.npz');geo=np.load(a.lingbot/'sparse-geometry-corrected.npz');cfg=read(a.cache/'config.json')
    assert cfg['video_sha256']==sample['video_sha256']
    assert [c['source_frame_index'] for c in cand]==[r['source_frame_index'] for r in sample['frames']]
    assert len(cand)==len(pose['c2w'])==950 and np.array_equal(pose['frame_ids'],np.arange(950))
    matrices=pose['c2w'].astype(float);K=pose['intrinsics'].astype(float);points=geo['points'];gids=geo['frame_ids'];gi=np.unique(gids);centers=matrices[:,:3,3]
    # Confirm correct camera-to-world conventions and rigid rotations.
    rotation_error=float(np.max(np.abs(matrices[:,:3,:3].transpose(0,2,1)@matrices[:,:3,:3]-np.eye(3))))
    assert rotation_error<.01 and np.max(abs(np.linalg.det(matrices[:,:3,:3])-1))<.01
    frame_index={c['source_frame_index']:i for i,c in enumerate(cand)};images={};features={};sift=cv2.SIFT_create(nfeatures=300)
    def image(k):
        if k not in images:images[k]=cv2.imread(str(a.cache/cand[k]['frame']))
        return images[k]
    def appearance(p):
        k=p['anchor']['frame'];b=np.asarray(p['anchor']['box_xyxy']);xy0=np.floor(b[:2]).astype(int);xy1=np.ceil(b[2:]).astype(int);im=image(k);crop=im[max(0,xy0[1]):min(480,xy1[1]),max(0,xy0[0]):min(640,xy1[0])]
        if crop.size==0:crop=im
        hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV);hist=cv2.calcHist([hsv],[0,1,2],None,[8,4,4],[0,180,0,256,0,256]).ravel();hist=np.sqrt(hist/max(hist.sum(),1))
        gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY);kp,desc=sift.detectAndCompute(gray,None)
        features[p['peak_id']]={'hist':hist,'points':np.asarray([k.pt for k in kp],np.float32),'desc':desc}
    def project_geo(k,box):
        # Only points originating in nearest sampled depth frame (<=1s), not the whole scene cloud.
        nearest=int(gi[np.argmin(abs(gi-k))]);world=points[gids==nearest];R=matrices[k,:3,:3];C=centers[k];cam=(world-C)@R
        z=cam[:,2];uv=cam@K[k].T;uv=uv[:,:2]/np.maximum(uv[:,2:],1e-9);uv*=np.array([640/518,480/392])
        b=np.asarray(box);pad=.15*(b[2:]-b[:2]);lo=b[:2]+pad;hi=b[2:]-pad
        mask=(z>0)&np.all(uv>=lo,axis=1)&np.all(uv<=hi,axis=1)
        if mask.sum()<6:return None
        pp=world[mask];depth=float(np.median(z[mask]));center=np.median(pp,axis=0);radius=float(np.quantile(np.linalg.norm(pp-center,axis=1),.75))
        spread=float(np.quantile(z[mask],.8)/max(np.quantile(z[mask],.2),1e-6))
        if spread>2 or radius>depth*.8:return None
        return {'center':center.tolist(),'radius':max(radius,.03*depth),'depth':depth,'samples':int(mask.sum()),'point_source_sample':nearest,'temporal_offset_s':abs(cand[nearest]['timestamp_s']-cand[k]['timestamp_s']),'confidence':'heuristic sparse ROI proxy; background/occlusion contamination possible'}
    geometry={}
    for p in peaks:appearance(p);geometry[p['peak_id']]=project_geo(p['anchor']['frame'],p['anchor']['box_xyxy'])
    # Same-frame disjoint high-score boxes are a hard cannot-link constraint.
    observations={p['peak_id']:{r['frame']:r['box_xyxy'] for r in p['observations'] if r['score']>=.5} for p in peaks}
    matcher=cv2.BFMatcher();pair_cache={}
    def pair(p,q):
        key=tuple(sorted([p['peak_id'],q['peak_id']]))
        if key in pair_cache:return pair_cache[key]
        f,g=features[p['peak_id']],features[q['peak_id']];hist=float(np.dot(f['hist'],g['hist']));gp,gq=geometry[p['peak_id']],geometry[q['peak_id']];cooccur=False
        op,oq=observations[p['peak_id']],observations[q['peak_id']]
        for k in op.keys()&oq.keys():
            if iou(op[k],oq[k])<.1:cooccur=True;break
        ratio=None
        if gp and gq:ratio=float(np.linalg.norm(np.asarray(gp['center'])-gq['center'])/max(gp['radius']+gq['radius'],1e-6))
        inliers=0;inlier_ratio=0.
        if not cooccur and f['desc'] is not None and g['desc'] is not None and len(f['desc'])>=8 and len(g['desc'])>=8:
            pairs=matcher.knnMatch(f['desc'],g['desc'],k=2);good=[m[0] for m in pairs if len(m)==2 and m[0].distance<.7*m[1].distance]
            if len(good)>=8:
                x=np.asarray([f['points'][m.queryIdx] for m in good]);y=np.asarray([g['points'][m.trainIdx] for m in good]);_,mask=cv2.findHomography(x,y,cv2.RANSAC,3)
                if mask is not None:inliers=int(mask.sum());inlier_ratio=inliers/len(good)
        conflict=cooccur or (ratio is not None and ratio>1.5)
        geometric_match=ratio is not None and ratio<=.65 and hist>=.80
        feature_match=inliers>=10 and inlier_ratio>=.6 and hist>=.65 and (ratio is None or ratio<=1.)
        merge=not conflict and (geometric_match or feature_match)
        result={'peak_a':p['peak_id'],'peak_b':q['peak_id'],'histogram_similarity':hist,'sift_inliers':inliers,'sift_inlier_ratio':inlier_ratio,'roi_geometry_separation_ratio':ratio,'same_frame_disjoint':cooccur,'cannot_link':conflict,'merge_evidence':merge,'rule':'geometry+appearance' if geometric_match else ('SIFT+appearance' if feature_match else 'insufficient evidence')}
        pair_cache[key]=result;return result
    groups=[]
    for obj in qs['objects']:
        pp=sorted([p for p in peaks if p['object']==obj],key=lambda p:-p['anchor']['score']);localgroups=[]
        for p in pp:
            matches=[]
            for g in localgroups:
                pr=pair(p,g['peaks'][0])
                if pr['merge_evidence'] and not any(pair(p,other)['cannot_link'] for other in g['peaks']):matches.append((pr['histogram_similarity'],g))
            if matches:max(matches,key=lambda x:x[0])[1]['peaks'].append(p)
            else:localgroups.append({'object':obj,'peaks':[p]})
        for n,g in enumerate(localgroups,1):g['group_id']=obj.replace(' ','-')+f'-g{n:02d}';groups.append(g)
        print('GROUPS',obj,len(pp),'peaks ->',len(localgroups),'candidate instances',flush=True)
    # Unknown identity does not create a new four-frame budget. Keep association
    # hypotheses for audit, but share the default budget across the question term.
    # Only simultaneous disjoint anchor boxes reserve separate four-frame sets.
    strict_groups=groups;save(a.out/'identity-hypotheses.json',[{'group_id':g['group_id'],'object':g['object'],'peaks':[p['peak_id'] for p in g['peaks']]} for g in strict_groups])
    groups=[]
    for obj in qs['objects']:
        pp=[p for p in peaks if p['object']==obj];byframe={}
        if not pp:
            continue  # No detected peak: retain missing-target status, never fabricate a four-frame budget.
        for p in pp:
            if p['anchor']['score']>=.5:byframe.setdefault(p['anchor']['frame'],[]).append(p)
        simultaneous=[]
        for k,rows in byframe.items():
            disjoint=[]
            for p in sorted(rows,key=lambda p:-p['anchor']['score']):
                if all(iou(p['anchor']['box_xyxy'],q['anchor']['box_xyxy'])<.1 for q in disjoint):disjoint.append(p)
            if len(disjoint)>=2:simultaneous.append(disjoint)
        reserves=max(simultaneous,key=lambda x:(len(x),sum(p['anchor']['score'] for p in x))) if simultaneous else []
        groups.append({'group_id':obj.replace(' ','-')+'-budget','object':obj,'peaks':pp,'reserved_peak_ids':[p['peak_id'] for p in reserves],'scope':'non_counting_object'})
    counting_objects={o for q in qs['questions'] if q['question_type']=='object_counting' for o in q['objects']}
    for g in strict_groups:
        if g['object'] in counting_objects:groups.append({**g,'group_id':g['group_id']+'-counting','reserved_peak_ids':[],'scope':'counting_hypothesis'})
    # Scale in native LingBot units. Avoid any Blender registration or metre claims.
    depths=[g['depth'] for g in geometry.values() if g];fallback_depth=float(np.median(depths)) if depths else float(np.linalg.norm(np.ptp(centers,axis=0)))
    final=[]
    for g in groups:
        geos=[geometry[p['peak_id']] for p in g['peaks'] if geometry[p['peak_id']]];target=np.median([r['center'] for r in geos],axis=0) if geos else None;scale=float(np.median([r['depth'] for r in geos])) if geos else fallback_depth
        candidates={}
        for p in g['peaks']:
            for r in p['selected']:
                k=r['frame'];box=r['box_xyxy'];C=centers[k];basis='box-ray proxy';direction=matrices[k,:3,2]
                if box:
                    mid=(np.asarray(box[:2])+box[2:])/2;pixel=np.r_[mid*np.array([518/640,392/480]),1];direction=unit(matrices[k,:3,:3]@np.linalg.solve(K[k],pixel))
                    if target is not None:
                        td=unit(target-C)
                        if np.dot(td,direction)>=math.cos(math.radians(20)):direction=td;basis='sparse target-center proxy'
                else:basis='camera-forward context proxy'
                clipped=bool(box and (box[0]<3 or box[1]<3 or box[2]>637 or box[3]>477));score=float(r['dino_score'] or 0)
                quality=(.35+.65*score)*r['quality']*(.65 if clipped else 1.)*(.15 if r['status']=='context_fallback' else 1.)
                row={**r,'source_peaks':[p['peak_id']],'source_frame_index':cand[k]['source_frame_index'],'camera_center':C.tolist(),'view_direction':unit(direction).tolist(),'view_basis':basis,'quality_score':quality,'clipped_box':clipped,'frame_sha256':cand[k]['frame_sha256']}
                key=row['source_frame_index']
                if key in candidates:
                    old=candidates[key];uses=old['source_peaks']+[p['peak_id']]
                    if row['quality_score']>old['quality_score']:candidates[key]=row
                    candidates[key]['source_peaks']=list(dict.fromkeys(uses))
                else:candidates[key]=row
        required=[{'peak_id':p['peak_id'],'source_frame_indices':[r['source_frame_index'] for r in p['selected']]} for p in g['peaks'] if p['peak_id'] in g['reserved_peak_ids']]
        cc=list(candidates.values());selected,additions=choose(cc,scale,required_frame_sets=[r['source_frame_indices'] for r in required])
        record={'group_id':g['group_id'],'object':g['object'],'scope':g['scope'],'peak_ids':[p['peak_id'] for p in g['peaks']],'identity_status':'shared question-object budget; unknown identities retained as hypotheses, not asserted merged' if g['scope']=='non_counting_object' else 'counting candidate instance; uncertain hypotheses retain separate minimums, not true object count','instance_reservations':required,'has_same_frame_distinct_evidence':bool(required),'target_proxy':None if target is None else target.tolist(),'depth_scale_native_units':scale,'candidates':cc,'selected':selected,'adaptive_additions':additions,'selected_count':len(selected),'context_count':sum(r['status']=='context_fallback' for r in selected),'pairwise_selected_geometry':[{'frame_a':x['source_frame_index'],'frame_b':y['source_frame_index'],**pose_distance(x,y,scale)} for i,x in enumerate(selected) for y in selected[i+1:]]};final.append(record)
    question_results=[];old=read(a.v3/'summary.json')
    for q in qs['questions']:
        scope='counting_hypothesis' if q['question_type']=='object_counting' else 'non_counting_object'
        gs=[g for g in final if g['object'] in q['objects'] and g['scope']==scope];ids=sorted({r['source_frame_index'] for g in gs for r in g['selected']});before=next(r for r in old['questions'] if r['question']['id']==q['id'])
        question_results.append({'question':q,'group_ids':[g['group_id'] for g in gs],'group_count':len(gs),'selected_slots':sum(g['selected_count'] for g in gs),'unique_frames':len(ids),'source_frame_indices':ids,'v3_unique_frames':before['unique_frames'],'context_slots':sum(g['context_count'] for g in gs)})
    config={'version':'v5','minimum_per_candidate_instance':4,'maximum_per_candidate_instance':8,'extra_view_min_novelty':.45,'question_hard_cap':None,'policy':'All V3 peaks compete within associated candidate instance groups; minimum four takes priority; uncertain groups kept separate; no automatic ground-truth instance claims.','association':{'geometry_separation_max':.65,'geometry_histogram_min':.8,'sift_min_inliers':10,'sift_min_inlier_ratio':.6,'sift_histogram_min':.65,'same_frame_disjoint_is_cannot_link':True,'geometry_disagreement_ratio_cannot_link':1.5},'pose_novelty':'.7 * min(target-observation angle/25deg,1) * min(baseline/depth/.08,1) + .3 * min(baseline/depth/.30,1). Marginal novelty is minimum vs selected views. Pure rotation has zero baseline gain. Extras require >=0.45 novelty, >=55% best quality, and <=65deg to an existing observation.','coordinate_contract':'Corrected raw OpenCV C2W; no Blender alignment; native arbitrary scale, normalized by ROI depth proxy','candidate_source':'V3 four-per-peak pool (including labeled context); no new DINO or SAM2 inference; 4fps LingBot pose matched by exact original video frame index','limitations':['Candidate groups are not confirmed physical instances. Similar objects may merge and repeat observations may remain separate. Same-frame distinct boxes cannot merge.','Sparse ROI depth (every eighth LingBot sample) can include background; unknown target uses world-space box bearing as a view proxy.','No proven uncertainty reduction or 3D reconstruction quality gain; no QA run.','Minimum four can require explicitly unverified context; 4 distinct frames need not be four independent viewpoints.','No arbitrary 32-frame question cap; per-group minimum takes precedence. No additional connection-frame search in this experiment.'],'source_sha256':{str(p):sha(p) for p in [a.v3/'peaks.json',a.v3/'questions.json',a.lingbot/'camera-poses.npz',a.lingbot/'sampling.json',a.lingbot/'sparse-geometry-corrected.npz',Path(__file__)]}}
    config.update(minimum_per_candidate_instance=None,minimum_per_question_object=4,maximum_per_candidate_instance=None,default_maximum_per_question_object=8,instance_reservation_policy='At the strongest maximal simultaneous-disjoint ANCHOR frame, preserve each observed target seed with four source observations. Shared image can satisfy multiple seeds. Budget increases if these reservations exceed eight. Does not guarantee all physical instances discovered.',policy='Unknown identity hypotheses do not each get four frames. Default budget is shared across all peaks for a question object. Same-frame separated targets reserve four each; no claim that unassociated peaks are the same physical instance.')
    config['limitations'][0]='Physical instance re-ID remains unresolved; 82-style fragmented hypotheses are audited but do not each create a budget. Different instances never seen together can be missed by this budget policy. Same-frame detection separation is evidence, not GT verification.'
    config['limitations'].append('Object-wide target center is a rough ROI proxy and may mix instances; bearing fallback and pose coverage are heuristics, not triangulation certainty.')
    config['question_type_policy']='Non-counting: one shared budget per question object, explicit co-occurrence reservations. Counting: separate minimum four for each unresolved identity hypothesis. A hypothesis count is never reported as ground-truth object count.'
    save(a.out/'config.json',config);save(a.out/'groups.json',final);save(a.out/'questions.json',qs);save(a.out/'association-audit.json',list(pair_cache.values()));save(a.out/'anchor-geometry.json',geometry)
    summary={'version':'v5','scene':qs['scene'],'questions':question_results,'v3_peak_count':len(peaks),'candidate_groups':len(final),'groups_with_multiple_peaks':sum(len(g['peak_ids'])>1 for g in final),'group_selected_slots':sum(g['selected_count'] for g in final),'global_unique_frames':len({r['source_frame_index'] for g in final for r in g['selected']}),'context_slots':sum(g['context_count'] for g in final),'groups_above_four':sum(g['selected_count']>4 for g in final),'elapsed_seconds':time.perf_counter()-start,'pose_rotation_max_error':rotation_error,'pose_frames':950,'no_new_model_inference':True,'no_qa':True}
    summary.update(identity_hypotheses=len(strict_groups),budget_scope='non-counting question-object budgets; counting separate hypothesis budgets',object_budgets=sum(g['scope']=='non_counting_object' for g in final),counting_hypothesis_budgets=sum(g['scope']=='counting_hypothesis' for g in final))
    save(a.out/'summary.json',summary);print('DONE',json.dumps({k:v for k,v in summary.items() if k!='questions'}),flush=True)


if __name__=='__main__':main()
