"""Multi-observation, constrained cross-peak association and instance budgets.

All identities are hypotheses. Existing optical-flow peak tracklets and frozen
DINOv2 ROI features are reused; no ground-truth labels or QA answers are read.
"""
import argparse
import copy
import itertools
import math
import time
from pathlib import Path

import numpy as np
from bvb_v5.dinov2_appearance import read, save, sha, roi_key, pair_evidence
from bvb_v5.pose_association import enhanced_support, pair_signals, position_evidence, refine_groups
from bvb_v5.select_frames import choose, unit, pose_distance


def match_observations(left, right, features, minimum_separation=.5, require_anchors=True):
    """Best two disjoint frame pairs; both anchors must participate.

    Maximize the weaker match, not one cherry-picked high similarity. Same source
    image on both sides is not independent evidence of cross-time identity.
    """
    edges=[]
    for i,a in enumerate(left):
        for j,b in enumerate(right):
            if a['source_frame_index']==b['source_frame_index']:continue
            score=float(np.clip(features[a['feature_index']] @ features[b['feature_index']],-1,1))
            edges.append((i,j,score))
    best=None
    for e,f in itertools.combinations(edges,2):
        i,j,s=e;k,l,t=f
        if i==k or j==l:continue
        if abs(left[i]['timestamp_s']-left[k]['timestamp_s'])<minimum_separation:continue
        if abs(right[j]['timestamp_s']-right[l]['timestamp_s'])<minimum_separation:continue
        if require_anchors and not (left[i]['anchor'] or left[k]['anchor']):continue
        if require_anchors and not (right[j]['anchor'] or right[l]['anchor']):continue
        rank=(min(s,t),(s+t)/2)
        if best is None or rank>best['rank']:
            best={'rank':rank,'bottleneck':min(s,t),'mean':(s+t)/2,
                  'matches':[{'left_roi':left[a]['roi_id'],'right_roi':right[b]['roi_id'],
                              'left_frame':left[a]['source_frame_index'],'right_frame':right[b]['source_frame_index'],
                              'cosine':v} for a,b,v in [e,f]]}
    if best:best.pop('rank')
    return best


def constrained_clusters(names, pairs):
    """Complete-link support: a bridge cannot transitively merge weak pairs."""
    groups=[[n] for n in sorted(names)];history=[]
    while True:
        options=[]
        for i,a in enumerate(groups):
            for j in range(i+1,len(groups)):
                b=groups[j];cross=[pairs[tuple(sorted((x,y)))] for x in a for y in b]
                if all(r['merge_supported'] for r in cross):
                    options.append((min(r['multi_view_score'] for r in cross),i,j))
        if not options:break
        score,i,j=max(options,key=lambda t:(t[0],-t[1],-t[2]))
        history.append({'left':groups[i][:],'right':groups[j][:],'minimum_cross_pair_score':score})
        groups[i]=sorted(groups[i]+groups[j]);groups.pop(j)
    return groups,history


def representative_support(names,pairs,mean_floor=.60):
    """One representative must support every member; no cannot-link anywhere."""
    if len(names)==1:return {'representative':names[0],'minimum_support':1.,'mean_pair_score':1.}
    records=[pairs[tuple(sorted((x,y)))] for x,y in itertools.combinations(names,2)]
    if any(p['same_frame_distinct_evidence'] for p in records):return None
    mean=float(np.mean([p['multi_view_score'] if p['multi_view_score'] is not None else 0. for p in records]))
    if mean<mean_floor:return None
    representatives=[]
    for n in names:
        edges=[pairs[tuple(sorted((n,m)))] for m in names if m!=n]
        if all(e['merge_supported'] for e in edges):
            representatives.append((min(e['multi_view_score'] for e in edges),n))
    if not representatives:return None
    score,n=max(representatives)
    return {'representative':n,'minimum_support':score,'mean_pair_score':mean}


def representative_clusters(names,pairs):
    groups=[[n] for n in sorted(names)];history=[]
    while True:
        candidates=[]
        for i,a in enumerate(groups):
            for j in range(i+1,len(groups)):
                proof=representative_support(a+groups[j],pairs)
                if proof:candidates.append((proof['minimum_support'],proof['mean_pair_score'],i,j,proof))
        if not candidates:break
        _,_,i,j,proof=max(candidates,key=lambda r:(r[0],r[1],-r[2],-r[3]))
        history.append({'left':groups[i][:],'right':groups[j][:],**proof})
        groups[i]=sorted(groups[i]+groups[j]);groups.pop(j)
    return groups,history


def run(a):
    start=time.perf_counter();a.out.mkdir(parents=True,exist_ok=False)
    pp=read(a.v3/'peaks.json');peaks={p['peak_id']:p for p in pp};qs=read(a.v3/'questions.json')
    rois=read(a.appearance/'rois.json');runmeta=read(a.appearance/'extraction.json');feats=np.load(a.appearance/'features.npz')
    assert sha(a.appearance/'rois.json')==runmeta['rois_sha256']
    assert sha(a.appearance/'features.npz')==runmeta['features_sha256']
    assert sha(a.v3/'peaks.json')==rois['provenance']['peaks_sha256']
    rows=rois['rows'];assert list(feats['ids'])==[r['id'] for r in rows]
    use_pose = getattr(a, 'pose_association', False)
    if use_pose:
        spatial = position_evidence(pp, a.cache, a.lingbot, a.intrinsics_size)
        roi_by_id = {r['id']: r for r in rows}

    def support(names, pairs):
        if use_pose:
            return enhanced_support(names, pairs, roi_by_id, representative_support)
        return representative_support(names, pairs)
    rid={r['id']:i for i,r in enumerate(rows)};anchors={p:i for i,r in enumerate(rows) for p in r['anchor_for']}
    geometry=read(a.v4/'anchor-geometry.json');cand=read(a.cache/'candidates.json')
    sampling=read(a.lingbot/'sampling.json');pose=np.load(a.lingbot/'camera-poses.npz')
    assert rois['video_sha256']==sampling['video_sha256']==read(a.cache/'config.json')['video_sha256']
    assert [r['source_frame_index'] for r in cand]==[r['source_frame_index'] for r in sampling['frames']]
    matrices=pose['c2w'];K=pose['intrinsics'];assert len(cand)==len(matrices)
    peakinfo={};within=[]
    for p in pp:
        valid=[]
        for r in p['selected']:
            if not r['box_xyxy'] or r['status']=='context_fallback':continue
            key=roi_key(p['object'],r['source_frame_index'],r['box_xyxy']);idx=rid[key]
            valid.append({'roi_id':key,'feature_index':idx,'source_frame_index':r['source_frame_index'],
                          'timestamp_s':r['timestamp_s'],'anchor':p['peak_id'] in rows[idx]['anchor_for']})
        scores=[float(feats['cls'][u['feature_index']]@feats['cls'][v['feature_index']]) for u,v in itertools.combinations(valid,2)]
        within+=scores
        peakinfo[p['peak_id']]={'object':p['object'],'observations':valid,'verified_roi_count':len(valid),
              'within_peak_median_cosine':float(np.median(scores)) if scores else None,
              'tracklet_warning': 'insufficient observations' if len(valid)<2 else
                                 ('appearance inconsistent; identity not guaranteed' if np.median(scores)<.5 else None),
              'association_source':'existing V3 optical-flow verified tracklet; not ground-truth identity'}
    pairmap={};pairlist=[]
    for p,q in itertools.combinations(pp,2):
        if p['object']!=q['object']:continue
        x,y=p['peak_id'],q['peak_id'];i,j=anchors[x],anchors[y]
        evidence=pair_evidence(x,y,float(feats['cls'][i]@feats['cls'][j]),
                               float(feats['patch_mean'][i]@feats['patch_mean'][j]),peaks,geometry)
        aggressive=a.mode=='aggressive'
        matching=match_observations(peakinfo[x]['observations'],peakinfo[y]['observations'],feats['cls'],
                                    minimum_separation=.25 if aggressive else .5,require_anchors=not aggressive)
        score=matching['bottleneck'] if matching else None
        if evidence['same_frame_distinct_evidence']:reason='same_frame_distinct_do_not_merge'
        elif evidence['geometry_conflict_heuristic'] and not aggressive:reason='sparse_geometry_conflict_defer'
        elif matching is None:reason='insufficient_independent_frame_pairs'
        elif score<a.threshold:reason='insufficient_multi_view_similarity'
        else:reason='two_view_supported_matches' if aggressive else 'two_independent_anchor_supported_matches'
        record={**evidence,'multi_view_score':score,'matching':matching,
                'merge_supported':reason in ['two_independent_anchor_supported_matches','two_view_supported_matches'],'reason':reason,
                'geometry_warning_only':aggressive and evidence['geometry_conflict_heuristic']}
        record.pop('automatic_merge',None);record.pop('decision',None)
        key = tuple(sorted((x, y)))
        if use_pose:
            record.update(pair_signals(spatial[key], evidence['cosine_cls']))
        pairmap[key]=record;pairlist.append(record)
    clusters=[];history=[]
    for obj in qs['objects']:
        names=[p['peak_id'] for p in pp if p['object']==obj]
        cc,hh=(representative_clusters if a.mode=='aggressive' else constrained_clusters)(names,pairmap)
        if use_pose:
            cc,hh=refine_groups(cc,hh,pairmap,support)
        for k,ids in enumerate(cc,1):clusters.append({'group_id':obj.replace(' ','-')+f'-instance-{k:02d}',
                              'object':obj,'peak_ids':ids,'identity_status':'multi_peak_supported_hypothesis' if len(ids)>1 else 'unresolved_single_peak'})
        history+=hh
    membership={p:g['group_id'] for g in clusters for p in g['peak_ids']}
    for record in pairlist:
        record['assigned_same_hypothesis']=membership[record['peak_a']]==membership[record['peak_b']]
    fallback=float(np.median([g['depth'] for g in geometry.values() if g]))
    final=[]
    for cluster in clusters:
        gs=[geometry[p] for p in cluster['peak_ids'] if geometry[p]]
        target=np.median([g['center'] for g in gs],axis=0) if gs else None
        scale=float(np.median([g['depth'] for g in gs])) if gs else fallback
        cc={}
        for pid in cluster['peak_ids']:
            for original in peaks[pid]['selected']:
                r=copy.deepcopy(original);k=r['frame'];box=r['box_xyxy'];C=matrices[k,:3,3]
                direction=matrices[k,:3,2];basis='camera-forward context proxy'
                if box:
                    mid=(np.array(box[:2])+box[2:])/2;pixel=np.r_[mid*np.array([518/640,392/480]),1]
                    direction=unit(matrices[k,:3,:3]@np.linalg.solve(K[k],pixel));basis='box-ray proxy'
                    if target is not None and np.dot(unit(target-C),direction)>=math.cos(math.radians(20)):
                        direction=unit(target-C);basis='cluster sparse target-center proxy'
                clipped=bool(box and (box[0]<3 or box[1]<3 or box[2]>637 or box[3]>477))
                quality=(.35+.65*float(r['dino_score'] or 0))*r['quality']*(.65 if clipped else 1)*(.15 if r['status']=='context_fallback' else 1)
                r.update(source_peaks=[pid],camera_center=C.tolist(),view_direction=unit(direction).tolist(),
                         view_basis=basis,quality_score=quality,clipped_box=clipped,frame_sha256=cand[k]['frame_sha256'])
                fid=r['source_frame_index'];assert fid==cand[k]['source_frame_index']
                if fid in cc:
                    uses=sorted(set(cc[fid]['source_peaks']+[pid]))
                    if quality>cc[fid]['quality_score']:cc[fid]=r
                    cc[fid]['source_peaks']=uses
                else:cc[fid]=r
        selected,extras=choose(list(cc.values()),scale)
        for r in selected:
            r['selection_reason']=r['selection_reason'].replace('question object','instance hypothesis')
        final.append({**cluster,'scope':'instance_hypothesis','candidates':list(cc.values()),'selected':selected,
                      'selected_count':len(selected),'context_count':sum(r['status']=='context_fallback' for r in selected),
                      'depth_scale_native_units':scale,'target_proxy':target.tolist() if target is not None else None,
                      'adaptive_additions':extras,'selected_peak_coverage':sorted({p for r in selected for p in r['source_peaks']}),
                      'pairwise_selected_geometry':[{'frame_a':x['source_frame_index'],'frame_b':y['source_frame_index'],**pose_distance(x,y,scale)} for x,y in itertools.combinations(selected,2)]})
    old={q['question']['id']:q for q in read(a.v4/'summary.json')['questions']}
    result=[]
    for q in qs['questions']:
        groups=[g for g in final if g['object'] in q['objects']]
        ids=sorted({r['source_frame_index'] for g in groups for r in g['selected']})
        result.append({'question':q,'group_ids':[g['group_id'] for g in groups],
                       'group_count':len(groups),'unique_frames':len(ids),'source_frame_indices':ids,
                       'selected_slots':sum(g['selected_count'] for g in groups),'context_slots':sum(g['context_count'] for g in groups),
                       'v3_unique_frames':old[q['id']]['v3_unique_frames'],'previous_v4_frames':old[q['id']]['unique_frames']})
    # Assert coverage and complete-link safeguards before emitting artifacts.
    assert sorted(p for g in final for p in g['peak_ids'])==sorted(peaks)
    for g in final:
        assert 4<=len({r['source_frame_index'] for r in g['selected']})<=8
        if a.mode=='aggressive':
            proof=support(g['peak_ids'],pairmap);assert proof
            g['representative_evidence']=proof
        else:
            for x,y in itertools.combinations(g['peak_ids'],2):assert pairmap[tuple(sorted((x,y)))]['merge_supported']
    summary={'version':'v5-pose-clustering','scene':qs['scene'],'questions':result,'original_peaks':len(pp),
        'instance_hypotheses':len(final),'multi_peak_groups':sum(len(g['peak_ids'])>1 for g in final),
        'unresolved_single_peaks':sum(len(g['peak_ids'])==1 for g in final),
        'global_unique_frames':len({r['source_frame_index'] for g in final for r in g['selected']}),
        'question_image_copies':sum(q['unique_frames'] for q in result),'context_slots':sum(g['context_count'] for g in final),
        'elapsed_seconds':time.perf_counter()-start,'new_model_inference':False,'qa_run':False,'physical_instance_count':None}
    config={'mode':a.mode,'threshold':a.threshold,'threshold_status':'heuristic; not a calibrated identity probability or fitted classifier',
            'matching':'two disjoint cross-time frame pairs; >=0.5s separation within each side; both anchors participate; score=min(pair cosines)',
            'clustering':'complete-link: all cross-cluster peak pairs require positive evidence; prevents single-link bridges',
            'geometry':'same-frame disjoint detection veto; sparse LingBot geometry conflict defers merge, not proof of different identity',
            'budget':'4 to 8 frames per hypothesis, including unresolved single-peak hypotheses; no object-count claims',
            'limitations':['Existing optical-flow associations may drift; incorrect detection classes persist.',
                          'Conservative clustering can leave repeated objects fragmented; comparable instances can still merge incorrectly.',
                          'Minimum four may include explicitly marked unverified context images.',
                          'Only existing V3 four-per-peak candidates are used; no new frames or SAM2.'],
            'within_peak_proxy_quantiles':np.quantile(within,[.1,.25,.5,.75,.9]).tolist(),
            'calibration_note':'Within-peak samples are noisy proxy positives. Only one same-frame-distinct peak pair is available; insufficient for calibrated thresholds.',
            'source_sha256':{str(p):sha(p) for p in [a.v3/'peaks.json',a.appearance/'rois.json',a.appearance/'features.npz',a.v4/'anchor-geometry.json',a.lingbot/'camera-poses.npz',Path(__file__)]}}
    if a.mode=='aggressive':
        config.update(matching='two disjoint cross-time frame pairs; >=0.25s separation; anchors optional; min(pair cosines)>=threshold',
                      clustering='representative-star: a member must directly support every other member; whole-group mean pair score >=0.60 (missing=0); no same-frame cannot-link anywhere',
                      geometry='sparse LingBot geometry conflict is warning only; same-frame distinct remains a hard veto',
                      group_mean_floor=.60)
        config['limitations'].append('Aggressive policy increases risk of merging similar furniture and tolerates geometry conflicts. Unknown single-observation peaks still cannot merge.')
    if use_pose:
        config['pose_association'] = {
            'intrinsics_size': list(a.intrinsics_size),
            'rule': 'After appearance grouping, require a direct cross-group pose link or a qualifying expanded small-ROI pair.',
            'small_pair_floor': .45,
            'depth_min_shared_points': 24,
            'depth_min_shared_fraction': .60,
            'keypoint_track_max_error_px': 12,
        }
        config['source_sha256'][str(a.lingbot/'sparse-geometry-corrected.npz')] = sha(
            a.lingbot/'sparse-geometry-corrected.npz')
    config['version']='v5-pose-clustering'
    config['appearance_crop_policy']=rois.get('crop_policy','tight')
    config['upstream']=read(Path(__file__).with_name('UPSTREAM.json'))
    for name in ['pose_association.py','dinov2_appearance.py']:
        config['source_sha256'][str(Path(__file__).with_name(name))]=sha(Path(__file__).with_name(name))
    if use_pose:
        config['limitations']=[x for x in config['limitations'] if 'single-observation peaks still cannot merge' not in x]
        config['limitations'].append('Pose-supported merges may use recovered context observations; sparse background points and keypoint drift can still cause wrong association. priority_only values are ranking constants, not measured cosine scores.')
    summary['mode']=a.mode
    summary['pose_association']=use_pose
    summary['appearance_crop_policy']=rois.get('crop_policy','tight')
    for name,data in [('groups',final),('summary',summary),('questions',qs),('config',config),('peak-association',pairlist),('peak-observations',peakinfo),('merge-history',history)]:save(a.out/(name+'.json'),data)
    print({k:v for k,v in summary.items() if k!='questions'})


def build_parser():
    ap=argparse.ArgumentParser()
    for k in ['v3','v4','appearance','cache','lingbot','out']:ap.add_argument('--'+k,type=Path,required=True)
    ap.add_argument('--threshold',type=float)
    ap.add_argument('--mode',choices=['conservative','aggressive'],default='aggressive')
    ap.add_argument('--pose-association', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--intrinsics-size', type=int, nargs=2, default=(518, 392), metavar=('WIDTH', 'HEIGHT'))
    return ap


def main():
    ap=build_parser();args=ap.parse_args()
    if args.threshold is None:args.threshold=.40 if args.mode=='aggressive' else .82
    if not -1<=args.threshold<=1:ap.error('threshold must be in [-1,1]')
    if args.pose_association and args.mode != 'aggressive':
        ap.error('--pose-association requires --mode aggressive')
    if min(args.intrinsics_size) <= 0:
        ap.error('--intrinsics-size requires positive dimensions')
    run(args)


if __name__=='__main__':main()
