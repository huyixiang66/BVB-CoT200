"""discovery-rate discovery, optical-flow box association, four frames per visibility peak.

No SAM2, answer labels, LLM calls, or inferred physical instance counts.
Local 4fps detections verify proposed neighbors; they do not create new peaks.
"""
import argparse, csv, functools, hashlib, html, json, math, shutil, time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(p): return json.loads(Path(p).read_text())
def save(p, d): Path(p).write_text(json.dumps(d, ensure_ascii=False, indent=2)+'\n')
def slug(s): return s.replace(' ', '-')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    inter = np.maximum(np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2]), 0).prod()
    return float(inter/max(np.maximum(a[2:]-a[:2], 0).prod()+np.maximum(b[2:]-b[:2], 0).prod()-inter, 1e-9))


def associate(predicted, boxes, threshold=.2):
    """Conservative one-to-one greedy matching, with area-ratio guard."""
    edges=[]
    for tid, box in predicted.items():
        ar=max(float(np.prod(np.asarray(box)[2:]-np.asarray(box)[:2])), 1)
        for j,b in enumerate(boxes):
            ratio=float(np.prod(np.asarray(b['box_xyxy'])[2:]-np.asarray(b['box_xyxy'])[:2]))/ar
            score=iou(box,b['box_xyxy'])
            if score>=threshold and .25<=ratio<=4: edges.append((score,tid,j))
    result={};used=set()
    for score,tid,j in sorted(edges,reverse=True):
        if tid not in result and j not in used:result[tid]=j;used.add(j)
    return result


def peak_tracks(rows, obj, index, propagate, hardcut=.5):
    """One strongest discovery-rate detection per flow-linked visibility tracklet.

    No cross-gap re-ID; two adjacent waves may be the same physical instance.
    Low-score observations can maintain a track but cannot create a valid peak.
    """
    tracks=[];active={}
    for row in rows:
        k=index[row['source_frame_index']]
        boxes=[b for b in row['detections'] if b['object']==obj and b['score']>=.25]
        predicted={}
        for tid,state in list(active.items()):
            if row['timestamp_s']-state['last_seen_s']>2.01:
                del active[tid];continue
            box,_=propagate(state['frame'],k,state['box'])
            state.update(frame=k,box=box);predicted[tid]=box
        matches=associate(predicted,boxes);used=set(matches.values())
        for tid,j in matches.items():
            b=boxes[j];tracks[tid]['observations'].append({'frame':k,**b})
            active[tid].update(last_seen_s=row['timestamp_s'],box=b['box_xyxy'])
        for j,b in enumerate(boxes):
            if j in used:continue
            tid=len(tracks);tracks.append({'tracklet':tid,'object':obj,'observations':[{'frame':k,**b}]})
            active[tid]={'frame':k,'last_seen_s':row['timestamp_s'],'box':b['box_xyxy']}
    passing=[t for t in tracks if max(r['score'] for r in t['observations'])>=hardcut]
    fallback=False
    if not passing and tracks:
        passing=[max(tracks,key=lambda t:max(r['score'] for r in t['observations']))];fallback=True
    result=[]
    for t in passing:
        anchor=max(t['observations'],key=lambda r:r['score'])
        result.append({**t,'anchor':anchor,'fallback':fallback,'physical_instance_id':None})
    return result,tracks


def merge_peak_fragments(peaks,times,min_separation=6.):
    """Temporal NMS of anchor proposals; simultaneous disjoint boxes survive.

    This merges sampling waves, NOT physical identities. Different instances in
    nearby separate shots may be grouped; the retained same-frame boxes only
    protect directly observed co-occurrence.
    """
    retained=[];suppressed=[]
    for p in sorted(peaks,key=lambda p:(-p['anchor']['score'],p['anchor']['frame'])):
        blockers=[]
        for q in retained:
            same_frame=p['anchor']['frame']==q['anchor']['frame']
            cooccurs=same_frame and iou(p['anchor']['box_xyxy'],q['anchor']['box_xyxy'])<.2
            if abs(times[p['anchor']['frame']]-times[q['anchor']['frame']])<min_separation and not cooccurs:blockers.append(q)
        if blockers:
            q=blockers[0];q.setdefault('merged_anchor_proposals',[]).append({'tracklet':p['tracklet'],'frame':p['anchor']['frame'],'score':p['anchor']['score']});suppressed.append(p)
        else:retained.append(p)
    return sorted(retained,key=lambda p:p['anchor']['frame']),suppressed


def main():
    ap=argparse.ArgumentParser()
    for name in ['coarse','local','piano','questions','native-flow','out']:ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--peak-threshold',type=float,default=.5)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);report=a.out/'report';report.mkdir(exist_ok=False)
    if not 0<=a.peak_threshold<=1:ap.error('peak-threshold must lie in [0,1]')
    start=time.perf_counter();cv2.setNumThreads(4)
    cfg=read(a.coarse/'config.json');lc=read(a.local/'config.json');pc=read(a.piano/'config.json')
    discovery_fps=float(cfg['candidate_fps'])
    local_fps=float(lc['candidate_fps'])
    assert discovery_fps in (1.,2.,4.) and local_fps in (2.,4.) and pc['candidate_fps']==local_fps and local_fps>=discovery_fps
    assert cfg['video_sha256']==lc['video_sha256']==pc['video_sha256']
    cand=read(a.local/'candidates.json');piano_cand=read(a.piano/'candidates.json')
    assert [(c['source_frame_index'],c['frame_sha256']) for c in cand]==[(c['source_frame_index'],c['frame_sha256']) for c in piano_cand]
    index={c['source_frame_index']:k for k,c in enumerate(cand)};times=np.array([c['timestamp_s'] for c in cand]);fps=cfg['fps']
    coarse=[json.loads(s) for s in (a.coarse/'detections.jsonl').read_text().splitlines()]
    local=[json.loads(s) for s in (a.local/'detections.jsonl').read_text().splitlines()]
    piano={r['source_frame_index']:r for r in map(json.loads,(a.piano/'detections.jsonl').read_text().splitlines())}
    def replace_keyboard(rows):
        merged=[]
        for r in rows:
            # Alias scores belong to a different prompt. Keep provenance; no cross-prompt calibration claim.
            found=sorted(piano[r['source_frame_index']]['detections'],key=lambda b:b['score'],reverse=True);kept=[]
            for b in found:
                if all(iou(b['box_xyxy'],x['box_xyxy'])<.5 for x in kept):kept.append(b)
            merged.append({**r,'detections':[b for b in r['detections'] if b['object']!='keyboard']+[{**b,'detection_alias':b['object'],'object':'keyboard'} for b in kept]})
        return merged
    coarse=replace_keyboard(coarse);local=replace_keyboard(local)
    qs=read(a.questions);save(report/'questions.json',qs)
    semantic={'keyboard':{'meaning':'piano / musical keyboard','aliases':pc['objects'],'source':'User scene-specific clarification; not an automatic semantic disambiguator','applies_to_questions':[q['id'] for q in qs['questions'] if 'keyboard' in q['objects']]}}
    save(report/'semantic-mapping.json',semantic)
    save(a.out/'merged-coarse-detections.json',coarse)
    grays=[cv2.resize(cv2.imread(str(a.local/c['frame']),0),(320,240)) for c in cand]
    params=dict(pyr_scale=.5,levels=3,winsize=15,iterations=3,poly_n=5,poly_sigma=1.2,flags=0)
    @functools.lru_cache(maxsize=192)
    def flow(u,v):return cv2.calcOpticalFlowFarneback(grays[u],grays[v],None,**params)
    def step(u,v,box):
        field=flow(u,v);b=np.asarray(box,dtype=float)/2;x0,y0=np.floor(b[:2]).astype(int);x1,y1=np.ceil(b[2:]).astype(int)
        x0,x1=np.clip([x0,x1],0,320);y0,y1=np.clip([y0,y1],0,240)
        roi=field[y0:y1,x0:x1]
        if roi.size<8:return list(box),0.
        shift=np.median(roi.reshape(-1,2),axis=0);motion=float(np.quantile(np.linalg.norm(roi,axis=2),.75)/max(math.sqrt(max((x1-x0)*(y1-y0),1)),4))
        nb=np.asarray(box)+np.tile(shift*2,2);nb=np.clip(nb,[0,0,0,0],[640,480,640,480]);return nb.tolist(),motion
    def propagate(u,v,box):
        distance=0.;sgn=1 if v>u else -1
        for k in range(u,v,sgn):box,m=step(k,k+sgn,box);distance+=m
        return box,distance
    # Original native-adjacent CUDA Farneback curve, not recomputed from sparse frames.
    native=list(csv.DictReader(a.native_flow.open()));global_cum=np.zeros(cfg['frame_count'])
    for r in native:global_cum[int(r['current_source_frame'])]=float(r['p75_magnitude_px'])/400
    global_cum=np.cumsum(global_cum)
    sharp=np.asarray([c['sharpness'] for c in cand]);quality=.5+.5*sharp/(sharp+max(np.median(sharp),1))
    allpeaks=[];alltracks={};discovery_start=time.perf_counter();trackcache=a.out/'track-cache';trackcache.mkdir(exist_ok=True);suppressed_peaks={}
    for obj in qs['objects']:
        signature=hashlib.sha256(json.dumps({'hardcut':a.peak_threshold,'rows':[[r['source_frame_index'],[b for b in r['detections'] if b['object']==obj]] for r in coarse]},sort_keys=True).encode()).hexdigest();cp=trackcache/(slug(obj)+'.json')
        if cp.exists() and read(cp)['signature']==signature:
            cached=read(cp);peaks,tracks=cached['peaks'],cached['tracks']
        else:
            peaks,tracks=peak_tracks(coarse,obj,index,propagate,hardcut=a.peak_threshold);save(cp,{'signature':signature,'peaks':peaks,'tracks':tracks})
        alltracks[obj]=tracks;peaks,suppressed=merge_peak_fragments(peaks,times);suppressed_peaks[obj]=suppressed
        for p in peaks:p['peak_id']=f'{slug(obj)}-p{len([x for x in allpeaks if x["object"]==obj])+1:02d}';allpeaks.append(p)
        print('PEAKS',obj,len(peaks),'all candidate tracks',len(tracks),flush=True)
    discovery_seconds=time.perf_counter()-discovery_start
    save(a.out/'candidate-tracklets.json',alltracks)
    save(report/'merged-peak-proposals.json',suppressed_peaks)
    fontpath='/System/Library/Fonts/Supplemental/Arial.ttf';font=ImageFont.truetype(fontpath,20);small=ImageFont.truetype(fontpath,16)
    images=report/'frames';images.mkdir();peakdir=report/'peaks';peakdir.mkdir();selected_unique=set()
    for pi,p in enumerate(allpeaks):
        obj=p['object'];anchor=p['anchor'];ak=anchor['frame'];at=times[ak];anchorbox=anchor['box_xyxy'];neighbors=[]
        for direction in [-1,1]:
            box=anchorbox;distance=0.;last_verified=at
            for k in range(ak+direction,len(cand) if direction>0 else -1,direction):
                if abs(times[k]-at)>4:break
                box,m=step(k-direction,k,box);distance+=m
                area=float(np.prod(np.asarray(box)[2:]-np.asarray(box)[:2]));candidates=[]
                for b in local[k]['detections']:
                    ratio=float(np.prod(np.asarray(b['box_xyxy'])[2:]-np.asarray(b['box_xyxy'])[:2]))/max(area,1)
                    if b['object']==obj and b['score']>=.25 and iou(box,b['box_xyxy'])>=.2 and .25<=ratio<=4:candidates.append(b)
                match=max(candidates,key=lambda b:iou(box,b['box_xyxy']),default=None)
                # Stop after >1s without a detection confirmation. No unchecked propagation through gaps.
                if abs(times[k]-last_verified)>1.01 and match is None:break
                if match:box=match['box_xyxy'];last_verified=times[k]
                if area<8:break
                g=abs(global_cum[cand[k]['source_frame_index']]-global_cum[cand[ak]['source_frame_index']])
                neighbors.append({'frame':k,'timestamp_s':float(times[k]),'box_xyxy':box,'dino_score':None if match is None else match['score'],'matched_detection':match is not None,'flow_coordinate':direction*(.75*distance+.25*g),'roi_cumulative_motion':distance,'global_cumulative_motion':g,'quality':float(quality[k]),'status':'locally_verified' if match else 'flow_only'})
        ar={'frame':ak,'timestamp_s':float(at),'box_xyxy':anchorbox,'dino_score':anchor['score'],'matched_detection':True,'flow_coordinate':0.,'quality':float(quality[ak]),'status':'low_score_anchor' if p['fallback'] else 'anchor','reason':f'{discovery_fps:g}fps tracklet maximum score'}
        selected=[ar];pool=[r for r in neighbors if r['matched_detection'] and abs(r['timestamp_s']-at)>=.23]
        def utility(r):
            novelty=min(abs(r['flow_coordinate']-s['flow_coordinate']) for s in selected)
            gap=min(abs(r['timestamp_s']-s['timestamp_s']) for s in selected)
            if gap<.23:return -1.
            return min(novelty,.75)*r['quality']*(.5+.5*r['dino_score'])
        # Prefer evidence on both sides when available; spacing is set by flow, not fixed offsets.
        for direction in [-1,1]:
            options=[r for r in pool if (r['timestamp_s']-at)*direction>0 and r['frame'] not in {s['frame'] for s in selected}]
            if options:
                r=max(options,key=utility);selected.append({**r,'selection_utility':utility(r),'reason':'flow-diverse neighbor before anchor' if direction<0 else 'flow-diverse neighbor after anchor'})
        while len(selected)<4:
            options=[r for r in pool if r['frame'] not in {s['frame'] for s in selected} and utility(r)>=0]
            if not options:break
            r=max(options,key=utility);selected.append({**r,'selection_utility':utility(r),'reason':'max marginal flow coverage among verified neighbors'})
        # Exactly four unique frames requested. If evidence is insufficient, label context rather than certify it.
        while len(selected)<4:
            used={s['frame'] for s in selected};ks=[k for k,t in enumerate(times) if k not in used and abs(t-at)<=2 and all(abs(t-s['timestamp_s'])>=.23 for s in selected)]
            k=min(ks,key=lambda k:(abs(times[k]-at),-quality[k]));selected.append({'frame':k,'timestamp_s':float(times[k]),'box_xyxy':None,'dino_score':None,'matched_detection':False,'flow_coordinate':None,'quality':float(quality[k]),'status':'context_fallback','reason':'insufficient verified local neighbors; nearest distinct context frame, not target evidence'})
        selected.sort(key=lambda r:r['frame']);p['selected']=selected;p['context_fallback_count']=sum(r['status']=='context_fallback' for r in selected);p['low_novelty_pairs']=[]
        for j in range(1,4):
            x,y=selected[j-1],selected[j]
            if x['flow_coordinate'] is not None and y['flow_coordinate'] is not None and abs(x['flow_coordinate']-y['flow_coordinate'])<.03:p['low_novelty_pairs'].append([x['frame'],y['frame']])
        p['anchor_time_s']=float(at);p['episode_time_s']=[float(times[p['observations'][0]['frame']]),float(times[p['observations'][-1]['frame']])]
        folder=peakdir/p['peak_id'];folder.mkdir();canvas=Image.new('RGB',(1280,1120),'white');draw=ImageDraw.Draw(canvas)
        for j,r in enumerate(selected):
            k=r['frame'];name=f'{cand[k]["source_frame_index"]:07d}.jpg';r.update(source_frame_index=cand[k]['source_frame_index'],image='frames/'+name,frame_sha256=cand[k]['frame_sha256']);selected_unique.add(k)
            if not (images/name).exists():shutil.copy2(a.local/cand[k]['frame'],images/name)
            im=Image.open(images/name).convert('RGB');d=ImageDraw.Draw(im)
            if r['box_xyxy']:d.rectangle(r['box_xyxy'],outline='#eb3d35' if k==ak else '#178b59',width=3)
            x=(j%2)*640;y=(j//2)*560;canvas.paste(im,(x,y+80));draw.text((x+8,y+6),f'{obj} | {p["peak_id"].split("-p")[-1]} | {r["timestamp_s"]:.2f}s | frame {r["source_frame_index"]}',font=font,fill='black');draw.text((x+8,y+34),r['status']+(f' | DINO {r["dino_score"]:.3f}' if r['dino_score'] is not None else ''),font=small,fill='#a33322' if r['status']=='context_fallback' else '#444444')
        canvas.save(folder/'four-frames.jpg',quality=92);save(folder/'selection.json',p)
        print('SELECTED',pi+1,'/',len(allpeaks),p['peak_id'],'context',p['context_fallback_count'],flush=True)
    byobj={o:[p for p in allpeaks if p['object']==o] for o in qs['objects']};questions=[]
    for number,q in enumerate(qs['questions'],1):
        ps=[p for o in q['objects'] for p in byobj[o]];folder=report/f'{number:02d}-q{q["id"]}';folder.mkdir();unique=sorted({r['frame'] for p in ps for r in p['selected']});record={'question':q,'peaks':[p['peak_id'] for p in ps],'peak_count':len(ps),'slots':4*len(ps),'unique_frames':len(unique),'source_frames':[cand[k]['source_frame_index'] for k in unique],'context_fallback_slots':sum(p['context_fallback_count'] for p in ps)};save(folder/'selection.json',record);questions.append({**record,'folder':folder.name})
        save(folder/'input-frames.json',{'question':q,'frames':[{'source_frame_index':cand[k]['source_frame_index'],'timestamp_s':float(times[k]),'image':'../frames/'+f'{cand[k]["source_frame_index"]:07d}.jpg','sha256':cand[k]['frame_sha256'],'used_by_peaks':[p['peak_id'] for p in ps if any(r['frame']==k for r in p['selected'])]} for k in unique],'chronological':True,'deduplicated':True,'contains_unverified_context':record['context_fallback_slots']>0})
        fig,axs=plt.subplots(len(q['objects']),1,figsize=(14,2.6*len(q['objects'])),squeeze=False,sharex=True)
        for j,o in enumerate(q['objects']):
            ax=axs[j,0];ys=[max((b['score'] for b in row['detections'] if b['object']==o),default=0) for row in coarse];ts=[r['timestamp_s'] for r in coarse];ax.plot(ts,ys,color='#687a95',lw=1,label=f'{discovery_fps:g}fps DINO score');ax.axhline(a.peak_threshold,color='#aaaaaa',ls='--',lw=.8)
            for n,p in enumerate(byobj[o]):
                ax.scatter([p['anchor_time_s']],[p['anchor']['score']],color='#ce3325',marker='^',s=45);ax.text(p['anchor_time_s'],min(.95,p['anchor']['score']+.07),p['peak_id'].split('-p')[-1],fontsize=8)
                for r in p['selected']:ax.plot([r['timestamp_s']]*2,[-.12,-.03],color='#d68c19' if r['status']=='context_fallback' else '#19865b',lw=1.5)
            ax.set_ylabel(o);ax.set_ylim(-.16,1);ax.grid(alpha=.2)
        axs[0,0].set_title(f'Q{q["id"]}: {len(ps)} peaks x4 = {4*len(ps)} slots; {len(unique)} unique frames\nred triangles: anchors / green ticks: selected evidence / orange: unverified context');axs[-1,0].set_xlabel('Original video seconds');fig.tight_layout();fig.savefig(folder/'distribution.png',dpi=130);plt.close(fig)
        cards=''.join(f'<section><h3>{html.escape(p["object"])} · {p["peak_id"]} · 锚点{p["anchor_time_s"]:.2f}s</h3><p>{"低分兜底锚点；" if p["fallback"] else ""}上下文补位{p["context_fallback_count"]}/4。{("存在低变化帧对；" if p["low_novelty_pairs"] else "")}</p><a href="../peaks/{p["peak_id"]}/four-frames.jpg"><img loading="lazy" src="../peaks/{p["peak_id"]}/four-frames.jpg"></a><a href="../peaks/{p["peak_id"]}/selection.json">选帧依据</a></section>' for p in ps)
        (folder/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:18px system-ui;max-width:1280px;margin:24px auto;line-height:1.6}img{max-width:100%}section{border-top:1px solid #bbb}</style><a href="../index.html">返回十题</a><h1>Q'+str(q['id'])+'</h1><p>'+html.escape(q['question'])+f'</p><p>{len(ps)}个候选波峰，{4*len(ps)}个帧位，去重后{len(unique)}帧。波峰不代表独立物体；补位帧不保证目标可见。</p><img src="distribution.png">'+cards)
        # One compact overview per question, all peak strips; originals remain linked above.
        thumb=Image.new('RGB',(1280,70+len(ps)*300),'white');dr=ImageDraw.Draw(thumb);dr.text((12,15),f'Q{q["id"]} | {len(ps)} peaks x 4 | {len(unique)} unique frames',font=font,fill='black')
        for j,p in enumerate(ps):
            dr.text((8,70+j*300),f'{p["peak_id"]} | anchor {p["anchor_time_s"]:.2f}s',font=small,fill='black')
            for col,r in enumerate(p['selected']):
                im=Image.open(report/r['image']).resize((320,240));thumb.paste(im,(320*col,100+j*300));dr.text((320*col+5,342+j*300),f'{r["timestamp_s"]:.2f}s '+r['status'],font=small,fill='#a33322' if r['status']=='context_fallback' else 'black')
        thumb.save(folder/'all-peaks.jpg',quality=91)
    config={'version':'bvb_v3 adaptive peaks4 prototype','scene':qs['scene'],'discovery_fps':discovery_fps,'local_candidate_fps':local_fps,'frames_per_peak':4,'hardcut':a.peak_threshold,'association_detection_min_score':.25,'association_iou':.2,'max_detection_gap_s':2,'local_radius_s':4,'local_unconfirmed_gap_s':1,'local_min_spacing_s':.23,'peak_definition':'Maximum discovery-rate score per flow-linked visibility tracklet, followed by 6s temporal nonmaximum suppression across fragments; simultaneous disjoint anchor boxes survive. Grouped waves are not unique instances','peak_min_separation_s':6,'fallback':f'If category has no >={a.peak_threshold:g} discovery-rate detection, retain only its best discovery-rate tracklet/anchor; fill missing local neighbors as explicitly unverified context','flow':'Local Farneback at 320x240 on local-rate adjacent JPEG frames; ROI p75 / sqrt(box area), integrated outward. Global term reuses native-adjacent CUDA Farneback p75/400 integral. Signed coordinate = 0.75 ROI + 0.25 global. Selection uses distance to already selected coordinates, capped 0.75, times sharpness quality and (0.5+0.5*score).','limitations':['Tracklets are not physical instances; cross-gap re-ID is absent. Temporal NMS may group different instances visible at nearby times; only simultaneous disjoint anchor boxes are protected.','No automatic semantic verification; only keyboard is user-resolved to piano. Other known false positives retained.','Flow path length is not proof of 3D baseline or geometric overlap; camera shake can contribute.','Exactly four distinct frames per peak; low-change pairs and unverified context explicitly flagged.','No SAM2, no LingBot, no extra spatial connection frames, no automatic difficulty budget in this fixed-four-per-peak experiment.'],'source_sha256':{str(p):sha(p) for p in [a.coarse/'detections.jsonl',a.local/'detections.jsonl',a.piano/'detections.jsonl',a.native_flow,a.questions,Path(__file__)]}}
    save(report/'config.json',config);save(report/'peaks.json',allpeaks)
    summary={'questions':questions,'peak_count':len(allpeaks),'slots':len(allpeaks)*4,'unique_source_frames':len(selected_unique),'context_fallback_slots':sum(p['context_fallback_count'] for p in allpeaks),'low_score_peaks':sum(p['fallback'] for p in allpeaks),'low_novelty_peak_count':sum(bool(p['low_novelty_pairs']) for p in allpeaks),'object_peak_counts':{o:len(ps) for o,ps in byobj.items()},'discovery_seconds':discovery_seconds,'local_processing_seconds':time.perf_counter()-start,'piano_detector':read(a.piano/'detector-info.json'),'no_qa_performed':True};save(report/'summary.json',summary)
    links=''.join(f'<tr><td><a href="{r["folder"]}/index.html">Q{r["question"]["id"]}</a></td><td>{html.escape(", ".join(r["question"]["objects"]))}</td><td>{r["peak_count"]}</td><td>{r["slots"]}</td><td>{r["unique_frames"]}</td><td>{r["context_fallback_slots"]}</td><td><a href="{r["folder"]}/all-peaks.jpg">全部帧组图</a> · <a href="{r["folder"]}/distribution.png">分布图</a></td></tr>' for r in questions)
    description='discovery-rate发现波峰，先取光流关联片段最高分，再用6秒时间抑制合并相邻碎峰，同帧分离框保留；每峰再选3张局部补充帧。局部候选使用4fps缓存；钢琴重新检测。绿色框为局部检测确认，红框为锚点，橙色文字为未经确认的上下文补位。每峰4张不同帧，跨峰可重合，输入重构前按原视频帧号去重。没有SAM2；波峰数量不等于实际物体数；已知误检仍可能存在。'
    (report/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:17px system-ui;max-width:1400px;margin:30px auto;line-height:1.6}td,th{padding:9px;border-bottom:1px solid #ddd}</style><h1>BVB_v3：10题 adaptive 抽帧，每峰4帧</h1><p>'+description+'</p><table><tr><th>题目</th><th>相关物体</th><th>波峰</th><th>帧位</th><th>去重帧</th><th>上下文补位</th><th>图片</th></tr>'+links+'</table><p><a href="config.json">算法与限制</a> · <a href="summary.json">统计</a> · <a href="peaks.json">全部选帧依据</a> · <a href="semantic-mapping.json">语义映射</a></p>')
    (report/'README.txt').write_text(description+'\n\n'+json.dumps(config,ensure_ascii=False,indent=2)+'\n')
    print('DONE',json.dumps({k:v for k,v in summary.items() if k not in ['questions','piano_detector']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
