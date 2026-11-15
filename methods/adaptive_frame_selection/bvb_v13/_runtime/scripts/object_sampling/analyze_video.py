"""Shared video cache: GroundingDINO detections and native-frame Farneback motion.

No answers, scene geometry, or model-generated QA responses are read.
"""
import argparse, hashlib, json, math, time
from pathlib import Path
import cv2
import numpy as np

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def save(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)

def nms(boxes,scores,threshold=.5):
    order=np.argsort(-scores,kind='stable');keep=[]
    while len(order):
        i=int(order[0]);keep.append(i);rest=order[1:]
        if not len(rest):break
        lo=np.maximum(boxes[i,:2],boxes[rest,:2]);hi=np.minimum(boxes[i,2:],boxes[rest,2:])
        inter=np.maximum(hi-lo,0).prod(1);area=np.maximum(boxes[:,2:]-boxes[:,:2],0).prod(1)
        iou=inter/np.maximum(area[i]+area[rest]-inter,1e-9);order=rest[iou<=threshold]
    return keep

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--video',required=True,type=Path);p.add_argument('--questions',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--candidate-fps',type=float,default=1.);p.add_argument('--model',default='IDEA-Research/grounding-dino-tiny');p.add_argument('--revision',default='main');p.add_argument('--device',default='cuda:0')
    p.add_argument('--box-threshold',type=float,default=.25);p.add_argument('--text-threshold',type=float,default=.20);p.add_argument('--stage',choices=['all','detect','flow','frames'],default='all');a=p.parse_args()
    if a.candidate_fps<=0:raise ValueError('candidate fps must be positive')
    out=a.out;out.mkdir(parents=True,exist_ok=True);frames=out/'frames';frames.mkdir(exist_ok=True)
    objects=json.loads(a.questions.read_text())['objects'];cap=cv2.VideoCapture(str(a.video));fps=cap.get(cv2.CAP_PROP_FPS);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));w=int(cap.get(3));h=int(cap.get(4))
    assert fps>0 and count>0 and a.candidate_fps<=fps
    indices=sorted(set(round(t*fps/a.candidate_fps) for t in range(math.ceil(count/fps*a.candidate_fps)) if round(t*fps/a.candidate_fps)<count))
    config={'video_sha256':digest(a.video),'fps':fps,'frame_count':count,'width':w,'height':h,'duration_s':count/fps,'candidate_fps':a.candidate_fps,'objects':objects,'model':a.model,'revision_requested':a.revision,'box_threshold':a.box_threshold,'text_threshold':a.text_threshold,'shortest_edge':640,'longest_edge':1024,'version':1}
    if (out/'config.json').exists() and json.loads((out/'config.json').read_text())!=config:raise ValueError('Cache configuration differs; use a new output directory.')
    save(out/'config.json',config)
    if not (out/'candidates.json').exists():
        wanted=set(indices);rows=[];n=0
        while True:
            ok,bgr=cap.read()
            if not ok:break
            if n in wanted:
                path=frames/f'{n:07d}.jpg';assert cv2.imwrite(str(path),bgr,[cv2.IMWRITE_JPEG_QUALITY,95])
                gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
                rows.append({'index':len(rows),'source_frame_index':n,'timestamp_s':n/fps,'frame':f'frames/{n:07d}.jpg','frame_sha256':digest(path),'sharpness':float(cv2.Laplacian(gray,cv2.CV_32F).var())})
            n+=1
        assert n==count and len(rows)==len(indices);save(out/'candidates.json',rows)
    cap.release();rows=json.loads((out/'candidates.json').read_text())
    if a.stage in ['all','detect']:
        import torch,transformers
        from PIL import Image
        from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
        torch.set_num_threads(4);start=time.perf_counter()
        processor=AutoProcessor.from_pretrained(a.model,revision=a.revision)
        model=AutoModelForZeroShotObjectDetection.from_pretrained(a.model,revision=a.revision).to(a.device).eval()
        # Token spans give exact category attribution, including multiword objects.
        # Unlike free decoded phrases, no category is silently merged with another.
        caption='';spans=[]
        for obj in objects:
            begin=len(caption);caption+=obj;spans.append((begin,len(caption)));caption+=' . '
        caption=caption.strip();encoded=processor.tokenizer(caption,return_offsets_mapping=True)
        offsets=encoded['offset_mapping'];token_indices=[[i for i,(x,y) in enumerate(offsets) if y>x and x<end and y>begin] for begin,end in spans]
        assert all(token_indices) and len(offsets)<=model.config.max_text_len
        detections=out/'detections.jsonl';done={}
        if detections.exists():
            for s in detections.read_text().splitlines():
                r=json.loads(s);done[r['source_frame_index']]=r
        if str(a.device).startswith('cuda'):torch.cuda.synchronize(a.device)
        loop_start=time.perf_counter();processed_count=0
        with detections.open('a') as f:
            for i,row in enumerate(rows):
                if row['source_frame_index'] in done:continue
                img=Image.open(out/row['frame']).convert('RGB')
                inputs=processor(images=img,text=caption,return_tensors='pt',size={'shortest_edge':640,'longest_edge':1024}).to(a.device)
                assert inputs['input_ids'][0].tolist()==encoded['input_ids']
                with torch.inference_mode():prediction=model(**inputs)
                probs=prediction.logits[0].sigmoid().float().cpu().numpy();raw=prediction.pred_boxes[0].float().cpu().numpy()
                boxes=np.column_stack((raw[:,:2]-raw[:,2:]/2,raw[:,:2]+raw[:,2:]/2))*[w,h,w,h]
                means=np.stack([probs[:,ids].mean(1) for ids in token_indices],1);peaks=np.stack([probs[:,ids].max(1) for ids in token_indices],1)
                category=means.argmax(1);found=[]
                for c,obj in enumerate(objects):
                    valid=np.flatnonzero((category==c)&(means[:,c]>=a.text_threshold)&(peaks[:,c]>=a.box_threshold))
                    bs=boxes[valid].clip([0,0,0,0],[w,h,w,h]);scores=means[valid,c]
                    for j in nms(bs,scores)[:20]:
                        if np.any(bs[j,2:]-bs[j,:2]<2):continue
                        found.append({'object':obj,'score':float(scores[j]),'peak_token_score':float(peaks[valid[j],c]),'box_xyxy':bs[j].tolist()})
                r={'source_frame_index':row['source_frame_index'],'timestamp_s':row['timestamp_s'],'detections':found};f.write(json.dumps(r)+'\n');f.flush()
                processed_count+=1
                if i%20==0:print('DETECT',i+1,'/',len(rows),'boxes',len(found),flush=True)
        if str(a.device).startswith('cuda'):torch.cuda.synchronize(a.device)
        loop_elapsed=time.perf_counter()-loop_start
        save(out/'detector-info.json',{'model':a.model,'resolved_commit':getattr(model.config,'_commit_hash',None),'transformers_version':transformers.__version__,'torch_version':torch.__version__,'device':a.device,'caption':caption,'category_token_indices':dict(zip(objects,token_indices)),'postprocess':'Assign each query to highest mean token-probability category; mean>=text_threshold and peak>=box_threshold; per-category NMS IoU 0.5. Scores are not calibrated probabilities.','elapsed_seconds_this_run':time.perf_counter()-start,'detector_loop_seconds':loop_elapsed,'processed_frames_this_run':processed_count,'detector_loop_scope':'image loading, preprocessing, inference, CPU postprocessing and JSONL writes; excludes model initialization and video extraction; includes first inference warmup'})
        del model;torch.cuda.empty_cache()
    if a.stage in ['all','flow']:
        detect=[json.loads(s) for s in (out/'detections.jsonl').read_text().splitlines()]
        by_frame={r['source_frame_index']:r for r in detect};assert set(by_frame)==set(indices)
        if (out/'flow.json').exists():print('FLOW cache reused',flush=True);return
        cv2.setNumThreads(4);cap=cv2.VideoCapture(str(a.video));fw=320;fh=round(h*fw/w);diag=math.hypot(fw,fh)
        params={'pyr_scale':.5,'levels':3,'winsize':15,'iterations':3,'poly_n':5,'poly_sigma':1.2,'flags':0}
        prev=None;n=0;cumulative=0.;interval=[];results=[];cuts=0;start=time.perf_counter()
        while True:
            ok,bgr=cap.read()
            if not ok:break
            gray=cv2.cvtColor(cv2.resize(bgr,(fw,fh)),cv2.COLOR_BGR2GRAY);mag=np.zeros((fh,fw),np.float32);is_cut=False
            if prev is not None:
                # Backward field is indexed in current-frame coordinates, matching current boxes.
                flow=cv2.calcOpticalFlowFarneback(gray,prev,None,**params);mag=np.linalg.norm(flow,axis=2)
                diff=float(np.mean(np.abs(gray.astype(float)-prev.astype(float)))/255)
                ha=cv2.calcHist([gray],[0],None,[32],[0,256]);hb=cv2.calcHist([prev],[0],None,[32],[0,256]);ha/=ha.sum();hb/=hb.sum()
                is_cut=diff>.35 and float(np.abs(ha-hb).sum())>.7
                if is_cut:mag.fill(0);cuts+=1
                step=float(np.quantile(mag,.75)/diag);cumulative+=step;interval.append(step)
            if n in by_frame:
                object_motion={}
                for d in by_frame[n]['detections']:
                    x0,y0,x1,y1=d['box_xyxy'];x0=max(0,int(x0*fw/w));y0=max(0,int(y0*fh/h));x1=min(fw,math.ceil(x1*fw/w));y1=min(fh,math.ceil(y1*fh/h))
                    if x1>x0 and y1>y0:
                        v=float(np.quantile(mag[y0:y1,x0:x1],.75)/diag*fps)
                        object_motion[d['object']]=max(object_motion.get(d['object'],0),v)
                results.append({'source_frame_index':n,'timestamp_s':n/fps,'global_motion_since_previous_candidate':float(sum(interval)),'cumulative_global_motion':cumulative,'object_roi_motion_per_second':object_motion,'scene_cut_at_frame':is_cut})
                interval=[]
                if len(results)%40==0:print('FLOW',len(results),'/',len(rows),flush=True)
            prev=gray;n+=1
        cap.release();assert n==count and len(results)==len(rows)
        save(out/'flow.json',{'rows':results,'parameters':params,'flow_size':[fw,fh],'opencv_version':cv2.__version__,'native_frame_pairs':count-1,'hard_cut_count':cuts,'elapsed_seconds':time.perf_counter()-start,'semantics':'Backward dense flow at every adjacent native frame. Global p75 magnitude/diagonal integrated between candidates. Object ROI p75 at candidate native pair multiplied by FPS, in image-diagonals/second. Camera motion is included; this is image change, not object 3D speed. Sharpness and capped normalization handled by selector.'})
    print('VIDEO_ANALYSIS_OK',out,flush=True)

if __name__=='__main__':main()
