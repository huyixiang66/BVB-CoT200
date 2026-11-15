"""Run a pinned grounding detector on immutable cached video frames.

Preserves the existing token-mean/category-argmax/NMS scoring contract. Thresholds
default to the source cache and can be overridden explicitly before NMS.
No segmentation, QA labels, or frame re-extraction.
"""
import argparse,hashlib,json,time,shutil
from pathlib import Path
import numpy as np
from PIL import Image
from .analyze_video import nms

def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def merge_detections(boxes,means,peaks,labels,aliases,width,height,text_threshold,box_threshold):
 category=means.argmax(1);indices=np.arange(len(boxes))
 chosen_mean=means[indices,category];chosen_peak=peaks[indices,category]
 canonical=[aliases.get(label,label) for label in labels];found=[]
 for obj in dict.fromkeys(canonical):
  classes=[i for i,c in enumerate(canonical) if c==obj]
  valid=np.flatnonzero(np.isin(category,classes)&(chosen_mean>=text_threshold)&(chosen_peak>=box_threshold))
  bs=boxes[valid].clip([0,0,0,0],[width,height,width,height]);scores=chosen_mean[valid]
  for j in nms(bs,scores)[:20]:
   if np.any(bs[j,2:]-bs[j,:2]<2):continue
   item=dict(object=obj,score=float(scores[j]),peak_token_score=float(chosen_peak[valid[j]]),box_xyxy=bs[j].tolist())
   if aliases:item['detection_alias']=labels[category[valid[j]]]
   found.append(item)
 return found
def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--questions',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--model',required=True);p.add_argument('--revision',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--include-piano',action='store_true');p.add_argument('--box-threshold',type=float);p.add_argument('--text-threshold',type=float);p.add_argument('--label-aliases',type=Path);a=p.parse_args()
 for value in [a.box_threshold,a.text_threshold]:
  if value is not None and not 0<=value<=1:p.error('Detection thresholds must lie in [0,1]')
 import torch,transformers
 from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection,AutoConfig
 from .mm_grounding_compat import prepare,restore_checkpoint_heads,audit
 torch.set_num_threads(4);start=time.perf_counter();model_cfg=AutoConfig.from_pretrained(a.model,revision=a.revision,local_files_only=True);compat=prepare(model_cfg);processor=AutoProcessor.from_pretrained(a.model,revision=a.revision,local_files_only=True);model=AutoModelForZeroShotObjectDetection.from_pretrained(a.model,revision=a.revision,local_files_only=True).to(a.device).eval()
 restore_checkpoint_heads(model,a.model,a.revision,compat)
 compat['weight_audit']=audit(model,a.model,a.revision)
 loaded=time.perf_counter();print('MODEL',type(model).__name__,type(processor).__name__,flush=True)
 cfg=read(a.cache/'config.json');cand=read(a.cache/'candidates.json');objects=read(a.questions)['objects'];w,h=cfg['width'],cfg['height'];aliases=read(a.label_aliases) if a.label_aliases else {}
 assert all(k in objects and v in objects for k,v in aliases.items())
 for r in cand:assert sha(a.cache/r['frame'])==r['frame_sha256']
 conditions=[('local-cache',objects)]+([('piano-cache',['piano'])] if a.include_piano else [])
 a.out.mkdir(parents=True,exist_ok=True)
 for name,labels in conditions:
  active_aliases=aliases if name=='local-cache' else {}
  out=a.out/name;out.mkdir(exist_ok=True);newcfg={**cfg,'objects':list(dict.fromkeys(active_aliases.get(label,label) for label in labels)),'model':a.model,'revision_requested':a.revision}
  if active_aliases:newcfg.update(detection_labels=labels,label_aliases=active_aliases)
  if a.box_threshold is not None:newcfg['box_threshold']=a.box_threshold
  if a.text_threshold is not None:newcfg['text_threshold']=a.text_threshold
  if (out/'config.json').exists():assert read(out/'config.json')==newcfg,'Refuse incompatible resume'
  save(out/'config.json',newcfg);shutil.copy2(a.cache/'candidates.json',out/'candidates.json')
  if not (out/'frames').exists():(out/'frames').symlink_to((a.cache/'frames').resolve(),target_is_directory=True)
  caption='';spans=[]
  for obj in labels:
   begin=len(caption);caption+=obj;spans.append((begin,len(caption)));caption+=' . '
  caption=caption.strip();encoded=processor.tokenizer(caption,return_offsets_mapping=True);offsets=encoded['offset_mapping'];ids=[[i for i,(x,y) in enumerate(offsets) if y>x and x<end and y>begin] for begin,end in spans]
  assert all(ids) and len(offsets)<=model.config.max_text_len
  file=out/'detections.jsonl';done={r['source_frame_index'] for r in map(json.loads,file.read_text().splitlines())} if file.exists() else set()
  assert len(done)<=len(cand);torch.cuda.synchronize();begin=time.perf_counter();count=0
  with file.open('a') as f:
   for i,row in enumerate(cand):
    if row['source_frame_index'] in done:continue
    im=Image.open(a.cache/row['frame']).convert('RGB');assert im.size==(w,h)
    inputs=processor(images=im,text=caption,return_tensors='pt',size={'shortest_edge':640,'longest_edge':1024}).to(a.device)
    assert inputs['input_ids'][0].tolist()==encoded['input_ids']
    with torch.inference_mode():pred=model(**inputs)
    assert pred.logits.shape[-1]>=len(offsets)
    probs=pred.logits[0].sigmoid().float().cpu().numpy();raw=pred.pred_boxes[0].float().cpu().numpy()
    boxes=np.column_stack((raw[:,:2]-raw[:,2:]/2,raw[:,:2]+raw[:,2:]/2))*[w,h,w,h]
    means=np.stack([probs[:,idx].mean(1) for idx in ids],1);peaks=np.stack([probs[:,idx].max(1) for idx in ids],1)
    found=merge_detections(boxes,means,peaks,labels,active_aliases,w,h,newcfg['text_threshold'],newcfg['box_threshold'])
    f.write(json.dumps(dict(source_frame_index=row['source_frame_index'],timestamp_s=row['timestamp_s'],detections=found))+'\n');f.flush();count+=1
    if i%50==0:print(name,i+1,'/',len(cand),'detections',len(found),flush=True)
  torch.cuda.synchronize();elapsed=time.perf_counter()-begin
  save(out/'detector-info.json',dict(model=a.model,resolved_commit=getattr(model.config,'_commit_hash',None),model_class=type(model).__name__,processor_class=type(processor).__name__,torch_version=torch.__version__,transformers_version=transformers.__version__,device=a.device,gpu=torch.cuda.get_device_name(),caption=caption,category_token_indices=dict(zip(labels,ids)),alias_mapping=active_aliases,box_threshold=newcfg['box_threshold'],text_threshold=newcfg['text_threshold'],postprocess='Assign each query to highest mean token-probability phrase; mean>=text_threshold and peak>=box_threshold; map explicit aliases to canonical category before NMS IoU .5; preserve winning phrase and score',dtype='float32',processed_frames_this_run=count,total_frames=len(cand),model_load_seconds=loaded-start,compatibility=compat,detector_loop_seconds=elapsed,frame_hashes_verified=True,script_sha256=sha(Path(__file__)),resumed_frame_count=len(done)))
  assert len(file.read_text().splitlines())==len(cand)
  print('DONE',name,elapsed,flush=True)
if __name__=='__main__':main()
