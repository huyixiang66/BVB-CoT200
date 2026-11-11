"""Portable raw-video adapter for the existing 2fps MM-DINO/LingBot/Farneback recipe."""
import csv,json,os,shutil,subprocess,sys,time
from pathlib import Path
from bvb_v10.pipeline import ROOT,read,save,sha
MODEL='openmmlab-community/mm_grounding_dino_base_all'
REVISION='b32293f50a56f7218d9f73ff940ffb9eba84d63c'

def command(args):subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,check=True)
def detect(job,questions):
 from transformers import AutoProcessor
 tokenizer=AutoProcessor.from_pretrained(MODEL,revision=REVISION,local_files_only=True).tokenizer
 labels=read(questions)['objects'];batches=[];batch=[]
 for label in labels:
  if len(tokenizer(' . '.join(batch+[label])+' .')['input_ids'])>240 and batch:batches.append(batch);batch=[]
  batch.append(label)
 if batch:batches.append(batch)
 if any(len(tokenizer(' . '.join(b)+' .')['input_ids'])>256 for b in batches):raise ValueError('Detector label exceeds token limit')
 out=job/'detector';out.mkdir(exist_ok=True)
 for i,objects in enumerate(batches):
  qfile=out/f'batch-{i:03d}-objects.json';save(qfile,dict(objects=objects))
  command([ROOT/'scripts/object_sampling/detect_cached_frames.py','--cache',job/'cache','--questions',qfile,'--out',out/f'batch-{i:03d}','--model',MODEL,'--revision',REVISION,'--box-threshold','.25','--text-threshold','.20'])
 cand=read(job/'cache/candidates.json');merged=[dict(source_frame_index=r['source_frame_index'],timestamp_s=r['timestamp_s'],detections=[]) for r in cand]
 for i in range(len(batches)):
  rows=[json.loads(x) for x in (out/f'batch-{i:03d}/local-cache/detections.jsonl').read_text().splitlines()]
  for a,b in zip(merged,rows,strict=True):
   assert a['source_frame_index']==b['source_frame_index'];a['detections']+=b['detections']
 final=out/'local-cache';final.mkdir(exist_ok=True)
 (final/'detections.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in merged));shutil.copy2(job/'cache/candidates.json',final/'candidates.json')
 save(final/'config.json',dict(read(job/'cache/config.json'),objects=labels,model=MODEL,revision_requested=REVISION))
 if not (final/'frames').exists():(final/'frames').symlink_to((job/'cache/frames').resolve())
 save(final/'detector-info.json',dict(model=MODEL,revision=REVISION,label_batches=batches,score_note='Scores are conditioned on the caption batch; phrase means, peak threshold and per-category NMS match the existing detector.'))

def flow(video,out):
 import cv2,numpy as np
 cap=cv2.VideoCapture(str(video));expected=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));prev=None;n=0;start=time.time();tmp=out.with_suffix('.tmp')
 with tmp.open('w') as f:
  w=csv.writer(f);w.writerow(['current_source_frame','p75_magnitude_px'])
  while True:
   ok,im=cap.read()
   if not ok:break
   gray=cv2.cvtColor(cv2.resize(im,(320,240)),cv2.COLOR_BGR2GRAY)
   if prev is not None:
    a=cv2.calcOpticalFlowFarneback(prev,gray,None,.5,3,15,3,5,1.2,0);w.writerow([n,float(np.quantile(np.linalg.norm(a,axis=2),.75))])
   prev=gray;n+=1
 cap.release();assert n==expected;tmp.replace(out)
 save(out.parent/'flow-info.json',dict(backend='OpenCV CPU Farneback',size=[320,240],pairs=n-1,elapsed_s=time.time()-start))

def run(video,questions,out,lingbot_root,stage='all',keyframe_interval='auto'):
 out=out.resolve();out.mkdir(parents=True,exist_ok=True);video=video.resolve();questions=questions.resolve()
 identity=dict(keyframe_interval_policy=keyframe_interval,video_sha256=sha(video),questions_sha256=sha(questions),candidate_fps=2,model=MODEL,revision=REVISION)
 if (out/'feature-input.json').exists() and read(out/'feature-input.json')!=identity:raise ValueError('Incompatible feature inputs; use a new output directory')
 save(out/'feature-input.json',identity);os.environ.setdefault('HF_ENDPOINT','https://hf-mirror.com')
 if stage in ('all','frames'):
  command([ROOT/'scripts/object_sampling/analyze_video.py','--video',video,'--questions',questions,'--out',out/'cache','--candidate-fps','2','--stage','frames'])
  cfg=read(out/'cache/config.json')
  if [cfg['width'],cfg['height']]!=[640,480]:raise ValueError('This frozen geometry recipe expects 640x480 inputs; normalize the video before all feature stages')
 if stage in ('all','dino'):detect(out,questions)
 if stage in ('all','poses') and not (out/'poses/validation.json').exists():
  if not lingbot_root:raise ValueError('--lingbot-root is required for pose extraction')
  from . import pose_exporter
  pose_exporter.INSTALL=lingbot_root.resolve();pose_exporter.poses(out,keyframe_interval=keyframe_interval)
 if stage in ('all','flow') and not (out/'native-flow.csv').exists():flow(video,out/'native-flow.csv')
