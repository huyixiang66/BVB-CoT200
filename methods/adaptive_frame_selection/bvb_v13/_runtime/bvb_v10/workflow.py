"""Complete per-video V10 workflow. Features may be produced on a separate GPU host."""
import argparse,json,sys,subprocess,time
from pathlib import Path
from .pipeline import ROOT,prepare,infer,execute,read,save,sha

def state(out,stage,**extra):save(out/'status.json',dict(stage=stage,updated=time.time(),**extra))
def run(questions,features,out,blender,device):
 from .frontend import sample,audit
 from .qa import qa
 from .report import render
 from .entities import public_questions
 if any(len(q.get('options',[]))<2 for q in public_questions(questions)):raise ValueError('This QA adapter requires multiple-choice options')
 out=out.resolve();out.mkdir(parents=True,exist_ok=True);features=features.resolve()
 identity=dict(questions_sha256=sha(questions),feature_config_sha256=sha(features/'detector/local-cache/config.json'),method_lock_sha256=sha(ROOT/'bvb_v10/method-lock.json'),rounds=[0],provider='subscription_direct',model='gpt-6-astra',reasoning='high')
 if (out/'run-config.json').exists() and read(out/'run-config.json')!=identity:raise ValueError('Inputs changed; use a fresh run')
 save(out/'run-config.json',identity)
 selection=out/'sampling/clusters/selection.json'
 if not selection.exists():sample(features/'detector/local-cache',features/'poses',features/'native-flow.csv',questions,out/'sampling',device)
 reviewed=out/'audit/input/review/reviewed-selection.json'
 if not (out/'audit/status.json').exists():audit(selection,out/'audit')
 elif read(out/'audit/status.json')['stage']!='completed':
  sys.path.insert(0,str(ROOT/'scripts'));import review_revsi_first10_astra as ar
  ar.OUT=out/'audit';ar.REVIEW=ar.OUT/'input/review';ar.main()
 rows=read(reviewed)['questions']
 failures=[]
 for qid,row in rows.items():
  if not str(qid).replace('-','').replace('_','').isalnum():raise ValueError('Unsafe question ID')
  prepared=out/'prepared'/str(qid);folder=out/'questions'/str(qid)
  try:
   if not (prepared/'manifest.json').exists():subprocess.run([sys.executable,'-m','bvb_v6','prepare','--selection',str(reviewed),'--qid',str(qid),'--cache',str(features/'detector/local-cache'),'--poses',str(features/'poses'),'--out',str(prepared)],cwd=ROOT,check=True)
   if not (folder/'input.json').exists():prepare(prepared,folder)
   if not (folder/'generated.py').exists():state(folder,'reconstructing');render(out);infer(folder,'attempt-01')
   if not (folder/'scope-validation.json').exists():state(folder,'building');execute(folder,blender)
   if not (folder/'validation.json').exists():state(folder,'rendering');execute(folder,blender,True)
   state(folder,'selecting-view');render(out);qa(folder);state(folder,'completed')
  except Exception as e:
   state(folder,'failed',error=str(e));failures.append(str(qid))
  render(out)
 save(out/'status.json',dict(stage='completed' if not failures else 'needs-attention',failed_questions=failures,questions=len(rows)))
 if failures:raise RuntimeError('Inspect failed questions before retry: '+', '.join(failures))

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
 a=s.add_parser('entities');a.add_argument('--questions',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
 a=s.add_parser('features')
 for key in ['video','questions','out']:a.add_argument('--'+key,type=Path,required=True)
 a.add_argument('--lingbot-root',type=Path);a.add_argument('--stage',choices=['all','frames','dino','poses','flow'],default='all')
 a=s.add_parser('run')
 for key in ['questions','features','out']:a.add_argument('--'+key,type=Path,required=True)
 a.add_argument('--device',choices=['cpu','cuda'],default='cuda');a.add_argument('--blender',default='/Applications/Blender.app/Contents/MacOS/Blender')
 a=s.add_parser('qa');a.add_argument('--out',type=Path,required=True)
 a=s.add_parser('report');a.add_argument('--out',type=Path,required=True);a.add_argument('--reference',type=Path,help='Optional display-only {qid: option-letter}; never used by inference')
 args=p.parse_args(argv)
 from .__main__ import verify
 verify()
 if args.command=='entities':
  from .entities import prepare as extract
  extract(args.questions,args.out)
 elif args.command=='features':
  from .features import run as features
  features(args.video,args.questions,args.out,args.lingbot_root,args.stage)
 elif args.command=='run':run(args.questions,args.features,args.out,args.blender,args.device)
 elif args.command=='qa':
  from .qa import qa
  qa(args.out.resolve())
 else:
  from .report import render
  render(args.out,args.reference)
if __name__=='__main__':main()
