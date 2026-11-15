import argparse,json,subprocess,sys
from pathlib import Path
from . import __version__
from .pipeline import ROOT,HERE,METHOD,sha,read,prepare,infer,execute

def verify():
    lock=read(HERE/'method-lock.json')
    changed=[p for p,h in lock['files'].items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
    if changed:raise RuntimeError('V10 dependencies changed; review/version explicitly: '+', '.join(changed[:8]))
    return len(lock['files'])

def main():
    p=argparse.ArgumentParser(description='BVB V10: reviewed per-question frames -> ONE shared-pose minimal bpy scene')
    p.add_argument('--version',action='version',version=__version__)
    sub=p.add_subparsers(dest='command',required=True);sub.add_parser('verify')
    a=sub.add_parser('sample',help='Reuse 2fps DINO/LingBot/flow caches; extract question peaks, DINOv2 and clusters')
    for k in ['cache','poses','flow','questions','out']:a.add_argument('--'+k,type=Path,required=True)
    a.add_argument('--device',default='cpu',choices=['cpu','cuda'])
    a=sub.add_parser('audit',help='Astra question policy, clean-image verification and coverage; calls API')
    for k in ['selection','out']:a.add_argument('--'+k,type=Path,required=True)
    a=sub.add_parser('prepare',help='Freeze all reviewed groups for one joint call; no model request')
    a.add_argument('--prepared',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
    a=sub.add_parser('infer',help='Joint Astra high via subscription_direct; no paid API fallback')
    a.add_argument('--out',type=Path,required=True);a.add_argument('--attempt',default='attempt-01',choices=['attempt-01','attempt-02','attempt-03'])
    a=sub.add_parser('iterate',help='Fresh round 0 plus two fixed-LingBot-pose visual geometry refinements')
    a.add_argument('--prepared',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
    a.add_argument('--blender',default='/Applications/Blender.app/Contents/MacOS/Blender')
    for name in ['build','render']:
        a=sub.add_parser(name);a.add_argument('--out',type=Path,required=True)
        a.add_argument('--blender',default='/Applications/Blender.app/Contents/MacOS/Blender')
    # Use the same reviewed-selection and shared-pose preparation as V8.
    if len(sys.argv)>1 and sys.argv[1]=='prepare-regions':
        verify();bootstrap="import sys,runpy;sys.path.insert(0,sys.argv.pop(1));runpy.run_module('bvb_v6',run_name='__main__')"
        subprocess.run([sys.executable,'-c',bootstrap,str(METHOD),'prepare',*sys.argv[2:]],check=True);return
    args=p.parse_args();n=verify()
    if args.command=='verify':print(json.dumps(dict(version=__version__,locked_files=n,status='verified')))
    elif args.command=='prepare':prepare(args.prepared,args.out)
    elif args.command=='infer':infer(args.out,args.attempt)
    elif args.command=='iterate':
        from .refine import run
        run(args.prepared,args.out,args.blender)
    elif args.command in ['build','render']:execute(args.out,args.blender,args.command=='render')
    elif args.command=='sample':
        from .frontend import sample
        sample(args.cache,args.poses,args.flow,args.questions,args.out,args.device)
    elif args.command=='audit':
        from .frontend import audit
        audit(args.selection,args.out)
if __name__=='__main__':main()
