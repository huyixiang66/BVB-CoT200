import argparse
import json
from pathlib import Path
from . import __version__
from .common import BLENDER

def main():
    parser=argparse.ArgumentParser(description='BVB V11: seven task adapters, reviewed evidence and reproducible receipts')
    parser.add_argument('--version',action='version',version=__version__)
    parser.add_argument('--measure',action='store_true',help='Record API attempts/tokens/timing in sibling OUT-usage/cost-calls (no monetary estimate)')
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('verify')
    p=sub.add_parser('features',help='Raw video -> 2fps DINO/LingBot/flow cache; GPU environment required')
    for name in ['video','questions','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--lingbot-root',type=Path);p.add_argument('--keyframe-interval',default='auto')
    p.add_argument('--stage',choices=['all','frames','dino','poses','flow'],default='all')
    p=sub.add_parser('sample',help='Question peaks and object clusters from existing feature caches')
    for name in ['cache','poses','flow','questions','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--counting-mode',action='store_true')
    p=sub.add_parser('audit',help='Fresh Astra question policy and clean-image candidate review')
    for name in ['selection','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--counting-mode',action='store_true')
    p=sub.add_parser('workflow',help='Reviewed selection -> recovery -> reference -> reconstruction -> QA')
    for name in ['selection','cache','poses','video','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--reference-selection',type=Path,help='Optional positively reviewed same-scene clusters for third-object reference candidates')
    p.add_argument('--scale-policy',choices=['stable','width-largest'],default='stable')
    p.add_argument('--blender',default=BLENDER)
    p=sub.add_parser('prepare',help='Prepare a task adapter from an existing reviewed region packet')
    for name in ['source','poses','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--context',type=Path,help='Required for abs dist: structural context plan JSON')
    for action in ['run','qa']:
        p=sub.add_parser(action);p.add_argument('--out',type=Path,required=True)
        if action=='run':p.add_argument('--blender',default=BLENDER)
    p=sub.add_parser('room',help='Existing blend -> pinned geometry selector -> room-area QA')
    for name in ['blend','video','question','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--blender',default=BLENDER)
    args=parser.parse_args()
    from .verify import verify
    n=verify()
    if args.measure:
        if not hasattr(args,'out'):parser.error('--measure requires a command with --out')
        import sys
        from bvb_v10.pipeline import ROOT
        sys.path.insert(0,str(ROOT/'scripts'))
        usage=args.out.resolve().with_name(args.out.name+'-usage');usage.mkdir(parents=True,exist_ok=True)
        from .cost_meter import install
        install(usage,usage)(args.command)
    if args.command=='verify':print(json.dumps(dict(version=__version__,locked_files=n,status='verified')))
    elif args.command=='features':
        from .features import run
        run(args.video,args.questions,args.out,args.lingbot_root,args.stage,args.keyframe_interval)
    elif args.command=='sample':
        from .frontend import sample
        sample(args.cache,args.poses,args.flow,args.questions,args.out,args.device,args.counting_mode)
    elif args.command=='audit':
        if args.counting_mode:
            from .counting_rejection import install_review_policy
            install_review_policy()
        from .frontend import audit
        audit(args.selection,args.out)
    elif args.command=='workflow':
        from .workflow import run
        run(args.selection,args.cache,args.poses,args.video,args.out,args.blender,args.reference_selection,args.scale_policy)
    elif args.command=='prepare':
        from .driver import prepare
        prepare(args.source.resolve(),args.out.resolve(),args.poses.resolve(),context=args.context)
    elif args.command=='run':
        from .driver import run
        run(args.out.resolve(),args.blender)
    elif args.command=='qa':
        from .driver import qa
        qa(args.out.resolve())
    elif args.command=='room':
        from .room import run
        run(args.blend,args.video,args.question,args.out,args.blender)

if __name__=='__main__':main()
