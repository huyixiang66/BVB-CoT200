import argparse,json
from pathlib import Path
from .pipeline import prepare,run,infer
from .verify import verify

def main():
    p=argparse.ArgumentParser(description='V10_2 soft trajectory/anchor joint reconstruction; full workflow: python -m bvb_v10_2.workflow')
    sub=p.add_subparsers(dest='action',required=True);sub.add_parser('verify')
    a=sub.add_parser('prepare')
    for k in ['source','out','poses']:a.add_argument('--'+k,type=Path,required=True)
    a.add_argument('--replace-poses',action='store_true',help='Rebind frozen selected images to new poses; exact index/time/hash checks')
    a=sub.add_parser('run');a.add_argument('--out',type=Path,required=True)
    a.add_argument('--blender',default='/Applications/Blender.app/Contents/MacOS/Blender')
    a=sub.add_parser('infer');a.add_argument('--out',type=Path,required=True);a.add_argument('--attempt',default='attempt-01')
    args=p.parse_args();n=verify()
    if args.action=='verify':print(json.dumps(dict(version='10.2.0',locked_files=n,status='verified')))
    elif args.action=='prepare':prepare(args.source.resolve(),args.out.resolve(),args.poses.resolve(),args.replace_poses)
    elif args.action=='infer':infer(args.out.resolve(),args.attempt)
    else:run(args.out.resolve(),args.blender)
if __name__=='__main__':main()
