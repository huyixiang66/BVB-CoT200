import argparse
import json
from pathlib import Path
from .pipeline import run, STAGES
from .backends import SubscriptionBackend


def main():
    parser=argparse.ArgumentParser(description='Portable room-size pipeline; run on the experiment server')
    parser.add_argument('command',choices=['plan','run'])
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--until',choices=STAGES,default='qa')
    parser.add_argument('--subscription-client',type=Path)
    parser.add_argument('--profile')
    parser.add_argument('--expected-email')
    parser.add_argument('--pro-confirmed',action='store_true')
    args=parser.parse_args();config=json.loads(args.config.read_text(encoding='utf8'))
    for key in ['video','out','v10_root','poses','frames_root','detections']:
        if key in config:config[key]=str((args.config.parent/config[key]).resolve())
    if args.command=='plan':
        print(json.dumps({'stages':STAGES,'config':config,'model_calls':3,'model':'gpt-6-astra','effort':'high',
                          'reference_fps_is_soft':True,'network_requests':0},ensure_ascii=False,indent=2));return
    backend=None
    if args.subscription_client:
        backend=SubscriptionBackend(args.subscription_client,args.profile,args.expected_email,pro_confirmed=args.pro_confirmed)
    result=run(config,backend,until=args.until)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
