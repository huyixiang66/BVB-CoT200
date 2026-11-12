"""Explicit-stage object-size batch runner; never launches work on import."""
import argparse
import json
from pathlib import Path
from .workflow import batch,prepare_case,build_server_case,render_server_views,prepare_reviewed_qa
from .handoff import write


def advance(cases, stage, backend=None):
    if stage not in ('generate','build','views','qa'):raise ValueError('Unknown stage')
    if stage in ('generate','qa') and backend is None:raise ValueError('Explicit subscription backend required')
    def one(case):
        work=Path(case['work']).resolve()
        if stage=='generate':
            return prepare_case(case['evidence'],case['base_prompt'],work,backend,case['runtime_root'])
        if stage=='build':
            return build_server_case(work,case['runtime_root'],case['blender'],case['validator_dir'])
        if stage=='views':
            joint=json.loads((work/'joint/input.json').read_text(encoding='utf8'))
            objects=joint['question']['objects']
            return render_server_views(work/'joint/scene.blend',objects[0],objects[1] if len(objects)>1 else None,
                                       work/'views-run',case['blender'],case.get('target_spec'))
        return str(prepare_reviewed_qa(work/'joint/input.json',work/'views-run/views/result.json',
                                       work/'visual-review.json',work,backend))
    return batch(cases,one)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',required=True);p.add_argument('--stage',choices=['generate','build','views','qa'],required=True)
    p.add_argument('--output',required=True);p.add_argument('--subscription-client');p.add_argument('--profile')
    p.add_argument('--expected-email');p.add_argument('--pro-confirmed',action='store_true')
    a=p.parse_args();cases=json.loads(Path(a.cases).read_text(encoding='utf8'))
    backend=None
    if a.stage in ('generate','qa'):
        if not a.subscription_client:raise ValueError('Missing subscription client')
        from room_size_pipeline.backends import SubscriptionBackend
        backend=SubscriptionBackend(a.subscription_client,a.profile,a.expected_email,pro_confirmed=a.pro_confirmed)
    result=advance(cases,a.stage,backend);write(a.output,result)
    print(json.dumps({'total':len(result),'completed':sum(x['status']=='completed' for x in result)},ensure_ascii=False))


if __name__=='__main__':main()
