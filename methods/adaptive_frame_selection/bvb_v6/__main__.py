"""Explicit stages; only the submit command makes model requests."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

from . import __version__
from .reconstruction import read, prepare, submit

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description='BVB V6 regional selection and minimal reconstruction')
    parser.add_argument('--version', action='version', version=__version__)
    sub = parser.add_subparsers(dest='command', required=True)
    select = sub.add_parser('select', help='Group frozen peaks; select pose-diverse frames, no model calls')
    for key in ['v3', 'cache', 'poses', 'appearance', 'out']:
        select.add_argument('--'+key, type=Path, required=True)
    select.add_argument('--qid', nargs='+')
    select.add_argument('--preset', type=Path, default=ROOT/'preset/geometry-appearance-075.json')
    select.add_argument('--frame-cap', type=int, default=8)
    select.add_argument('--image-size', nargs=2, type=int, default=[640,480])
    select.add_argument('--reference-cameras', type=Path)
    select.add_argument('--allow-missing-appearance', action='store_true')
    prep = sub.add_parser('prepare', help='Freeze reconstruction inputs and prompts, no model calls')
    for key in ['selection', 'cache', 'poses', 'out']:
        prep.add_argument('--'+key, type=Path, required=True)
    prep.add_argument('--qid', required=True)
    prep.add_argument('--arm', help='Defaults to the reviewed selection_arm (semantic_selected with coverage), otherwise v6')
    prep.add_argument('--reference-cameras', type=Path)
    prep.add_argument('--frame-cap', type=int, default=8)
    prep.add_argument('--object-size-reference-mode', action='store_true',
                      help='Require one Astra-selected object-size cluster and a visible scale reference with meter ruler')
    prep.add_argument('--allow-incomplete-evidence', action='store_true',
                      help='Prepare a static-context draft for supplemental-evidence questions')
    call = sub.add_parser('submit', help='Explicit paid model requests; serial and fail-fast')
    call.add_argument('--out', type=Path, required=True)
    call.add_argument('--provider', choices=['openai','boyue','subscription','subscription_direct','subscription_direct_4'], default=None)
    call.add_argument('--key-env', help='Defaults to LLM_API_KEY or BOYUE_API_KEY')
    call.add_argument('--env-file', type=Path, default=ROOT.parent/'.env')
    call.add_argument('--model', default='gpt-6-astra')
    call.add_argument('--reasoning', choices=['low','medium','high'], default='high')
    call.add_argument('--max-tokens', type=int, default=None,
                      help='Optional API completion cap; omitted by default, prompt budget unchanged')
    call.add_argument('--limit', type=int, default=1)
    call.add_argument('--group', help='One group ID from the prepared manifest')
    build = sub.add_parser('build', help='Validate/compile cached scene JSON and merge; no model calls')
    build.add_argument('--source', type=Path, required=True)
    build.add_argument('--out', type=Path, required=True)
    build.add_argument('--blender', default=shutil.which('blender') or '/Applications/Blender.app/Contents/MacOS/Blender')
    args = parser.parse_args()
    if args.command == 'select':
        from .selection import run
        run(args)
    elif args.command == 'prepare':
        data = read(args.selection);row = data['questions'][args.qid]
        policy=data.get('semantic_review',{});safe_arm=policy.get('selection_arm','semantic_accepted')
        arm=args.arm or (safe_arm if policy.get('required') else 'v6')
        if policy.get('required') and arm!=safe_arm:
            raise ValueError('Reviewed selections must use '+safe_arm+'; original hypotheses are audit-only')
        if policy.get('coverage_required') and not row.get('coverage_complete'):
            raise ValueError('Category coverage is incomplete; finish comparisons or obtain missing candidates first')
        requirements = row.get('retention_policy')
        if requirements and requirements.get('category') == 'supplemental' and not args.allow_incomplete_evidence:
            raise ValueError('This question requires temporal or other supplemental evidence; '
                             'use --allow-incomplete-evidence only for a static-context draft')
        groups = row['arms'][arm]
        if not groups:
            raise ValueError('No observed target groups; provide context explicitly for global questions')
        if args.frame_cap < 4:
            raise ValueError('frame-cap must be at least four')
        question=dict(row['question'])
        if args.object_size_reference_mode:
            question['object_size_reference_mode']=True
            question['instance_limits']={obj:1 for obj in question['objects']}
        prepare(args.out.resolve(), question, groups, None, args.frame_cap,
                args.reference_cameras, cache=args.cache.resolve(), poses=args.poses.resolve())
        if requirements:
            from .reconstruction import write
            manifest_path=args.out.resolve()/'manifest.json';manifest=read(manifest_path)
            manifest['retention_policy']=requirements
            manifest['supplemental_evidence_verified']=False
            write(manifest_path,manifest)
    elif args.command == 'submit':
        from .api_provider import resolve
        connection=resolve(args.provider,args.env_file,args.key_env)
        connection.token_kwargs(args.max_tokens)
        if args.limit < 1 or (args.max_tokens is not None and args.max_tokens < 1):
            raise ValueError('limit and max-tokens must be positive')
        manifest = read(args.out/'manifest.json')
        if args.group and args.group not in {g['group_id'] for g in manifest['groups']}:
            raise ValueError('Unknown group')
        folders = [args.out/g['folder'] for g in manifest['groups'] if not args.group or g['group_id']==args.group]
        folders = [f for f in folders if not (f/'assets/request.json').exists() and not (f/'assets/scene-spec.json').exists()]
        for folder in folders[:args.limit]:
            record = submit(folder, connection.key, args.max_tokens, '', connection.provider, args.reasoning, base_url=connection.base_url, model=args.model, provider_config=connection)
            print(record, flush=True)
            if record['status'] != 'schema_valid':
                raise SystemExit('Stopped after failed request/validation; inspect ledger before retrying')
    else:
        from .build import run
        run(args)


if __name__ == '__main__':
    main()
