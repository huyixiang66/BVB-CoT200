"""Data-only semantic asset pilot on the same immutable 45 input images."""
import argparse
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import reconstruction_runtime as B
sys.path.insert(0,str(B.ROOT/'scripts/assets'))
from schema import validate

OUT=B.ROOT/'results/bvb-asset-ablation-20261026/3db0a1c8f3-first90s'
OLD=B.ROOT/'results/legacy-fps/0.5fps'
SOURCES=[B.ROOT/'scripts/presets/preset_runtime.py']+list((B.ROOT/'scripts/assets').glob('*.py'))+[B.ROOT/'scripts/assets/interface.txt',B.ROOT/'scripts/compact_pose_input.py']


def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--max-visible-tokens',type=int,default=9999)
    parser.add_argument('--compact',action='store_true')
    parser.add_argument('--question-id',type=int)
    parser.add_argument('--target-object',help='Restrict this independent reconstruction to one parsed category/cluster')
    parser.add_argument('--local-context',action='store_true',help='Only one-hop target context; no generated camera')
    parser.add_argument('--walls-only',action='store_true',help='Local context variant: structural geometry is walls only')
    parser.add_argument('--reasoning',choices=['low','medium','high'],default='high')
    parser.add_argument('--prompt-total-token-budget',type=int,help='Prompt-only budget for reasoning plus final answer; does not change the API cap')
    parser.add_argument('--input-manifest',type=Path,help='Explicit immutable frame list instead of the 45-frame baseline')
    parser.add_argument('--pose-reference',type=Path,help='Known per-frame camera references, used as input only')
    parser.add_argument('--pose-input-format',choices=['compact','legacy'],default='compact',help='VLM transport only; full-precision references stay unchanged')
    parser.add_argument('--out',type=Path,help='Independent experiment directory')
    parser.add_argument('--questions',type=Path,required=True)
    args=parser.parse_args()
    if args.prompt_total_token_budget is not None and args.prompt_total_token_budget<=0:
        raise ValueError('--prompt-total-token-budget must be positive')
    if args.walls_only and not args.local_context:raise ValueError('--walls-only requires --local-context')
    if args.local_context and args.question_id is None:raise ValueError('--local-context requires --question-id')
    if (args.input_manifest or args.pose_reference) and not args.out:raise ValueError('Custom frame/pose input requires --out')
    if args.pose_reference and not args.local_context:raise ValueError('Pose reference pilot requires --local-context')
    question=None
    if args.question_id is not None:
        from object_sampling.questions import extract_question
        from question_scene import prompt_for,expand
        if args.local_context:
            from local_context_scene import prompt_for,expand
            if args.walls_only:
                from functools import partial
                prompt_for=partial(prompt_for,walls_only=True)
                expand=partial(expand,walls_only=True)
        rows=[json.loads(line) for line in args.questions.read_text().splitlines() if line.strip()]
        selected=[q for q in rows if q['id']==args.question_id ]
        if len(selected)!=1:raise ValueError('Question ID must uniquely identify a row in --questions')
        question=extract_question(selected[0]);args.compact=True
        if args.target_object:
            if args.target_object not in question['objects']:raise ValueError('Target must be a parsed question object')
            question['all_question_objects']=list(question['objects'])
            question['objects']=[args.target_object]
    elif args.target_object:raise ValueError('--target-object requires --question-id')
    if args.compact:OUT=B.ROOT/'results/bvb-asset-compact-ablation-20261026/3db0a1c8f3-first90s'
    if question:OUT=B.ROOT/f'results/bvb-question-assets-20261026/3db0a1c8f3-first90s/question-{args.question_id}'
    if args.local_context:OUT=B.ROOT/f'results/bvb-local-context-20261026/3db0a1c8f3-first90s/question-{args.question_id}'
    if args.walls_only:OUT=B.ROOT/f'results/bvb-walls-only-20261026/3db0a1c8f3-first90s/question-{args.question_id}'
    if args.reasoning!='high':OUT=OUT/f'reasoning-{args.reasoning}'
    if args.prompt_total_token_budget is not None:OUT=OUT/f'prompt-total-{args.prompt_total_token_budget}'
    if args.out:OUT=args.out.resolve()
    if args.max_visible_tokens<1000:raise ValueError('Visible budget must be at least 1000')
    folder=OUT/'assets';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():raise RuntimeError('Run already has a result; do not replay paid calls')
    if list((folder/'requests').glob('attempt-*.json')):raise RuntimeError('Partial paid run exists; inspect before continuing')
    cfg=json.loads((args.input_manifest or OLD/'input.json').read_text())
    assert cfg['frame_count']==len(cfg['frames']) and cfg['frame_count']>0
    interval=cfg.get('interval_s',[0,90])
    poses=json.loads(args.pose_reference.read_text()) if args.pose_reference else None
    pose_transport=None
    if poses:
        assert [r['timestamp_s'] for r in poses['prompt_rows']]==[f['timestamp_s'] for f in cfg['frames']]
        B.write(OUT/'pose-reference-input.json',poses)
        if args.pose_input_format=='compact':
            from compact_pose_input import write_transport
            assert [(r['timestamp_s'],r['source_frame_index']) for r in poses['rows']]==[(f['timestamp_s'],f['source_frame_index']) for f in cfg['frames']]
            pose_transport=write_transport(args.pose_reference,OUT/'pose-reference-vlm.json')
    for f in cfg['frames']:assert B.sha(f['path'])==f['sha256']
    B.write(folder/'input.json',cfg)
    prompt=(B.ROOT/'scripts/assets/interface.txt').read_text()
    if args.compact:
        prompt+='\nCOMPACT TRANSPORT (overrides the object dictionary format above): each object MUST be an array [id,kind,p,s,r,color,style,openings]. Only the first four entries are required. Omit trailing default entries; use null for an interior default. Keep ids short, e.g. bath_sink or bottle07, not verbose descriptions. All geometric conventions and asset kinds stay the same. Top-level keys are still objects and camera. Example: {"objects":[["table1","table",[0,0,0.4],[1.2,0.7,0.8],null,"wood"]],"camera":[[0,[0,-3,1.6],[0,0,0.5]],[90,[1,-3,1.6],[0,0,0.5]]]}\n'
    prompt+=f'\nOUTPUT BUDGET: aim for 4000–8000 visible tokens, with a strict acceptance ceiling of {args.max_visible_tokens} for the ENTIRE JSON (objects plus camera). Use compact JSON, short meaningful ids, omit default fields. Do not omit observed objects or replace multiple objects by one to meet the budget. Reasoning tokens are accounted separately.\n'
    if question:
        if args.local_context:
            prompt=prompt_for(question)+f'\nAcceptance ceiling: {args.max_visible_tokens} visible tokens. No minimum length.\n'
        else:prompt+=prompt_for(question)
        B.write(OUT/'question-public.json',question)
        provenance=json.loads((B.ROOT/'scripts/object_sampling/vendor/provenance.json').read_text())
        B.write(OUT/'parser-provenance.json',provenance)
    if args.input_manifest:
        prompt=prompt.replace('Use only the observed first 90 seconds.',
            'Use only the supplied, timestamped frames; they may span the full video.')
    if args.target_object:
        prompt+='\nCLUSTER SCOPE: This is an independent reconstruction of one candidate instance group for '+args.target_object+'. Only this category is a detailed target. Other question objects are not required; include only directly visible one-hop context as white boxes. The cluster and optional detector boxes are hypotheses, not verified identities or labels. Do not invent unseen objects to answer the full question.\n'
    if any(f.get('bbox_overlay') for f in cfg['frames']):
        prompt+='\nIMAGE ANNOTATIONS: Some input images have a colored bounding box and a candidate category label drawn over the original pixels. Inspect these regions carefully for the requested target. These are detector suggestions, not verified object identities. Reject a box if the visible content does not support its label; do not invent the target. Box borders, label text and overlay colors are annotations, not physical scene geometry or object colors. Frames without boxes are context only.\n'
    if poses:
        prompt+='\nCAMERA REFERENCE OVERRIDE: Each input image has its measured/predicted LingBot camera position, right/up/forward unit axes and fx,fy,cx,cy for 640x480 pixels. Reconstruct all objects and walls in THIS shared world coordinate system; do not recenter, rotate or mirror the scene independently. Use viewpoint correspondences to reconcile observations and avoid duplicated or mirrored rooms. World Z is up; lengths are approximate scene units, not calibrated meters. These poses were globally aligned to an older model-authored BVB camera trajectory, with alignment disagreement recorded in pose-reference.json; they are noisy references, not ground truth. No old object geometry is supplied. Infer depth and geometry from the supplied images and these references. Do not output camera poses or re-estimate a trajectory; cameras will be attached locally. Return the same structure/targets/context JSON only.\n'
        if pose_transport:
            prompt+='\n'+pose_transport['explanation']+'\n'
    if args.prompt_total_token_budget is not None:
        prompt+=f'\nKeep your combined reasoning and final answer within {args.prompt_total_token_budget} tokens in total. The final answer must be the complete scene JSON.\n'
    B.write(OUT/'experiment.json',dict(model=B.MODEL,reasoning=args.reasoning,frames=cfg['frame_count'],fps=cfg['fps'],interval_s=interval,
        provider='openai_official',key_source='LLM_API_KEY',image_detail='high',image_size=[640,480],
        max_output_tokens=B.CAP,cost_limit_usd=B.COST_LIMIT,max_turns=20,wall_clock_limit_s=1800,
        max_visible_tokens=args.max_visible_tokens,visible_token_target=None if args.walls_only else ([1000,3000] if args.local_context else [4000,8000]),
        schema_only=True,compact_transport=args.compact,old_scene_geometry_used=False,ground_truth_used=False,
        public_question_used=bool(question),question_id=args.question_id,
        cluster_target=args.target_object,
        local_context_only=args.local_context,camera_trajectory_generated=not args.local_context,
        walls_only=args.walls_only,
        **(dict(prompt_total_token_budget=args.prompt_total_token_budget,prompt_budget_only=True) if args.prompt_total_token_budget is not None else {}),
        **(dict(custom_frame_manifest=str(args.input_manifest.resolve()),custom_frame_manifest_sha256=B.sha(args.input_manifest)) if args.input_manifest else {}),
        **(dict(camera_pose_reference=True,pose_reference_sha256=B.sha(args.pose_reference),pose_reference_is_ground_truth=False,
                pose_input_format=args.pose_input_format,pose_input_decimals=3 if pose_transport else None,
                existing_full_video_lingbot_reused=True,old_bvb_camera_used_for_alignment=True) if poses else {}),
        method_change='JSON-only asset compiler; repair only on validation/execution failure, stop after first successful build. No autonomous bash inspection loop.',
        comparison='First-call output is primary; cumulative usage also changes because stopping rules differ.',
        source_hashes={str(p.relative_to(B.ROOT)):B.sha(p) for p in SOURCES}))
    for p in SOURCES:
        dest=OUT/'library'/p.relative_to(B.ROOT/'scripts');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    (OUT/'system-prompt.txt').write_text(prompt)
    if args.prepare_only:return
    from dotenv import dotenv_values
    from openai import OpenAI
    key=os.environ.get('LLM_API_KEY') or dotenv_values(B.ROOT/'.env').get('LLM_API_KEY')
    if not key:raise RuntimeError('Missing LLM_API_KEY')
    client=OpenAI(api_key=key,base_url='https://api.openai.com/v1',timeout=600,max_retries=0)
    intro='First 90 seconds, interval [0,90), 0.5 FPS, 45 original frames. Reconstruct the observed scene using the asset schema.'
    if args.input_manifest:intro=f'{cfg["frame_count"]} video frames (candidate box overlays where indicated), timestamps span {interval} seconds, selected by {cfg.get("sampling","question-conditioned sampling")}. Reconstruct only observed geometry using the asset schema.'
    content=[{'type':'input_text','text':intro}]
    for index,f in enumerate(cfg['frames']):
        content += [{'type':'input_text','text':f'[frame t={f["timestamp_s"]:.2f}s]'},
          {'type':'input_image','detail':'high','image_url':'data:image/jpeg;base64,'+base64.b64encode(Path(f['path']).read_bytes()).decode()}]
        if pose_transport:content.append({'type':'input_text','text':json.dumps(pose_transport['rows'][index],separators=(',',':'))})
        elif poses:content.append({'type':'input_text','text':'LingBot camera reference for this image: '+json.dumps(poses['prompt_rows'][index],separators=(',',':'))})
        if args.target_object and 'observation_status' in f:
            content.append({'type':'input_text','text':'Detector hint (not ground truth): '+json.dumps({'target':args.target_object,'box_xyxy':f.get('target_box_xyxy'),'status':f['observation_status']},separators=(',',':'))})
    messages=[{'role':'user','content':content}];records=[];cost=0.;done=False;error=None;stop='max_turns';start=time.monotonic()
    print('START assets',cfg['frame_count'],flush=True)
    with B.Sandbox(image='bvb-sandbox:latest',video_path=None,cpus=2,memory='3g',network=False) as sandbox:
        subprocess.run(['docker','cp',str(OUT/'library'),f'{sandbox.name}:/workspace/library'],check=True,capture_output=True)
        try:
            for turn in range(1,21):
                if cost>=B.COST_LIMIT:stop='cost';break
                if time.monotonic()-start>1800:stop='wall_clock';break
                req=folder/'requests'
                B.write(req/f'attempt-{turn:02d}.json',dict(turn=turn,model=B.MODEL,reasoning=args.reasoning,started_unix=time.time(),image_count=cfg['frame_count'],max_output_tokens=B.CAP))
                response=client.responses.create(model=B.MODEL,reasoning={'effort':args.reasoning},instructions=prompt,input=messages,max_output_tokens=B.CAP,store=False)
                B.write(req/f'raw-response-{turn:02d}.json',response.model_dump())
                if not response.usage:raise RuntimeError('Missing usage')
                row=dict(turn=turn,response_id=response.id,response_model=response.model,status=response.status,reply=response.output_text,usage=response.usage.model_dump())
                row.update(B.usage_row(row['usage']));records.append(row);cost+=row['estimated_cost_usd'];B.write(req/f'response-{turn:02d}.json',row)
                (req/f'response-{turn:02d}-verbatim.txt').write_text(row['reply'])
                print('TURN',turn,'input',row['input_tokens'],'output',row['output_tokens'],'reasoning',row['reasoning_tokens'],'cost',cost,flush=True)
                if response.status!='completed':stop='incomplete';break
                messages.append({'role':'assistant','content':row['reply']})
                visible=row['output_tokens']-row['reasoning_tokens']
                if visible>args.max_visible_tokens:
                    B.write(folder/f'budget-{turn:02d}.json',dict(accepted=False,visible_tokens=visible,max_visible_tokens=args.max_visible_tokens))
                    messages.append({'role':'user','content':f'Visible JSON was {visible} tokens, over the {args.max_visible_tokens} ceiling. Return compact complete JSON, shorten ids and omit default fields; preserve all semantic objects. Do not write code or explanations.'})
                    continue
                try:
                    if question:data,audit=expand(json.loads(row['reply']),question)
                    else:data=validate(json.loads(row['reply']))
                    if not args.local_context and 'camera' not in data:
                        raise ValueError('This condition still requires camera samples; only local-context mode omits them')
                except (ValueError,TypeError,KeyError) as exc:
                    feedback=str(exc);B.write(folder/f'validation-{turn:02d}.json',dict(error=feedback))
                    messages.append({'role':'user','content':'Validation failed: '+feedback+'. Return the corrected complete JSON.'});continue
                spec=folder/f'scene-spec-{turn:02d}.json';B.write(spec,data)
                if question:B.write(folder/f'question-audit-{turn:02d}.json',audit)
                (folder/f'scene-transport-{turn:02d}.json').write_text(row['reply'])
                subprocess.run(['docker','cp',str(spec),f'{sandbox.name}:/workspace/scratch/scene.json'],check=True,capture_output=True)
                result=sandbox.exec('blender_run --python /workspace/library/assets/asset_runtime.py -- /workspace/scratch/scene.json /workspace/output/result.blend',timeout=300)
                B.write(folder/f'execution-{turn:02d}.json',result.__dict__)
                print('EXEC',turn,result.returncode,flush=True)
                if result.returncode==0 and sandbox.output_exists():
                    done=True;stop='done';B.write(folder/'scene-spec.json',data)
                    if question:B.write(folder/'question-audit.json',audit)
                    break
                messages.append({'role':'user','content':'Compiler failed:\n'+B._truncate(result.stdout+'\n'+result.stderr,8000)+'\nCorrect complete scene JSON; do not write code.'})
        except Exception as exc:
            error=dict(type=type(exc).__name__,status_code=getattr(exc,'status_code',None));stop='error';B.write(folder/'error.json',error)
        finally:
            if sandbox.output_exists():sandbox.copy_out('/workspace/output/result.blend',folder/'scene.blend')
    totals={k:sum(r[k] for r in records) for k in ['input_tokens','output_tokens','total_tokens','cached_input_tokens','cache_write_tokens','ordinary_input_tokens','reasoning_tokens','estimated_cost_usd']}
    result=dict(condition='assets',done=done,stop_reason=stop,produced_blend=(folder/'scene.blend').exists(),turns=len(records),elapsed_s=time.monotonic()-start,error=error,**totals)
    B.write(folder/'result.json',result);print('RESULT',json.dumps(result),flush=True)
    if not done:raise RuntimeError('Asset pilot incomplete; inspect trace')


if __name__=='__main__':main()
