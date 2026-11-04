"""Reconstruct each selected V5 group independently using the minimal asset pipeline."""
import argparse
import html
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
BLENDER=shutil.which('blender') or '/Applications/Blender.app/Contents/MacOS/Blender'
POSE=ROOT/'results/lingbot-trajectory-3db0a1c8f3-20261024'
CACHE=ROOT/'results/dino-sam2-4fps-20261101/cache'

def load_font(size):
    for name in ['DejaVuSans.ttf','/System/Library/Fonts/Supplemental/Arial.ttf']:
        try:return ImageFont.truetype(name,size)
        except OSError:pass
    return ImageFont.load_default()

def read(p):return json.loads(Path(p).read_text())
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def sha(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def prepare(a):
    a.out.mkdir(parents=True,exist_ok=False)
    q=read(a.question_folder/'question.json');groups=read(a.question_folder/'instance-groups.json')
    assert groups, 'No groups supplied'
    save(a.out/'question-public.json',q)
    (a.out/'questions.jsonl').write_text(json.dumps(q,ensure_ascii=False)+'\n')
    sampling=read(POSE/'sampling.json');alignment=read(POSE/'alignment.json')
    raw=np.load(POSE/'camera-poses.npz');aligned=np.load(POSE/'aligned-poses.npz')['c2w_blender']
    cand=read(CACHE/'candidates.json');lookup={r['source_frame_index']:i for i,r in enumerate(cand)}
    assert [r['source_frame_index'] for r in cand]==[r['source_frame_index'] for r in sampling['frames']]
    w,h=read(POSE/'geometry-info.json')['image_size_wh']
    rot=np.asarray(alignment['rotation']);scale=alignment['scale_lingbot_to_blender'];translation=np.asarray(alignment['translation'])
    colors={obj:[[.85,.08,.055,1],[.05,.48,.15,1],[.06,.30,.90,1],[.95,.43,.04,1]][i%4] for i,obj in enumerate(q['objects'])}
    catalog=[]
    for g in groups:
        folder=a.out/g['group_id'];folder.mkdir();frames=[];rows=[];prompt_rows=[]
        for r in sorted(g['selected'],key=lambda r:r['source_frame_index']):
            fid=r['source_frame_index'];i=lookup[fid];c=cand[i];sample=sampling['frames'][i];path=(CACHE/c['frame']).resolve()
            assert sha(path)==r['frame_sha256']==sample['sha256']
            with Image.open(path) as im:assert im.size==(640,480), 'Reconstruction protocol requires 640x480 input frames'
            assert c['timestamp_s']==sample['timestamp_s']==r['timestamp_s']
            matrix=aligned[i].astype(float)
            assert np.allclose(matrix[:3,:3],rot@raw['c2w'][i,:3,:3]@np.diag([1,-1,-1]),atol=1e-6)
            assert np.allclose(matrix[:3,3],scale*rot@raw['c2w'][i,:3,3]+translation,atol=1e-6)
            k=raw['intrinsics'][i].astype(float).copy();k[0,:]*=640/w;k[1,:]*=480/h
            frames.append(dict(timestamp_s=r['timestamp_s'],source_frame_index=fid,path=str(path),sha256=sha(path),
                               target_box_xyxy=[round(v,1) for v in r['box_xyxy']] if r['box_xyxy'] else None,observation_status=r['status']))
            if a.annotate_boxes and r['box_xyxy']:
                image=Image.open(path).convert('RGB');pen=ImageDraw.Draw(image)
                color=tuple(round(v*255) for v in colors[g['object']][:3])
                x1,y1,x2,y2=r['box_xyxy'];assert 0<=x1<x2<=image.width and 0<=y1<y2<=image.height
                pen.rectangle((x1,y1,x2,y2),outline=color,width=3)
                label=g['object']+'? (detector)';font=load_font(15)
                label_w=pen.textbbox((0,0),label,font=font)[2]+8
                x=min(max(0,x1),max(0,image.width-label_w));y=max(0,y1-23)
                pen.rectangle((x,y,x+label_w,y+22),fill=color);pen.text((x+4,y+2),label,fill='white',font=font)
                dest=folder/'annotated-frames'/f'{fid:07d}.jpg';dest.parent.mkdir(exist_ok=True);image.save(dest,quality=95)
                frames[-1].update(raw_path=str(path),raw_sha256=sha(path),path=str(dest.resolve()),sha256=sha(dest),
                                  bbox_overlay=True,bbox_color_rgb=list(color))
            rows.append(dict(timestamp_s=r['timestamp_s'],source_frame_index=fid,pose_sample_index=i,c2w_blender=matrix.tolist(),intrinsics_640x480=k.tolist()))
            prompt_rows.append(dict(timestamp_s=r['timestamp_s'],position=matrix[:3,3].tolist(),right=matrix[:3,0].tolist(),up=matrix[:3,1].tolist(),forward=(-matrix[:3,2]).tolist(),intrinsics_fx_fy_cx_cy=[k[0,0],k[1,1],k[0,2],k[1,2]]))
        save(folder/'input-frames.json',dict(fps=None,frame_count=len(frames),sampling='V5 per-instance DINOv2 + LingBot clustering and pose-diverse selection',
             interval_s=[frames[0]['timestamp_s'],frames[-1]['timestamp_s']],video_sha256=sampling['video_sha256'],frames=frames))
        save(folder/'pose-reference.json',dict(coordinate_system='Blender XYZ, Z up; camera right +X, up +Y, forward -Z',
             units='Approximate prior global Sim3 units, not calibrated meters',ground_truth_pose=False,
             old_bvb_object_geometry_supplied=False,old_bvb_camera_used_for_global_alignment=True,
             alignment_center_disagreement_rmse=alignment['center_rmse_blender_units'],rows=rows,prompt_rows=prompt_rows,
             source_hashes={n:sha(POSE/n) for n in ['camera-poses.npz','aligned-poses.npz','alignment.json','sampling.json']}))
        save(folder/'cluster.json',g)
        contact=Image.new('RGB',(1280,math.ceil(len(frames)/4)*280),'white');draw=ImageDraw.Draw(contact)
        for i,f in enumerate(frames):
            x=i%4*320;y=i//4*280;draw.text((x+5,y+5),f'{f["timestamp_s"]:.2f}s | {f["observation_status"]}',fill='black')
            contact.paste(Image.open(f['path']).convert('RGB').resize((320,240)),(x,y+30))
        contact.save(folder/'selected-frames.jpg',quality=93)
        catalog.append({'group_id':g['group_id'],'object':g['object'],'frames':len(frames),'folder':g['group_id']})
    save(a.out/'manifest.json',dict(question_id=q['id'],scene=q['scene_name'],groups=catalog,colors=colors,
         source_groups_sha256=sha(a.question_folder/'instance-groups.json'),selection_changed=False,
         annotated_detector_boxes=a.annotate_boxes,
         method='Minimal local-context assets, walls only, compressed LingBot poses, high effort, prompt-only total budget 2000'))

def logged(out,name,cmd):
    print(name,flush=True)
    with (out/(name+'.log')).open('w') as log:
        p=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if p.returncode:raise RuntimeError(f'{name} failed: {out/(name+".log")}')

def finish(out):
    m=read(out/'manifest.json');q=read(out/'question-public.json');results=[];sections=[]
    combined=Image.new('RGB',(900*len(m['groups']),960),'white');draw=ImageDraw.Draw(combined)
    font=load_font(24)
    for i,g in enumerate(m['groups']):
        folder=out/g['folder'];assets=folder/'assets';r=read(assets/'result.json');assert r['done']
        replies=[read(p) for p in sorted((assets/'requests').glob('response-*.json'))]
        for key in ['input_tokens','output_tokens','reasoning_tokens']:assert r[key]==sum(x[key] for x in replies)
        geometry=read(assets/'geometry-validation.json');assert geometry['status']=='passed' and geometry['walls_only_verified']
        inspection=read(assets/'scene-inspection.json');assert sha(assets/'scene.blend')==inspection['source_sha256']
        assert inspection['preview_views']['oblique']['rotation_degrees']==[20,-20,0]
        audit=read(assets/'question-audit.json')
        row={**g,**r,'visible_output_tokens':r['output_tokens']-r['reasoning_tokens'],'targets':len(audit['targets']),
             'context_boxes':len(audit['background']),'walls':len(audit['structure']),'missing_keywords':audit['missing_keywords'],
             'geometry_validation':geometry,'oblique_rotation_degrees':[20,-20,0]}
        row['target_reconstruction_status']='missing' if audit['missing_keywords'] else 'present_not_accuracy_verified'
        results.append(row)
        preview=Image.open(assets/'targets-highlight.png').convert('RGB');combined.paste(preview.resize((900,900)),(900*i,60))
        draw.text((900*i+15,10),f'{g["object"]} | {g["frames"]} frames | {r["output_tokens"]} output tokens'+(' | NO TARGET' if audit['missing_keywords'] else ''),font=font,fill='#a54300' if audit['missing_keywords'] else 'black')
        warning='<p style="color:#a54300;font-weight:bold">目标缺失：'+html.escape(', '.join(audit['missing_keywords']))+'。本组仅通过结构编译，没有重构出目标；请对照下方输入帧检查检测与选帧。</p>' if audit['missing_keywords'] else ''
        rel=g['folder'];sections.append(f'<section><h2>{html.escape(g["group_id"])} · {g["frames"]}帧</h2>{warning}<p>目标 {row["targets"]}，关联白块 {row["context_boxes"]}，墙体 {row["walls"]}。输入 {r["input_tokens"]} token；输出 {r["output_tokens"]}＝推理 {r["reasoning_tokens"]}＋可见 JSON {row["visible_output_tokens"]}。API轮数 {r["turns"]}。</p><p><a href="{rel}/assets/scene.blend">Blender 场景</a> · <a href="{rel}/assets/targets-highlight.png">斜俯视原图</a> · <a href="{rel}/assets/scene-spec.json">场景 JSON</a> · <a href="{rel}/input-frames.json">输入帧</a></p><img src="{rel}/assets/targets-highlight.png"><details open><summary>本组原视频输入</summary><img src="{rel}/selected-frames.jpg"></details></section>')
    combined.save(out/'comparison.jpg',quality=94)
    totals={k:sum(r[k] for r in results) for k in ['input_tokens','output_tokens','reasoning_tokens','visible_output_tokens','estimated_cost_usd']}
    save(out/'summary.json',dict(question=q,groups=results,totals=totals,qa_run=False,
         cost_note='Historical experimental rate estimate, not a verified bill',geometry_accuracy_verified=False))
    page='<!doctype html><html lang="zh"><meta charset="utf-8"><title>Cluster · 聚类最简重构</title><style>body{font:16px system-ui;max-width:1500px;margin:30px auto;padding:20px;line-height:1.7}img{max-width:100%}section{border:1px solid #ddd;padding:20px;margin:24px 0}section>img{width:850px}a{color:#185e9c}</style>'
    page+='<h1>Cluster：每个聚类独立重构</h1><p>'+html.escape(q['question'])+'</p><p>Astra high，官方 API，紧凑 JSON＋本地基础资产。只保留当前聚类目标、直接关联白块及墙体，不生成地板、屋顶或相机轨迹。各组分别重构，双轴20°斜俯视、自动取景；未拼接成统一场景，也未进行QA。</p><p>压缩LingBot位姿沿用已有近似全局配准，并非真实位姿。基础资产形状是近似（当前kettle映射为pot）。检测与聚类可能包含错误，构建成功不代表几何正确。</p><img src="comparison.jpg">'+''.join(sections)+'</html>'
    (out/'index.html').write_text(page,encoding='utf-8')
    save(out/'validation.json',dict(groups=len(results),all_compiled=True,geometry_schema_passed=True,source_blends_unchanged=True,usage_sums_verified=True,views=[20,-20,0],qa_run=False,
         groups_missing_target=[r['group_id'] for r in results if r['missing_keywords']]))
    print(totals,flush=True)

def main():
    global POSE,CACHE,BLENDER
    p=argparse.ArgumentParser();p.add_argument('--question-folder',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--poses',type=Path,required=True,help='Full precision aligned LingBot pose directory')
    p.add_argument('--cache',type=Path,required=True,help='Immutable detection frame cache')
    p.add_argument('--blender',default=BLENDER)
    p.add_argument('--annotate-boxes',action='store_true',help='Draw existing candidate detection boxes on model input images; retain raw frame hashes')
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--finish-only',action='store_true');a=p.parse_args();a.out=a.out.resolve();POSE=a.poses.resolve();CACHE=a.cache.resolve();BLENDER=a.blender
    if not a.out.exists():prepare(a)
    m=read(a.out/'manifest.json')
    assert bool(m.get('annotated_detector_boxes'))==a.annotate_boxes,'Input annotation mode differs from existing prepared run'
    if a.prepare_only:return
    for g in m['groups']:
        folder=a.out/g['folder'];assets=folder/'assets'
        if not a.finish_only and not (assets/'result.json').exists():
            logged(folder,'reconstruction',[sys.executable,str(ROOT/'scripts/reconstruct_asset_ablation.py'),
                '--questions',str(a.out/'questions.jsonl'),'--question-id',str(m['question_id']),'--target-object',g['object'],
                '--local-context','--walls-only','--reasoning','high','--prompt-total-token-budget','2000',
                '--input-manifest',str(folder/'input-frames.json'),'--pose-reference',str(folder/'pose-reference.json'),
                '--pose-input-format','compact','--out',str(folder)])
        assert read(assets/'result.json')['done'], 'Incomplete paid run; inspect before retrying'
        save(assets/'target-colors-input.json',m['colors'])
        logged(folder,'preview',[BLENDER,'--background','--disable-autoexec',str(assets/'scene.blend'),
            '--python-exit-code','1','--python',str(ROOT/'scripts/verify_question_scene.py'),
            '--python',str(ROOT/'scripts/render_fps_ablation.py'),'--python',str(ROOT/'scripts/render_question_highlight.py'),'--',str(assets)])
    finish(a.out)

if __name__=='__main__':main()
