"""Self-contained ten-question instance-aware sampling report."""
import argparse
import html
import itertools
import math
import re
import shutil
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote

import numpy as np
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bvb_v5.dinov2_appearance import read,save,sha


def esc(x):return html.escape(str(x))


def representative_caption(proof):
    if proof.get('priority_only'):
        return f' · 代表 {proof["representative"]} · 位姿证据支持（排序优先级不是外观相似度）'
    return f' · 代表 {proof["representative"]} · 代表最弱支持 {proof["minimum_support"]:.3f} · 组均分 {proof["mean_pair_score"]:.3f}'


def build(a):
    out=a.out
    if out.exists() and a.refresh:
        assert read(out/'manifest.json')['version']=='v5-pose-clustering'
    out.mkdir(parents=True,exist_ok=a.refresh)
    summary=read(a.result/'summary.json');groups=read(a.result/'groups.json');gm={g['group_id']:g for g in groups}
    config=read(a.result/'config.json');qs=read(a.result/'questions.json');peaks=read(a.v3/'peaks.json')
    review_warnings=read(a.result/'visual-review.json').get('group_warnings',{}) if (a.result/'visual-review.json').exists() else {}
    aggressive=config.get('mode')=='aggressive'
    baseline=read(a.conservative_result/'summary.json') if a.conservative_result else None
    baseline_q={r['question']['id']:r['unique_frames'] for r in baseline['questions']} if baseline else {}
    prior_aggressive=read(a.previous_aggressive_result/'summary.json') if a.previous_aggressive_result else None
    prior_threshold=read(a.previous_aggressive_result/'config.json')['threshold'] if prior_aggressive else None
    prior_q={r['question']['id']:r['unique_frames'] for r in prior_aggressive['questions']} if prior_aggressive else {}
    pp={p['peak_id']:p for p in peaks};pairs=read(a.result/'peak-association.json');pairmap={tuple(sorted((p['peak_a'],p['peak_b']))):p for p in pairs}
    old=read(a.previous/'manifest.json');oldq={q['id']:q for q in old['questions']}
    colors=read(a.previous/'object-colors.json')['palette'];cand=read(a.cache/'candidates.json');byframe={r['source_frame_index']:r for r in cand}
    for folder in ['01-original-video','02-questions','03-per-question','04-method-and-validation','images','peak-montages']:(out/folder).mkdir(exist_ok=True)
    vp=out/'01-original-video'/Path(old['video']['path']).name;shutil.copy2(a.previous/old['video']['path'],vp);assert sha(vp)==old['video']['sha256']
    shutil.copytree(a.previous/'02-questions',out/'02-questions',dirs_exist_ok=True)
    save(out/'object-colors.json',{'palette':colors})
    for f in a.result.glob('*.json'):shutil.copy2(f,out/'04-method-and-validation'/f.name)
    for name in ['instance_sampling.py','instance_report.py','test_instance_sampling.py','select_frames.py','dinov2_appearance.py','pose_association.py','UPSTREAM.json']:
        shutil.copy2(Path(__file__).with_name(name),out/'04-method-and-validation'/name)
    # Raw candidate frames are stored once and shared across all ten questions.
    allids=sorted({r['source_frame_index'] for g in groups for r in g['candidates']})
    for fid in allids:
        c=byframe[fid];target=out/'images'/f'frame{fid:07d}.jpg';shutil.copy2(a.cache/c['frame'],target);assert sha(target)==c['frame_sha256']
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',14)
    small=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',12)
    peakgroup={p:g for g in groups for p in g['peak_ids']}
    for p in peaks:
        g=peakgroup[p['peak_id']];selected={r['source_frame_index'] for r in g['selected']};color=colors[p['object']]
        canvas=Image.new('RGB',(960,220),'#f5f7fa');draw=ImageDraw.Draw(canvas)
        for j,r in enumerate(p['selected']):
            im=Image.open(out/'images'/f'frame{r["source_frame_index"]:07d}.jpg').convert('RGB');d=ImageDraw.Draw(im)
            if r['box_xyxy']:d.rectangle(r['box_xyxy'],outline=color,width=4)
            x=j*240;canvas.paste(im.resize((240,180)),(x,38));kept=r['source_frame_index'] in selected
            if kept:draw.rectangle((x+2,39,x+237,216),outline=color,width=4)
            draw.text((x+5,3),f'{r["timestamp_s"]:.2f}s | '+('KEEP' if kept else 'candidate'),font=font,fill='black')
            draw.text((x+5,21),r['status'],font=small,fill='#9a5a00' if r['status']=='context_fallback' else '#555555')
        canvas.save(out/'peak-montages'/(p['peak_id']+'.jpg'),quality=91)
    comparison=[];sections=[];question_catalog=[];total_candidates=total_kept=0
    C=np.load(a.lingbot/'camera-poses.npz')['c2w'][:,:3,3]
    for n,record in enumerate(summary['questions'],1):
        q=record['question'];qid=q['id'];gs=[gm[k] for k in record['group_ids']]
        rel=Path('03-per-question')/f'{n:02d}-q{qid}';folder=out/rel;folder.mkdir(exist_ok=True);(folder/'candidates-annotated').mkdir(exist_ok=True)
        previous=a.previous/oldq[qid]['folder'];pool=read(previous/'candidate-frames.json')['frames']
        save(folder/'question.json',q);shutil.copy2(previous/'question.txt',folder/'question.txt')
        for f in ['dino-bars.png','dino-scores.json']:shutil.copy2(previous/f,folder/f)
        selected_by_frame={};candidates_by_frame={}
        for g in gs:
            for r in g['selected']:selected_by_frame.setdefault(r['source_frame_index'],[]).append((g,r))
            for r in g['candidates']:candidates_by_frame.setdefault(r['source_frame_index'],[]).append((g,r))
        assert {r['source_frame_index'] for r in pool}==set(candidates_by_frame)
        positions=[];cards=[];gallery=[]
        for number,prior in enumerate(pool,1):
            fid=prior['source_frame_index'];selected=selected_by_frame.get(fid,[]);chosen=bool(selected)
            objs=[o for o in q['objects'] if any(g['object']==o for g,r in selected)]
            raw=out/'images'/f'frame{fid:07d}.jpg';im=Image.open(raw).convert('RGB');draw=ImageDraw.Draw(im)
            # Same palette as the previous delivery; exact cached boxes retained.
            boxes=prior['boxes']
            for b in boxes:
                color=colors[b['object']];x0,y0,x1,y1=b['box_xyxy'];draw.rectangle((x0,y0,x1,y1),outline=color,width=3)
                label=b['object']+(f' {b["score"]:.2f}' if b['score'] is not None else '')
                w=draw.textbbox((0,0),label,font=font)[2]+8;x=max(0,min(x0,640-w));y=max(0,y0-21)
                draw.rectangle((x,y,x+w,y+21),fill=color);draw.text((x+4,y+2),label,font=font,fill='black')
            context=[g['object'] for g,r in selected if r['status']=='context_fallback']
            if context:draw.rectangle((0,459,640,480),fill='#fff0d4');draw.text((5,462),'Unverified context: '+', '.join(sorted(set(context))),font=small,fill='#784800')
            canvas=Image.new('RGB',(640,510),'#edf0f3');canvas.paste(im,(0,30));draw=ImageDraw.Draw(canvas)
            if chosen:
                for j,o in enumerate(objs):
                    x=round(j*640/len(objs));xx=round((j+1)*640/len(objs));draw.rectangle((x,0,xx,29),fill=colors[o]);draw.text((x+5,7),'KEEP: '+o,font=small,fill='black')
                    pad=j*4;draw.rectangle((pad,30+pad,639-pad,509-pad),outline=colors[o],width=4)
            else:draw.text((6,7),'CANDIDATE - NOT RETAINED',font=font,fill='#555555')
            filename=f'{number:03d}__frame{fid:07d}.jpg';canvas.save(folder/'candidates-annotated'/filename,quality=92)
            display=str(rel/'candidates-annotated'/filename);source=str(Path('../../images')/raw.name)
            uses=[{'group_id':g['group_id'],'object':g['object'],'identity_status':g['identity_status'],
                   'status':r['status'],'reason':r['selection_reason'],'source_peaks':r['source_peaks']} for g,r in selected]
            entry={'number':number,'source_frame_index':fid,'timestamp_s':prior['timestamp_s'],
                   'image':source,'sha256':sha(raw),'display_image':'candidates-annotated/'+filename,
                   'final_selected':chosen,'selected_objects':objs,'uses':uses,'boxes':boxes}
            gallery.append(entry)
            if chosen:positions.append(entry)
            labels=' / '.join(g['group_id'] for g,r in selected) if chosen else '未保留'
            cards.append(f'<figure class="candidate" data-final="{str(chosen).lower()}"><a href="{display}" target="_blank"><img loading="lazy" width="640" height="510" src="{display}" alt="Q{qid} candidate {number}"></a><figcaption>#{number} · <button class="seek" data-time="{prior["timestamp_s"]}">{prior["timestamp_s"]:.2f}s</button><br>{esc(labels)}</figcaption></figure>')
        assert len(positions)==record['unique_frames'];total_candidates+=len(pool);total_kept+=len(positions)
        save(folder/'candidate-frames.json',{'question_id':qid,'frames':gallery})
        save(folder/'sampling-positions.json',{'question_id':qid,'frames':positions,'physical_instance_count':None})
        save(folder/'input-frames.json',{'question':q,'version':summary['version'],'chronological':True,'deduplicated':True,
             'frames':[{'source_frame_index':r['source_frame_index'],'timestamp_s':r['timestamp_s'],'image':r['image'],'sha256':r['sha256'],'selection_uses':r['uses']} for r in positions]})
        save(folder/'instance-groups.json',gs)
        fig,ax=plt.subplots(figsize=(15,2+.65*len(q['objects'])))
        oldpos=read(previous/'sampling-positions.json')['frames']
        for k,o in enumerate(q['objects']):
            ax.vlines([r['timestamp_s'] for r in pool if any(b['object']==o for b in r['boxes'])],k-.3,k-.15,color='#bcc2cb',lw=1)
            ax.vlines([r['timestamp_s'] for r in oldpos if o in r['objects']],k-.08,k+.07,color='#333333',lw=1.5)
            for g in gs:
                if g['object']==o:
                    for r in g['selected']:ax.vlines(r['timestamp_s'],k+.15,k+.3,color='#d28a13' if r['status']=='context_fallback' else colors[o],lw=2)
        ax.set_yticks(range(len(q['objects'])),q['objects']);ax.invert_yaxis();ax.set_xlim(-1,238);ax.grid(axis='x',alpha=.2)
        ax.set_xlabel('Original video seconds');ax.set_title(f'Q{qid}: gray=candidates / black=previous V4 / color=new instance selection / orange=context');fig.tight_layout();fig.savefig(folder/'sampling-positions.png',dpi=130);plt.close(fig)
        fig=plt.figure(figsize=(10,7));ax=fig.add_subplot(111,projection='3d');ax.plot(*C.T,color='#bbbbbb',alpha=.45,lw=.7)
        for o in q['objects']:
            rs=[r for g in gs if g['object']==o for r in g['selected']];loc=np.array([r['camera_center'] for r in rs]);dirs=np.array([r['view_direction'] for r in rs])
            ax.scatter(*loc.T,c=colors[o],label=o,s=18);ax.quiver(*loc.T,*dirs.T,length=.1,normalize=True,color=colors[o],alpha=.45)
        ax.set_title(f'Q{qid}: selected LingBot cameras / ROI view directions');ax.set_xlabel('X');ax.set_ylabel('Y');ax.set_zlabel('Z');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(folder/'lingbot-views.png',dpi=130);plt.close(fig)
        groupcards=[]
        for g in gs:
            multi=len(g['peak_ids'])>1
            scores=[pairmap[tuple(sorted((x,y)))]['multi_view_score'] for x,y in itertools.combinations(g['peak_ids'],2)]
            observed_scores=[s for s in scores if s is not None]
            score=min(observed_scores) if observed_scores else None
            tag='多波峰支持的实例假设' if multi else '单波峰：身份未确定'
            details=[]
            for pid in g['peak_ids']:
                details.append(f'<div><strong>{pid}</strong> · 锚点 {cand[pp[pid]["anchor"]["frame"]]["timestamp_s"]:.2f}s<img class="montage" loading="lazy" src="peak-montages/{pid}.jpg" alt="{pid} four observations"></div>')
            matching=''
            if multi:
                for x,y in itertools.combinations(g['peak_ids'],2):
                    evidence=pairmap[tuple(sorted((x,y)))];ms=evidence['matching']['matches'] if evidence['matching'] else []
                    note=(' · 稀疏几何冲突（本版仅警告）' if aggressive and evidence['geometry_conflict_heuristic'] else '')
                    if evidence.get('pose_depth'):note+=' · 三维点重投影支持'
                    if evidence.get('pose_keypoints'):note+=' · 局部特征三视角支持'
                    note+=(' · 外观配对未通过' if not evidence['merge_supported'] else '')
                    matching+=f'<li>{x} ↔ {y}：'+(', '.join(f'{m["left_frame"]/30:.2f}s ↔ {m["right_frame"]/30:.2f}s = {m["cosine"]:.3f}' for m in ms) or '没有有效双图匹配')+note+'</li>'
            status=esc(tag)+(f' · 组内最弱双图匹配 {score:.3f}' if score is not None else (' · 无有效双图分数，以几何证据关联' if multi else ''))
            if aggressive and multi:
                status+=representative_caption(g['representative_evidence'])
            if g['group_id'] in review_warnings:status+='<br><strong style="color:#a14c00">人工抽查警告：'+esc(review_warnings[g['group_id']])+'</strong>'
            outside=[p for p in pairs if (p['peak_a'] in g['peak_ids']) != (p['peak_b'] in g['peak_ids'])]
            closest=sorted(outside,key=lambda p:p['multi_view_score'] if p['multi_view_score'] is not None else -2,reverse=True)[:3]
            reasons={'same_frame_distinct_do_not_merge':'同帧两个独立框，禁止合并',
                     'sparse_geometry_conflict_defer':'稀疏几何冲突，暂不合并',
                     'insufficient_independent_frame_pairs':'不足两对独立且含锚点的匹配',
                     'insufficient_multi_view_similarity':'双图匹配未达阈值',
                     'two_independent_anchor_supported_matches':'此对通过，但整组其他波峰对未全部通过',
                     'two_view_supported_matches':'此对通过，但整组代表、均分或同帧独立约束未通过'}
            if aggressive:reasons['insufficient_independent_frame_pairs']='不足两对满足间隔要求的一对一匹配'
            review=''
            for p in closest:
                v=f'{p["multi_view_score"]:.3f}' if p['multi_view_score'] is not None else '无有效双图匹配'
                review+=f'<li>{p["peak_a"]} ↔ {p["peak_b"]}：{v}；{reasons[p["reason"]]}</li>'
            groupcards.append(f'<article class="instance" style="border-left:5px solid {colors[g["object"]]}"><h4>{g["group_id"]}</h4><p>{status} · {len(g["peak_ids"])} 个波峰 → {g["selected_count"]} 帧，其中上下文 {g["context_count"]} 帧</p>'+('<details><summary>查看归并的配对证据</summary><ul>'+matching+'</ul></details>' if matching else '')+('<details><summary>为什么未与其他波峰合并：最近候选</summary><ul>'+review+'</ul></details>' if review else '')+''.join(details)+'</article>')
        opts='<ul>'+''.join('<li>'+esc(o)+'</li>' for o in (q.get('options') or []))+'</ul>' if q.get('options') else ''
        row={'id':qid,'v3':record['v3_unique_frames'],'previous_v4':record['previous_v4_frames'],'new':record['unique_frames'],'instance_hypotheses':len(gs),'context_slots':record['context_slots']};comparison.append(row)
        if qid in baseline_q:row['conservative']=baseline_q[qid]
        if qid in prior_q:row['previous_aggressive']=prior_q[qid]
        question_catalog.append({'id':qid,'folder':str(rel),'unique_frames':len(positions),'candidate_frames':len(pool),'group_ids':record['group_ids']})
        sections.append(f'''<section id="q{qid}"><h2>Q{qid} · {esc(q['question_type'])}</h2><p class="question">{esc(q['question'])}</p>{opts}
<p>原始逐峰 {record['v3_unique_frames']} 帧 → 上版 V4 {record['previous_v4_frames']} 帧 → 本版 {record['unique_frames']} 帧。共 {len(gs)} 个实例候选组（不是真实物体数量）。</p>
<p><a href="{rel}/input-frames.json">重构输入清单</a> · <a href="{rel}/instance-groups.json">实例候选完整记录</a> · <a href="{rel}/candidate-frames.json">全部候选框及选择记录</a></p>
<h3>波峰归并与每组四张原始观察</h3>{''.join(groupcards)}
<h3>GroundingDINO 分数与抽帧位置</h3><img class="chart" loading="lazy" src="{rel}/dino-bars.png" alt="DINO bars"><img class="chart" loading="lazy" src="{rel}/sampling-positions.png" alt="sampling positions">
<h3>LingBot 机位与方向</h3><p>类别颜色与检测框一致。坐标为任意尺度；框内中心仍为稀疏几何代理。</p><img class="chart" loading="lazy" src="{rel}/lingbot-views.png" alt="camera poses">
<h3>全部 {len(pool)} 张预选帧，最终保留 {len(positions)} 张</h3><div class="filters" data-target="gallery-{qid}"><button data-filter="all">全部</button><button data-filter="true">已保留</button><button data-filter="false">未保留</button></div><div id="gallery-{qid}" class="gallery">{''.join(cards)}</div></section>''')
    table='<table><tr><th>题目</th><th>原始逐峰</th><th>上版 V4</th><th>本版</th><th>实例候选组</th></tr>'+''.join(f'<tr><td><a href="#q{r["id"]}">{r["id"]}</a></td><td>{r["v3"]}</td><td>{r["previous_v4"]}</td><td>{r["new"]}</td><td>{r["instance_hypotheses"]}</td></tr>' for r in comparison)+'</table>'
    if baseline:
        table='<table><tr><th>题目</th><th>原始逐峰</th><th>按类别预算</th><th>保守归并</th><th>本版积极归并</th><th>实例候选组</th></tr>'+''.join(f'<tr><td><a href="#q{r["id"]}">{r["id"]}</a></td><td>{r["v3"]}</td><td>{r["previous_v4"]}</td><td>{r["conservative"]}</td><td>{r["new"]}</td><td>{r["instance_hypotheses"]}</td></tr>' for r in comparison)+'</table>'
    if baseline and prior_aggressive:
        table=f'<table><tr><th>题目</th><th>原始逐峰</th><th>按类别预算</th><th>保守归并</th><th>积极版 {prior_threshold:.2f}</th><th>本版 {config["threshold"]:.2f}</th><th>实例候选组</th></tr>'+''.join(f'<tr><td><a href="#q{r["id"]}">{r["id"]}</a></td><td>{r["v3"]}</td><td>{r["previous_v4"]}</td><td>{r["conservative"]}</td><td>{r["previous_aggressive"]}</td><td>{r["new"]}</td><td>{r["instance_hypotheses"]}</td></tr>' for r in comparison)+'</table>'
    page='''<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V5 · DINOv2 实例归并抽帧</title><style>
body{font:16px/1.6 system-ui;background:#f3f5f8;color:#202937;margin:0}main{max-width:1400px;margin:auto;padding:25px}section,.intro{background:white;border-radius:12px;margin:20px 0;padding:24px}h1,h2,h3{line-height:1.3}a{color:#235ab5}video{width:100%;max-height:560px;background:#111}.question{white-space:pre-line}.instance{margin:18px 0;padding:12px 18px;background:#fafbfc}.montage{width:min(960px,100%);height:auto;display:block;margin:8px 0 20px}.chart{width:100%;height:auto}.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:16px}.gallery img{width:100%;height:auto}figure{margin:0;background:#f5f7fa;border-radius:8px;overflow:hidden}figcaption{padding:9px;font-size:13px;overflow-wrap:anywhere}button{cursor:pointer;padding:7px 12px;border:1px solid #ccd3df;background:white;border-radius:6px}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:9px;border-bottom:1px solid #ddd}.note{background:#fff4db;padding:14px;border-radius:8px}[hidden]{display:none!important}nav{display:flex;gap:12px;flex-wrap:wrap}.filters{margin:14px 0}summary{cursor:pointer}</style></head><body><main>'''
    intro=f'''<div class="intro"><h1>V5 · DINOv2 多波峰实例归并</h1><p>场景 3db0a1c8f3 · 同十题 · 每峰四张观察 → 跨峰外观/几何关联 → 每个实例候选至少四帧、最多八帧。</p>
<p>{summary['original_peaks']} 个波峰 → {summary['instance_hypotheses']} 个实例候选组：{summary['multi_peak_groups']} 个多波峰组，{summary['unresolved_single_peaks']} 个单波峰未确定组。全局保留 {summary['global_unique_frames']} 张不同源帧；跨十题展示 {total_kept} 张入选副本，共 {total_candidates} 张预选副本。</p>
<p class="note">组数不是物体真实数量。同类相似物体仍可能误合并，重复观察也可能没有归并。未确定组仍各保留至少四帧，因此可能比上一版“按类别共享预算”保留更多图片。没有重跑 QA 或重构。</p>
<details open><summary>具体如何分组</summary><ol><li>沿用光流验证的波峰内框轨迹；没有可靠框的上下文不提取物体特征。</li><li>比较两峰的多张 DINOv2 特征，至少两对不同帧的一对一匹配，双方均包含锚点，组内两帧相隔至少 0.5 秒。两对中较低相似度须 ≥ {config['threshold']:.2f}。</li><li>同帧独立检测框禁止合并；LingBot 稀疏几何冲突时暂不合并。合并整组需要所有跨组波峰对都通过，避免链式合并。</li><li>每组以清晰度、检测分数和 LingBot 互补视角选 4～8 帧；不足可靠目标帧时上下文补位明确标注。</li></ol><p>0.82 是本版保守启发式阈值，未用真实身份标签标定。现有同帧独立负例不足，不能称为同实例概率。keyboard 沿用本场景 piano 的语义修正；其他检测错误仍可能存在。</p></details>
<p><a href="04-method-and-validation/peak-association.json">全部跨峰匹配与不合并原因</a> · <a href="04-method-and-validation/merge-history.json">归并过程</a> · <a href="04-method-and-validation/config.json">完整策略</a> · <a href="02-questions/questions.txt">十题原文</a></p>
{table}<h2>原视频</h2><video id="source-video" controls preload="metadata" src="{vp.relative_to(out)}"></video><nav>'''+''.join(f'<a href="#q{r["id"]}">Q{r["id"]}</a>' for r in comparison)+'</nav></div>'
    page+=intro+''.join(sections)+'''</main><script>document.addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;if(b.classList.contains('seek')){const v=document.getElementById('source-video');v.currentTime=Number(b.dataset.time);v.scrollIntoView({behavior:'smooth'});return;}if(b.dataset.filter){const root=document.getElementById(b.parentElement.dataset.target);for(const f of root.children)f.hidden=b.dataset.filter!=='all'&&f.dataset.final!==b.dataset.filter;}});</script></body></html>'''
    if aggressive:
        rules=f'''<details open><summary>积极版如何分组</summary><ol><li>同类物体比较，每峰最多四图，有效框才参与。</li><li>两对不同帧的一对一匹配，不再强制锚点参与；每侧两图间隔至少0.25秒，两对中的较低相似度 ≥ {config['threshold']:.2f}。</li><li>同帧独立框仍禁止合并。稀疏 LingBot 几何冲突仅警告。</li><li>外观归并阶段：一个共同代表必须与组内每个成员直接匹配通过；全部波峰对的平均分至少0.60，缺失匹配计0分。允许非代表成员彼此不相似，禁止无共同代表的长链合并。</li><li>每个实例候选继续保留4～8帧，优先质量与位姿互补；上下文补位明确标注。</li></ol><p>阈值为启发式，未经身份标签标定。更积极地合并意味着相似家具误合并的风险上升；组数不是真实物体数，没有验证QA或重构提升。</p></details>'''
        page=re.sub(r'<details open><summary>具体如何分组</summary>.*?</details>',lambda m:rules,page,count=1,flags=re.S)
        page=page.replace('V5 · DINOv2 多波峰实例归并','V5 · DINOv2 积极归并版').replace('<title>V5 · DINOv2 实例归并抽帧</title>','<title>V5 · DINOv2 积极归并版</title>')
        if baseline:
            page=page.replace(table,f'<p><strong>保守版 {baseline["instance_hypotheses"]} 组 / {baseline["global_unique_frames"]} 张全局源帧 → 本版 {summary["instance_hypotheses"]} 组 / {summary["global_unique_frames"]} 张。</strong></p>'+table,1)
        if prior_aggressive:
            page=page.replace(table,f'<p><strong>积极版阈值 {prior_threshold:.2f}：{prior_aggressive["instance_hypotheses"]} 组 / {prior_aggressive["global_unique_frames"]} 张 → 阈值 {config["threshold"]:.2f}：{summary["instance_hypotheses"]} 组 / {summary["global_unique_frames"]} 张。</strong>普通外观组均分仍须 ≥ 0.60；若开启位姿关联，还会应用下述几何归并规则。</p>'+table,1)
        page=page.replace('V5 · DINOv2 积极归并版',f'V5 · DINOv2 积极归并版 · 阈值 {config["threshold"]:.2f}')
    if config.get('pose_association'):
        page=page.replace('V5 · DINOv2 积极归并版', 'V5 · 外观与位姿联合归并')
        page=page.replace(f'积极版 {prior_threshold:.2f}' if prior_threshold is not None else '上版积极归并', 'V4 外观归并')
        page=page.replace('本版积极归并', 'V5 联合归并')
        explanation='<p class="note"><strong>V5 位姿关联已启用：</strong>外观分组后，检查每次新增合并的直接跨组证据：稀疏点重投影、局部特征三视角验证，或符合条件的扩框小物体配对。位姿合并可绕过外观组均分0.60要求；显示为 priority_only 的0.60只是排序常数，不能解释为实际相似度。原检测框及同帧独立约束保留。</p>'
        page=page.replace('<details open><summary>',explanation+'<details open><summary>',1)
    (out/'index.html').write_text(page);save(out/'comparison.json',comparison)
    save(out/'manifest.json',{'version':summary['version'],'mode':config.get('mode','conservative'),'video':{'path':str(vp.relative_to(out)),'sha256':sha(vp)},'questions':question_catalog,'candidate_copies':total_candidates,'selected_copies':total_kept,'global_unique_selected':summary['global_unique_frames'],'source_result':str(a.result.resolve())})
    (out/'README.txt').write_text('打开 index.html 查看原视频、十题、逐峰四图、实例归并证据、DINO条形图、抽帧位置和全部候选。\n每题 input-frames.json 是新选帧输入，路径相对此文件所在目录。\n实例组是假设；单峰组身份未定，不把组数作为计数题答案。原版交付保留。\n')
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[];self.ids=set();self.cards=0;self.kept=0
        def handle_starttag(self,t,attrs):
            d=dict(attrs);self.ids.update([d['id']] if 'id' in d else [])
            self.links.extend(d[k] for k in ['href','src'] if d.get(k))
            if t=='figure' and d.get('class')=='candidate':self.cards+=1;self.kept+=d.get('data-final')=='true'
    parser=Links();parser.feed(page)
    for link in parser.links:
        u=urlsplit(link)
        if u.scheme or u.netloc:continue
        if u.path:assert (out/unquote(u.path)).exists(),link
        elif u.fragment:assert u.fragment in parser.ids,link
    for q in question_catalog:
        folder=out/q['folder'];inputs=read(folder/'input-frames.json')['frames']
        ids=[r['source_frame_index'] for r in inputs];assert ids==sorted(set(ids))
        for r in inputs:assert sha(folder/r['image'])==r['sha256']
    assert (parser.cards,parser.kept)==(total_candidates,total_kept)
    validation={'questions':len(question_catalog),'all_candidates':total_candidates,'highlighted_final':total_kept,
                'raw_frames_sha256_verified':len(allids),'video_sha256_verified':True,'html_links_checked':len(parser.links),'broken_links':[],
                'input_manifests_chronological_deduplicated_hash_verified':True,'identity_counts_are_hypotheses':True}
    save(out/'validation.json',validation);print(validation)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for k in ['result','previous','v3','cache','lingbot','out']:ap.add_argument('--'+k,type=Path,required=True)
    ap.add_argument('--refresh',action='store_true')
    ap.add_argument('--conservative-result',type=Path)
    ap.add_argument('--previous-aggressive-result',type=Path)
    build(ap.parse_args())
