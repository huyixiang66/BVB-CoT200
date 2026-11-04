"""Build a self-contained report for any number of scene questions, including context-only ones."""
import argparse
import html
import json
import shutil
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .dinov2_appearance import read, save, sha
from .global_context import select_context


def complete_questions(questions, summary, candidates, poses):
    records = {r['question']['id']: r for r in summary['questions']}
    result = []
    for q in questions['questions']:
        if q['objects']:
            result.append(records[q['id']])
        else:
            rows = select_context(candidates, poses)
            result.append(dict(question=q, group_ids=[], group_count=0,
                               source_frame_indices=[r['source_frame_index'] for r in rows],
                               unique_frames=len(rows), context_slots=len(rows),
                               context_frames=rows, context_search_candidates=len(candidates),
                               selection_policy='global_context_16'))
    return result


def build(a):
    a.out.mkdir(parents=True, exist_ok=False)
    qs=read(a.questions); summary=read(a.result/'summary.json'); groups=read(a.result/'groups.json')
    gm={g['group_id']:g for g in groups}; peaks=read(a.v3/'peaks.json')
    peak_threshold=read(a.v3/'config.json').get('hardcut',.5)
    cand=read(a.cache/'candidates.json'); cf={c['source_frame_index']:c for c in cand}
    poses=np.load(a.lingbot/'camera-poses.npz')['c2w']
    sampling=read(a.lingbot/'sampling.json')
    assert [c['source_frame_index'] for c in cand]==[r['source_frame_index'] for r in sampling['frames']]
    records=complete_questions(qs,summary,cand,poses)
    assert len(records)==len(qs['questions'])
    assert all(r['question']['question_type']!='obj_appearance_order' for r in records)
    detections=read(a.v3.parent/'merged-coarse-detections.json')
    palette=read(a.colors)['palette']
    extras=['#17becf','#bc6515','#556b2f','#008080']
    for o in qs['objects']:
        if o not in palette:palette[o]=extras.pop(0)
    save(a.out/'object-colors.json',{'palette':palette})
    for name in ['01-original-video','02-questions','03-per-question','04-method-and-validation','images']:
        (a.out/name).mkdir()
    video=a.out/'01-original-video'/a.video.name;shutil.copy2(a.video,video)
    assert sha(video)==sampling['video_sha256']
    save(a.out/'02-questions/questions.json',qs)
    (a.out/'02-questions/questions.txt').write_text('\n\n'.join(
        f'Q{q["id"]} [{q["question_type"]}]\n{q["question"]}\n'+ '\n'.join(q.get('options') or [])
        for q in qs['questions']),encoding='utf-8')
    for file in a.result.glob('*.json'):shutil.copy2(file,a.out/'04-method-and-validation'/file.name)
    for file in ['scene_report.py','global_context.py','instance_sampling.py','pose_association.py','UPSTREAM.json']:
        shutil.copy2(Path(__file__).with_name(file),a.out/'04-method-and-validation'/file)
    for record in records:
        gs=[gm[k] for k in record['group_ids']]
        record['candidate_ids']=sorted({r['source_frame_index'] for g in gs for r in g['candidates']} |
                                       {r['source_frame_index'] for r in record.get('context_frames',[])})
    raw_ids=sorted({fid for r in records for fid in r['candidate_ids']})
    for fid in raw_ids:
        source=a.cache/cf[fid]['frame'];dest=a.out/'images'/f'frame{fid:07d}.jpg'
        shutil.copy2(source,dest);assert sha(dest)==cf[fid]['frame_sha256']
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',15)
    sections=[];catalog=[];checked_inputs=0
    old={r['question']['id']:r for r in read(a.compare/'summary.json')['questions']} if a.compare else {}
    esc=lambda x:html.escape(str(x))
    for number,record in enumerate(records,1):
        q=record['question'];qid=q['id'];gs=[gm[k] for k in record['group_ids']]
        selected=set(record['source_frame_indices']);pool=record['candidate_ids'];folder=a.out/'03-per-question'/f'{number:02d}-q{qid}'
        folder.mkdir();(folder/'candidates-annotated').mkdir();rel=folder.relative_to(a.out)
        save(folder/'question.json',q);save(folder/'instance-groups.json',gs)
        gallery=[];inputs=[];cards=[]
        for i,fid in enumerate(pool,1):
            boxes=[];uses=[]
            for g in gs:
                for r in g['candidates']:
                    if r['source_frame_index']==fid and r.get('box_xyxy'):
                        item={'object':g['object'],'box_xyxy':r['box_xyxy']}
                        if item not in boxes:boxes.append(item)
                for r in g['selected']:
                    if r['source_frame_index']==fid:
                        uses.append({'object':g['object'],'group_id':g['group_id'],'status':r['status'],
                                     'reason':r['selection_reason'],'source_peaks':r['source_peaks']})
            if not gs:
                uses=[r for r in record.get('context_frames',[]) if r['source_frame_index']==fid]
            objects=list(dict.fromkeys(u['object'] for u in uses if 'object' in u))
            im=Image.open(a.out/'images'/f'frame{fid:07d}.jpg').convert('RGB');d=ImageDraw.Draw(im)
            for b in boxes:
                color=palette[b['object']];x0,y0,x1,y1=b['box_xyxy'];d.rectangle((x0,y0,x1,y1),outline=color,width=4)
                y=max(0,y0-20);d.rectangle((x0,y,x0+d.textlength(b['object'],font=font)+8,y+20),fill=color)
                d.text((x0+4,y+1),b['object'],font=font,fill='black')
            canvas=Image.new('RGB',(640,510),'#edf0f3');canvas.paste(im,(0,30));d=ImageDraw.Draw(canvas)
            if fid in selected:
                colors=[palette[o] for o in objects] or ['#355c9a']
                for j,color in enumerate(colors):
                    x=int(j*640/len(colors));xx=int((j+1)*640/len(colors));d.rectangle((x,0,xx,29),fill=color)
                    d.text((x+3,6),'KEEP '+(objects[j] if objects else 'SCENE CONTEXT'),font=font,fill='black')
                    pad=j*4;d.rectangle((pad,30+pad,639-pad,509-pad),outline=color,width=4)
            else:d.text((5,7),'CANDIDATE - NOT RETAINED',font=font,fill='#555555')
            name=f'{i:03d}__frame{fid:07d}.jpg';canvas.save(folder/'candidates-annotated'/name,quality=90)
            row={'source_frame_index':fid,'timestamp_s':cf[fid]['timestamp_s'],'image':f'../../images/frame{fid:07d}.jpg',
                 'sha256':cf[fid]['frame_sha256'],'final_selected':fid in selected,'boxes':boxes,'uses':uses,
                 'display_image':'candidates-annotated/'+name}
            gallery.append(row)
            if fid in selected:inputs.append(row)
            display=rel/'candidates-annotated'/name
            context_note=' · 含未验证上下文补位' if any(u['status']=='context_fallback' for u in uses) else ''
            cards.append(f'<figure data-final="{str(fid in selected).lower()}"><a href="{display}"><img loading="lazy" width="640" height="510" src="{display}"></a><figcaption>#{i} · <button data-seek="{row["timestamp_s"]}">{row["timestamp_s"]:.2f}s</button> · 原帧 {fid}<br>'+esc((' / '.join(objects) if objects else ('全局场景上下文' if not gs else '未选中'))+context_note)+'</figcaption></figure>')
        assert len(inputs)==record['unique_frames'] and [r['source_frame_index'] for r in inputs]==sorted(selected)
        checked_inputs+=len(inputs)
        save(folder/'input-frames.json',{'question':q,'version':'v5-pose-clustering','chronological':True,'deduplicated':True,'frames':inputs})
        save(folder/'candidate-frames.json',{'question_id':qid,'frames':gallery})
        save(folder/'sampling-positions.json',{'question_id':qid,'frames':inputs})
        (folder/'sampling-positions.txt').write_text('source_frame_index\ttime_seconds\n'+'\n'.join(f'{r["source_frame_index"]}\t{r["timestamp_s"]:.6f}' for r in inputs)+'\n')
        n=max(1,len(q['objects']));fig,axs=plt.subplots(n,1,figsize=(14,2.1*n),squeeze=False,sharex=True)
        scores=[]
        for ax,o in zip(axs[:,0],q['objects']):
            ys=[max((d['score'] for d in r['detections'] if d['object']==o),default=0) for r in detections]
            ts=[r['timestamp_s'] for r in detections];ax.bar(ts,ys,width=.82,color=palette[o]);ax.axhline(peak_threshold,color='#444',ls='--',lw=.8)
            ax.set_ylabel(o);ax.set_ylim(0,1);scores.append({'object':o,'times_s':ts,'max_detection_scores':ys})
            for p in peaks:
                if p['object']==o:ax.scatter(p['anchor_time_s'],p['anchor']['score'],color='#cf3030',marker='v',s=20)
        if not q['objects']:axs[0,0].text(.5,.5,'No object target: scene coverage selection',ha='center',transform=axs[0,0].transAxes)
        axs[0,0].set_title(f'Q{qid}: 1fps discovery scores / dashed line {peak_threshold:g} / triangles: candidate peaks')
        axs[-1,0].set_xlabel('Original video seconds');fig.tight_layout();fig.savefig(folder/'dino-bars.png',dpi=120);plt.close(fig)
        save(folder/'dino-scores.json',{'objects':scores,'context_only':not q['objects']})
        fig,ax=plt.subplots(figsize=(14,2.5));ax.vlines([cf[f]['timestamp_s'] for f in pool],.7,1,color='#bbbbbb');ax.vlines([cf[f]['timestamp_s'] for f in selected],0,.5,color='#326ba8')
        ax.set_yticks([.25,.85],['selected','candidates']);ax.set_xlim(-1,cand[-1]['timestamp_s']+1);ax.set_xlabel('Original video seconds');ax.set_title(f'Q{qid}: {len(pool)} candidates -> {len(selected)} selected');fig.tight_layout();fig.savefig(folder/'sampling-positions.png',dpi=120);plt.close(fig)
        gids={c['source_frame_index']:i for i,c in enumerate(cand)};fig=plt.figure(figsize=(9,6));ax=fig.add_subplot(111,projection='3d');centers=poses[:,:3,3];ax.plot(*centers.T,color='#cccccc',lw=.6)
        for obj in q['objects'] or [None]:
            ids=sorted({r['source_frame_index'] for g in gs if g['object']==obj for r in g['selected']}) if obj else sorted(selected)
            idx=[gids[f] for f in ids]
            if idx:ax.scatter(*centers[idx].T,color=palette[obj] if obj else '#355c9a',s=15,label=obj or 'scene context')
        ax.set_title(f'Q{qid}: selected LingBot camera positions');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(folder/'lingbot-views.png',dpi=120);plt.close(fig)
        group_rows=''.join(f'<tr><td>{esc(g["group_id"])}</td><td>{esc(", ".join(g["peak_ids"]))}</td><td>{g["selected_count"]}</td><td>{esc(g.get("representative_evidence",{}).get("support_type","appearance / unresolved"))}</td></tr>' for g in gs)
        policy='V5：扩框外观＋位姿关联，每个实例假设 4～8 帧。' if gs else f'房间面积题没有目标物体：从{len(cand)}张4fps候选中，8个时间段各选一张清晰帧，再按机位与朝向互补补到16帧。此题图库只展示16张最终帧。全局几何覆盖是代理指标，不代表已完整覆盖所有房间。'
        missing=[o for o in q['objects'] if not any(g['object']==o for g in gs)]
        if missing:
            policy='V5：仅对有候选波峰的实例假设选择 4～8 帧。' if not gs else policy
            policy+=' 缺少目标证据：'+esc(', '.join(missing))+'；当前检测与波峰规则未产生候选，没有用其他物体图片冒充补齐。'
            record['missing_objects']=missing
        warning='<p class="warning">部分 chair-p08/p11 观测此前发现目标切换或背景污染；聚类并未确认语义正确。</p>' if not getattr(a,'detector_label',None) and any(set(g['peak_ids'])&{'chair-p08','chair-p11'} for g in gs) else ''
        options='<ul>'+''.join('<li>'+esc(x)+'</li>' for x in (q.get('options') or []))+'</ul>'
        sections.append(f'<section id="q{qid}"><h2>{number:02d} · Q{qid} · {esc(q["question_type"])}</h2><p>{esc(q["question"])}</p>{options}<p>{policy} 共 {len(gs)} 个实例假设，{len(selected)} 张输入图片。</p>{warning}<p><a href="{rel}/input-frames.json">最终输入清单</a> · <a href="{rel}/sampling-positions.txt">时间与帧号</a> · <a href="{rel}/instance-groups.json">聚类证据</a></p><details><summary>查看实例假设分组</summary><table><tr><th>组</th><th>波峰</th><th>保留帧</th><th>证据</th></tr>{group_rows}</table></details><img class="chart" loading="lazy" src="{rel}/dino-bars.png"><img class="chart" loading="lazy" src="{rel}/sampling-positions.png"><img class="pose" loading="lazy" src="{rel}/lingbot-views.png"><p>相同颜色用于物体检测框、选中边框和机位图。上下文补位不保证目标可见。</p><div class="filters"><button data-filter="all">全部候选（{len(pool)}）</button><button data-filter="true">仅最终帧（{len(selected)}）</button></div><div class="gallery">'+''.join(cards)+'</div></section>')
        catalog.append({'id':qid,'type':q['question_type'],'objects':q['objects'],'groups':len(gs),'candidate_frames':len(pool),'selected_frames':len(selected),'folder':str(rel),'previous_10q_frames':old[qid]['unique_frames'] if qid in old else None})
        catalog[-1]['missing_objects']=missing
        print(f'REPORT Q{qid}: {len(selected)} / {len(pool)}',flush=True)
    total=sum(q['selected_frames'] for q in catalog);global_selected=len({f for r in records for f in r['source_frame_indices']})
    types=Counter(q['type'] for q in catalog)
    table='<table><tr><th>题号</th><th>类型</th><th>物体</th><th>实例假设</th><th>候选帧</th><th>选中帧</th></tr>'+''.join(f'<tr><td><a href="#q{q["id"]}">Q{q["id"]}</a></td><td>{q["type"]}</td><td>{esc(", ".join(q["objects"])) or "全局场景"}</td><td>{q["groups"]}</td><td>{q["candidate_frames"]}</td><td>{q["selected_frames"]}</td></tr>' for q in catalog)+'</table>'
    style='body{font:16px system-ui;margin:25px auto;max-width:1500px;padding:0 20px;background:#f5f7fa;color:#182235}section{background:white;padding:22px;margin:24px 0;border-radius:12px}table{border-collapse:collapse;width:100%}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:left}video{width:100%;max-width:900px}.chart{width:100%}.pose{width:65%}.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}figure{margin:0}figure img{width:100%;height:auto}button{cursor:pointer;padding:7px;margin:4px}a{color:#185e9c}.warning{color:#994e00}[hidden]{display:none!important}'
    detector_label=getattr(a,'detector_label',None)
    title=f'{qs["scene"]} · V5 · {len(catalog)}题抽帧'+(f' · {detector_label}' if detector_label else '')
    page=f'<!doctype html><html lang="zh"><meta charset="utf-8"><title>{title}</title><style>{style}</style><h1>{title}</h1><p>已排除 appearance order。{len(groups)}个目标实例假设；跨题去重后{global_selected}张源帧，逐题合计{total}张。组数不等于真实物体数。未运行重建或QA。</p><p>复用检测、光流及LingBot；重新提取扩框DINOv2特征，外观阈值0.4，启用几何关联。发现波峰1fps，局部候选4fps。keyboard沿用用户确认的钢琴含义。</p><p><a href="02-questions/questions.txt">全部问题原文</a> · <a href="manifest.json">交付清单</a> · <a href="04-method-and-validation/run.json">运行记录</a></p><video id="video" controls preload="metadata" src="01-original-video/{video.name}"></video>{table}'+''.join(sections)+'''<script>document.addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;if(b.dataset.seek){const v=document.getElementById('video');v.currentTime=+b.dataset.seek;v.scrollIntoView({behavior:'smooth'});}if(b.dataset.filter){for(const f of b.closest('section').querySelectorAll('figure'))f.hidden=b.dataset.filter!=='all'&&f.dataset.final!==b.dataset.filter;}});</script></html>'''
    if detector_label:
        page=page.replace('复用检测、光流及LingBot；',html.escape(detector_label)+'重新检测；复用光流及LingBot；')
    (a.out/'index.html').write_text(page,encoding='utf-8')
    manifest={'version':'v5-pose-clustering-all-questions','scene':qs['scene'],'questions':catalog,'question_types':dict(types),'question_count':len(catalog),'instance_hypotheses':len(groups),'question_image_copies':total,'global_unique_selected':global_selected,'global_unique_candidates':len(raw_ids),'video':{'path':'01-original-video/'+video.name,'sha256':sha(video)},'context_policy':{'budget':16,'temporal_bins':8,'translation_weight':.5,'orientation_weight':.5},'qa_run':False,'peak_threshold':peak_threshold}
    if detector_label:manifest['detector_label']=detector_label
    save(a.out/'manifest.json',manifest);save(a.out/'04-method-and-validation/all-question-selections.json',records)
    (a.out/'README.txt').write_text(f'Open index.html. {len(catalog)} questions, excluding appearance order. Clean input frames are shared in images/; per-question input-frames.json gives ordered exact frames. Annotation cards retain bounding boxes and highlight final selections. No reconstruction or QA. Context-only question uses explicit global coverage policy.\n')
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[];self.ids=set()
        def handle_starttag(self,tag,attrs):
            d=dict(attrs)
            if 'id' in d:self.ids.add(d['id'])
            self.links.extend(d[k] for k in ('href','src') if d.get(k))
    parser=Links();parser.feed(page)
    for link in parser.links:
        u=urlsplit(link)
        if u.scheme or u.netloc:continue
        if u.path:assert (a.out/unquote(u.path)).exists(),link
        elif u.fragment:assert u.fragment in parser.ids,link
    validation={'question_count':len(catalog),'excluded_appearance_order':True,'video_hash_verified':True,'unique_raw_frame_hashes_verified':len(raw_ids),'chronological_deduplicated_input_frames':checked_inputs,'html_links_checked':len(parser.links),'broken_links':[]}
    save(a.out/'validation.json',validation);print(validation)
    from .cluster_gallery import add_gallery
    add_gallery(a.out,a.v3)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['questions','result','v3','cache','lingbot','video','colors','out']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--compare',type=Path)
    p.add_argument('--detector-label',help='Fresh detector variant; suppress old detector-specific observation warnings')
    build(p.parse_args())
