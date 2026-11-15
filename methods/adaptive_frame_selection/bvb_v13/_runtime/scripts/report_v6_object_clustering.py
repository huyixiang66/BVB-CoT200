"""Portable visual comparison of frozen peaks, object evidence, and grouping."""
from pathlib import Path
import argparse,html,json,math,shutil,sys
from collections import Counter
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_v6_object_clustering import read,write,sha

LABELS={'baseline':'旧版：画面中心＋complete-link',
        'roi_complete':'物体证据＋complete-link',
        'roi_robust':'新版：物体证据＋多关联聚类'}
COLORS=['#d02f48','#1878d1','#189669','#ab6be3','#e29822','#1296a0','#a95a32','#cf46a2','#6c7623','#5043a8','#cc690d','#3a8394','#777777']
def esc(x):return html.escape(str(x))
def font(n):
    try:return ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',n)
    except OSError:return ImageFont.load_default()

def draw_scored_box(im,box,label,score,color):
    """Place the actual detector score inside the box; compact tiny labels."""
    draw=ImageDraw.Draw(im);x0,y0,x1,y1=box
    draw.rectangle(box,outline=color,width=3)
    value=f'{score:.3f}' if score is not None else 'n/a'
    width,height=x1-x0-6,y1-y0-6
    for text in [f'{label} {value}',value]:
        for size in range(24,7,-1):
            face=font(size);bounds=draw.textbbox((0,0),text,font=face)
            tw,th=bounds[2]-bounds[0],bounds[3]-bounds[1]
            if tw+6<=width and th+6<=height:
                draw.rectangle((x0+3,y0+3,x0+tw+9,y0+th+9),fill=color)
                draw.text((x0+6-bounds[0],y0+6-bounds[1]),text,font=face,fill='white')
                return
    # For a sub-character box, the caption below the image retains the score.

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--detector-label',default='Grounding DINO');ap.add_argument('--video',type=Path,required=True);ap.add_argument('--prepared-root',type=Path);a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);(a.out/'media').mkdir(exist_ok=True);(a.out/'frames').mkdir(exist_ok=True)
    s=read(a.source/'selection.json');ev=read(a.source/'evidence.json');cache=Path(s['inputs']['cache'])
    discovery_fps=read(Path(s['inputs']['v3'])/'config.json').get('discovery_fps',1)
    palette={}
    # Preserve established per-question 3D target colors when available.
    for qid in s['questions']:
        p=a.prepared_root/str(qid)/'manifest.json' if a.prepared_root else None
        if p is not None and p.exists():
            for cat,color in read(p)['colors'].items():
                value='#'+''.join(f'{round(v*255):02x}' for v in color[:3])
                if cat in ev and cat not in palette and value not in palette.values():palette[cat]=value
    for cat in sorted(ev):
        if cat not in palette:palette[cat]=next(c for c in COLORS if c not in palette.values())
    assert len(set(palette.values()))==len(palette)
    copies={};sheets={}
    def contact(group,mode):
        rows=group.get('candidate_frames',group['selected']);keep={r['source_frame_index'] for r in group['selected']}
        signature=sha_bytes(json.dumps(['boxed-score-v1',mode,group['object'],rows,sorted(keep)],sort_keys=True).encode())[:16]
        if signature in sheets:return sheets[signature]
        name=f'media/{signature}.jpg';color=palette[group['object']]
        sheet=Image.new('RGB',(1280,304*math.ceil(len(rows)/4)),'#eef2f6')
        for i,r in enumerate(rows):
            fid=r['source_frame_index'];src=cache/r['image'];dest=a.out/'frames'/f'{fid:07d}.jpg'
            if fid not in copies:
                assert sha(src)==r['frame_sha256'];shutil.copy2(src,dest);copies[fid]=r['frame_sha256']
            assert copies[fid]==r['frame_sha256']
            im=Image.open(src).convert('RGB');captions=[]
            for h in r['observations']:
                if h.get('box_xyxy'):
                    c=palette[h['object']];score=h.get('dino_score')
                    draw_scored_box(im,h['box_xyxy'],h['object'],score,c)
                    captions.append(h['object']+' '+(f'{score:.3f}' if score is not None else 'n/a'))
            x=i%4*320;y=i//4*304;sheet.paste(im.resize((320,240)),(x,y+32));d=ImageDraw.Draw(sheet)
            selected=fid in keep
            if selected:d.rectangle((x+2,y+1,x+317,y+301),outline=color,width=5)
            text=f'{"KEEP" if selected else "candidate"} | {r["timestamp_s"]:.2f}s | f{fid}'
            d.text((x+8,y+8),text,font=font(15),fill=color if selected else '#596579')
            d.text((x+8,y+278),' | '.join(dict.fromkeys(captions)) or 'No verified box / score',font=font(14),fill='#243148')
        sheet.save(a.out/name,quality=91);sheets[signature]=name;return name
    def groups_html(gs,mode):
        parts=[]
        for g in gs:
            img=contact(g,mode);rows=g.get('candidate_frames',g['selected'])
            links=' · '.join(f'<a href="frames/{r["source_frame_index"]:07d}.jpg">{r["timestamp_s"]:.2f}s</a>' for r in g['selected'])
            parts.append(f'<article style="border-left:5px solid {palette[g["object"]]}"><h4>{esc(g["group_id"])} <small>{len(g["peak_ids"])} 个波峰 · {len(rows)} 候选 → {len(g["selected"])} 输入帧</small></h4><p>{esc(" / ".join(g["peak_ids"]))}</p><a href="{img}"><img loading="lazy" src="{img}" alt="{esc(g["group_id"])} 的带检测框候选及选中帧"></a><p class="links">{links}</p></article>')
        return ''.join(parts)
    sections=[];summary=[]
    for qid,q in s['questions'].items():
        tabs=''.join(f'<button class="{"active" if mode=="roi_robust" else ""}" data-mode="{mode}">{label}</button>' for mode,label in LABELS.items())
        panels=''.join(f'<div data-panel="{mode}" {"hidden" if mode!="roi_robust" else ""}><p>{q["summary"][mode]["groups"]} 组 · {q["summary"][mode]["image_slots"]} 张次</p>{groups_html(q["arms"][mode],mode)}</div>' for mode in LABELS)
        options=' · '.join(q['question'].get('options') or [])
        absent=set(q['question']['objects'])-{g['object'] for g in q['arms']['roi_robust']}
        missing_note='<p class="note">没有候选波峰的类别：'+esc(', '.join(sorted(absent)))+'；未生成虚假补位实例。</p>' if absent else ''
        sections.append(f'<section id="q{qid}"><h2>Q{qid}</h2><p>{esc(q["question"]["question"])}</p><p>{esc(options)}</p>{missing_note}<nav>{tabs}</nav>{panels}</section>')
        summary.append(f'<tr><td><a href="#q{qid}">Q{qid}</a></td>'+''.join(f'<td>{q["summary"][m]["groups"]} / {q["summary"][m]["image_slots"]}</td>' for m in LABELS)+'</tr>')
    total={m:{k:sum(q['summary'][m][k] for q in s['questions'].values()) for k in ['groups','image_slots']} for m in LABELS}
    matrices=[]
    for cat,e in ev.items():
        pair={tuple(sorted(p['peaks'])):p for p in e['pairs']};ps=e['peaks']
        rows=[]
        for x in ps:
            cells=[]
            for y in ps:
                if x==y:cells.append('<td>—</td>');continue
                p=pair[tuple(sorted((x,y)))];color='#ffe0e2' if p['cannot_link'] else '#dff3e7' if p['score']>=.468 else '#f0f1f5'
                cells.append(f'<td style="background:{color}" title="{esc(p["conflict_reason"] or "融合分；不是概率")}">{p["score"]:.3f}{" ×" if p["cannot_link"] else ""}</td>')
            rows.append(f'<tr><th>{esc(x)}</th>'+''.join(cells)+'</tr>')
        valid=sum(r['geometry'] is not None for r in e['descriptors'])
        matrices.append(f'<details {"open" if cat=="sofa" else ""}><summary>{esc(cat)} · 有效 ROI 几何 {valid}/{len(e["descriptors"])}</summary><div class="scroll"><table><tr><th>波峰</th>'+''.join(f'<th>{esc(p)}</th>' for p in ps)+'</tr>'+''.join(rows)+'</table></div></details>')
    for name in ['selection.json','evidence.json','comparison.json']:shutil.copy2(a.source/name,a.out/name)
    video=a.video;shutil.copy2(video,a.out/'original-video.mp4')
    write(a.out/'media-manifest.json',dict(frames=copies,video_sha256=sha(video),palette=palette))
    write(a.out/'summary.json',dict(totals=total,questions=len(s['questions']),new_model_calls=False,reconstruction_run=False))
    css='body{margin:0;background:#f3f5f9;color:#243148;font:16px system-ui}main{max-width:1320px;margin:auto;padding:28px}header,section{background:white;border-radius:16px;padding:28px;margin-bottom:24px}h1{font-size:32px}h2{margin-top:0}h4{margin:0}small{font-weight:400;color:#627188}article{padding:16px;background:#f7f9fc;margin:16px 0;border-radius:8px}img{width:100%;height:auto;display:block}video{width:640px;max-width:100%}nav{display:flex;flex-wrap:wrap;gap:8px}button{padding:12px;border:1px solid #c5d0df;border-radius:8px;background:white;color:#30435f;cursor:pointer}button.active{background:#234ba0;color:white}table{border-collapse:collapse;width:100%}td,th{padding:10px;text-align:left;border-bottom:1px solid #dce2ed}a{color:#2055a9}details{padding:12px 0}.scroll{overflow:auto}.note{background:#fff3d9;padding:15px;border-radius:8px;line-height:1.7}.links{font-size:14px}p{line-height:1.65}[hidden]{display:none!important}'
    overview='<table><tr><th>问题</th>'+''.join(f'<th>{esc(v)}<br>组数 / 输入张次</th>' for v in LABELS.values())+'</tr>'+''.join(summary)+'<tr><th>合计</th>'+''.join(f'<th>{total[m]["groups"]} / {total[m]["image_slots"]}</th>' for m in LABELS)+'</tr></table>'
    intro='<p>检测器：<strong>'+esc(a.detector_label)+'</strong>。从已有检测、波峰、DINOv2 和 LingBot 缓存生成聚类对比。</p><p>比较原 V6、物体框证据＋complete-link、物体框证据＋多关联聚类。框颜色表示类别，同色 KEEP 边框表示选中帧。聚类为实例假设，检测分数不是正确概率。</p><p><a href="selection.json">选帧</a> · <a href="evidence.json">匹配证据</a> · <a href="comparison.json">组数对比</a></p>'
    js="document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{let s=b.closest('section');s.querySelectorAll('[data-panel]').forEach(p=>p.hidden=p.dataset.panel!==b.dataset.mode);s.querySelectorAll('button').forEach(x=>x.classList.toggle('active',x===b));});"
    intro+=f'<p>波峰发现采样率：{discovery_fps:g} fps。</p>'
    support=s['preset']['parameters']['minimum_frame_pairs']
    intro+=f'<p class="note">本轮允许 {support} 对有效物体帧提供合并支持；匹配门槛 0.468、空间冲突检查、组间平均分及支持比例条件保持不变。无检测框补帧不参与匹配。</p>'
    page='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>BVB V6 · 物体聚类对比</title><style>'+css+'</style></head><body><main><header><h1>物体聚类对比</h1>'+intro+overview+'<div class="original-video"><h2>原视频</h2><video controls preload="metadata" playsinline src="original-video.mp4"></video><p><a href="original-video.mp4">单独打开原视频</a></p></div></header>'+''.join(sections)+'<section><h2>波峰关联分矩阵</h2><p>绿色 ≥ 0.468；× 表示存在禁止合并证据。融合分支持数见本页顶部设置；单帧模式采用最高有效帧对分数。</p>'+''.join(matrices)+'</section></main><script>'+js+'</script></body></html>'
    (a.out/'index.html').write_text(page)
    (a.out/'index-base.html').write_text(page)
    print(a.out/'index.html',json.dumps(total),len(sheets),'contact sheets',flush=True)

def sha_bytes(b):
    import hashlib
    return hashlib.sha256(b).hexdigest()

if __name__=='__main__':main()
