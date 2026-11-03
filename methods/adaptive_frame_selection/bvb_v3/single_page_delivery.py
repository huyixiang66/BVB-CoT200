"""Render every delivered question, chart and extracted image on one page."""
import argparse
import html
import json
from pathlib import Path


def render(out):
    out=Path(out)
    manifest=json.loads((out/'manifest.json').read_text())
    esc=html.escape
    nav=[];sections=[];total=0
    for entry in manifest['questions']:
        folder=entry['folder'];qid=entry['id']
        q=json.loads((out/folder/'question.json').read_text())
        positions=json.loads((out/folder/'sampling-positions.json').read_text())
        nav.append(f'<a href="#q{qid}">{entry["number"]:02d} · Q{qid}</a>')
        cards=[]
        for f in positions['frames']:
            path=folder+'/'+f['image'];status=f['evidence_status']
            label={'detection_backed':'检测支持','context_only':'上下文补位','mixed_evidence_and_context':'检测支持＋上下文用途'}[status]
            cards.append(f'''<figure class="frame-card {status}">
<button class="enlarge" data-image="{esc(path)}" data-caption="Q{qid} · #{f['number']:03d} · {esc(f['timecode'])}" aria-label="放大第{f['number']}张图片">
<img loading="lazy" decoding="async" width="640" height="480" src="{esc(path)}" alt="Q{qid} 第{f['number']}张，原视频{esc(f['timecode'])}"></button>
<figcaption><strong>#{f['number']:03d}</strong> · <button class="seek" data-time="{f['timestamp_s']}">{esc(f['timecode'])}</button>
<span class="badge">{label}</span><br>{esc(', '.join(f['objects']))}<br><span class="muted">原视频帧索引 {f['source_frame_index_zero_based']}（从0开始）</span></figcaption></figure>''')
        total+=len(cards)
        options=''.join(f'<li>{esc(o)}</li>' for o in q.get('options') or [])
        sections.append(f'''<section class="question" id="q{qid}">
<div class="question-title"><h2>{entry['number']:02d} · Q{qid}</h2><a href="#top">回到顶部 ↑</a></div>
<p class="question-text">{esc(q['question']).replace(chr(10),'<br>')}</p>{('<ul class="options">'+options+'</ul>') if options else ''}
<p class="muted">相关物体：{esc(', '.join(q['objects']))}</p>
<p class="stats">{entry['peaks']} 个采样波峰 · {entry['frame_slots']} 个帧位 · <strong>{len(cards)} 张去重图片</strong> · {entry['context_slots']} 个上下文补位帧位</p>
<div class="question-nav"><a href="#q{qid}-dino">DINO条形图</a><a href="#q{qid}-positions">抽帧位置</a><a href="#q{qid}-frames">实际图片</a><a href="{folder}/sampling-positions.txt">下载时间和帧号清单</a></div>
<h3 id="q{qid}-dino">DINO 分数条形图</h3><p class="legend">蓝柱 ≥ 0.5；灰柱 &lt; 0.5；红三角为保留锚点。每秒、每类取最高原始检测分数。</p>
<img class="chart" loading="lazy" src="{folder}/dino-bars.png" alt="Q{qid} DINO分数条形图">
<h3 id="q{qid}-positions">抽帧位置</h3><p class="legend">绿色：检测支持；橙色：未确认的上下文；紫色：同帧兼有两种用途；红色：锚点。</p>
<img class="chart" loading="lazy" src="{folder}/sampling-positions.png" alt="Q{qid} 抽帧时间分布">
<h3 id="q{qid}-frames">实际抽出的全部 {len(cards)} 张图片</h3><p class="legend">按时间排序。点击图片在本页放大；点击时间跳到上方原视频对应位置。</p>
<div class="frame-grid">{''.join(cards)}</div></section>''')
    video=esc(manifest['video']['path'])
    page='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>十题自适应抽帧 · 全部内容</title><style>
:root{color-scheme:light;--ink:#1d2939;--muted:#667085;--line:#dde3eb;--accent:#1765a3}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:100px}body{margin:0;background:#f6f8fb;color:var(--ink);font:16px/1.65 system-ui,-apple-system,sans-serif}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}button{font:inherit;cursor:pointer}main{max-width:1440px;margin:auto;padding:28px 24px 80px}h1{font-size:30px;margin:0 0 12px}h2{font-size:26px;margin:0}h3{margin:32px 0 10px}.muted,.legend{color:var(--muted)}.legend{font-size:14px}
.jump-nav{position:sticky;top:0;z-index:10;background:rgba(255,255,255,.97);border-bottom:1px solid var(--line);padding:10px 20px;display:flex;gap:8px;flex-wrap:wrap;justify-content:center}.jump-nav a{padding:5px 11px;background:#edf3fa;border-radius:6px;font-size:14px;white-space:nowrap}
.intro,.question{background:white;border:1px solid var(--line);border-radius:12px;padding:26px;margin-bottom:32px}.intro video{display:block;width:min(100%,850px);max-height:65vh;margin:18px auto 0;background:#111;border-radius:8px}.question-title{display:flex;align-items:center;justify-content:space-between;gap:12px}.question-title>a{font-size:14px}.question-text{font-size:19px;line-height:1.7}.stats{padding:12px 16px;background:#f0f5fb;border-radius:6px}.question-nav{display:flex;gap:20px;flex-wrap:wrap;font-size:14px}.chart{display:block;width:100%;height:auto;border:1px solid #edf0f3}
.frame-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:18px}.frame-card{margin:0;border:1px solid var(--line);border-radius:8px;overflow:hidden;background:#fff}.enlarge{display:block;width:100%;padding:0;border:0;background:#eee}.enlarge img{display:block;width:100%;height:auto;aspect-ratio:4/3;object-fit:contain}.enlarge:hover{filter:brightness(.95)}figcaption{padding:10px 12px;font-size:13px}.seek{border:0;padding:0;color:var(--accent);background:none;font-weight:600}.badge{display:block;font-size:12px;color:#246d4c;margin:5px 0}.context_only .badge{color:#a5610e}.mixed_evidence_and_context .badge{color:#7a4c9e}
dialog{border:0;border-radius:10px;padding:14px;max-width:95vw;max-height:94vh}dialog::backdrop{background:rgba(0,0,0,.75)}dialog img{display:block;max-width:90vw;max-height:78vh;width:auto;height:auto;margin:auto}.modal-head{display:flex;justify-content:space-between;gap:24px;align-items:center;margin-bottom:10px}.modal-head button{background:#edf0f4;border:0;border-radius:4px;padding:5px 14px}
@media(max-width:1100px){.frame-grid{grid-template-columns:repeat(3,minmax(0,1fr))}}@media(max-width:750px){main{padding:16px 12px}.intro,.question{padding:16px}.frame-grid{grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}h1{font-size:24px}.jump-nav{flex-wrap:nowrap;overflow-x:auto;justify-content:flex-start}html{scroll-padding-top:70px}}@media(max-width:420px){.frame-grid{grid-template-columns:1fr}}
</style></head><body><nav class="jump-nav" aria-label="题目跳转"><a href="#top">原视频</a>'''+''.join(nav)+'''</nav><main id="top"><div class="intro">
<h1>3db0a1c8f3：十题自适应抽帧</h1><p>原视频、10个问题、DINO条形图、抽帧位置及所有实际图片均展示在本页，向下滚动即可查看。</p>
<p><a href="'''+video+'''">原视频文件</a> · <a href="02-questions/questions.txt">10个问题原文</a> · <a href="02-questions/questions.json">题目JSON</a></p>
<p class="muted">1fps发现波峰，每峰4帧；各题内去重。90个采样峰共360个帧位，对应全局290张源图；跨题复用后，本页共展示 '''+str(total)+''' 张图片。65个帧位为上下文补位。波峰数不等于物体数，检测分数不等于正确率。keyboard在本题中按钢琴检测。</p>
<video id="source-video" controls preload="metadata" src="'''+video+'''"></video><p id="video-status" class="legend" aria-live="polite">点击下方任意图片的时间，可定位到这里的原视频。</p></div>'''+''.join(sections)+'''</main>
<dialog id="image-dialog"><div class="modal-head"><span id="image-caption"></span><button id="close-dialog" aria-label="关闭图片">关闭 ×</button></div><img id="large-image" alt="抽帧原图"></dialog>
<script>
const video=document.getElementById('source-video'),modal=document.getElementById('image-dialog');
document.addEventListener('click',event=>{
const seekButton=event.target.closest('button.seek');
if(seekButton){const t=Number(seekButton.dataset.time);video.pause();if(video.readyState>=1){video.currentTime=t;}else{video.addEventListener('loadedmetadata',()=>{video.currentTime=t;},{once:true});video.load();}document.getElementById('video-status').textContent='定位时间：'+seekButton.textContent+'（暂停，可点击播放）';video.scrollIntoView({behavior:'smooth',block:'center'});}
const imageButton=event.target.closest('button.enlarge');
if(imageButton){document.getElementById('large-image').src=imageButton.dataset.image;document.getElementById('image-caption').textContent=imageButton.dataset.caption;modal.showModal();}
});
document.getElementById('close-dialog').addEventListener('click',()=>modal.close());
modal.addEventListener('click',event=>{if(event.target===modal)modal.close();});
</script></body></html>'''
    (out/'index.html').write_text(page)
    return {'questions':len(sections),'inline_frame_images':total,'inline_charts':2*len(sections)}


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);args=ap.parse_args();print(json.dumps(render(args.directory)))
