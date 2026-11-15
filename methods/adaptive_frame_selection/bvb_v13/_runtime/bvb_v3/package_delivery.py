"""Package the five requested adaptive-sampling deliverables, without inference."""
import argparse, hashlib, html, json, math, shutil
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(p):return json.loads(p.read_text())
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def stamp(t):return f'{int(t)//60:02d}:{t%60:06.3f}'


def main():
    ap=argparse.ArgumentParser()
    for k in ['source','video','out']:ap.add_argument('--'+k,type=Path,required=True)
    a=ap.parse_args();src=a.source;out=a.out;out.mkdir(parents=True,exist_ok=False)
    summary=read(src/'summary.json');peaks=read(src/'peaks.json');peakmap={p['peak_id']:p for p in peaks};qs=read(src/'questions.json');cfg=read(src/'config.json')
    detections=read(src/'provenance/merged-coarse-detections.json');times=np.asarray([r['timestamp_s'] for r in detections])
    video=out/'01-original-video';video.mkdir();vp=video/a.video.name;shutil.copy2(a.video,vp)
    assert digest(vp)==digest(a.video)=='cad5068450a546ade12eac8fa0ca9f568095cb235708ac5ccd9ec122064ded6e'
    qdir=out/'02-questions';qdir.mkdir();save(qdir/'questions.json',qs)
    qtext=[]
    for i,q in enumerate(qs['questions'],1):
        qtext.append(f'{i:02d}. Q{q["id"]} [{q["question_type"]}]\n{q["question"]}\n'+('\n'.join(q.get('options') or [])+'\n' if q.get('options') else '')+'相关物体：'+', '.join(q['objects'])+'\n')
    (qdir/'questions.txt').write_text('\n'.join(qtext))
    per=out/'03-per-question';per.mkdir();catalog=[];fileproof=[]
    style='<style>body{max-width:1400px;margin:24px auto;padding:0 18px;font:17px system-ui;line-height:1.65}img,video{max-width:100%}table{border-collapse:collapse}td,th{padding:8px 12px;border-bottom:1px solid #ddd}code{background:#f0f3f5}nav{padding:12px;background:#f0f3f5}.frames{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}.frames figure{margin:0}.frames img{width:100%}figcaption{font-size:14px}a{color:#1262aa}button{cursor:pointer}</style>'
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',17)
    for qi,entry in enumerate(summary['questions'],1):
        q=entry['question'];folder=per/entry['folder'];folder.mkdir();framesdir=folder/'frames';framesdir.mkdir()
        original=read(src/entry['folder']/'input-frames.json');ps=[peakmap[pid] for pid in entry['peaks']];frames=[]
        save(folder/'question.json',q);(folder/'question.txt').write_text(qtext[qi-1])
        for j,f in enumerate(original['frames'],1):
            uses=[{'peak_id':p['peak_id'],'object':p['object'],'status':r['status'],'reason':r['reason'],'dino_score':r['dino_score'],'is_anchor':r['frame']==p['anchor']['frame']} for p in ps for r in p['selected'] if r['source_frame_index']==f['source_frame_index']]
            assert uses
            name=f'{j:03d}__{f["timestamp_s"]:010.3f}s__frame{f["source_frame_index"]:07d}.jpg';target=framesdir/name;source=src/entry['folder']/f['image'];shutil.copy2(source,target)
            assert digest(target)==f['sha256']
            has_evidence=any(u['status']!='context_fallback' for u in uses);has_context=any(u['status']=='context_fallback' for u in uses)
            row={'number':j,'timecode':stamp(f['timestamp_s']),'timestamp_s':f['timestamp_s'],'source_frame_index_zero_based':f['source_frame_index'],'source_frame_number_one_based':f['source_frame_index']+1,'image':'frames/'+name,'sha256':f['sha256'],'objects':list(dict.fromkeys(u['object'] for u in uses)),'uses':uses,'evidence_status':'mixed_evidence_and_context' if has_evidence and has_context else ('context_only' if has_context else 'detection_backed')};frames.append(row)
            fileproof.append({'path':str(target.relative_to(out)),'sha256':f['sha256'],'source':str(source.resolve())})
        assert len(frames)==entry['unique_frames'];assert [f['timestamp_s'] for f in frames]==sorted(set(f['timestamp_s'] for f in frames))
        positions={'question_id':q['id'],'source_video':'../../01-original-video/'+vp.name,'source_video_fps':30,'discovery_fps':1,'local_candidate_fps':4,'frames_per_peak':4,'peak_count':len(ps),'frame_slots':entry['slots'],'unique_frames':len(frames),'context_fallback_slots':entry['context_fallback_slots'],'frames':frames,'peaks':[{'peak_id':p['peak_id'],'object':p['object'],'anchor_s':p['anchor_time_s'],'fallback':p['fallback'],'selected_source_frame_indices':[r['source_frame_index'] for r in p['selected']]} for p in ps]}
        save(folder/'sampling-positions.json',positions)
        lines=['序号\t原视频时间\t秒\t原视频帧索引(从0开始)\t物体\t状态\t图片']
        for f in frames:lines.append(f'{f["number"]}\t{f["timecode"]}\t{f["timestamp_s"]:.6f}\t{f["source_frame_index_zero_based"]}\t{", ".join(f["objects"])}\t{f["evidence_status"]}\t{f["image"]}')
        (folder/'sampling-positions.txt').write_text('\n'.join(lines)+'\n')
        scoredata=[];fig,axs=plt.subplots(len(q['objects']),1,figsize=(16,2.5*len(q['objects'])),sharex=True,squeeze=False)
        for oi,o in enumerate(q['objects']):
            vals=np.asarray([max((b['score'] for b in r['detections'] if b['object']==o),default=0) for r in detections]);ax=axs[oi,0]
            ax.bar(times,vals,width=.82,color=np.where(vals>=.5,'#297cb3','#c8cdd2'),edgecolor='none');ax.axhline(.5,color='#a54b41',ls='--',lw=1)
            for p in ps:
                if p['object']==o:ax.scatter(p['anchor_time_s'],p['anchor']['score'],s=35,marker='v',color='#dc382b',zorder=5)
            ax.set_ylabel('keyboard\n(piano)' if o=='keyboard' else o);ax.set_ylim(0,1);ax.set_xlim(-1,238);ax.grid(axis='y',alpha=.18)
            scoredata.append({'object':o,'detection_meaning':'piano' if o=='keyboard' else o,'rows':[{'timestamp_s':r['timestamp_s'],'source_frame_index':r['source_frame_index'],'score':float(v),'boxes':[b for b in r['detections'] if b['object']==o],'passes_hardcut':bool(v>=.5)} for r,v in zip(detections,vals)]})
        axs[0,0].set_title(f'Q{q["id"]} | DINO score bars, 1fps discovery\nblue >= 0.5 / grey < 0.5 / red triangles = retained anchors; raw scores, no area weighting')
        axs[-1,0].set_xlabel('Original video time (seconds)');fig.tight_layout();fig.savefig(folder/'dino-bars.png',dpi=150);plt.close(fig)
        save(folder/'dino-scores.json',{'question_id':q['id'],'semantics':'per-class maximum cached box score per 1fps frame; missing cached detection ->0, not proof of object absence; scores are not calibrated probabilities','objects':scoredata})
        # Separate position chart: all deduplicated input frames and per-object assignments.
        fig,ax=plt.subplots(figsize=(16,2.3+.65*len(q['objects'])))
        for f in frames:
            color='#d38b18' if f['evidence_status']=='context_only' else ('#8e5aa6' if f['evidence_status']=='mixed_evidence_and_context' else '#19865b')
            ax.vlines(f['timestamp_s'],-.25,.25,color=color,lw=1.2)
        for oi,o in enumerate(q['objects'],1):
            for p in ps:
                if p['object']!=o:continue
                ax.plot([min(r['timestamp_s'] for r in p['selected']),max(r['timestamp_s'] for r in p['selected'])],[oi,oi],color='#ccc',lw=1)
                for r in p['selected']:ax.scatter(r['timestamp_s'],oi,marker='|' if r['status']!='context_fallback' else 'x',s=70,color='#d38b18' if r['status']=='context_fallback' else '#19865b',linewidths=1)
                ax.scatter(p['anchor_time_s'],oi,marker='^',s=24,color='#dc382b',zorder=5)
        ax.set_yticks(range(len(q['objects'])+1),['ALL INPUT FRAMES']+[('keyboard (piano)' if o=='keyboard' else o) for o in q['objects']]);ax.invert_yaxis();ax.set_xlim(-1,238);ax.grid(axis='x',alpha=.2);ax.set_xlabel('Original video time (seconds)');ax.set_title(f'Q{q["id"]}: {len(ps)} peaks x 4 = {entry["slots"]} slots; {len(frames)} unique input frames\ngreen: detection-backed / orange: unverified context / purple: both roles / red: anchor');fig.tight_layout();fig.savefig(folder/'sampling-positions.png',dpi=150);plt.close(fig)
        contacts=folder/'contact-sheets';contacts.mkdir()
        for page in range(math.ceil(len(frames)/24)):
            subset=frames[page*24:(page+1)*24];canvas=Image.new('RGB',(1280,math.ceil(len(subset)/4)*285),'white');draw=ImageDraw.Draw(canvas)
            for j,f in enumerate(subset):
                x=j%4*320;y=j//4*285;im=Image.open(folder/f['image']).resize((320,240));canvas.paste(im,(x,y+45));draw.text((x+5,y+4),f'{f["number"]:03d} | {f["timecode"]}',font=font,fill='black');draw.text((x+5,y+24),f['evidence_status'],font=font,fill='#a6640b' if f['evidence_status']=='context_only' else '#22643c')
            canvas.save(contacts/f'page-{page+1:02d}.jpg',quality=90)
        cards=''.join(f'<figure><a href="{f["image"]}"><img loading="lazy" src="{f["image"]}"></a><figcaption>#{f["number"]:03d} · <button onclick="seek({f["timestamp_s"]})">{f["timecode"]}</button> · 原帧{f["source_frame_index_zero_based"]}<br>{html.escape(", ".join(f["objects"]))}<br>{f["evidence_status"]}</figcaption></figure>' for f in frames)
        page='<!doctype html><html lang="zh"><meta charset="utf-8">'+style+f'<title>Q{q["id"]} 抽帧交付</title><a href="../../index.html">返回总目录</a><h1>Q{q["id"]}</h1><p>{html.escape(q["question"]).replace(chr(10),"<br>")}</p><p>'+html.escape(' / '.join(q.get('options') or []))+f'</p><nav><a href="#dino">DINO条形图</a> · <a href="#positions">抽帧位置</a> · <a href="#frames">全部图片</a> · <a href="sampling-positions.txt">时间/帧号清单</a> · <a href="sampling-positions.json">完整JSON</a></nav><p>{len(ps)}个采样波峰 × 4 = {entry["slots"]}帧位，去重后{len(frames)}张。上下文补位{entry["context_fallback_slots"]}个帧位。波峰不等于物体数量。</p><video id="video" controls preload="metadata" src="../../01-original-video/{vp.name}"></video><h2 id="dino">DINO分数条形图</h2><p>每秒一个柱，每类取该帧最高检测分数；蓝色≥0.5，灰色&lt;0.5，红三角为保留峰值。不是分数×面积；keyboard按本题钢琴含义检测。</p><img src="dino-bars.png"><h2 id="positions">抽帧位置</h2><img src="sampling-positions.png"><h2 id="frames">实际抽出的原始图片</h2><p>按时间排序，点击时间可定位原视频；绿/橙仅表示选择依据，不证明检测语义正确。context_only为未确认的上下文，mixed表示同一帧对不同峰有不同用途。</p><div class="frames">'+cards+'</div><script>function seek(t){const v=document.getElementById("video");if(v.readyState>=1){v.currentTime=t;}else{v.addEventListener("loadedmetadata",()=>v.currentTime=t,{once:true});v.load();}v.scrollIntoView({behavior:"smooth"});}</script></html>'
        (folder/'index.html').write_text(page);catalog.append({'number':qi,'id':q['id'],'question':q['question'],'objects':q['objects'],'folder':str(folder.relative_to(out)),'peaks':len(ps),'frame_slots':entry['slots'],'unique_frames':len(frames),'context_slots':entry['context_fallback_slots']})
        print('PACKAGED',q['id'],len(frames),'images',flush=True)
    extra=out/'04-method-and-validation';extra.mkdir()
    for name in ['config.json','semantic-mapping.json','visual-review.json','validation.json']:shutil.copy2(src/name,extra/name)
    save(out/'manifest.json',{'scene':'3db0a1c8f3','video':{'path':str(vp.relative_to(out)),'sha256':digest(vp),'duration_s':7124/30,'fps':30,'native_frames':7124},'questions':catalog,'per_question_image_copies':len(fileproof),'global_unique_source_frames':summary['unique_source_frames'],'images':fileproof,'source_report':str(src.resolve()),'no_inference_rerun':True})
    rows=''.join(f'<tr><td><a href="{q["folder"]}/index.html">{q["number"]:02d} · Q{q["id"]}</a></td><td>{html.escape(", ".join(q["objects"]))}</td><td>{q["peaks"]}</td><td>{q["unique_frames"]}</td><td><a href="{q["folder"]}/dino-bars.png">DINO条形图</a> · <a href="{q["folder"]}/sampling-positions.png">抽帧位置</a> · <a href="{q["folder"]}/frames/">图片文件夹</a></td></tr>' for q in catalog)
    (out/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8">'+style+'<title>十题抽帧完整交付</title><h1>3db0a1c8f3：十题自适应抽帧交付</h1><p>本目录可单独复制。包含原视频、10个原题、每题DINO条形图、抽帧位置和实际原始图片。</p><p><a href="01-original-video/'+vp.name+'">原视频</a> · <a href="02-questions/questions.txt">10个问题原文</a> · <a href="02-questions/questions.json">题目JSON</a></p><video controls preload="metadata" src="01-original-video/'+vp.name+'"></video><table><tr><th>问题</th><th>相关物体</th><th>波峰数</th><th>实际图片数</th><th>文件</th></tr>'+rows+'</table><p>同一帧可被不同问题复用，因此各题图片数之和大于全局290张。90个采样峰共360个帧位，其中65个为未确认的上下文补位。未运行重构或QA；峰数不代表物体数量。详情见04-method-and-validation。</p></html>')
    (out/'README.txt').write_text('打开 index.html 查看全部内容。此文件夹自包含，可整体复制。\n\n01-original-video：原始MP4（237.467秒、30fps）。\n02-questions：10个问题原文、选项、相关物体。\n03-per-question：每题独立目录，包括 question.txt/json、dino-bars.png、dino-scores.json、sampling-positions.png/txt/json、frames/实际图片、contact-sheets/分页组图、index.html。\n04-method-and-validation：算法参数、钢琴语义映射、抽查说明和来源验证。\n\nDINO条形图：1fps，每类取该帧最高原始检测分数。蓝色通过0.5阈值，灰色未通过，红三角为保留波峰。无缓存检测画为0，不证明物体不存在。\n抽帧位置：秒数相对于原视频起点；同时提供从0开始的帧索引和从1开始的帧号。每峰4个帧位，每题文件夹按源帧去重，不重复复制同一张图片。图片文件名包含序号、秒数和从0开始的原帧索引。\n检测分数不是正确率。上下文补位不保证目标可见；部分目标截断和误检仍存在。keyboard只针对这题按用户确认解释为钢琴。\n本次仅整理已有结果和绘图，没有重新推理、重构或QA。\n')
    from .single_page_delivery import render
    render(out)
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[]
        def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ['src','href'] and v)
    broken=[];checked=0
    for p in out.rglob('*.html'):
        h=Links();h.feed(p.read_text())
        for link in h.links:
            u=urlsplit(link)
            if u.scheme or u.netloc or not u.path:continue
            checked+=1
            if not (p.parent/unquote(u.path)).exists():broken.append([str(p),link])
    assert not broken,broken
    save(out/'validation.json',{'video_sha256_verified':True,'question_count':len(catalog),'bar_charts':len(list(per.glob('*/dino-bars.png'))),'position_charts':len(list(per.glob('*/sampling-positions.png'))),'per_question_images_sha256_verified':len(fileproof),'html_links_checked':checked,'broken_links':broken,'symlinks':sum(p.is_symlink() for p in out.rglob('*'))})
    print('DONE',json.dumps(read(out/'validation.json')),flush=True)


if __name__=='__main__':main()
