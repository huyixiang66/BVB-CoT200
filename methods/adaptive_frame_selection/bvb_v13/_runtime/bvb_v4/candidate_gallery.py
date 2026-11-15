"""Show every preselected candidate; object-colored frames mark final choices."""
import argparse,hashlib,html,json,re,shutil
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
from PIL import Image,ImageDraw,ImageFont
from bvb_v3.adaptive_peaks import iou

def read(p):return json.loads(Path(p).read_text())
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def update(out,result,cache):
    out,result,cache=map(Path,[out,result,cache]);manifest=read(out/'manifest.json');summary=read(result/'summary.json');gm={g['group_id']:g for g in read(result/'groups.json')};colors=read(out/'object-colors.json')['palette'];cand={r['source_frame_index']:r for r in read(cache/'candidates.json')}
    det={r['source_frame_index']:r['detections'] for r in map(json.loads,(cache/'detections.jsonl').read_text().splitlines())};piano={r['source_frame_index']:r['detections'] for r in map(json.loads,(result.parent/'bvb-v3-adaptive-peaks4-20261101/piano-only-cache-4fps/detections.jsonl').read_text().splitlines())}
    page=(out/'index.html').read_text();font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',15);small=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',13);catalog=[];total=selected_total=multi=0;input_hashes={};boxes_total=0
    for entry,record in zip(manifest['questions'],summary['questions']):
        qid=entry['id'];q=record['question'];folder=out/entry['folder'];input_hashes[str(folder/'input-frames.json')]=sha(folder/'input-frames.json');gs=[gm[k] for k in record['group_ids']];byframe={}
        for g in gs:
            final_ids={r['source_frame_index'] for r in g['selected']}
            for r in g['candidates']:byframe.setdefault(r['source_frame_index'],[]).append((g,r,r['source_frame_index'] in final_ids))
        assert len(byframe)==record['v3_unique_frames'];dest=folder/'candidates-annotated';dest.mkdir(exist_ok=True);rawdir=folder/'candidate-originals';rawdir.mkdir(exist_ok=True);records=[];cards=[];nselected=0
        for number,(fid,uses) in enumerate(sorted(byframe.items()),1):
            c=cand[fid];raw=rawdir/f'frame{fid:07d}.jpg';source=cache/c['frame']
            if not raw.exists():shutil.copy2(source,raw)
            assert sha(raw)==c['frame_sha256']
            selected_objects=[o for o in q['objects'] if any(g['object']==o and chosen for g,r,chosen in uses)];chosen=bool(selected_objects);nselected+=chosen;multi+=len(selected_objects)>1
            boxes=[]
            for g,r,is_final in uses:
                if r['box_xyxy'] and not any(b['object']==g['object'] and iou(b['box_xyxy'],r['box_xyxy'])>=.9 for b in boxes):boxes.append({'object':g['object'],'box_xyxy':r['box_xyxy'],'score':r['dino_score'],'status':r['status'],'source':'preselected candidate'})
            rawboxes=[b for b in det[fid] if b['object']!='keyboard']+[{**b,'object':'keyboard'} for b in piano[fid]]
            for b in sorted(rawboxes,key=lambda b:-b['score']):
                if b['object'] in q['objects'] and b['score']>=.5 and not any(x['object']==b['object'] and iou(x['box_xyxy'],b['box_xyxy'])>=.7 for x in boxes):boxes.append({**b,'status':'cached_detection','source':'additional relevant detection >=0.5'})
            im=Image.open(raw).convert('RGB');draw=ImageDraw.Draw(im);labels=[]
            for b in boxes:
                color=colors[b['object']];x0,y0,x1,y1=b['box_xyxy'];draw.rectangle((x0,y0,x1,y1),outline=color,width=3);label=('keyboard (piano)' if b['object']=='keyboard' else b['object'])+(f' {b["score"]:.2f}' if b['score'] is not None else '');w=draw.textbbox((0,0),label,font=font)[2]+8;x=max(0,min(int(x0),640-w));y=max(0,int(y0)-22)
                while any(abs(y-py)<22 and not (x+w<px or px+pw<x) for px,py,pw in labels) and y<450:y+=22
                y=min(y,458);labels.append((x,y,w));draw.rectangle((x,y,x+w,y+22),fill=color);draw.text((x+4,y+1),label,font=font,fill='#111111')
            missing=[o for o in q['objects'] if any(g['object']==o and r['status']=='context_fallback' for g,r,_ in uses) and not any(b['object']==o for b in boxes)]
            if missing:draw.rectangle((0,459,640,480),fill='#fff0d4');draw.text((6,461),'Unverified context: '+', '.join(missing),font=small,fill='#784800')
            canvas=Image.new('RGB',(640,510),'#edf0f3');canvas.paste(im,(0,30));d=ImageDraw.Draw(canvas)
            if chosen:
                for j,o in enumerate(selected_objects):
                    x0=round(j*640/len(selected_objects));x1=round((j+1)*640/len(selected_objects));d.rectangle((x0,0,x1,29),fill=colors[o]);d.text((x0+7,6),'KEEP: '+o,font=small,fill='#111111');pad=j*4;d.rectangle((pad,30+pad,639-pad,509-pad),outline=colors[o],width=4)
            else:d.text((7,6),'CANDIDATE - NOT RETAINED',font=small,fill='#555b64');d.rectangle((0,30,639,509),outline='#c6ccd5',width=1)
            target=dest/f'{number:03d}__{c["timestamp_s"]:010.3f}s__frame{fid:07d}.jpg';canvas.save(target,quality=94);rel=str(target.relative_to(out));time=f'{int(c["timestamp_s"])//60:02d}:{c["timestamp_s"]%60:06.3f}'
            row={'question_id':qid,'number':number,'source_frame_index':fid,'timestamp_s':c['timestamp_s'],'original':str(raw.relative_to(out)),'original_sha256':c['frame_sha256'],'display_image':rel,'display_sha256':sha(target),'final_selected':chosen,'selected_objects':selected_objects,'highlight_colors':[colors[o] for o in selected_objects],'boxes':boxes,'uses':[{'object':g['object'],'group_id':g['group_id'],'selected':is_final,'status':r['status'],'reason':r['selection_reason']} for g,r,is_final in uses]};records.append(row);boxes_total+=len(boxes)
            badges=''.join(f'<span class="keep-badge" style="background:{colors[o]};color:#111">已保留 · {html.escape(o)}</span>' for o in selected_objects) if chosen else '<span class="not-kept">未保留</span>'
            stripe=''.join(f'<span style="background:{colors[o]}"></span>' for o in selected_objects)
            cards.append(f'<figure class="frame-card candidate-card" data-final="{str(chosen).lower()}"><div class="keep-stripes">{stripe}</div><button class="enlarge" data-image="{rel}" data-caption="Q{qid} · 候选#{number:03d} · {time} · {html.escape(", ".join(selected_objects)) if chosen else "未保留"}"><img loading="lazy" decoding="async" width="640" height="510" src="{rel}" alt="Q{qid}候选{number}，'+('已保留' if chosen else '未保留')+f'"></button><figcaption><strong>候选 #{number:03d}</strong> · <button class="seek" data-time="{c["timestamp_s"]}">{time}</button><div>{badges}</div><span class="muted">原帧 {fid}（从0开始）</span></figcaption></figure>')
        assert nselected==record['unique_frames'];total+=len(records);selected_total+=nselected
        save(folder/'candidate-frames.json',{'question_id':qid,'definition':'All V4 input candidates inherited from V3, deduplicated by original frame index; not all 950 4fps video frames','candidate_count':len(records),'final_count':nselected,'object_colors':{o:colors[o] for o in q['objects']},'frames':records});catalog.append({'question_id':qid,'candidates':len(records),'final_selected':nselected,'manifest':str((folder/'candidate-frames.json').relative_to(out))})
        block=f'<h3 id="q{qid}-frames">全部 {len(records)} 张预选帧 · 最终保留 {nselected} 张</h3><p class="legend">所有候选都按物体颜色显示检测框。<strong>同色外框＋KEEP／已保留</strong>表示该帧最终被对应物体选中；多色外框表示被多个物体选中。无彩色外框为未保留。颜色与上方3D图一致；上下文补位无可靠框时不补画。点击图片可在本页放大。</p><div class="candidate-filters" data-gallery="candidate-q{qid}"><button data-filter="all">全部 {len(records)}</button><button data-filter="true">仅已保留 {nselected}</button><button data-filter="false">仅未保留 {len(records)-nselected}</button><a href="{entry["folder"]}/candidate-frames.json">候选与选择记录</a></div><div class="frame-grid" id="candidate-q{qid}">'+''.join(cards)+'</div>'
        pattern=rf'<h3 id="q{qid}-frames">.*?</section>'
        page,n=re.subn(pattern,lambda m:block+'</section>',page,count=1,flags=re.S);assert n==1
    css='<style id="candidate-gallery-style">.candidate-card .enlarge img{aspect-ratio:auto}.keep-stripes{display:flex;height:5px}.keep-stripes span{flex:1}.candidate-card[data-final="false"] .keep-stripes{background:#e2e5e9}.keep-badge{display:inline-block;padding:2px 6px;border-radius:3px;margin:5px 4px 2px 0;font-size:12px;font-weight:600}.not-kept{display:inline-block;color:#697386;padding:3px 0}.candidate-filters{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:12px 0 20px}.candidate-filters button{border:1px solid #ccd5df;background:white;border-radius:5px;padding:6px 12px}.candidate-card[hidden]{display:none}</style>'
    if 'id="candidate-gallery-style"' not in page:page=page.replace('</head>',css+'</head>')
    js='<script id="candidate-gallery-script">document.addEventListener("click",event=>{const button=event.target.closest(".candidate-filters button");if(!button)return;const root=document.getElementById(button.parentElement.dataset.gallery);for(const card of root.children){card.hidden=button.dataset.filter!=="all"&&card.dataset.final!==button.dataset.filter;}});</script>'
    if 'id="candidate-gallery-script"' not in page:page=page.replace('</body>',js+'</body>')
    page=re.sub(r'跨题复用后本页展示\d+张。',f'跨题复用后，本页展示全部{total}张候选，其中{selected_total}张标记为最终保留。',page)
    page=page.replace('全部实际图片均展示在本页','全部预选帧及最终选择标记均展示在本页');(out/'index.html').write_text(page)
    for p,h in input_hashes.items():assert sha(p)==h
    manifest['candidate_gallery']={'definition':'V3 preselected pool used as V4 input','question_candidate_copies':total,'question_final_copies':selected_total,'global_unique_candidates':290,'global_unique_final':summary['global_unique_frames'],'questions':catalog};save(out/'manifest.json',manifest);save(out/'candidate-gallery.json',catalog)
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[];self.cards=0;self.selected=0
        def handle_starttag(self,t,attrs):
            d=dict(attrs)
            if t=='figure' and 'candidate-card' in d.get('class',''):self.cards+=1;self.selected+=d.get('data-final')=='true'
            self.links.extend(d[k] for k in ['src','href','data-image'] if d.get(k))
    h=Links();h.feed(page)
    for link in h.links:
        u=urlsplit(link)
        if not u.scheme and not u.netloc and u.path:assert (out/unquote(u.path)).exists(),link
    assert (h.cards,h.selected)==(total,selected_total)
    v=read(out/'validation.json');v.update(candidate_gallery_frames=total,highlighted_final_frames=selected_total,multiple_selected_object_frames=multi,candidate_box_records=boxes_total,final_input_manifests_unchanged=True,html_links_checked=len(h.links),broken_links=[]);save(out/'validation.json',v)
    for f in ['candidate_gallery.py','report.py']:shutil.copy2(Path(__file__).with_name(f),out/'04-method-and-validation'/f)
    print(json.dumps({'candidates':total,'selected':selected_total,'multiple_color_highlights':multi,'boxes':boxes_total,'unchanged_final_inputs':True},indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for k in ['out','result','cache']:ap.add_argument('--'+k,type=Path,required=True)
    a=ap.parse_args();update(a.out,a.result,a.cache)
