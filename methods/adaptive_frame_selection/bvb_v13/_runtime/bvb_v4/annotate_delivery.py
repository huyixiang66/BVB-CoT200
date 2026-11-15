"""Deterministic detection overlays with shared object colors in 2D and 3D.

Raw frame files and reconstruction-input hashes are kept unchanged.
"""
import argparse, hashlib, html, json, math, re, shutil
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from bvb_v3.adaptive_peaks import iou


def read(p):return json.loads(Path(p).read_text())
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def annotate(out,result,cache,lingbot):
    out,result,cache,lingbot=map(Path,[out,result,cache,lingbot])
    manifest=read(out/'manifest.json');summary=read(result/'summary.json');groups=read(result/'groups.json');gm={g['group_id']:g for g in groups};questions=read(result/'questions.json');palette={o:to_hex(plt.get_cmap('tab20')(i)) for i,o in enumerate(questions['objects'])}
    det={r['source_frame_index']:r for r in map(json.loads,(cache/'detections.jsonl').read_text().splitlines())}
    piano_path=result.parent/'bvb-v3-adaptive-peaks4-20261101/piano-only-cache-4fps/detections.jsonl'
    piano={r['source_frame_index']:r for r in map(json.loads,piano_path.read_text().splitlines())}
    poses=np.load(lingbot/'camera-poses.npz')['c2w'];C=poses[:,:3,3]
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',16);small=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',14)
    page=(out/'index.html').read_text();all_annotations=[];box_count=0;missing_count=0
    for entry,record in zip(manifest['questions'],summary['questions']):
        assert entry['id']==record['question']['id'];q=record['question'];folder=out/entry['folder'];positions=read(folder/'sampling-positions.json');gs=[gm[k] for k in record['group_ids']];display=folder/'frames-annotated';display.mkdir(exist_ok=True);frame_images={}
        for f in positions['frames']:
            fid=f['source_frame_index_zero_based'];boxes=[]
            # Keep exact boxes used by selection, including low-score/fallback anchors.
            for g in gs:
                for r in g['selected']:
                    if r['source_frame_index']!=fid or not r['box_xyxy']:continue
                    boxes.append({'object':g['object'],'box_xyxy':r['box_xyxy'],'score':r['dino_score'],'source':'selected observation','group_id':g['group_id'],'status':r['status']})
            # Also show other question-relevant, hardcut-passing detections in this frame.
            raw=[b for b in det[fid]['detections'] if b['object']!='keyboard']+[{**b,'object':'keyboard'} for b in piano[fid]['detections']]
            for b in sorted(raw,key=lambda b:-b['score']):
                if b['object'] not in q['objects'] or b['score']<.5:continue
                if any(x['object']==b['object'] and iou(x['box_xyxy'],b['box_xyxy'])>=.7 for x in boxes):continue
                boxes.append({'object':b['object'],'box_xyxy':b['box_xyxy'],'score':b['score'],'source':'additional question-relevant cached detection >=0.5','status':'cached_detection'})
            clean=[]
            for b in boxes:
                if not any(x['object']==b['object'] and iou(x['box_xyxy'],b['box_xyxy'])>=.9 for x in clean):clean.append(b)
            boxes=clean;im=Image.open(folder/f['image']).convert('RGB');draw=ImageDraw.Draw(im);labels=[]
            for b in boxes:
                color=palette[b['object']];x0,y0,x1,y1=[float(v) for v in b['box_xyxy']];x0,x1=np.clip([x0,x1],0,639);y0,y1=np.clip([y0,y1],0,479)
                if x1<=x0 or y1<=y0:continue
                draw.rectangle((x0,y0,x1,y1),outline=color,width=3)
                name='keyboard (piano)' if b['object']=='keyboard' else b['object'];label=name+(f' {b["score"]:.2f}' if b['score'] is not None else '')
                if b['status']=='low_score_anchor':label+=' [low]'
                bb=draw.textbbox((0,0),label,font=font);w=bb[2]-bb[0]+10;h=23;x=min(int(x0),640-w);y=max(0,int(y0)-h)
                while any(abs(y-py)<h and not (x+w<px or px+pw<x) for px,py,pw in labels) and y+h<480:y+=h
                y=min(y,480-h);labels.append((x,y,w));draw.rectangle((x,y,x+w,y+h),fill=color);draw.text((x+5,y+1),label,font=font,fill='black',stroke_width=0)
            context_targets=[u['object'] for u in f['uses'] if u['status']=='context_fallback']
            if context_targets:
                note='CONTEXT ONLY for '+', '.join(dict.fromkeys(context_targets));draw.rectangle((0,457,640,480),fill='#fff0d4');draw.text((7,460),note,font=small,fill='#784800')
            if not boxes and not context_targets:
                draw.rectangle((0,457,640,480),fill='#fff0d4');draw.text((7,460),'No verified detection box available',font=small,fill='#784800')
            target=display/Path(f['image']).name;im.save(target,quality=94);rel=str(target.relative_to(out));old=entry['folder']+'/'+f['image'];page=page.replace(old,rel)
            frame_images[fid]=target;f['display_image']=str(target.relative_to(folder));f['display_sha256']=sha(target);f['display_detections']=boxes
            assert sha(folder/f['image'])==f['sha256'];box_count+=len(boxes);missing_count+=not bool(boxes)
            all_annotations.append({'question_id':q['id'],'source_frame_index':fid,'original':old,'original_sha256':f['sha256'],'annotated':rel,'annotated_sha256':f['display_sha256'],'boxes':boxes,'context_targets':context_targets})
        positions['object_colors']={o:palette[o] for o in q['objects']};save(folder/'sampling-positions.json',positions)
        # Same object => same color across questions, image boxes and all camera groups.
        fig=plt.figure(figsize=(11,7));ax=fig.add_subplot(111,projection='3d');ax.plot(C[:,0],C[:,1],C[:,2],color='#bbb',lw=.7,alpha=.5)
        for g in gs:
            color=palette[g['object']];rs=g['selected'];p=np.array([r['camera_center'] for r in rs]);dirs=np.array([r['view_direction'] for r in rs]);ax.scatter(p[:,0],p[:,1],p[:,2],color=color,s=30,label=g['group_id']);ax.quiver(p[:,0],p[:,1],p[:,2],dirs[:,0],dirs[:,1],dirs[:,2],length=.12,normalize=True,color=color,alpha=.75)
        ax.set_xlabel('LingBot X');ax.set_ylabel('LingBot Y');ax.set_zlabel('LingBot Z');ax.set_title(f'Q{q["id"]}: camera locations / observation directions\nObject colors match image bounding boxes; native LingBot units');ax.legend(fontsize=7,loc='upper left');fig.tight_layout();fig.savefig(folder/'lingbot-views.png',dpi=140);plt.close(fig)
        for g in gs:
            rs=g['selected'];canvas=Image.new('RGB',(1280,math.ceil(len(rs)/4)*285),'white');draw=ImageDraw.Draw(canvas)
            for j,r in enumerate(rs):
                x=j%4*320;y=j//4*285;im=Image.open(frame_images[r['source_frame_index']]).resize((320,240));canvas.paste(im,(x,y+45));draw.text((x+5,y+3),f'{g["object"]} | {r["timestamp_s"]:.2f}s',font=font,fill=palette[g['object']]);draw.text((x+5,y+24),r['status'],font=small,fill='#444444')
            canvas.save(folder/(g['group_id']+'.jpg'),quality=92)
        chips=' '.join(f'<span style="display:inline-block;padding:3px 10px;margin:3px;border:3px solid {palette[o]};border-radius:6px;color:#18202c;background:white">{html.escape(o)}'+('（钢琴）' if o=='keyboard' else '')+'</span>' for o in q['objects'])
        marker=f'<!--object-colors-Q{q["id"]}-->'
        block=marker+'<div class="object-color-legend"><strong>物体颜色（与3D机位图一致）：</strong>'+chips+'</div>'+marker
        if marker in page:page=re.sub(re.escape(marker)+'.*?'+re.escape(marker),lambda m:block,page,flags=re.S)
        else:page=page.replace(f'<h3 id="q{q["id"]}-frames">',block+f'<h3 id="q{q["id"]}-frames">',1)
    page=page.replace('实际抽出的全部','带检测框的全部抽帧')
    note='图片显示用于选帧的检测框，以及同帧中其他题目相关且分数≥0.5的检测框；颜色与3D机位图逐物体一致。标框不代表识别正确；上下文目标无可靠框时不补画。原图与重构输入清单保持不变。'
    if '<!--bbox-note-->' not in page:page=page.replace('<video id="source-video"','<!--bbox-note--><p class="legend">'+note+'</p><video id="source-video"',1)
    (out/'index.html').write_text(page)
    colors={'scope':'global per object across all questions; same colors for boxes and LingBot 3D plots','palette':palette};save(out/'object-colors.json',colors);save(out/'bounding-box-annotations.json',all_annotations)
    manifest['annotated_images']=[{'path':r['annotated'],'sha256':r['annotated_sha256'],'original':r['original']} for r in all_annotations];manifest['object_colors']=palette;save(out/'manifest.json',manifest)
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[]
        def handle_starttag(self,t,attrs):
            d=dict(attrs)
            for k in ['href','src','data-image']:
                if d.get(k):self.links.append(d[k])
    parser=Links();parser.feed(page)
    for link in parser.links:
        u=urlsplit(link)
        if not u.scheme and not u.netloc and u.path:assert (out/unquote(u.path)).exists(),link
    for r in all_annotations:
        assert r['annotated'] in page and sha(out/r['annotated'])==r['annotated_sha256']
    v=read(out/'validation.json');v.update(annotated_images=len(all_annotations),drawn_box_records=box_count,frames_without_box=missing_count,object_colors_shared_between_2d_and_3d=True,original_images_unchanged=True,html_links_checked=len(parser.links));save(out/'validation.json',v)
    provenance=out/'04-method-and-validation';shutil.copy2(Path(__file__),provenance/'annotate_delivery.py');shutil.copy2(Path(__file__).with_name('report.py'),provenance/'report.py')
    print(json.dumps({k:v[k] for k in ['annotated_images','drawn_box_records','frames_without_box','object_colors_shared_between_2d_and_3d','original_images_unchanged']},indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for key in ['out','result','cache','lingbot']:ap.add_argument('--'+key,type=Path,required=True)
    a=ap.parse_args();annotate(a.out,a.result,a.cache,a.lingbot)
