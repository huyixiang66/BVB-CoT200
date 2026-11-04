"""Add per-instance, per-peak image montages to an existing scene delivery."""
import argparse
import html
import itertools
import re
import shutil
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

from PIL import Image, ImageDraw, ImageFont

from .dinov2_appearance import read, save, sha


def add_gallery(out, v3):
    manifest=read(out/'manifest.json')
    groups=read(out/'04-method-and-validation/groups.json')
    gm={g['group_id']:g for g in groups}
    peaks={p['peak_id']:p for p in read(v3/'peaks.json')}
    pairs=read(out/'04-method-and-validation/peak-association.json')
    pairmap={tuple(sorted((p['peak_a'],p['peak_b']))):p for p in pairs}
    palette=read(out/'object-colors.json')['palette']
    input_paths=[out/q['folder']/'input-frames.json' for q in manifest['questions']]
    before={str(p):sha(p) for p in input_paths}
    page=(out/'index.html').read_text()
    backup=out/'index.before-cluster-gallery.html'
    if not backup.exists():shutil.copy2(out/'index.html',backup)
    page=re.sub(r'<!-- cluster-gallery-\d+:start -->.*?<!-- cluster-gallery-\d+:end -->','',page,flags=re.S)
    page=re.sub(r'<style id="cluster-gallery-style">.*?</style>','',page,flags=re.S)
    (out/'peak-montages').mkdir(exist_ok=True)
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',14)
    esc=lambda value:html.escape(str(value))
    peak_records=[]
    for g in groups:
        selected={r['source_frame_index'] for r in g['selected']}
        color=palette[g['object']]
        for pid in g['peak_ids']:
            p=peaks[pid];rows=p['selected'];canvas=Image.new('RGB',(320*len(rows),290),'#f2f4f7');d=ImageDraw.Draw(canvas)
            slots=[]
            for j,r in enumerate(rows):
                fid=r['source_frame_index'];source=out/'images'/f'frame{fid:07d}.jpg'
                assert sha(source)==r['frame_sha256']
                im=Image.open(source).convert('RGB');draw=ImageDraw.Draw(im)
                if r.get('box_xyxy'):draw.rectangle(r['box_xyxy'],outline=color,width=4)
                x=j*320;canvas.paste(im.resize((320,240)),(x,50))
                keep=fid in selected
                if keep:d.rectangle((x+2,50,x+317,287),outline=color,width=5)
                d.text((x+6,4),f'{r["timestamp_s"]:.2f}s | '+('KEEP' if keep else 'candidate'),font=font,fill='black')
                d.text((x+6,25),r['status'],font=font,fill='#985600' if r['status']=='context_fallback' else '#555555')
                slots.append({'source_frame_index':fid,'selected_for_group':keep,'status':r['status']})
            name=f'peak-montages/{pid}.jpg';canvas.save(out/name,quality=93)
            peak_records.append({'group_id':g['group_id'],'peak_id':pid,'image':name,'slots':slots})
    copies=0;question_counts={}
    reasons={'same_frame_distinct_do_not_merge':'同帧独立框，禁止合并',
             'sparse_geometry_conflict_defer':'稀疏几何冲突',
             'insufficient_independent_frame_pairs':'不足两对满足时间间隔要求的一对一匹配',
             'insufficient_multi_view_similarity':'外观双图匹配未达阈值',
             'two_independent_anchor_supported_matches':'配对通过，但未满足整组归并约束',
             'two_view_supported_matches':'外观配对通过，但未满足整组归并约束'}
    for q in manifest['questions']:
        qid=q['id'];local=read(out/q['folder']/'instance-groups.json');cards=[]
        for g in local:
            assert g['peak_ids']==gm[g['group_id']]['peak_ids']
            copies+=1;multi=len(g['peak_ids'])>1;proof=g.get('representative_evidence',{})
            status='多波峰实例假设' if multi else '单波峰：身份尚未确认'
            if proof.get('priority_only'):status+=' · 位姿证据支持（排序常数不是外观相似度）'
            elif multi and proof:
                status+=f' · 代表 {proof["representative"]} · 最弱支持 {proof["minimum_support"]:.3f} · 组均分 {proof["mean_pair_score"]:.3f}'
            evidence=[]
            for x,y in itertools.combinations(g['peak_ids'],2):
                pair=pairmap[tuple(sorted((x,y)))];matches=pair['matching']['matches'] if pair.get('matching') else []
                scores=', '.join(f'{m["left_frame"]/30:.2f}s ↔ {m["right_frame"]/30:.2f}s = {m["cosine"]:.3f}' for m in matches) or '无有效双图外观匹配'
                notes=[]
                if pair.get('pose_depth'):notes.append('三维点重投影支持')
                if pair.get('pose_keypoints'):notes.append('局部特征三视角支持')
                if pair.get('geometry_conflict_heuristic'):notes.append('稀疏几何冲突警告')
                if not pair['merge_supported']:notes.append('外观配对未通过；由几何或共同代表关联')
                evidence.append('<li>'+esc(f'{x} ↔ {y}：{scores}'+(' · '+' / '.join(notes) if notes else ''))+'</li>')
            outside=[p for p in pairs if (p['peak_a'] in g['peak_ids']) != (p['peak_b'] in g['peak_ids'])]
            closest=sorted(outside,key=lambda p:p['multi_view_score'] if p['multi_view_score'] is not None else -2,reverse=True)[:3]
            reviews=[]
            for p in closest:
                score=f'{p["multi_view_score"]:.3f}' if p['multi_view_score'] is not None else '无有效双图分数'
                reviews.append('<li>'+esc(f'{p["peak_a"]} ↔ {p["peak_b"]}：{score}；{reasons.get(p["reason"],p["reason"])}')+'</li>')
            strips=[]
            for pid in g['peak_ids']:
                peak=peaks[pid];image=f'peak-montages/{pid}.jpg'
                strips.append(f'<div class="cluster-peak"><strong>{esc(pid)}</strong> · 锚点 {peak["anchor_time_s"]:.2f}s<a href="{image}" target="_blank"><img class="cluster-montage" loading="lazy" src="{image}" width="1280" height="290" alt="{esc(pid)} 四帧候选，KEEP表示本组保留"></a></div>')
            warning='<p class="warning">此前人工抽查：chair-p08/p11 部分观测可能切换目标或混入背景，聚类未修正检测语义。</p>' if not manifest.get('detector_label') and set(g['peak_ids'])&{'chair-p08','chair-p11'} else ''
            cards.append(f'<article class="cluster-instance" data-cluster-group="{esc(g["group_id"])}" style="border-left-color:{palette[g["object"]]}"><h4>{esc(g["group_id"])}</h4><p>{esc(status)} · {len(g["peak_ids"])} 个波峰 → {g["selected_count"]} 帧，其中 {g["context_count"]} 帧为上下文补位。</p>{warning}'+
                         ('<details><summary>查看归并的配对证据</summary><ul>'+''.join(evidence)+'</ul></details>' if evidence else '')+
                         ('<details><summary>为什么未与其他波峰合并：最近候选</summary><ul>'+''.join(reviews)+'</ul></details>' if reviews else '')+''.join(strips)+'</article>')
        body='<h3>聚类图片结果</h3><p>每个卡片是一组实例假设，组内每行是一个原始波峰的四张候选图。彩色框标记目标；同色外边框和 KEEP 表示该帧被本组保留。组数不代表真实物体数量。</p>'+''.join(cards) if cards else '<h3>全局场景帧</h3><p>本题没有物体目标，因此不进行物体聚类。16张全局场景帧见下方图库。</p>'
        if not cards and q['objects']:
            body='<h3>缺少目标证据</h3><p>当前检测与波峰规则没有产生可用候选，未生成实例分组或伪造目标帧。</p>'
        block=f'<!-- cluster-gallery-{qid}:start --><div class="cluster-gallery" id="q{qid}-clusters">{body}</div><!-- cluster-gallery-{qid}:end -->'
        pattern=rf'(<section id="q{qid}">.*?)(<details><summary>查看实例假设分组)'
        page,n=re.subn(pattern,lambda m:m[1]+block+m[2],page,count=1,flags=re.S)
        assert n==1,f'Missing insertion point Q{qid}'
        question_counts[str(qid)]=len(cards)
    style='<style id="cluster-gallery-style">.cluster-instance{border:1px solid #d8dfe8;border-left:6px solid;padding:16px;margin:18px 0;border-radius:8px;background:#fafbfc}.cluster-instance h4{margin:0;font-size:20px}.cluster-peak{margin:18px 0}.cluster-montage{display:block;width:100%;height:auto;margin-top:6px}.cluster-instance details{margin:8px 0}.cluster-gallery{margin:24px 0}</style>'
    page=page.replace('</style>','</style>'+style,1)
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[];self.ids=set();self.cards=0;self.montages=0
        def handle_starttag(self,tag,attrs):
            d=dict(attrs)
            if 'id' in d:self.ids.add(d['id'])
            if 'data-cluster-group' in d:self.cards+=1
            if tag=='img' and d.get('class')=='cluster-montage':self.montages+=1
            self.links.extend(d[k] for k in ('src','href') if d.get(k))
    parser=Links();parser.feed(page)
    for link in parser.links:
        u=urlsplit(link)
        if u.scheme or u.netloc:continue
        if u.path:assert (out/unquote(u.path)).exists(),link
        elif u.fragment:assert u.fragment in parser.ids,link
    assert parser.cards==copies
    assert all(sha(Path(p))==h for p,h in before.items())
    (out/'index.html').write_text(page,encoding='utf-8')
    save(out/'peak-montages/manifest.json',peak_records)
    info={'unique_groups':len(groups),'unique_peak_montages':len(peak_records),'question_group_cards':copies,
          'question_peak_image_rows':parser.montages,'question_group_counts':question_counts,'selection_manifests_unchanged':True}
    manifest['cluster_gallery']=info;save(out/'manifest.json',manifest)
    validation=read(out/'validation.json');validation.update(cluster_gallery=info,html_links_checked=len(parser.links),broken_links=[])
    save(out/'validation.json',validation)
    shutil.copy2(Path(__file__),out/'04-method-and-validation/cluster_gallery.py')
    print(info)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True);p.add_argument('--v3',type=Path,required=True)
    a=p.parse_args();add_gallery(a.out,a.v3)
