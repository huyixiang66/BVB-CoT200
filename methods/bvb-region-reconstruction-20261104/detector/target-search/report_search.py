"""Paired diagnostic metrics and original-image overlays, with cache provenance."""
from html import escape
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
D = HERE.parent
sys.path.insert(0,str(D))
from evaluate_pilot import iou, REFERENCES, NEGATIVES


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()]


def rect(box,color):
    x,y,xx,yy=box
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{xx-x:.1f}" height="{yy-y:.1f}" fill="none" stroke="{color}" stroke-width="3"/>'


def evaluate_label(data,label):
    refs=REFERENCES[label]
    counts=dict(positives=len(refs),hit_at_05=0,best_box_hit_at_05=0,
                negatives=len(NEGATIVES[label]),negative_frames_with_fp=0)
    for fid in list(refs)+NEGATIVES[label]:
        boxes=sorted([d for d in data[fid] if d['object']==label],key=lambda d:d['score'],reverse=True)
        if fid in refs:
            overlaps=[iou(d['box_xyxy'],[v*2 for v in refs[fid]]) for d in boxes]
            counts['hit_at_05']+=int(max(overlaps,default=0)>=.5)
            counts['best_box_hit_at_05']+=int(bool(overlaps) and overlaps[0]>=.5)
        else:
            counts['negative_frames_with_fp']+=int(bool(boxes))
    return dict(summary=counts)


def main():
    assert "SEARCH_DONE" in (HERE/"search.log").read_text(encoding="utf-8")
    panel=read(HERE/"panel/candidates.json")
    images={r["source_frame_index"]:r for r in panel}
    conditions={}
    for name,path in [("Tiny 原词表",D/"tiny/base.jsonl"),("Tiny 加 built-in",D/"tiny-built-in/aliases.jsonl")]:
        data=rows(path)
        for field in ["normal","loose"]:
            conditions[name+" / "+field]={r["source_frame_index"]:r[field] for r in data}
    # The existing local Tiny cache supplies only its original thresholds for additional frames.
    cache=D.parent.parent/"code-selected-reconstruction-20261103/inputs/cache"
    tiny={r["source_frame_index"]:r["detections"] for r in rows(cache/"detections.jsonl")}
    conditions["Tiny 原词表 / normal"].update({fid:tiny[fid] for fid in images if fid in tiny})
    for tag,old in [("base","base"),("mm","mm-base")]:
        title="Base" if tag=="base" else "MM Base"
        data=rows(D/old/"base.jsonl")+rows(HERE/tag/"base-delta.jsonl")
        for field in ["normal","loose"]:
            conditions[title+" 原词表 / "+field]={r["source_frame_index"]:r[field] for r in data}
        for mode,note in [("built-in","加 built-in"),("focus-refrigerator","只提示 refrigerator"),("focus-built-in","只提示 built-in refrigerator")]:
            data=rows(HERE/tag/(mode+".jsonl"))
            for field in ["normal","loose"]:
                conditions[title+" "+note+" / "+field]={r["source_frame_index"]:r[field] for r in data}
    metrics={}
    for name,data in conditions.items():
        tested=['refrigerator'] if '只提示' in name else ['refrigerator','blanket']
        metrics[name]={label:evaluate_label(data,label) for label in tested}
    write=lambda p,v:p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    write(HERE/"metrics.json",metrics)
    write(HERE/"detections-combined.json",conditions)
    page=['<!doctype html><html lang="zh"><meta charset="utf-8">',
      '<meta name="viewport" content="width=device-width,initial-scale=1">',
      '<title>1585 / 1758 · 目标搜寻对照</title>',
      '<style>body{font:16px/1.6 system-ui;max-width:1600px;margin:28px auto;padding:0 20px;color:#17212b}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:8px;border-bottom:1px solid #ddd}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:10px}figure{margin:0;border:1px solid #ddd}figcaption,figure p{padding:6px;margin:0;font-size:13px}svg{width:100%;display:block}.note{background:#f3f6f8;padding:14px}a{color:#275e95}</style>',
      '<h1>1585 / 1758：目标搜寻对照</h1>',
      '<p>本轮只检查检测与搜寻，不改重建、不跑QA。冰箱为微波炉下方灰色柜门；毯子为卧室床上的花纹床毯。</p>',
      '<p class="note">原50张诊断图全部复用，另补13张原入选／邻域图，共63张。Base和MM补跑加提示词；同时比较仅提示单一冰箱类别。',
      '常规文本均值／token峰值阈值0.20／0.25，放宽0.12／0.17。四种提示：原20类、原20类加built-in、单独refrigerator、单独built-in refrigerator。',
      '单类提示仅在同一25张冰箱诊断图运行，不作为全视频召回率。绿色人工框只用于评价，不参与模型输入、筛选或重建。</p>',
      '<nav><a href="#fridge">真冰箱与误检对照</a> · <a href="#neighbors">115秒邻域</a> · <a href="#blanket">1758毯子</a> · <a href="#mouse">1758鼠标</a> · <a href="REPORT.md">结论</a></nav>',
      '<table><tr><th>条件</th><th>冰箱任一框命中</th><th>冰箱最高分命中</th><th>冰箱负帧误报</th><th>毯子最高分命中</th><th>毯子负帧误报</th></tr>']
    for name,data in metrics.items():
        a=data["refrigerator"]["summary"];b=data.get("blanket",{}).get("summary")
        vals=[f'{a["hit_at_05"]}/{a["positives"]}',f'{a["best_box_hit_at_05"]}/{a["positives"]}',f'{a["negative_frames_with_fp"]}/{a["negatives"]}',f'{b["best_box_hit_at_05"]}/{b["positives"]}' if b else '未测',f'{b["negative_frames_with_fp"]}/{b["negatives"]}' if b else '未测']
        page.append('<tr><td>'+escape(name)+'</td>'+''.join('<td>'+v+'</td>' for v in vals)+'</tr>')
    page.append('</table><p>命中为IoU≥0.50。红框为该类别最高分预测，黄／紫为第2／3框；没有绿色框的邻域或鼠标图为未量化观察，不自动记负例。</p>')

    def show(fid,label,names):
        row=images[fid]
        relative=os.path.relpath(row['frame'],HERE).replace(os.sep,'/')
        page.append(f'<h3>{row["timestamp_s"]:.2f}s · frame {fid}</h3><div class="grid">')
        for name in names:
            data=conditions[name]
            if fid not in data:
                continue
            boxes=sorted([d for d in data[fid] if d['object']==label],key=lambda r:r['score'],reverse=True)[:3]
            gt=REFERENCES.get(label,{}).get(fid)
            overlay=rect([v*2 for v in gt],'#46bd73') if gt else ''
            overlay+=''.join(rect(d['box_xyxy'],color) for d,color in zip(boxes,['#e63b35','#da9a1b','#b638ba']))
            captions=' · '.join(f'{d.get("detection_alias",label)} {d["score"]:.3f}' for d in boxes) or '无该类别检测框'
            page.append(f'<figure><figcaption>{escape(name)}</figcaption><svg viewBox="0 0 640 480"><image href="{escape(relative,quote=True)}" width="640" height="480"/>{overlay}</svg><p>{escape(captions)}</p></figure>')
        page.append('</div>')
    names=['Tiny 原词表 / normal','Tiny 加 built-in / normal','Base 原词表 / normal','Base 加 built-in / normal','MM Base 原词表 / normal','MM Base 加 built-in / normal']
    page.append('<h2 id="fridge">1585 · 真冰箱与易混淆负例</h2>')
    for fid in [3420,3450,4080,3960]:show(fid,'refrigerator',names)
    page.append('<details><summary>单类别提示及放宽阈值</summary>')
    for fid in [3450,3960]:show(fid,'refrigerator',['Base 只提示 refrigerator / normal','Base 只提示 built-in refrigerator / normal','MM Base 只提示 refrigerator / normal','MM Base 只提示 built-in refrigerator / normal','Base 加 built-in / loose','MM Base 加 built-in / loose'])
    page.append('</details><h2 id="neighbors">115秒波峰附近：逐帧复检</h2><p>该锚点来自已有完整238帧MM发现序列。下面核查的是邻域帧检测，不代表已经复跑全部4 FPS波峰选择。</p>')
    for fid in [3435,3442,3450,3458]:show(fid,'refrigerator',['Base 原词表 / normal','Base 加 built-in / normal','MM Base 原词表 / normal','MM Base 加 built-in / normal'])
    page.append('<h2 id="blanket">1758 · 卧室毯子与浴室毛巾误检</h2>')
    for fid in [4800,5580,330]:show(fid,'blanket',['Tiny 原词表 / normal','Base 原词表 / normal','MM Base 原词表 / normal','Base 原词表 / loose'])
    page.append('<h2 id="mouse">1758 · 鼠标候选：160 / 166秒，以及195秒耳机混淆</h2><p>本节只看框位置，不以原Tiny框作为真实标注，不报告鼠标准确率。</p>')
    for fid in [4800,4980,5850]:show(fid,'computer mouse',['Tiny 原词表 / normal','Base 原词表 / normal','MM Base 原词表 / normal','Base 原词表 / loose','MM Base 原词表 / loose'])
    page.append('</html>')
    (HERE/'review.html').write_text('\n'.join(page),encoding='utf-8')
    for name,data in metrics.items():
        a=data['refrigerator']['summary'];b=data.get('blanket',{}).get('summary',{})
        print(name,a['best_box_hit_at_05'],a['negative_frames_with_fp'],'blanket',b.get('best_box_hit_at_05'),b.get('negative_frames_with_fp'))


if __name__=='__main__':main()
