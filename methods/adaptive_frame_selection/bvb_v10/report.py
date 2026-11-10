"""Portable HTML with source clusters, rendered scene, chosen view, and QA receipts."""
import html,json,shutil,hashlib
from pathlib import Path
from .pipeline import read
E=lambda x:html.escape(str(x),quote=True)
def render(out,reference=None):
 out=out.resolve();page=out/'report';assets=page/'assets';assets.mkdir(parents=True,exist_ok=True)
 def get(p,default=None):return read(p) if p.exists() else default
 def copy(p):
  h=hashlib.sha256(p.read_bytes()).hexdigest();dest=assets/(h+p.suffix)
  if not dest.exists():shutil.copy2(p,dest)
  return 'assets/'+dest.name
 def fig(p,label,box=None,color='#e33'):
  if not p.exists():return '<p>图片尚未生成</p>'
  overlay=''
  if box:
   x,y,x2,y2=box;overlay=f'<svg viewBox="0 0 640 480"><rect x="{x}" y="{y}" width="{x2-x}" height="{y2-y}" fill="none" stroke="{color}" stroke-width="3"/></svg>'
  return f'<figure><a href="{copy(p)}"><img loading="lazy" src="{copy(p)}">{overlay}</a><figcaption>{E(label)}</figcaption></figure>'
 selection=get(out/'audit/input/review/reviewed-selection.json',{})
 cache=Path(selection.get('inputs',{}).get('cache','.'));cand={r['source_frame_index']:r for r in get(cache/'candidates.json',[])}
 refs=get(reference,{}) if reference else {};body=[]
 for qid,row in selection.get('questions',{}).items():
  q=row['question'];folder=out/'questions'/qid;status=get(folder/'status.json',{'stage':'pending'});answer=get(folder/'qa/answer.json');views=get(folder/'views/views.json',{}).get('views',[]);choice=get(folder/'view-selection/selection.json');colors=dict(zip(q['objects'],['#d93636','#09965c','#2563eb','#9b44c4','#ef8b12']))
  text=f'<article id="q{E(qid)}"><h2>Q{E(qid)} · {E(status["stage"])}</h2><pre>{E(q["question"])}\n{E(chr(10).join(q.get("options",[])))}</pre>'
  if status.get('error'):text+='<pre>'+E(status['error'])+'</pre>'
  if views:
   text+='<div class="grid">'+fig(folder/'views'/next(v['image'] for v in views if v['id']=='top'),'完整场景 · 世界Z轴俯视')
   selected=next((v for v in views if choice and v['id']==choice['view_id']),None)
   if selected:text+=fig(folder/'views'/selected['image'],'QA选中视角：'+selected['id'])
   text+='</div>'
  if answer:text+='<h3>单图 QA：'+E(answer['option']+'. '+answer['answer'])+'</h3><p>'+E(answer['reason'])+'</p><p>'+E(answer.get('limitations',''))+'</p>'
  if qid in refs:text+='<p>展示用 GT：'+E(refs[qid])+'；得分：'+str(int(answer['option']==refs[qid]) if answer else '待作答')+'</p>'
  for f in ['scene.blend','scene-with-reference-cameras.blend','generated.py']:
   if (folder/f).exists():text+=f'<a href="{copy(folder/f)}">{f}</a> · '
  for arm,title in [('roi_robust','候选聚类'),('semantic_selected','Astra筛选后聚类')]:
   text+='<details '+('open' if arm=='semantic_selected' else '')+'><summary>'+title+'</summary>'
   for g in row['arms'].get(arm,[]):
    text+='<h4>'+E(g['group_id'])+'</h4><p>'+E(g.get('semantic_review',{}).get('status','未审核'))+'</p><div class="grid">'
    for f in g['selected']:
     r=cand[f['source_frame_index']];text+=fig(cache/r['frame'],f'{f["timestamp_s"]:.2f}s · {g["object"]} · score {f.get("dino_score",f.get("score",""))}',f.get('box_xyxy'),colors.get(g['object'],'#e33'))
    text+='</div>'
   text+='</details>'
  body.append(text+'</article>')
 text='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>BVB V10</title><style>body{font:16px/1.6 system-ui;background:#eef2f6;max-width:1300px;margin:auto;padding:20px}article{background:white;padding:24px;margin:20px 0;border-radius:12px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}figure{margin:0}figure a{display:block;position:relative}img{width:100%;display:block}svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}pre{white-space:pre-wrap}summary{cursor:pointer}a{color:#2563eb}</style><h1>BVB V10 · 全流程结果</h1><p>原图/聚类 → Astra审核 → 联合位姿约束重构（第0轮）→ 九视图选一 → 单图QA。几何准确性不由结构校验保证。</p>'+''.join(body)+'</html>'
 tmp=page/'index.tmp';tmp.write_text(text);tmp.replace(page/'index.html')
