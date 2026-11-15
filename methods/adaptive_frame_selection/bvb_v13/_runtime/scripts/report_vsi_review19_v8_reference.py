"""Build a local evidence report from actual V8/Astra/Blender artifacts."""
import html
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/vsi-review19-v8-reference-20261109'
def read(p):return json.loads(p.read_text())
def esc(x):return html.escape(str(x),quote=True)
def rel(p):return esc(os.path.relpath(p,BASE))
def link(p,label):return f'<a href="{rel(p)}">{esc(label)}</a>'

def trend(delta):
    if delta > 1e-9:
        return f'<span class="trend up">↑ 上升 {delta*100:.2f} 个百分点</span>'
    if delta < -1e-9:
        return f'<span class="trend down">↓ 下降 {abs(delta)*100:.2f} 个百分点</span>'
    return '<span class="trend flat">→ 持平 0.00 个百分点</span>'

COMPARISON_HEAD='<th>正确答案 (cm)</th><th>原回答 · 均匀抽帧 (cm)</th><th>本次回答 (cm)</th><th>均匀抽帧 MRA</th><th>本次 MRA</th><th>MRA 变化</th>'

def comparison_cells(row):
    return (f'<td>{row["ground_truth_cm"]:g}</td><td data-baseline-answer="{row["id"]}">{row["prior_answer_cm"]:g}</td>'
            f'<td><b>{row["answer_cm"]:g}</b></td><td data-baseline-mra="{row["id"]}">{row["prior_mra"]:.2%}</td>'
            f'<td><b>{row["mra"]:.2%}</b></td><td data-baseline-trend="{row["id"]}">{trend(row["mra_delta"])}</td>')

def uniform_comparison(qa_rows):
    baselines={};summaries={}
    current={str(r['id']):{'answer_cm':r['answer_cm'],'mra':r['mra']} for r in qa_rows}
    for fps in (1,.5,.25,.125):
        key=f'{fps:g}';folder=ROOT/f'results/vsi-review19-uniform-{key}fps-20261109'
        source={str(r['id']):r for r in read(folder/'answers.json')}
        rows={}
        for q in qa_rows:
            qid=str(q['id']);r=source[qid]
            assert r['question']==q['question'] and float(r['ground_truth'])==q['ground_truth_cm']
            assert r['fps']==fps and r['reason'].strip()
            rows[qid]={'answer_cm':r['answer'],'mra':r['mra'],'reason':r['reason'],
                'frame_count':r['frame_count'],'sufficient_evidence':r['sufficient_evidence'],
                'source_page':os.path.relpath(folder/'index.html',BASE)+f'#q{qid}'}
        baselines[key]=rows
        deltas=[current[qid]['mra']-r['mra'] for qid,r in rows.items()]
        summaries[key]={'mean_mra':sum(r['mra'] for r in rows.values())/len(rows),
            'improved':sum(d>1e-9 for d in deltas),'unchanged':sum(abs(d)<=1e-9 for d in deltas),
            'decreased':sum(d < -1e-9 for d in deltas)}
    return {'default_fps':'1','baseline_protocol':'Full-video uniform original frames only',
        'current_protocol':'New reference reconstruction + selected V8 original frames',
        'current':current,'baselines':baselines,'summaries':summaries}


def main():
    qs=read(BASE/'questions.public.json');cards=[];complete=0;rulers=0
    old={str(q['id']):q for q in read(ROOT/'exports/vsi-review19-uniform-20261109/questions.json')}
    qa_rows=read(BASE/'qa/results.json') if (BASE/'qa/results.json').exists() else []
    qa_summary=read(BASE/'qa/summary.json') if (BASE/'qa/summary.json').exists() else None
    comparison=uniform_comparison(qa_rows) if qa_rows else None
    if comparison:
        (BASE/'qa/uniform-baseline-comparison.json').write_text(json.dumps(comparison,ensure_ascii=False,indent=2)+'\n')
        qa_rows=[dict(r,prior_answer_cm=comparison['baselines']['1'][str(r['id'])]['answer_cm'],
            prior_mra=comparison['baselines']['1'][str(r['id'])]['mra'],
            mra_delta=r['mra']-comparison['baselines']['1'][str(r['id'])]['mra']) for r in qa_rows]
    qa_by_id={str(r['id']):r for r in qa_rows}
    for q in qs:
        qid=str(q['id']);folder=BASE/'questions'/qid;prepared=folder/'prepared'
        decision=read(folder/'cluster-comparison/decision.json') if (folder/'cluster-comparison/decision.json').exists() else None
        result=read(folder/'result.json') if (folder/'result.json').exists() else None
        contents=''
        if result:
            complete+=1
            contents+=f'<p class="ok">已完成 · {result["source_clusters"]} 个候选 → {result["retained_clusters"]} 个目标聚类</p>'
        else:contents+='<p>处理中；仅展示已经落盘的实际结果。</p>'
        if decision:
            contents+=f'<p><b>Astra 选择：</b>{esc(decision["selected_group_id"])} · {esc(decision["status"])}</p><p>{esc(decision["reason"])}</p>'
            contents+='<details><summary>全部候选聚类的比较依据</summary><ul>'+''.join(f'<li><b>{esc(c["group_id"])}</b> · {esc(c["assessment"])}<br>{esc(c["reason"])}</li>' for c in decision['clusters'])+'</ul></details>'
        if qid in qa_by_id:
            qa=qa_by_id[qid]
            prior_reason=comparison['baselines']['1'][qid]
            original_reason=esc(prior_reason['reason'])
            reasoning_comparison=(f'<div class="reasoning-comparison"><div class="prior-reason"><h4>原 reasoning · <span class="baseline-fps">1</span> FPS 均匀抽帧</h4><p data-baseline-reason="{qid}">{original_reason}</p><p>全视频均匀抽帧 · <span data-baseline-count="{qid}">{prior_reason["frame_count"]}</span> 帧 · <a data-baseline-link="{qid}" href="{esc(prior_reason["source_page"])}">查看该轮完整输入与结果</a></p></div>'
                f'<div class="current-reason"><h4>本次 reasoning · 判断依据</h4><p>{esc(qa["reason"])}</p><p><b>本次不确定性：</b>{esc(qa["limitations"])}</p></div></div>')
            contents+=f'<div class="qa-answer"><h3>本次新重构 QA</h3><div class="table-scroll"><table class="answer-comparison"><thead><tr>{COMPARISON_HEAD}</tr></thead><tbody><tr>{comparison_cells(qa)}</tr></tbody></table></div>{reasoning_comparison}<p>模型自报：置信度 {esc(qa["confidence"])}；证据充分：{"是" if qa["sufficient_evidence"] else "否"}。输入：2 张重构图 + {qa["original_frame_count"]} 张该聚类的 V8 原帧。</p><p>'+link(BASE/'qa/answers'/f'{qid}.json','本题新回答 JSON')+' · '+link(BASE/'qa/inputs'/f'{qid}.json','本题 QA 输入清单')+' · '+link(BASE/'qa/requests'/qid/'response.txt','模型原始回答')+'</p></div>'
        manifest=read(prepared/'manifest.json') if (prepared/'manifest.json').exists() else None
        if manifest:
            for g in manifest['groups']:
                gf=prepared/g['folder'];auditpath=gf/'assets/question-audit.json'
                audit=read(auditpath) if auditpath.exists() else {}
                ref=audit.get('scale_reference')
                if ref:
                    contents+=f'<aside><b>参照物：{esc(ref["object"][1])}</b> · 假设长度 {ref["assumed_length_m"]:g} m · {esc(ref["confidence"])} confidence<br>{esc(ref["rationale"])}<br><b>不确定性：</b>{esc(ref["uncertainty"])}<br>来源帧：{esc(ref["source_timestamps_s"])} s。米尺依赖此估计；不是实测标定。</aside>'
                elif audit.get('reference_unavailable_reason'):
                    contents+=f'<aside class="warn">未找到可靠参照物，未生成米尺：{esc(audit["reference_unavailable_reason"])}</aside>'
                merged=folder/'build/merged'
                if (merged/'reference-ruler.png').exists():
                    rulers+=1
                    contents+=f'<figure class="render"><a href="{rel(merged/"reference-ruler.png")}"><img loading="lazy" src="{rel(merged/"reference-ruler.png")}"></a><figcaption>目标红色，参照物蓝色；参照物边缘标注米尺。此视图隐藏其他上下文，完整模型保留墙体与支撑物。</figcaption></figure>'
                if (merged/'reference-detail.png').exists():
                    contents+=f'<details><summary>放大参照物与米尺刻度</summary><figure class="render"><a href="{rel(merged/"reference-detail.png")}"><img loading="lazy" src="{rel(merged/"reference-detail.png")}"></a><figcaption>仅放大参照物；刻度以米（m）为单位。</figcaption></figure></details>'
                if (merged/'merged-oblique.png').exists():
                    contents+=f'<details><summary>完整重构场景（含墙体与支撑物）</summary><figure class="render"><a href="{rel(merged/"merged-oblique.png")}"><img loading="lazy" src="{rel(merged/"merged-oblique.png")}"></a></figure></details>'
                assets=[(merged/'scene-with-ruler.blend','下载带米尺的 Blender 场景'),(merged/'merged-scene.blend','下载原始重构场景'),(gf/'assets/scene-spec.json','场景 JSON'),(merged/'scale-reference.json','米尺与尺度记录'),(gf/'assets/response.txt','Astra 重构输出')]
                contents+='<p class="links">'+' · '.join(link(p,label) for p,label in assets if p.exists())+'</p>'
                frames=read(gf/'input-frames.json')
                contents+=f'<h3>实际输入 Astra 的 V8 抽帧 · {len(frames)} 帧</h3><div class="frames">'
                for f in frames:
                    path=Path(f['path'])
                    contents+=f'<figure><a href="{rel(path)}"><img loading="lazy" src="{rel(path)}"></a><figcaption>{f["timestamp_s"]:g}s · frame {f["source_frame_index"]}</figcaption></figure>'
                contents+='</div>'
        # Keep all rejected candidate images available for review as well.
        delivery=BASE/'inputs'/q['scene_name']/'delivery'/f'q{qid}-frames.json'
        if delivery.exists():
            packet=read(delivery);contents+='<details><summary>全部 V8 候选聚类抽帧（含未选中）</summary><div class="frames">'
            for f in packet['frames']:
                path=delivery.parent/f['image']
                contents+=f'<figure><a href="{rel(path)}"><img loading="lazy" src="{rel(path)}"></a><figcaption>{esc(f["group"])} · {f["timestamp_s"]:g}s</figcaption></figure>'
            contents+='</div></details>'
        previous=old[qid]
        qa_note='旧结果仅用于对照，未输入本次重构或 QA。' if qa_summary else '本轮未重新进行 QA。'
        contents+=f'<details><summary>更早的 review（历史记录，非当前均匀抽帧基线）</summary><p>正确答案 {previous["ground_truth_cm"]:g} cm；原 VLM 回答 {previous["model_answer_cm"]:g} cm；原 MRA {previous["mra_percent_reported"]:g}%。{qa_note}</p></details>'
        video=ROOT/'upstream/VSI-Bench/scannetpp'/f'{q["scene_name"]}.mp4'
        contents+=f'<details><summary>原视频 · {esc(q["scene_name"])}</summary><video controls preload="none" src="{rel(video)}"></video></details>'
        cards.append(f'<section data-scene="{q["scene_name"]}" data-search="{esc(qid+" "+q["objects"][0]+" "+previous["object_zh"])}" id="q{qid}"><h2>Q{qid} · {esc(q["objects"][0])}</h2><p>{esc(q["question"])}</p><p class="muted">VSI · {esc(q["scene_name"])}</p>{contents}</section>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>19 题 · V8 单聚类与参照米尺重构</title><style>
body{margin:auto;max-width:1320px;padding:24px;background:#f2f5f9;color:#172535;font:16px/1.65 system-ui}header,section{background:white;padding:28px;border-radius:14px;margin:20px 0}h1{line-height:1.3}h2{margin:0}.muted{color:#607082}.ok{color:#197342}a{color:#1565ad}nav{display:flex;flex-wrap:wrap;gap:14px}.frames{display:grid;grid-template-columns:repeat(auto-fill,minmax(265px,1fr));gap:12px;margin:16px 0}figure{margin:0}img,video{max-width:100%;height:auto}figcaption{font-size:13px;color:#56616d}.render{max-width:920px;margin:24px auto}aside{background:#edf5ff;padding:18px;border-left:4px solid #377fc7;margin:20px 0}.warn{background:#fff4df}summary{cursor:pointer;padding:10px 0;font-weight:600}details{border-top:1px solid #e7ecf2;margin-top:12px}li{margin:12px 0}.toolbar{position:sticky;top:0;z-index:2;background:#eef3f9f5;padding:12px 18px;display:flex;gap:20px;align-items:center;flex-wrap:wrap;border-radius:10px}input,select{font:inherit;border:1px solid #b9c8d8;border-radius:7px;padding:8px;margin-left:8px}html{scroll-behavior:smooth;scroll-padding-top:100px}.links{display:flex;gap:16px;flex-wrap:wrap}
</style>'''
    toolbar='<div class="toolbar"><label>搜索题号 / 物体 <input id="search" placeholder="例如 2341、显示器、monitor"></label><label>场景 <select id="scene"><option value="">全部 3 个场景</option>'+''.join(f'<option>{sid}</option>' for sid in sorted({q["scene_name"] for q in qs}))+'</select></label><span id="shown">19 题</span></div>'
    qa_intro=''
    if qa_summary:
        qa_intro=f'<div class="qa-answer"><h2>新重构 QA · {qa_summary["completed"]}/19 题</h2><p>Astra high · 输入：新目标与米尺图、米尺近景、该聚类实际选中的 V8 原帧。正确答案与旧回答未进入模型输入。</p>'
        if qa_rows:
            base_summary=comparison['summaries']['1']
            qa_intro+='<p><label>原结果：全视频均匀抽帧 <select id="baseline-fps">'+''.join(f'<option value="{fps}">{fps} FPS</option>' for fps in comparison['baselines'])+'</select></label> · 切换后，原回答、原 MRA、原 reasoning 和上升／下降标记同步更新。</p>'
            qa_intro+=f'<p>本次平均 MRA <b>{qa_summary["mean_mra"]:.2%}</b> · 均匀抽帧 <span class="baseline-fps">1</span> FPS <b id="baseline-mean">{base_summary["mean_mra"]:.2%}</b> · <span id="baseline-counts">{base_summary["improved"]} 题上升 / {base_summary["unchanged"]} 题持平 / {base_summary["decreased"]} 题下降</span>。</p>'
            qa_intro+='<div class="table-scroll"><table class="answer-comparison"><thead><tr><th>题号 / 物体</th>'+COMPARISON_HEAD+'</tr></thead><tbody>'+''.join(f'<tr><td><a href="#q{r["id"]}">Q{r["id"]} · {esc(r["object_zh"])}</a></td>{comparison_cells(r)}</tr>' for r in qa_rows)+'</tbody></table></div>'
        qa_intro+='<p>MRA 沿用原 review 的 11 个阈值（linspace(0.5, 0.95, 11)）。本次重构、参照物、抽帧和提示词均有变化，分数对照不是单因素实验。</p><p>'+link(BASE/'qa/results.json','原始 QA 结果 JSON（历史 review 字段保留）')+' · '+link(BASE/'qa/reasoning.txt','本次判断依据 TXT')+' · '+(link(BASE/'qa/uniform-baseline-comparison.json','均匀抽帧四档对照与 reasoning JSON')+' · ' if comparison else '')+link(BASE/'qa/experiment.json','QA 配置')+'</p></div>'
    page+='<style>.qa-answer{background:#eef8f2;border:1px solid #c6e1d0;border-radius:10px;padding:20px;margin:20px 0}.qa-answer h3{margin-top:0}.reasoning-comparison{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:16px;margin:18px 0}.reasoning-comparison>div{border-radius:8px;padding:16px;background:#fff}.reasoning-comparison h4{margin:0 0 12px;font-size:16px}.reasoning-comparison p{white-space:pre-wrap;overflow-wrap:anywhere}.reasoning-comparison .prior-reason{background:#f1f4f8;border-left:3px solid #94a3b8}.reasoning-comparison .current-reason{border-left:3px solid #42966a}@media(max-width:760px){.reasoning-comparison{grid-template-columns:1fr}}.table-scroll{overflow-x:auto;margin:14px 0}.trend{display:inline-block;padding:3px 9px;border-radius:6px;font-weight:650}.trend.up{color:#166534;background:#dcfce7}.trend.down{color:#991b1b;background:#fee2e2}.trend.flat{color:#475569;background:#e2e8f0}.answer-comparison td{font-variant-numeric:tabular-nums}table{width:100%;border-collapse:collapse;white-space:nowrap}th,td{padding:8px 12px;text-align:left;border-bottom:1px solid #ccdfd3}</style>'
    page+=f'<header><h1>19 题 · V8 单聚类与参照米尺重构</h1><p>已完成 {complete}/19 题；已生成米尺 {rulers}/19 题。</p><p>本次开启 <code>--object-size-reference-mode</code>：2 FPS 候选帧 → V8 峰值与几何/外观聚类 → Astra 比较并保留一个聚类 → 单目标与可见参照物重构 → 最终实例审核 → Blender 米尺。</p><p>使用 V8 简化几何资产：红色为目标，蓝色为参照物。参照物尺寸来自明确记录的视觉先验假设，米尺不是实测标定。原题答案与 MRA 仅在报告中展示，没有送入重构模型。</p><nav>'+''.join(f'<a href="#q{q["id"]}">Q{q["id"]}</a>' for q in qs)+'</nav></header>'+toolbar+''.join(cards)
    page=page.replace('</nav></header>','</nav>'+qa_intro+'</header>')
    page+='''<script>const cards=[...document.querySelectorAll('section')],query=document.querySelector('#search'),scene=document.querySelector('#scene');function filter(){let n=0;for(const card of cards){const ok=(!scene.value||card.dataset.scene===scene.value)&&card.dataset.search.toLowerCase().includes(query.value.trim().toLowerCase());card.hidden=!ok;n+=ok;}document.querySelector('#shown').textContent=n+' / 19 题';}query.addEventListener('input',filter);scene.addEventListener('change',filter);document.querySelectorAll('a[href^="#q"]').forEach(a=>a.addEventListener('click',()=>{query.value='';scene.value='';filter();}));</script>'''
    if comparison:
        page+='<script type="application/json" id="uniform-baselines">'+json.dumps(comparison,ensure_ascii=False).replace('<','\\u003c')+'</script>'
        page+='''<script>
const baselineData=JSON.parse(document.getElementById('uniform-baselines').textContent);
const baselineSelect=document.getElementById('baseline-fps');
function baselineTrend(delta){
  if(delta>1e-9)return '<span class="trend up">↑ 上升 '+(delta*100).toFixed(2)+' 个百分点</span>';
  if(delta < -1e-9)return '<span class="trend down">↓ 下降 '+(-delta*100).toFixed(2)+' 个百分点</span>';
  return '<span class="trend flat">→ 持平 0.00 个百分点</span>';
}
function updateBaseline(){
  const fps=baselineSelect.value,rows=baselineData.baselines[fps],summary=baselineData.summaries[fps];
  document.querySelectorAll('.baseline-fps').forEach(el=>el.textContent=fps);
  document.querySelectorAll('[data-baseline-answer]').forEach(el=>el.textContent=rows[el.dataset.baselineAnswer].answer_cm);
  document.querySelectorAll('[data-baseline-mra]').forEach(el=>el.textContent=(rows[el.dataset.baselineMra].mra*100).toFixed(2)+'%');
  document.querySelectorAll('[data-baseline-trend]').forEach(el=>{const id=el.dataset.baselineTrend;el.innerHTML=baselineTrend(baselineData.current[id].mra-rows[id].mra);});
  document.querySelectorAll('[data-baseline-reason]').forEach(el=>el.textContent=rows[el.dataset.baselineReason].reason);
  document.querySelectorAll('[data-baseline-count]').forEach(el=>el.textContent=rows[el.dataset.baselineCount].frame_count);
  document.querySelectorAll('[data-baseline-link]').forEach(el=>el.setAttribute('href',rows[el.dataset.baselineLink].source_page));
  document.getElementById('baseline-mean').textContent=(summary.mean_mra*100).toFixed(2)+'%';
  document.getElementById('baseline-counts').textContent=summary.improved+' 题上升 / '+summary.unchanged+' 题持平 / '+summary.decreased+' 题下降';
}
baselineSelect.addEventListener('change',updateBaseline);
</script>'''
    (BASE/'index.html').write_text(page)
    print(f'Report: {complete}/19 reconstructions, {rulers}/19 rulers')


if __name__=='__main__':main()
