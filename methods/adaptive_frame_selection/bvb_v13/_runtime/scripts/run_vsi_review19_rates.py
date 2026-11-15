"""Run three additional frame rates and compare with the completed 1 FPS run."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import html
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/vsi-review19-uniform-comparison-20261109'
RATES = [1, .5, .25, .125]


def read(p):
    return json.loads(p.read_text())


def report():
    OUT.mkdir(parents=True,exist_ok=True)
    priors=read(ROOT/'exports/vsi-review19-uniform-20261109/questions.json')
    results={}
    for rate in RATES:
        p=ROOT/f'results/vsi-review19-uniform-{rate:g}fps-20261109/answers.json'
        results[rate]={q['id']:q for q in read(p)} if p.exists() else {}
    summary={f'{r:g}':dict(completed=len(rows),expected=len(priors),mean_mra=sum(q['mra'] for q in rows.values())/len(rows) if rows else None) for r,rows in results.items()}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (OUT/'answers.json').write_text(json.dumps([q for r in RATES for q in results[r].values()],ensure_ascii=False,indent=2)+'\n')
    lines=['VSI 19 题 · 1 / 0.5 / 0.25 / 0.125 FPS 对比与判断依据','']
    table='<table><tr><th>题目</th><th>GT (cm)</th>'+''.join(f'<th>{r:g} FPS<br>答案 (cm) / MRA</th>' for r in RATES)+'</tr>'
    sections=[]
    for p in priors:
        qid=p['id']
        table+=f'<tr><td><a href="#q{qid}">Q{qid} {p["object_zh"]}</a></td><td>{p["ground_truth_cm"]:g}</td>'
        details=[]
        lines += [f'Q{qid} · {p["object_zh"]} · {p["scene_name"]}',p['question'],f'GT：{p["ground_truth_cm"]:g} cm']
        for rate in RATES:
            q=results[rate].get(qid)
            if not q:
                table+='<td>待完成</td>'
                continue
            table+=f'<td>{q["answer"]} / {q["mra"]:.2%}</td>'
            link=f'../vsi-review19-uniform-{rate:g}fps-20261109/index.html#q{qid}'
            details.append(f'<div class="result"><h3>{rate:g} FPS · {q["frame_count"]} 帧</h3><p>答案 <b>{q["answer"]} cm</b> · MRA <b>{q["mra"]:.2%}</b></p><p>{html.escape(q["reason"])}</p><p class="meta">模型自报证据充分：{"是" if q["sufficient_evidence"] else "否"} · <a href="{link}">查看全部输入帧</a></p></div>')
            lines += [f'{rate:g} FPS · {q["frame_count"]} 帧 · 答案 {q["answer"]} cm · MRA {q["mra"]:.2%}',q['reason'],f'模型自报证据充分：{q["sufficient_evidence"]}','']
        table+='</tr>'
        sections.append(f'<section id="q{qid}"><h2>Q{qid} · {p["object_zh"]}</h2><p class="meta">{p["scene_name"]} · GT {p["ground_truth_cm"]:g} cm</p><p>{html.escape(p["question"])}</p><div class="grid">{"".join(details)}</div></section>')
    table+='<tr><th>平均 MRA</th><td>—</td>'+''.join(f'<td>{summary[f"{r:g}"]["mean_mra"]:.2%}</td>' if summary[f'{r:g}']['mean_mra'] is not None else '<td>—</td>' for r in RATES)+'</tr></table>'
    (OUT/'reasoning.txt').write_text('\n'.join(lines))
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>19 题均匀抽帧四帧率对比</title><style>body{font:16px/1.7 system-ui;color:#192b3c;background:#f2f5f8;max-width:1400px;margin:24px auto;padding:20px}header,section{padding:24px;border-radius:12px;background:white;margin-bottom:24px}table{width:100%;border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #dce4eb;text-align:left}.scroll{overflow:auto}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px}.result{padding:16px;background:#f4f7fa;border-radius:8px}.meta{color:#536475;font-size:14px}a{color:#146c9e}</style><header><h1>19 题 · 均匀抽帧四帧率对比</h1><p>Astra high；只输入全视频均匀原帧和当前问题。各档使用相同提示词，仅帧率说明和帧数随输入变化。包含 review 全部 19 题；此前七题的 28 条结果复用，本次为其余 12 题新增 48 次作答。MRA 使用与 review 一致的 11 阈值。</p><p><a href="reasoning.txt">完整判断依据 TXT</a> · <a href="answers.json">结果 JSON</a></p><div class="scroll">'''+table+'</div></header>'+''.join(sections)+'</html>'
    (OUT/'index.html').write_text(page)
    return summary


def run_rate(rate):
    log=OUT/f'fps-{rate:g}.log'
    with log.open('w') as f:
        result=subprocess.run([sys.executable,str(ROOT/'scripts/rerun_vsi_low_mra_uniform.py'),'--fps',f'{rate:g}','--workers','1','--questions-file',str(ROOT/'exports/vsi-review19-uniform-20261109/questions.json'),'--suite','vsi-review19-uniform','--run'],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
    print('RATE_FINISHED',rate,'exit',result.returncode,flush=True)
    return rate,result.returncode


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    report()
    if args.run:
        failed=[]
        with ThreadPoolExecutor(max_workers=4) as pool:
            for f in as_completed([pool.submit(run_rate,r) for r in RATES]):
                rate,code=f.result()
                if code:failed.append(rate)
                report()
        if failed:raise SystemExit('Failed rates: '+str(failed))
    print(json.dumps(report(),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
