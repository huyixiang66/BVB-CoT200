"""Fresh seven-question 1 FPS inference, with public evidence explanations."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo
import numpy as np

from subscription_direct_client import request, image_item

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/vsi-three-scenes-subscription-fps-20261107'
PRIOR = ROOT / 'exports/vsi-low-mra-v8-frames-20261109/questions.json'
FPS = 1.0
OUT = ROOT / 'results/vsi-low-mra-uniform-1fps-20261109'
PROMPT = '''Answer the object-size question using ONLY the supplied original video frames.
The frames cover the full video at 1 FPS, in chronological order; each timestamp precedes one image.
Estimate the longest dimension of the target in centimeters. No metric scale or dimensions are supplied.
Return ONLY a JSON object with keys:
"answer": a nonnegative number in centimeters;
"reason": a concise Chinese explanation of the visible evidence supporting your estimate (at most 300 Chinese characters), including relevant timestamps where possible, reference objects or size assumptions, and the main uncertainty;
"sufficient_evidence": a boolean.
If evidence is insufficient, still give your best estimate and set sufficient_evidence to false.
Provide a brief evidence-based explanation, not a detailed internal thought process.'''


def score(answer, question):
    target = float(question['ground_truth'])
    return float((abs(float(answer)-target)/target <= 1-np.linspace(.5,.95,11)).mean())


def read(p):
    return json.loads(p.read_text())


def save(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def prepare():
    prior = read(PRIOR)
    targets = {q['id']:q for q in read(SOURCE/'evaluation-targets.json')}
    rows = [targets[p['id']] for p in prior]
    assert rows and len(rows) == len({q["id"] for q in rows})
    save(OUT/'evaluation-targets.json', rows)
    save(OUT/'questions-public.json', [{k:q[k] for k in ('id','question','scene_name','question_type')} for q in rows])
    for sid in dict.fromkeys(q['scene_name'] for q in rows):
        original = read(SOURCE/sid/'sampling.json')
        source = read(SOURCE/sid/f'fps-{FPS:g}/sampling.json')
        assert source['fps'] == FPS and source['video_sha256'] == original['video_sha256']
        assert hashlib.sha256(Path(original['video_path']).read_bytes()).hexdigest() == source['video_sha256']
        frames = []
        for i, f in enumerate(source['frames']):
            assert f['source_frame_index'] == i*round(30/FPS) and f['timestamp_s'] == i/FPS
            path = Path(f['path'])
            assert hashlib.sha256(path.read_bytes()).hexdigest() == f['sha256']
            dest = OUT/'frames'/sid/path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                shutil.copy2(path, dest)
            frames.append(dict(f, path=str(dest.relative_to(OUT))))
        assert len(frames) == math.ceil(source['duration_s']*FPS)
        save(OUT/'sampling'/f'{sid}.json', dict(source, frames=frames))
    save(OUT/'experiment.json', dict(model='gpt-6-astra', effort='high', fps=FPS,
         created_at=datetime.now(ZoneInfo('America/New_York')).isoformat(), question_ids=[q['id'] for q in rows],
         prompt=PROMPT, inputs='Original uniform video frames and current public question only.',
         fresh_requests=True, prior_answers_sent=False, ground_truth_sent=False,
         explanation='Model public reason output; not hidden chain-of-thought.',
         scoring='Same 11 NumPy confidence thresholds as supplied review: linspace(.5,.95,11). Verified against all seven prior review scores.',
         confidence_thresholds=np.linspace(.5,.95,11).tolist(),
         comparison_note='Prior review uses reconstruction plus V8 frames; this run uses uniform original frames only. Both visual inputs and explanation prompt differ.'))
    return rows


def infer(q):
    cfg = read(OUT/'sampling'/f'{q["scene_name"]}.json')
    content = [{'type':'input_text','text':PROMPT+f'\nDuration: {cfg["duration_s"]:.6f} seconds. Frames: {len(cfg["frames"])}.'}]
    for f in cfg['frames']:
        content += [{'type':'input_text','text':f'Timestamp: {f["timestamp_s"]:g} s'}, image_item(OUT/f['path'])]
    content.append({'type':'input_text','text':'Question: '+q['question']})
    response = request(OUT/'requests'/str(q['id']), content, model='gpt-6-astra', effort='high')
    raw = response['answer'].strip()
    if raw.startswith('```'):
        raw = raw.split('\n',1)[1].rsplit('```',1)[0]
    answer = json.loads(raw)
    assert type(answer['answer']) in (int,float) and math.isfinite(answer['answer']) and answer['answer'] >= 0
    assert isinstance(answer['reason'],str) and answer['reason'].strip()
    assert type(answer['sufficient_evidence']) is bool
    result = dict(q, **answer, fps=FPS, frame_count=len(cfg['frames']),
                  mra=score(answer['answer'],q), usage=response['usage'], response_id=response['response_id'])
    save(OUT/'answers'/f'{q["id"]}.json',result)
    print('QUESTION_COMPLETED', q['id'], 'answer',answer['answer'],'MRA',result['mra'],flush=True)
    return result


def report():
    priors = read(PRIOR)
    rows = [read(OUT/'answers'/f'{p["id"]}.json') for p in priors if (OUT/'answers'/f'{p["id"]}.json').exists()]
    for q in rows:
        q['mra'] = score(q['answer'],q)
        save(OUT/'answers'/f'{q["id"]}.json',q)
    save(OUT/'answers.json', rows)
    summary = dict(completed=len(rows),expected=len(priors),mean_mra=sum(q['mra'] for q in rows)/len(rows) if rows else None)
    save(OUT/'summary.json',summary)
    blocks, lines = [], [f'VSI {len(priors)} 题 · 全视频 {FPS:g} FPS 重新作答与判断依据','']
    for q in rows:
        prior = next(p for p in priors if p['id']==q['id'])
        frames = read(OUT/'sampling'/f'{q["scene_name"]}.json')['frames']
        images = ''.join(f'<figure><a href="{f["path"]}" target="_blank"><img loading="lazy" src="{f["path"]}"></a><figcaption>{f["timestamp_s"]:g}s · 帧 {f["source_frame_index"]}</figcaption></figure>' for f in frames)
        blocks.append(f'<section id="q{q["id"]}"><h2>Q{q["id"]} · {prior["object_zh"]}</h2><p>{q["scene_name"]} · {FPS:g} FPS · {q["frame_count"]} 帧</p><p>{html.escape(q["question"])}</p><p>GT <b>{q["ground_truth"]} cm</b> · 本次答案 <b>{q["answer"]} cm</b> · MRA <b>{q["mra"]:.2%}</b></p><p>此前建模图＋V8帧：{prior["model_answer_cm"]:g} cm · MRA {prior["mra_percent_reported"]:g}%</p><h3>模型返回的判断依据</h3><p>{html.escape(q["reason"])}</p><p>模型自报证据充分：{"是" if q["sufficient_evidence"] else "否"}</p><details><summary>查看本次完整 {q["frame_count"]} 张输入帧</summary><div class="grid">{images}</div></details></section>')
        lines += [f'Q{q["id"]} · {prior["object_zh"]} · {q["scene_name"]}',q['question'],
                  f'GT {q["ground_truth"]} cm；本次答案 {q["answer"]} cm；MRA {q["mra"]:.2%}；{q["frame_count"]} 帧',
                  '判断依据：'+q['reason'],f'证据充分：{q["sufficient_evidence"]}','']
    (OUT/'reasoning.txt').write_text('\n'.join(lines))
    (OUT/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>七题 1 FPS 重新作答</title><style>body{font:16px/1.7 system-ui;background:#f2f5f8;max-width:1350px;margin:24px auto;padding:20px}header,section{background:white;border-radius:12px;padding:24px;margin-bottom:24px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:12px}figure{margin:0}img{width:100%}summary{cursor:pointer}a{color:#146c9e}</style><header><h1>VSI 七题 · 全视频 1 FPS 重新作答</h1>'+f'<p>Astra high · 已完成 {len(rows)} / {len(priors)}</p><p>本次仅输入均匀原视频帧与问题；此前为建模图＋V8选帧，因此分数差异包含输入方案与提示词的变化。MRA 沿用 review 的 11 阈值口径。以下展示模型输出的简短判断依据。</p><p><a href="reasoning.txt">判断依据 TXT</a> · <a href="answers.json">全部结果 JSON</a> · <a href="experiment.json">实验配置</a></p></header>'+''.join(blocks)+'</html>')
    page = OUT/'index.html'
    page.write_text(page.read_text().replace('1 FPS',f'{FPS:g} FPS').replace('七题',f'{len(priors)} 题'))
    return summary


def main():
    global FPS, OUT, PROMPT, PRIOR
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--fps',type=float,choices=[1,.5,.25,.125],default=1)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--questions-file',type=Path,default=PRIOR)
    parser.add_argument('--suite',default='vsi-low-mra-uniform')
    args = parser.parse_args()
    FPS = args.fps
    assert args.suite.replace('-', '').isalnum()
    OUT = ROOT/f'results/{args.suite}-{FPS:g}fps-20261109'
    PRIOR = args.questions_file.resolve()
    PROMPT = PROMPT.replace('1 FPS',f'{FPS:g} FPS')
    rows = prepare()
    if args.run:
        pending = [q for q in rows if not (OUT/'answers'/f'{q["id"]}.json').exists()]
        failures=[]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(infer,q):q['id'] for q in pending}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    failures.append(futures[future])
                    print('QUESTION_FAILED',futures[future],type(exc).__name__,flush=True)
                report()
        if failures:
            print('FAILED_IDS',failures,flush=True)
            raise SystemExit(1)
    print(json.dumps(report(),ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
