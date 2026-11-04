"""Make a portable side-by-side review for the annotated detection panel."""
import base64
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CASES = {
    "refrigerator": [3420, 3435, 3458, 3472, 4050, 4080, 3960, 4448],
    "blanket": [4770, 4875, 5100, 5580, 6060, 6120, 330, 345],
}
NAMES = {
    "historical_tiny": "原始 Tiny",
    "tiny/base/normal": "Tiny · 同词同阈值复跑",
    "tiny/base/loose": "Tiny · 放宽阈值",
    "tiny/aliases/normal": "Tiny · 加别名",
    "tiny/aliases/loose": "Tiny · 加别名＋放宽",
    "tiny-built-in/aliases/normal": "Tiny · 只加 built-in refrigerator",
    "tiny-built-in/aliases/loose": "Tiny · 只加 built-in refrigerator＋放宽",
    "base/base/normal": "Grounding DINO Base",
    "base/base/loose": "Grounding DINO Base · 放宽",
    "mm-base/base/normal": "MM Grounding DINO Base",
    "mm-base/base/loose": "MM Grounding DINO Base · 放宽",
}
OBJECTS = {"refrigerator": "嵌入式冰箱", "blanket": "卧室花纹毯"}


def rect(box, color):
    x0, y0, x1, y1 = box
    return (f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{x1-x0:.1f}" height="{y1-y0:.1f}"'
            f' fill="none" stroke="{color}" stroke-width="4"/>')


def figure(uri, row, title):
    overlay = rect(row["gt_box"], "#62f08a") if row["gt_box"] else ""
    boxes = row["detections"][:3]
    colors = ["#ff625f", "#e5bb4f", "#ed8de4"]
    overlay += "".join(rect(box["box_xyxy"], color) for box, color in zip(boxes, colors))
    labels = [f'{box.get("detection_alias", box["object"])} {box["score"]:.2f}' for box in boxes]
    note = " · ".join(labels) if labels else "无检测框"
    return (f'<figure><figcaption>{html.escape(title)}</figcaption>'
            f'<svg viewBox="0 0 640 480"><image href="{uri}" width="640" height="480"/>'
            f'{overlay}</svg><p>{html.escape(note)}'
            f'{" · 最大 IoU %.2f" % row["best_iou"] if row["gt_box"] else " · 负例"}</p></figure>')


def main():
    metrics = json.loads((ROOT / "metrics.json").read_text())
    frames = {r["source_frame_index"]: r for r in json.loads((ROOT / "cache/candidates.json").read_text())}
    page = ["""<!doctype html><html lang="zh"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>3db 冰箱与毯子检测对照</title>
<style>
body{font:16px/1.55 system-ui,sans-serif;margin:32px auto;max-width:1440px;padding:0 20px;color:#17212b;background:#f6f8fa}
table{width:100%;border-collapse:collapse;background:white}th,td{padding:9px 12px;text-align:left;border-bottom:1px solid #d9e1e9}
.row{display:grid;grid-template-columns:repeat(auto-fit,minmax(205px,1fr));gap:12px}
figure{background:white;border:1px solid #d9e1e9;border-radius:8px;overflow:hidden;margin:0}
figure svg{display:block;width:100%;background:#18232e}
figure p,figcaption{padding:6px 9px;margin:0;font-size:13px}
h3{margin-bottom:7px}.stat{max-width:960px}
</style>
<h1>3db0a1c8f3 冰箱与毯子检测对照</h1>
<p class="stat">50 张冻结原帧。正例按框与人工参考框的 IoU ≥ 0.50 计中。负例统计有任一同类框的帧数。
每个模型使用相同视频帧、原词表、输入分辨率和检测后处理。阈值为文本均值／峰值，
原版 0.20／0.25，放宽 0.12／0.17。样本特意包含难帧和混淆项，用于诊断。</p>
<table><thead><tr><th>检测条件</th><th>冰箱命中／最高分命中</th><th>冰箱负帧误报</th>
<th>毯子命中／最高分命中</th><th>毯子负帧误报</th></tr></thead><tbody>"""]
    for condition, report in metrics.items():
        cells = []
        for label in OBJECTS:
            r = report[label]["summary"]
            cells.extend([f'{r["hit_at_05"]}/{r["positives"]} · {r["best_box_hit_at_05"]}/{r["positives"]}',
                          f'{r["negative_frames_with_fp"]}/{r["negatives"]}'])
        page.append('<tr><td>' + html.escape(NAMES.get(condition, condition)) + '</td>' +
                    ''.join(f'<td>{cell}</td>' for cell in cells) + '</tr>')
    page.append('</tbody></table><p>绿色是人工参考框。红、黄、粉分别是预测分数前三的框。</p>')
    display = ["historical_tiny", "tiny/base/loose", "tiny-built-in/aliases/normal",
               "tiny/aliases/normal",
               "base/base/normal", "mm-base/base/normal"]
    for label, cases in CASES.items():
        page.append(f'<h2>{OBJECTS[label]}</h2>')
        for fid in cases:
            frame = frames[fid]
            uri = 'data:image/jpeg;base64,' + base64.b64encode((ROOT / "cache" / frame["frame"]).read_bytes()).decode()
            page.append(f'<h3>{frame["timestamp_s"]:.2f} s</h3><div class="row">')
            for condition in display:
                if condition in metrics:
                    row = next(r for r in metrics[condition][label]["frames"] if r["frame"] == fid)
                    page.append(figure(uri, row, NAMES[condition]))
            page.append('</div>')
    page.append('</html>')
    (ROOT / "review.html").write_text('\n'.join(page), encoding="utf-8")


if __name__ == "__main__":
    main()
