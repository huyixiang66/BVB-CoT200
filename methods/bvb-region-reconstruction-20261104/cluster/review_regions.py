"""Write a small local review of the fixed-frame region grouping pilot."""

import argparse
from hashlib import sha256
from html import escape
from pathlib import Path
import os

from group_regions import read


def gallery(groups, cache, destination):
    cards = []
    for group in groups:
        selected = group["selected"]
        images = []
        for row in selected:
            path = os.path.relpath(cache / row["image"], destination).replace(os.sep, "/")
            times = f'{row["timestamp_s"]:.2f} s'
            labels = ", ".join(sorted({h["object"] for h in row["observations"]}))
            images.append(f'<figure><img loading="lazy" src="{escape(path)}" alt="{times}">'
                          f'<figcaption>{times} · {escape(labels)}</figcaption></figure>')
        title = f'{group["group_id"]} · {", ".join(group["objects"])} · {len(selected)} 帧'
        cards.append(f'<details><summary>{escape(title)}</summary><div class="images">'
                     + "".join(images) + '</div></details>')
    return "".join(cards)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    primary = read(args.run / "verified/results.json")
    peak = read(args.run / "verified/peak-results.json")
    ray = read(args.run / "verified/ray-results.json")
    roi = read(args.run / "roi-focus/results.json")
    checked = {}
    for item in primary["questions"].values():
        for group in item["arms"]["geometry"]:
            for frame in group["selected"]:
                name = frame["image"]
                if name not in checked:
                    checked[name] = sha256((args.cache / name).read_bytes()).hexdigest()
                assert checked[name] == frame["frame_sha256"]
    heads = ("V5 成员映射", "纯位姿与深度", "加 0.20 外观", "加 0.35 外观",
             "峰级纯几何 .55", "峰级纯几何 .45", "峰级 .55 加检测框射线")
    rows, details = [], []
    for qid, item in primary["questions"].items():
        arms = list(item["arms"].values()) + [
            peak["questions"][qid]["arms"][k] for k in ("peak_geometry_55", "peak_geometry_45")
        ] + [ray["questions"][qid]["selected_groups"]]
        numbers = "".join(f"<td>{len(groups)}</td>" for groups in arms)
        roi_count = roi["questions"][qid]["metrics"]["geometry"]["groups"]
        rows.append(f'<tr><th>{qid}</th><td>{item["frame_count"]}</td><td>{item["valid_depth"]}</td>'
                    + numbers + f"<td>{roi_count}</td></tr>")
        if qid in {"1585", "1754", "2607"}:
            bodies = "".join(f"<section><h3>{head}</h3>{gallery(groups, args.cache, args.out.parent)}</section>"
                             for head, groups in zip(heads, arms))
            details.append(f'<article><h2>{qid}</h2><p>{escape(item["question"]["question"])}</p>'
                           + f'<div class="arms">{bodies}</div></article>')
    columns = "".join(f"<th>{name}</th>" for name in heads)
    page = """<!doctype html><html lang="zh"><meta charset="utf-8">
<title>3db 区域聚类对照</title><style>
body{max-width:1250px;margin:40px auto;padding:0 20px;font:16px/1.6 system-ui;color:#202725}
table{width:100%;border-collapse:collapse}td,th{padding:7px 9px;border-bottom:1px solid #d2dcd8}
.arms{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}
section{border:1px solid #d2dcd8;border-radius:8px;padding:0 12px 12px}
details{padding:5px 0;border-bottom:1px solid #eee}summary{cursor:pointer}
.images{display:grid;grid-template-columns:repeat(2,1fr);gap:8px}
figure{margin:0}figure img{width:100%;display:block}figcaption{font-size:12px}
article{margin:34px 0}</style><h1>3db0a1c8f3 区域聚类对照</h1>
<p>每题各方法使用同一批 V3 入选原帧。表里是候选组数。V5 列把既有实例成员映射回这一批帧。
纯几何由中央视野里的位姿与稀疏深度决定。峰级方法保留每个原波峰，用至少两对独立帧判断合并。
检测框射线只有在两侧独立定位、再双向投进检测框时才增加合并依据。
最后一列是逐帧改用检测框周边深度的诊断。各组展开后可查看原图与时间。</p>
"""
    page += (f'<p>已核对 {len(checked)} 张不同源帧的文件哈希。</p><table><thead><tr>'
             f'<th>题号</th><th>原帧</th><th>中央深度可用</th>{columns}<th>检测框辅助</th>'
             f'</tr></thead><tbody>{"".join(rows)}</tbody></table>{"".join(details)}')
    args.out.write_text(page, encoding="utf-8")
    print(f"Verified {len(checked)} image hashes and wrote {args.out}")


if __name__ == "__main__":
    main()
