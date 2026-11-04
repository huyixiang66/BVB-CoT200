"""Evaluate a visually labelled diagnostic panel, separate from detector inputs."""
import argparse
import json
from pathlib import Path


# Annotations at half resolution on the frozen original frames.
# Targets are the microwave-base refrigerator and the patterned bedspread.
REFERENCES = {
    "refrigerator": {
        3390: [135, 139, 204, 240],
        3405: [106, 80, 183, 198],
        3420: [155, 63, 235, 180],
        3435: [195, 61, 291, 196],
        3450: [205, 63, 296, 202],
        3458: [225, 54, 312, 203],
        3472: [260, 49, 320, 196],
        3480: [275, 47, 320, 114],
        4050: [89, 73, 132, 180],
        4065: [91, 84, 148, 198],
        4080: [95, 102, 160, 240],
        4095: [55, 103, 155, 240],
        4110: [37, 117, 151, 240],
    },
    "blanket": {
        4770: [191, 0, 320, 43],
        4800: [203, 0, 320, 86],
        4845: [257, 163, 320, 240],
        4875: [240, 196, 320, 240],
        5100: [215, 128, 320, 240],
        5160: [226, 147, 320, 240],
        5520: [0, 20, 94, 240],
        5580: [0, 56, 253, 240],
        6060: [105, 145, 320, 240],
        6120: [121, 163, 320, 240],
    },
}
NEGATIVES = {
    "refrigerator": [0, 330, 1590, 2160, 3000, 3840, 3900, 3960, 4448, 5580, 5932, 6060],
    "blanket": [0, 330, 345, 360, 3420, 3450, 3840, 3900, 3960, 4448, 5640, 5932],
}


def iou(a, b):
    ax, ay, bx, by = a
    cx, cy, dx, dy = b
    overlap = max(0, min(bx, dx) - max(ax, cx)) * max(0, min(by, dy) - max(ay, cy))
    union = (bx - ax) * (by - ay) + (dx - cx) * (dy - cy) - overlap
    return overlap / union if union else 0


def evaluate(rows, field):
    index = {r["source_frame_index"]: r[field] for r in rows}
    report = {}
    for label, refs in REFERENCES.items():
        counts = {"positives": len(refs), "hit_at_04": 0, "hit_at_05": 0,
                  "best_box_hit_at_05": 0, "negatives": len(NEGATIVES[label]), "negative_frames_with_fp": 0}
        detailed = []
        for fid in list(refs) + NEGATIVES[label]:
            detections = sorted((r for r in index[fid] if r["object"] == label),
                                key=lambda row: row["score"], reverse=True)
            if fid in refs:
                box = [2 * v for v in refs[fid]]
                overlaps = [iou(r["box_xyxy"], box) for r in detections]
                counts["hit_at_04"] += max(overlaps, default=0) >= .4
                counts["hit_at_05"] += max(overlaps, default=0) >= .5
                counts["best_box_hit_at_05"] += bool(overlaps) and overlaps[0] >= .5
            else:
                box, overlaps = None, []
                counts["negative_frames_with_fp"] += bool(detections)
            detailed.append(dict(frame=fid, gt_box=box, best_iou=max(overlaps, default=0.),
                                 detections=detections))
        report[label] = {"summary": counts, "frames": detailed}
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--model-results", type=Path, nargs="*", default=[])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    results = {"historical_tiny": evaluate(
        list(map(json.loads, args.baseline.read_text().splitlines())), "detections")}
    for path in args.model_results:
        for file in sorted(path.glob("*.jsonl")):
            rows = list(map(json.loads, file.read_text().splitlines()))
            for field in ("normal", "loose"):
                results[f"{path.name}/{file.stem}/{field}"] = evaluate(rows, field)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for condition, report in results.items():
        print(condition, *(f"{name} {r['summary']}" for name, r in report.items()))


if __name__ == "__main__":
    main()
