"""Add a corroborating geometric link when two peaks independently locate the same ROI."""

import argparse
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from group_regions import format_group, read, write


def project_error(point, camera):
    center, rotation, k, box, _ = camera
    uvz = k @ (rotation.T @ (point - center))
    if uvz[2] <= 0:
        return float("inf")
    lo, hi = box.reshape(2, 2)
    return float(np.max(np.abs((uvz[:2] / uvz[2] - (lo + hi) / 2) /
                               ((hi - lo) / 2 + 6.))))


def fit_anchor(views, scale):
    best = None
    for left, right in combinations(views, 2):
        a, _, _, _, u = left
        b, _, _, _, v = right
        if np.linalg.norm(np.cross(u, v)) < .07 or np.linalg.norm(a - b) < .05 * scale:
            continue
        depths = np.linalg.lstsq(np.column_stack([u, -v]), b - a, rcond=None)[0]
        if np.any(depths <= .05 * scale) or np.any(depths > 5. * scale):
            continue
        x, y = a + depths[0] * u, b + depths[1] * v
        if np.linalg.norm(x - y) > .06 * scale:
            continue
        p = (x + y) / 2
        errors = np.array([project_error(p, c) for c in views])
        inliers = errors[errors <= 1.]
        if len(inliers) < 3:
            continue
        key = (len(inliers), -float(np.median(inliers)))
        if best is None or key > best[0]:
            best = (key, p)
    return best[1] if best else None


def extract_views(question, c2w, k):
    rows, peak_views, peak_frames = {}, defaultdict(dict), defaultdict(set)
    for group in question["arms"]["geometry"]:
        for row in group["selected"]:
            fid, idx = row["source_frame_index"], row["frame"]
            rows[fid] = row
            r, p = c2w[idx, :3, :3], c2w[idx, :3, 3]
            for hint in row["observations"]:
                peak = hint["peak_id"]
                peak_frames[peak].add(fid)
                if hint["box_xyxy"] is None:
                    continue
                box = np.array(hint["box_xyxy"])
                uv = box.reshape(2, 2).mean(axis=0)
                ray = r @ np.linalg.solve(k[idx], np.r_[uv, 1.])
                peak_views[peak][fid] = (p, r, k[idx], box, ray / np.linalg.norm(ray))
    return rows, peak_frames, {p: list(views.values()) for p, views in peak_views.items()}


def regroup(question, prior, c2w, k, scale):
    rows, peak_frames, views = extract_views(question, c2w, k)
    peaks = sorted(peak_frames)
    anchors = {p: fit_anchor(views.get(p, []), scale) for p in peaks}
    affinity = np.eye(len(peaks))
    extra_links = []
    for i, j in combinations(range(len(peaks)), 2):
        a, b = peaks[i], peaks[j]
        score = prior[a][b]
        p, q = anchors[a], anchors[b]
        if p is not None and q is not None:
            distance = float(np.linalg.norm(p - q) / scale)
            ab = sum(project_error(p, c) <= 1. for c in views[b])
            ba = sum(project_error(q, c) <= 1. for c in views[a])
            if distance <= .25 and min(ab, ba) >= 2 and score < .55:
                extra_links.append(dict(peaks=[a, b], original_score=score, distance=distance,
                                        reciprocal_views=[ab, ba]))
                score = .55
        affinity[i, j] = affinity[j, i] = score
    labels = fcluster(linkage(squareform(1 - affinity), method="complete"), .45,
                      criterion="distance") if len(peaks) > 1 else [1]
    sets = defaultdict(set)
    for peak, label in zip(peaks, labels):
        sets[int(label)].add(peak)
    groups = [format_group(f"peak-ray-{i:02d}", sorted({f for p in members for f in peak_frames[p]}),
                           rows, members)
              for i, members in enumerate(sorted(sets.values(), key=min), start=1)]
    assert {r["source_frame_index"] for g in groups for r in g["selected"]} == set(rows)
    return dict(question=question["question"], selected_groups=groups,
                anchor_count=sum(p is not None for p in anchors.values()), peak_count=len(peaks),
                anchor_points={k: p.tolist() if p is not None else None for k, p in anchors.items()},
                extra_links=extra_links,
                multi_category_groups=sum(len(g["objects"]) > 1 for g in groups))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames", type=Path)
    parser.add_argument("peaks", type=Path)
    parser.add_argument("poses", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--intrinsics-size", nargs=2, type=int, default=(518, 392))
    parser.add_argument("--image-size", nargs=2, type=int, default=(640, 480))
    args = parser.parse_args()
    frames = read(args.frames)
    peaks = read(args.peaks)
    pose = np.load(args.poses)
    k = np.diag([*np.divide(args.image_size, args.intrinsics_size), 1.]) @ pose["intrinsics"]
    output = {}
    for qid, question in frames["questions"].items():
        old = peaks["questions"][qid]
        output[qid] = regroup(question, old["evidence"]["geometry"], pose["c2w"], k, frames["depth_scale"])
        print(qid, len(old["arms"]["peak_geometry_55"]), len(output[qid]["selected_groups"]),
              output[qid]["anchor_count"], output[qid]["peak_count"], output[qid]["extra_links"])
    write(args.out, dict(scene=frames["scene"], questions=output))


if __name__ == "__main__":
    main()
