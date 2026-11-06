"""Keep each original V3 peak intact and group only corroborated peak pairs."""

import argparse
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .group_regions import format_group, read, write


def two_view_support(a, b, scores, minimum_pairs=2):
    # Find the strongest association supported by two distinct views per side.
    ranked = sorted(((scores[tuple(sorted((x, y)))] if x != y else 1., x, y)
                     for x in a for y in b), reverse=True)
    if minimum_pairs == 1:
        return ranked[0][0] if ranked else 0.
    for i, (score, x, y) in enumerate(ranked):
        if any(px != x and py != y for _, px, py in ranked[:i]):
            return score
    return 0.


def group_peaks(question, arm, threshold, minimum_pairs=2):
    rows = {}
    peak_frames = defaultdict(set)
    # Geometry arms partition the frozen input so the full observation list
    # on each frame is available exactly once.
    for group in question["arms"]["geometry"]:
        for row in group["selected"]:
            fid = row["source_frame_index"]
            rows[fid] = row
            for hint in row["observations"]:
                peak_frames[hint["peak_id"]].add(fid)
    peaks = sorted(peak_frames)
    scores = {tuple(p["frames"]): p[arm] for p in question["pairs"]}
    affinity = np.eye(len(peaks))
    for i, j in combinations(range(len(peaks)), 2):
        score = two_view_support(peak_frames[peaks[i]], peak_frames[peaks[j]], scores, minimum_pairs)
        affinity[i, j] = affinity[j, i] = score
    labels = fcluster(linkage(squareform(1 - affinity), method="complete"),
                      1 - threshold, criterion="distance") if len(peaks) > 1 else [1]
    sets = defaultdict(set)
    for p, label in zip(peaks, labels):
        sets[int(label)].add(p)
    groups = []
    for i, members in enumerate(sorted(sets.values(), key=min), start=1):
        frames = sorted({f for p in members for f in peak_frames[p]})
        groups.append(format_group(f"peak-{arm}-{i:02d}", frames, rows, members))
    assert {r["source_frame_index"] for g in groups for r in g["selected"]} == set(rows)
    return groups, {a: {b: float(affinity[i, j]) for j, b in enumerate(peaks) if i != j}
                    for i, a in enumerate(peaks)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    data = read(args.source)
    out = {"scene": data["scene"], "questions": {}}
    for qid, question in data["questions"].items():
        arms, evidence = {}, {}
        for arm in ("geometry", "hybrid", "hybrid_035"):
            for threshold in (.55, .45):
                key = f"peak_{arm}_{round(threshold * 100)}"
                groups, evidence[arm] = group_peaks(question, arm, threshold)
                arms[key] = groups
        metrics = {arm: dict(groups=len(groups), slots=sum(len(g["selected"]) for g in groups),
                             multi_category_groups=sum(len(g["objects"]) > 1 for g in groups),
                             singleton_groups=sum(len(g["selected"]) == 1 for g in groups))
                   for arm, groups in arms.items()}
        out["questions"][qid] = dict(question=question["question"], arms=arms, metrics=metrics,
                                    frame_count=question["frame_count"], evidence=evidence)
        print(qid, [m["groups"] for m in metrics.values()], flush=True)
    write(args.out, out)


if __name__ == "__main__":
    main()
