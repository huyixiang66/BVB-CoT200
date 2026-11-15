"""Compare region grouping against the frozen V5 instance memberships."""

import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import cKDTree
from scipy.spatial.distance import squareform


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def describe_view(index, c2w, intrinsics, cloud, image_size, box=None):
    """Estimate the surface region seen by the central part of the image."""
    frames = np.unique(cloud["frame_ids"])
    nearby = sorted(frames, key=lambda fid: (abs(int(fid) - index), fid))
    sources = [int(fid) for fid in nearby[:3] if abs(int(fid) - index) <= 12]
    points = cloud["points"][np.isin(cloud["frame_ids"], sources)]
    rotation, center = c2w[:3, :3], c2w[:3, 3]
    xyz = (points - center) @ rotation
    positive = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 1e-4)
    xyz, points = xyz[positive], points[positive]
    projected = xyz @ intrinsics.T
    uv = projected[:, :2] / projected[:, 2:3] / image_size
    focus = np.full(2, .5)
    extent = np.full(2, .30)
    if box is not None:
        low, high = np.asarray(box, dtype=float).reshape(2, 2) / image_size
        focus = (low + high) / 2
        extent = np.maximum(.10, .85 * (high - low))
    central = np.all(np.abs(uv - focus) <= extent, axis=1) & np.all((uv >= 0) & (uv < 1), axis=1)
    uv, xyz, points = uv[central], xyz[central], points[central]
    # One frontmost sample per cell reduces density bias and approximates visibility.
    cells = np.floor(uv * [32, 24]).astype(int)
    front = {}
    for i in np.argsort(xyz[:, 2]):
        front.setdefault(tuple(cells[i]), i)
    kept = np.array(list(front.values()), dtype=int)
    if len(kept) < 12:
        return None
    depth = float(np.median(xyz[kept, 2]))
    kept = kept[(xyz[kept, 2] >= depth * .45) & (xyz[kept, 2] <= depth * 1.55)]
    if len(kept) < 12:
        return None
    surface = points[kept]
    ray = np.linalg.solve(intrinsics, np.r_[focus * image_size, 1.])
    return dict(anchor=np.median(surface, axis=0), depth=depth, samples=surface, tree=cKDTree(surface),
                look_at=center + depth * (rotation @ ray), sources=sources)


def geometric_affinity(left, right, scale):
    if left is None or right is None:
        return 0., None, None
    distance = float(np.linalg.norm(left["anchor"] - right["anchor"]) / scale)
    radius = .18 * scale
    a, b = left["samples"], right["samples"]
    overlap_a = np.mean(left["tree"].query(b)[0] <= radius)
    overlap_b = np.mean(right["tree"].query(a)[0] <= radius)
    overlap = float(min(overlap_a, overlap_b))
    score = float(.70 * np.exp(-.5 * (distance / .45) ** 2) + .30 * overlap)
    return score, distance, overlap


def semantic_affinity(left, right):
    similarities = [float(a @ b) for label, aa in left.items()
                    for a in aa for b in right.get(label, ())]
    return max(0., min(1., max(similarities, default=0.)))


def complete_link(affinity, frame_ids):
    if len(frame_ids) <= 1:
        return [frame_ids] if frame_ids else []
    labels = fcluster(linkage(squareform(1 - affinity), method="complete"), .45,
                      criterion="distance")
    clusters = defaultdict(list)
    for frame, label in zip(frame_ids, labels):
        clusters[int(label)].append(frame)
    return sorted(clusters.values())


def frame_rows(peaks, candidates):
    observations = defaultdict(list)
    for p in peaks:
        for r in p["selected"]:
            observations[r["source_frame_index"]].append({
                "object": p["object"], "peak_id": p["peak_id"],
                **{k: r.get(k) for k in ("box_xyxy", "status", "dino_score")},
            })
    rows = {}
    for fid, hints in sorted(observations.items()):
        image = candidates[fid]
        best = max(hints, key=lambda h: (h["box_xyxy"] is not None, h["dino_score"] or 0))
        rows[fid] = dict(source_frame_index=fid, timestamp_s=image["timestamp_s"],
                         frame_sha256=image["frame_sha256"], frame=image["index"],
                         image=image["frame"], observations=hints,
                         object=best["object"], box_xyxy=best["box_xyxy"], status=best["status"])
    return rows


def format_group(group_id, frames, rows, peaks=None):
    selected = []
    for fid in frames:
        row = rows[fid]
        hints = [h for h in row["observations"] if peaks is None or h["peak_id"] in peaks]
        best = max(hints, key=lambda h: (h["box_xyxy"] is not None, h["dino_score"] or 0))
        selected.append(dict(row, observations=hints, object=best["object"],
                             box_xyxy=best["box_xyxy"], status=best["status"]))
    hints = [h for r in selected for h in r["observations"]]
    objects = sorted({h["object"] for h in hints})
    return dict(group_id=group_id, objects=objects, object=objects[0] if len(objects) == 1 else None,
                peak_ids=sorted({h["peak_id"] for h in hints}), selected=selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--intrinsics-size", nargs=2, type=int, default=(518, 392))
    parser.add_argument("--image-size", nargs=2, type=int, default=(640, 480))
    parser.add_argument("--roi-focus", action="store_true",
                        help="Diagnostic only: locate the depth patch around the highest-confidence selected box.")
    args = parser.parse_args()
    root = args.source
    work = root / "pose-crop-ten"
    questions = read(work / "v3/report/questions.json")["questions"]
    peaks = read(work / "v3/report/peaks.json")
    baseline = read(work / "groups/groups.json")
    candidates = {r["source_frame_index"]: r for r in read(root / "inputs/cache/candidates.json")}
    samples = read(root / "inputs/lingbot/sampling.json")["frames"]
    poses = dict(np.load(root / "inputs/lingbot/camera-poses.npz"))
    cloud = dict(np.load(root / "inputs/lingbot/sparse-geometry-corrected.npz"))
    all_rows = frame_rows(peaks, candidates)
    intrinsics_resize = np.diag([*np.divide(args.image_size, args.intrinsics_size), 1.])
    central_geometry = {}
    for fid, row in all_rows.items():
        i = row["frame"]
        assert samples[i]["source_frame_index"] == fid and samples[i]["sha256"] == row["frame_sha256"]
        central_geometry[fid] = describe_view(i, poses["c2w"][i], intrinsics_resize @ poses["intrinsics"][i],
                                              cloud, np.array(args.image_size))
    scale = float(np.median([r["depth"] for r in central_geometry.values() if r]))
    assert scale > 0
    appearance = np.load(work / "appearance/features.npz")
    features = dict(zip(appearance["ids"], appearance["cls"]))
    roi_features = {}
    for row in read(work / "appearance/rois.json")["rows"]:
        v = features[row["id"]]
        key = (row["source_frame_index"], row["object"], tuple(row["box_xyxy"]))
        roi_features[key] = v / max(np.linalg.norm(v), 1e-12)

    weights = {"geometry": 0., "hybrid": .20, "hybrid_035": .35}
    output = dict(scene=questions[0]["scene_name"], question_count=len(questions),
                  focus="detector-box diagnostic" if args.roi_focus else "central view only",
                  depth_scale=scale, hybrid_weights=weights, questions={})
    focus_cache = {}
    for question in questions:
        relevant = [p for p in peaks if p["object"] in question["objects"]]
        by_frame = frame_rows(relevant, candidates)
        frames = list(by_frame)
        geometry = central_geometry.copy()
        if args.roi_focus:
            for fid, row in by_frame.items():
                box = row["box_xyxy"]
                if box is not None:
                    key = (fid, tuple(box))
                    if key not in focus_cache:
                        i = row["frame"]
                        focus_cache[key] = describe_view(i, poses["c2w"][i], intrinsics_resize @ poses["intrinsics"][i],
                                                         cloud, np.array(args.image_size), box)
                    geometry[fid] = focus_cache[key]
        descriptors = defaultdict(lambda: defaultdict(list))
        for fid, row in by_frame.items():
            for hint in row["observations"]:
                if hint["box_xyxy"] is not None:
                    key = (fid, hint["object"], tuple(hint["box_xyxy"]))
                    if key in roi_features:
                        descriptors[fid][hint["object"]].append(roi_features[key])
        pairs = []
        affinity = {arm: np.eye(len(frames)) for arm in weights}
        for i, j in combinations(range(len(frames)), 2):
            a, b = frames[i], frames[j]
            g, distance, overlap = geometric_affinity(geometry[a], geometry[b], scale)
            s = semantic_affinity(descriptors[a], descriptors[b])
            pair = dict(frames=[a, b], geometry=g, depth_normalized_distance=distance,
                        surface_overlap=overlap, semantic=s)
            for arm, weight in weights.items():
                score = g + weight * s * (1 - g) if g >= .35 else g
                affinity[arm][i, j] = affinity[arm][j, i] = score
                pair[arm] = score
            pairs.append(pair)
        arms = {}
        arms["v5_projection"] = []
        peak_ids = {p["peak_id"] for p in relevant}
        covered_peaks = set()
        for group in baseline:
            members = set(group["peak_ids"]) & peak_ids
            if members:
                covered_peaks.update(members)
                members_frames = [f for f, row in by_frame.items() if
                                  any(h["peak_id"] in members for h in row["observations"])]
                arms["v5_projection"].append(format_group(group["group_id"], members_frames, by_frame, members))
        assert covered_peaks == peak_ids
        for arm in weights:
            arms[arm] = [format_group(f"{arm}-{i + 1:02d}", group, by_frame)
                         for i, group in enumerate(complete_link(affinity[arm], frames))]
        for groups in arms.values():
            assert {r["source_frame_index"] for g in groups for r in g["selected"]} == set(frames)
        metrics = {arm: dict(groups=len(groups), slots=sum(len(g["selected"]) for g in groups),
                             multi_category_groups=sum(len(g["objects"]) > 1 for g in groups),
                             singleton_groups=sum(len(g["selected"]) == 1 for g in groups))
                   for arm, groups in arms.items()}
        output["questions"][str(question["id"])] = dict(question=question, arms=arms, metrics=metrics,
            frame_count=len(frames), valid_depth=sum(geometry[f] is not None for f in frames),
            geometry={str(f): (dict(anchor=geometry[f]["anchor"].tolist(), look_at=geometry[f]["look_at"].tolist(),
                              depth=geometry[f]["depth"], visible_cells=len(geometry[f]["samples"]),
                              sample_indices=geometry[f]["sources"]) if geometry[f] else None) for f in frames},
            pairs=pairs)

    args.out.mkdir(parents=True, exist_ok=True)
    write(args.out / "results.json", output)
    for qid, result in output["questions"].items():
        print(qid, result["frame_count"], result["valid_depth"], result["metrics"], flush=True)


if __name__ == "__main__":
    main()
