"""Independent position evidence when one peak has no usable depth."""

from functools import lru_cache
from itertools import combinations
import json

import cv2
import numpy as np

from bvb_v5.dinov2_appearance import read


def observations(peak):
    return [(r["frame"], np.asarray(r["box_xyxy"], dtype=float)) for r in peak["selected"]
            if r["box_xyxy"] and r["status"] != "context_fallback"]


def position_evidence(peaks, cache, lingbot, intrinsics_size):
    """Return one-sided depth reprojection and independent keypoint matches."""
    cv2.setNumThreads(2)
    with np.load(lingbot / "camera-poses.npz") as pose:
        mats = pose["c2w"]
        native_intrinsics = pose["intrinsics"]
    with np.load(lingbot / "sparse-geometry-corrected.npz") as sparse:
        ids, points = sparse["frame_ids"], sparse["points"]
    clouds = {k: points[ids == k] for k in np.unique(ids)}
    frames = read(cache / "candidates.json")
    detections = {r["source_frame_index"]: r["detections"] for r in
                  map(json.loads, (cache / "detections.jsonl").read_text(encoding="utf-8").splitlines())}
    sift = cv2.SIFT_create(nfeatures=900, contrastThreshold=.015)
    matcher = cv2.BFMatcher()

    @lru_cache(maxsize=96)
    def gray(frame):
        path = cache / frames[frame]["frame"]
        image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Cannot decode frame: {path}")
        return image

    if not peaks:
        return {}
    height, width = gray(peaks[0]["anchor"]["frame"]).shape
    reference_width, reference_height = intrinsics_size
    intrinsics = np.diag([width / reference_width, height / reference_height, 1]) @ native_intrinsics

    def project(k, points):
        camera = (points - mats[k, :3, 3]) @ mats[k, :3, :3]
        uv = camera @ intrinsics[k].T
        return uv[:, :2] / np.maximum(uv[:, 2, None], 1e-9), camera[:, 2]

    def box_mask(uv, box, margin):
        border = (box[2:] - box[:2]) * margin
        return np.all((uv >= box[:2] - border) & (uv <= box[2:] + border), axis=1)

    def iou(a, b):
        def area(p):
            return np.prod(np.maximum(p[2:] - p[:2], 0))
        shared = area(np.r_[np.maximum(a[:2], b[:2]), np.minimum(a[2:], b[2:])])
        return shared / max(area(a) + area(b) - shared, 1e-6)

    def recover(peak, views):
        if len(views) != 1:
            return []
        k, box = views[0]
        start = gray(k)
        mask = np.zeros_like(start)
        x0, y0, x1, y1 = np.round(box).astype(int)
        mask[y0:y1, x0:x1] = 255
        pts = cv2.goodFeaturesToTrack(start, 60, .01, 5, mask=mask)
        if pts is None or len(pts) < 6:
            return []
        out = []
        for row in peak["selected"]:
            j = row["frame"]
            if row["status"] != "context_fallback" or abs(j-k) > 3:
                continue
            end = gray(j)
            tracked, ok, _ = cv2.calcOpticalFlowPyrLK(start, end, pts, None, winSize=(31, 31), maxLevel=4)
            if tracked is None or ok is None:
                continue
            back, reverse_ok, _ = cv2.calcOpticalFlowPyrLK(end, start, tracked, None, winSize=(31, 31), maxLevel=4)
            if back is None or reverse_ok is None:
                continue
            good = ok.ravel().astype(bool) & reverse_ok.ravel().astype(bool)
            good &= np.linalg.norm((pts - back)[:, 0], axis=1) <= 2
            if good.sum() < 6:
                continue
            affine, inliers = cv2.estimateAffinePartial2D(pts[good], tracked[good], ransacReprojThreshold=4)
            if affine is None or inliers.sum() < 6:
                continue
            corners = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]]) @ affine[:, :2].T + affine[:, 2]
            predicted = np.r_[corners.min(axis=0), corners.max(axis=0)].clip(0, [width, height, width, height])
            boxes = [np.array(d["box_xyxy"]) for d in detections[frames[j]["source_frame_index"]]
                     if d["object"] == peak["object"]]
            best = max(boxes, key=lambda b: iou(predicted, b), default=None)
            if best is not None and iou(predicted, best) >= .35:
                out.append((j, best))
        return out

    obs, recovered = {}, {}
    for peak in peaks:
        name = peak["peak_id"]
        primary = observations(peak)
        recovered[name] = recover(peak, primary)
        obs[name] = primary + recovered[name]
    seeds = {}
    for name, views in obs.items():
        source = {}
        for frame, box in views:
            for k in sorted(clouds, key=lambda k: abs(k - frame))[:2]:
                if abs(k - frame) > 5:
                    continue
                uv, z = project(frame, clouds[k])
                pts = clouds[k][box_mask(uv, box, -.10) & (z > 0)]
                if len(pts) >= 6:
                    source.setdefault(k, []).append(pts)
        seeds[name] = {int(k): np.unique(np.concatenate(sets), axis=0) for k, sets in source.items()}

    def reprojection(first, second):
        views = obs[second]
        independent = [(a, b) for a, b in combinations(range(len(views)), 2)
                       if abs(views[a][0] - views[b][0]) >= 2]
        if not independent:
            return []
        stats = []
        for sample, points in seeds[first].items():
            inside = []
            for frame, box in views:
                uv, z = project(frame, points)
                inside.append(box_mask(uv, box, 0) & (z > 0))
            best = max(int(np.sum(inside[a] & inside[b])) for a, b in independent)
            stats.append(dict(sample=sample, points=len(points), shared=best, fraction=best / len(points)))
        return stats

    @lru_cache(maxsize=96)
    def features(frame, box):
        image = gray(frame)
        mask = np.zeros_like(image)
        b = np.array(box)
        margin = np.maximum((b[2:] - b[:2]) * .45, 18)
        lo = np.maximum(np.floor(b[:2] - margin).astype(int), 0)
        hi = np.minimum(np.ceil(b[2:] + margin).astype(int), [image.shape[1], image.shape[0]])
        mask[lo[1]:hi[1], lo[0]:hi[0]] = 255
        kp, desc = sift.detectAndCompute(image, mask)
        return np.asarray([k.pt for k in kp]), desc

    def point_matches(a, b):
        ka, ba = a
        kb, bb = b
        ua, da = features(ka, tuple(ba))
        ub, db = features(kb, tuple(bb))
        if da is None or db is None or min(len(da), len(db)) < 2:
            return None
        forward = [(u.queryIdx, u.trainIdx) for u, v in matcher.knnMatch(da, db, k=2) if u.distance < .82 * v.distance]
        reverse = {(u.trainIdx, u.queryIdx) for u, v in matcher.knnMatch(db, da, k=2) if u.distance < .82 * v.distance}
        links = sorted(set(forward) & reverse)
        if not links:
            return None
        aidx, bidx = np.array(links).T
        xa, xb = ua[aidx], ub[bidx]
        ma = np.linalg.inv(mats[ka])[:3]
        mb = np.linalg.inv(mats[kb])[:3]
        world = cv2.triangulatePoints(intrinsics[ka] @ ma, intrinsics[kb] @ mb, xa.T, xb.T).T
        valid = np.abs(world[:, 3]) > 1e-8
        pts = world[valid, :3] / world[valid, 3, None]
        xa, xb = xa[valid], xb[valid]
        pa, za = project(ka, pts)
        pb, zb = project(kb, pts)
        errors = np.maximum(np.linalg.norm(pa - xa, axis=1), np.linalg.norm(pb - xb, axis=1))
        va, vb = pts - mats[ka, :3, 3], pts - mats[kb, :3, 3]
        cos = np.sum(va * vb, axis=1) / np.maximum(np.linalg.norm(va, axis=1) * np.linalg.norm(vb, axis=1), 1e-9)
        near = box_mask(xa, ba, .25) & box_mask(xb, bb, .25)
        core = box_mask(xa, ba, 0) | box_mask(xb, bb, 0)
        valid = (za > 0) & (zb > 0) & (errors <= 6) & (cos < np.cos(np.radians(4))) & near & core
        if not np.any(valid):
            return None
        # Deduplicate SIFT orientations at essentially the same image location.
        keep = []
        for i in np.flatnonzero(valid):
            if all(np.linalg.norm(xa[i] - xa[j]) >= 7 and np.linalg.norm(xb[i] - xb[j]) >= 7 for j in keep):
                keep.append(i)
        return dict(frames=[ka, kb], count=len(keep), extent_a=float(np.linalg.norm(np.ptp(xa[keep], axis=0))),
                    extent_b=float(np.linalg.norm(np.ptp(xb[keep], axis=0))),
                    median_error=float(np.median(errors[keep])),
                    pixels=[xa[keep].round(1).tolist(), xb[keep].round(1).tolist()])

    def tracks(matches):
        verified = []
        for left, right in combinations(matches, 2):
            common = set(left["frames"]) & set(right["frames"])
            if len(common) != 1:
                continue
            frame = common.pop()
            a, b = left["frames"].index(frame), right["frames"].index(frame)
            ka, kb = left["frames"][1-a], right["frames"][1-b]
            if abs(ka - kb) < 1:
                continue
            for i, u in enumerate(left["pixels"][a]):
                for j, v in enumerate(right["pixels"][b]):
                    if np.linalg.norm(np.subtract(u, v)) > 4:
                        continue
                    views = [(frame, u), (ka, left["pixels"][1-a][i]), (kb, right["pixels"][1-b][j])]
                    projections = [intrinsics[k] @ np.linalg.inv(mats[k])[:3] for k, _ in views]
                    equations = np.stack([p[axis] - xy[axis]*p[2] for p, (_, xy) in zip(projections, views)
                                          for axis in (0, 1)])
                    w = np.linalg.svd(equations)[2][-1]
                    if abs(w[3]) < 1e-8:
                        continue
                    world = w[None, :3] / w[3]
                    errors = []
                    for k, xy in views:
                        uv, z = project(k, world)
                        errors.append(float(np.linalg.norm(uv[0] - xy)) if z[0] > 0 else float("inf"))
                    if max(errors) <= 12:
                        verified.append(dict(frames=[k for k, _ in views], max_error_px=max(errors)))
        return sorted(verified, key=lambda r: r["max_error_px"])

    output = {}
    for first, second in combinations(peaks, 2):
        if first["object"] != second["object"]:
            continue
        a, b = sorted((first["peak_id"], second["peak_id"]))
        matches = [m for av in obs[a] for bv in obs[b] if (m := point_matches(av, bv))]
        output[a, b] = dict(
            point_reprojection=[reprojection(a, b), reprojection(b, a)],
            keypoint_pairs=sorted(matches, key=lambda m: (m["count"], -m["median_error"]), reverse=True),
            three_view_tracks=tracks(matches),
            seed_samples=[len(seeds[a]), len(seeds[b])],
            recovered_frames=[[k for k, _ in recovered[n]] for n in (a, b)])
    return output


def pair_signals(spatial, anchor_similarity):
    depth = any(sum(r["shared"] >= 24 and r["fraction"] >= .60 for r in side) >= 2
                for side in spatial["point_reprojection"])
    spread = [r for r in spatial["keypoint_pairs"] if r["count"] >= 2 and
              min(r["extent_a"], r["extent_b"]) >= 24]
    tracks = bool(spatial["three_view_tracks"]) and any(
        max(abs(a-b) for a, b in zip(u["frames"], v["frames"])) >= 2
        for u, v in combinations(spread, 2))
    return dict(spatial=spatial, pose_depth=depth and anchor_similarity >= .35, pose_keypoints=tracks)


def enhanced_support(names, pairs, rois, original_support):
    proof = original_support(names, pairs)
    if proof:
        return proof
    records = [pairs[tuple(sorted((a, b)))] for a, b in combinations(names, 2)]
    if any(p["same_frame_distinct_evidence"] for p in records):
        return None
    if len(names) == 2:
        pair = records[0]
        matching = pair["matching"]
        if matching and pair["merge_supported"] and matching["bottleneck"] >= .45 and all(
            rois[m[key]].get("crop_expansion_each_side") == .75
            for m in matching["matches"] for key in ("left_roi", "right_roi")
        ):
            score = matching["bottleneck"]
            return dict(representative=names[0], minimum_support=score,
                        mean_pair_score=score, support_type="expanded_small_roi")
    for name in sorted(names):
        edges = [pairs[tuple(sorted((name, other)))] for other in names if other != name]
        pose_edges = [p for p in edges if has_pose(p)]
        if pose_edges and all(p["merge_supported"] or has_pose(p) for p in edges):
            return dict(representative=name, minimum_support=.60, mean_pair_score=.60,
                        priority_only=True, support_type="pose",
                        pose_edges=[dict(peaks=[p["peak_a"], p["peak_b"]],
                                         depth=p["pose_depth"], keypoints=p["pose_keypoints"])
                                    for p in pose_edges])
    return None


def has_pose(pair):
    return pair.get("pose_depth", False) or pair.get("pose_keypoints", False)


def refine_groups(groups, history, pairs, support):
    """Every extra merge requires direct new evidence across current groups."""
    while True:
        options = []
        for i, j in combinations(range(len(groups)), 2):
            first, second = groups[i], groups[j]
            proof = support(first + second, pairs)
            if proof is None:
                continue
            cross = [pairs[tuple(sorted((a, b)))] for a in first for b in second]
            pose_cross = [pair for pair in cross if has_pose(pair)]
            if pose_cross or proof.get("support_type") == "expanded_small_roi":
                options.append((len(pose_cross), proof["minimum_support"], i, j, proof, pose_cross))
        if not options:
            return groups, history
        _, _, i, j, proof, pose_cross = max(options, key=lambda r: (r[0], r[1], -r[2], -r[3]))
        history.append(dict(left=groups[i], right=groups[j], **proof, cross_pose_edges=[
            [p["peak_a"], p["peak_b"]] for p in pose_cross]))
        groups[i] = sorted(groups[i] + groups[j])
        groups.pop(j)
