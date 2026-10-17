"""Conservative background-relative 2D residuals; no images or GT are read here."""

import math
import statistics

POLICY = {
    "max_background_offset": 0.04,
    "max_background_residual": 0.025,
    "max_baseline_box_error": 0.06,
    "min_improvement": 0.015,
}


def coordinates(value, count):
    return (isinstance(value, list) and len(value) == count
            and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in value))


def box_valid(box):
    return coordinates(box, 4) and box[2] > box[0] and box[3] > box[1]


def top_left_box(box):
    if not box_valid(box):
        return None
    return [box[0], 1 - box[3], box[2], 1 - box[1]]


def background_measurement(row):
    """Only small background translations are tolerated; never fit the target."""
    landmarks = row.get("landmarks", [])
    if not isinstance(landmarks, list) or len(landmarks) < 3:
        return None
    if any(not isinstance(p, dict) or not isinstance(p.get("id"), str)
           or not coordinates(p.get("original"), 2) or not coordinates(p.get("baseline"), 2)
           for p in landmarks):
        return None
    if len({p["id"] for p in landmarks}) != len(landmarks):
        return None
    points = [p["baseline"] for p in landmarks]
    area = max(abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
               for a in points for b in points for c in points)
    if area < 0.02:
        return None
    offsets = [[p["original"][axis] - p["baseline"][axis] for axis in range(2)] for p in landmarks]
    offset = [statistics.median(p[axis] for p in offsets) for axis in range(2)]
    residual = max(math.dist(p, offset) for p in offsets)
    if max(map(abs, offset)) > POLICY["max_background_offset"] or residual > POLICY["max_background_residual"]:
        return None
    return {"background_offset": offset, "background_residual": residual}


def measurement(row, geometry_box):
    original, baseline = row.get("target_original"), row.get("target_baseline")
    geometry = top_left_box(geometry_box)
    if not all(box_valid(box) for box in (original, baseline, geometry)):
        return None
    background = background_measurement(row)
    if background is None:
        return None
    offset = background["background_offset"]
    if max(abs(a - b) for a, b in zip(baseline, geometry)) > POLICY["max_baseline_box_error"]:
        return None
    # Reject truncated boxes: clipping makes candidate motion unidentifiable.
    if any(box[0] <= 0.005 or box[1] <= 0.005 or box[2] >= 0.995 or box[3] >= 0.995
           for box in (original, baseline, geometry)):
        return None
    desired = [g + o - b - offset[i % 2] for i, (g, o, b) in
               enumerate(zip(geometry, original, baseline))]
    if not box_valid(desired):
        return None
    return {"desired_box_top_left": desired, **background, "geometry_baseline_top_left": geometry}


def partial_measurement(row, bound, align_background=False):
    """Freeze shared visible feature targets; complete-object boxes are irrelevant."""
    if align_background:
        import refine_alignment as alignment
        result = alignment.background_alignment(row)
        background = {"background_alignment": result} if result["accepted"] else None
    else:
        background = background_measurement(row)
    features = row.get("features", [])
    if row.get("same_physical_target") is not True or background is None:
        return None
    if not isinstance(features, list) or not 2 <= len(features) <= 6:
        return None
    if any(not isinstance(f, dict) or not isinstance(f.get("id"), str) or not f["id"].strip()
           or not coordinates(f.get("original"), 2) or not coordinates(f.get("baseline"), 2)
           or f.get("same_physical_point") is not True for f in features):
        return None
    ids = [f["id"] for f in features]
    if len(set(ids)) != len(ids) or len({f["id"] for f in bound}) != len(bound):
        return None
    by_id = {f["id"]: f for f in bound}
    retained = []
    for feature in features:
        anchor = by_id.get(feature["id"])
        if (not anchor or anchor.get("visible") is not True or not coordinates(anchor.get("xy"), 2)
                or math.dist(anchor["xy"], feature["baseline"]) > 0.005):
            continue
        if align_background:
            desired = alignment.inverse_point(feature["original"], result)
            if not alignment.supports_point(anchor["xy"], result) or not alignment.supports_point(desired, result):
                continue
        else:
            desired = [feature["original"][i] - background["background_offset"][i] for i in range(2)]
        if coordinates(desired, 2):
            retained.append({**anchor, "desired_xy": desired})
    if len(retained) < 2 or max(math.dist(a["xy"], b["xy"]) for a in retained for b in retained) < 0.03:
        return None
    return {**background, "features": retained}


def partial_loss(projected, evidence):
    lookup = {f["id"]: f for f in projected}
    if len(lookup) != len(projected):
        return 1.0
    errors = []
    for feature in evidence["features"]:
        actual = lookup.get(feature["id"])
        if not actual or actual.get("visible") is not True or not coordinates(actual.get("xy"), 2):
            return 1.0
        errors.extend(abs(actual["xy"][i] - feature["desired_xy"][i]) for i in range(2))
    return sum(errors) / len(errors) if errors else 1.0


def loss(geometry_box, evidence):
    box = top_left_box(geometry_box)
    if box is None:
        return math.inf
    return sum(abs(a - b) for a, b in zip(box, evidence["desired_box_top_left"])) / 4


def choose(scores):
    baseline = scores["no_change"]
    if not baseline or any(not math.isfinite(v) for v in baseline):
        return "no_change"
    eligible = [
        key for key, values in scores.items()
        if key != "no_change" and len(values) == len(baseline)
        and all(math.isfinite(v) and b - v >= POLICY["min_improvement"]
                for b, v in zip(baseline, values))
    ]
    return min(eligible, key=lambda key: (sum(scores[key]), key)) if eligible else "no_change"
