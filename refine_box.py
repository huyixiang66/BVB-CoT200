"""Minimal target-box alignment. No background registration, QA, or GT."""

import itertools
import math

from refine_measurement import box_valid, top_left_box

POLICY = {"translation_bound_m": 0.9, "scale_bounds": [0.7, 1.3],
          "steps": [[0.3, 0.2], [0.15, 0.1]], "min_gain": 0.002}
EDGE_EPSILON = 1e-6


def extent_valid(box):
    return (isinstance(box, (list, tuple)) and len(box) == 4 and
            all(isinstance(v, (float, int)) and math.isfinite(v) for v in box) and
            box[0] < box[2] and box[1] < box[3])


def censored(value, index):
    return value <= EDGE_EPSILON if index < 2 else value >= 1-EDGE_EPSILON


def edge_error(actual, desired, edge_types=None):
    """Known occlusion/crop edges are bounds; unknown edges provide no constraint."""
    if not extent_valid(actual) or not box_valid(desired):
        return 1.0
    residuals = []
    for index, (a, d) in enumerate(zip(actual, desired)):
        if edge_types is not None and edge_types[index] == "unknown":
            residuals.append(0.0)
        elif censored(d, index) or (edge_types is not None and edge_types[index] != "silhouette"):
            residuals.append(max(0, a-d) if index < 2 else max(0, d-a))
        else:
            residuals.append(abs(a-d))
    return sum(residuals) / 4


def predict_edges(projected, initial, measured, edge_types=None):
    if not (extent_valid(projected) and extent_valid(initial) and box_valid(measured)):
        return None
    # A cropped baseline edge has no observed offset to calibrate.
    result = [p if censored(m, i) or (edge_types is not None and edge_types[i] != "silhouette") else p+m-b
              for i, (p, b, m) in enumerate(zip(projected, initial, measured))]
    return result if extent_valid(result) else None


def vector(box):
    return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2,
            box[2] - box[0], box[3] - box[1]]


def error(actual, desired):
    if not box_valid(actual) or not box_valid(desired):
        return 1.0
    return sum(abs(a - b) for a, b in zip(vector(actual), vector(desired))) / 4


def predicted_box(projected, initial_projected, measured_baseline):
    """Use projected changes, keeping the observed baseline box offset fixed."""
    if not all(box_valid(b) for b in (projected, initial_projected, measured_baseline)):
        return None
    current, initial, observed = map(vector, (projected, initial_projected, measured_baseline))
    cx, cy, width, height = [v + c - i for v, c, i in zip(observed, current, initial)]
    if width <= 0 or height <= 0:
        return None
    box = [max(0, cx - width / 2), max(0, cy - height / 2),
           min(1, cx + width / 2), min(1, cy + height / 2)]
    return box if box_valid(box) else None


def candidates(center, round_index):
    step, scale_step = POLICY["steps"][round_index]
    offsets = (-2, -1, 0, 1, 2) if round_index == 0 else (-1, 0, 1)
    values = {(0.0, 0.0, 1.0), tuple(center)}
    for x, y, s in itertools.product(offsets, offsets, (-1, 0, 1)):
        value = (round(center[0] + step * x, 6), round(center[1] + step * y, 6),
                 round(center[2] + scale_step * s, 6))
        if (math.hypot(value[0], value[1]) <= POLICY["translation_bound_m"] and
                POLICY["scale_bounds"][0] <= value[2] <= POLICY["scale_bounds"][1]):
            values.add(value)
    return [{"id": f"x{x:+.3f}_y{y:+.3f}_s{s:.3f}", "parameters": [x, y, s]}
            for x, y, s in sorted(values)]


def usable(row):
    return row.get("same_physical_target") is True and all(
        box_valid(row.get(k)) for k in ("original_box", "baseline_box"))


def select(rows, observation, initial_projected, metric="center_size"):
    if metric not in ("center_size", "censored_edges"):
        raise ValueError("Unknown box metric")
    scored = []
    for row in rows:
        if metric == "censored_edges":
            predicted = predict_edges(row["unclipped_bbox_top_left"], initial_projected,
                                      observation["baseline_box"], observation.get("baseline_edge_types"))
            loss = edge_error(predicted, observation["original_box"], observation.get("original_edge_types"))
        else:
            predicted = predicted_box(top_left_box(row["projected_bbox_xyxy"]),
                                      initial_projected, observation["baseline_box"])
            loss = error(predicted, observation["original_box"])
        if not row["first_hit_group_samples"]:
            loss = 1.0
        params = row["candidate"]["parameters"]
        magnitude = math.hypot(*params[:2]) + abs(params[2] - 1)
        scored.append({"candidate": row["candidate"], "loss": loss, "predicted_box": predicted,
                       "magnitude": magnitude})
    return min(scored, key=lambda p: (p["loss"], p["magnitude"])), scored


def select_multiview(records, observations, initial_projected, objective="mean", metric="center_size"):
    """Equal-weight fixed views; no distance labels or heldout views enter selection."""
    if not observations or len(observations) != len(initial_projected):
        raise ValueError("Need matching nonempty observation and projection lists")
    if objective not in ("mean", "worst_gain"):
        raise ValueError("Unknown multiview objective")
    baseline = [(edge_error(predict_edges(initial, initial, o["baseline_box"], o.get("baseline_edge_types")),
                            o["original_box"], o.get("original_edge_types"))
                 if metric == "censored_edges" else error(o["baseline_box"], o["original_box"]))
                for o, initial in zip(observations, initial_projected)]
    scored = []
    for rows in records:
        if len(rows) != len(observations) or any(r["candidate"] != rows[0]["candidate"] for r in rows):
            raise ValueError("Candidate/view mismatch")
        views = [select([row], observation, initial, metric=metric)[0] for row, observation, initial
                 in zip(rows, observations, initial_projected)]
        scored.append({"candidate": rows[0]["candidate"],
                       "loss": sum(v["loss"] for v in views) / len(views),
                       "view_losses": [v["loss"] for v in views],
                       "worst_gain": min(b-v["loss"] for b, v in zip(baseline, views)),
                       "predicted_boxes": [v["predicted_box"] for v in views],
                       "magnitude": views[0]["magnitude"]})
    key = (lambda p: (-round(p["worst_gain"], 12), p["loss"], p["magnitude"])) if objective == "worst_gain" else (
        lambda p: (p["loss"], p["magnitude"]))
    return min(scored, key=key), scored
