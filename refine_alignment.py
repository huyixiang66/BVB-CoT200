"""Background-only 2D nuisance alignment. Not a recovered physical camera pose."""

import itertools
import math

import numpy as np

POLICY = {
    "min_landmarks": 4, "max_landmarks": 12,
    "min_triangle_double_area": 0.01,
    "max_heldout_error": 0.02,
    "min_scale": 0.75, "max_scale": 1.33,
    "max_rotation_degrees": 15,
    "max_support_distance": 0.25,
    "support_bbox_margin": 0.05,
}


def _point(value):
    return (isinstance(value, list) and len(value) == 2 and
            all(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1 for x in value))


def _area(points):
    return max(abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
               for a, b, c in itertools.combinations(points, 3))


def _fit(source, target, model):
    if model == "translation":
        delta = np.mean(np.asarray(target) - np.asarray(source), axis=0)
        return [1.0, 0.0, float(delta[0]), float(delta[1])]
    design, values = [], []
    for (x, y), (u, v) in zip(source, target):
        design.extend(([x, -y, 1, 0], [y, x, 0, 1]))
        values.extend((u, v))
    solution, _, rank, _ = np.linalg.lstsq(design, values, rcond=None)
    return solution.tolist() if rank == 4 else None


def _forward(point, parameters):
    a, b, tx, ty = parameters
    x, y = point
    return [a * x - b * y + tx, b * x + a * y + ty]


def _plausible(parameters):
    if parameters is None:
        return False
    a, b, _, _ = parameters
    return (POLICY["min_scale"] <= math.hypot(a, b) <= POLICY["max_scale"] and
            abs(math.degrees(math.atan2(b, a))) <= POLICY["max_rotation_degrees"])


def background_alignment(row):
    landmarks = row.get("landmarks", [])
    if not isinstance(landmarks, list) or not POLICY["min_landmarks"] <= len(landmarks) <= POLICY["max_landmarks"]:
        return {"accepted": False, "reason": "insufficient_or_excess_landmarks"}
    if any(not isinstance(p, dict) or not isinstance(p.get("id"), str) or not p["id"].strip()
           or not _point(p.get("original")) or not _point(p.get("baseline")) for p in landmarks):
        return {"accepted": False, "reason": "invalid_landmarks"}
    if len({p["id"] for p in landmarks}) != len(landmarks):
        return {"accepted": False, "reason": "duplicate_landmark_ids"}
    source = [p["baseline"] for p in landmarks]
    target = [p["original"] for p in landmarks]
    if min(_area(source), _area(target)) < POLICY["min_triangle_double_area"]:
        return {"accepted": False, "reason": "degenerate_landmark_geometry"}
    attempts = []
    # Every landmark is predicted once without participating in that fit.
    for model in ("translation", "similarity"):
        errors = []
        valid = True
        for heldout in range(len(source)):
            train = [i for i in range(len(source)) if i != heldout]
            train_source = [source[i] for i in train]
            train_target = [target[i] for i in train]
            if min(_area(train_source), _area(train_target)) < POLICY["min_triangle_double_area"]:
                valid = False
                break
            parameters = _fit(train_source, train_target, model)
            if not _plausible(parameters):
                valid = False
                break
            errors.append(math.dist(_forward(source[heldout], parameters), target[heldout]))
        error = max(errors) if valid else None
        attempts.append({"model": model, "max_heldout_error": error, "valid_folds": valid})
        if valid and error <= POLICY["max_heldout_error"]:
            parameters = _fit(source, target, model)
            if not _plausible(parameters):
                continue
            return {"accepted": True, "model": model, "parameters": parameters,
                    "max_heldout_error": error, "support_baseline": source,
                    "landmark_ids": [p["id"] for p in landmarks], "attempts": attempts}
    return {"accepted": False, "reason": "heldout_landmark_mismatch", "attempts": attempts}


def inverse_point(point, alignment):
    a, b, tx, ty = alignment["parameters"]
    x, y = point[0] - tx, point[1] - ty
    denominator = a * a + b * b
    return [(a * x + b * y) / denominator, (-b * x + a * y) / denominator]


def supports_point(point, alignment):
    points = alignment["support_baseline"]
    margin = POLICY["support_bbox_margin"]
    return (all(min(p[i] for p in points) - margin <= point[i] <= max(p[i] for p in points) + margin
                for i in range(2)) and
            min(math.dist(point, p) for p in points) <= POLICY["max_support_distance"])
