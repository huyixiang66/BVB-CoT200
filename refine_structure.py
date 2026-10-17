"""Explicit development heuristic for structural background candidates, not identity truth."""

import re

POLICY = {
    "name_tokens": ["post", "finial", "frame", "headboard", "door", "window"],
    "excluded_tokens": ["cloth", "curtain", "pillow", "duvet", "pleat", "fold", "rib", "scroll"],
    "min_projected_span": 0.04,
    "max_points_per_mesh": 2,
}


def structural_name(name):
    words = set(re.findall(r"[a-z]+", name.lower()))
    return bool(words.intersection(POLICY["name_tokens"])) and not words.intersection(POLICY["excluded_tokens"])


def point_description(point, bounds, collections, kind):
    low, high = bounds["min_xyz"], bounds["max_xyz"]
    fractions = [(point[i] - low[i]) / (high[i] - low[i]) if high[i] > low[i] else 0.5
                 for i in range(3)]
    height = "upper" if fractions[2] >= 0.8 else "lower" if fractions[2] <= 0.2 else "middle-height"
    return {"kind": kind, "world_height_band": height,
            "collections": sorted(collections),
            "warning": "Geometry/name hints only; match the exact source point or reject."}
