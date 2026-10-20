"""Shared action validation for the model runner and Blender's Python runtime."""

import math


def blender_intrinsic_settings(width, height, fx, fy, cx=None, cy=None):
    if min(width, height, fx, fy) <= 0:
        raise ValueError("Image size and focal lengths must be positive")
    cx, cy = width / 2 if cx is None else cx, height / 2 if cy is None else cy
    aspect = fx / fy
    return {"lens": fx * 36.0 / width, "sensor_width": 36.0,
            "pixel_aspect_x": max(1.0, 1/aspect), "pixel_aspect_y": max(1.0, aspect),
            "shift_x": (width / 2 - cx) / width,
            "shift_y": (cy - height / 2) * aspect / width}


def action_components(value):
    if value.get("subject") != "both":
        return [value]
    parts = []
    for subject in ("camera", "target"):
        part = value.get(subject)
        if not isinstance(part, dict):
            raise ValueError("Joint action requires camera and target objects")
        parts.append({"action": "adjust", "subject": subject,
                      "translation_local": part.get("translation_local"),
                      "rotation_local_degrees": part.get("rotation_local_degrees", [0, 0, 0]),
                      "scale_ratio": part.get("scale_ratio", 1)})
    return parts


def validate_action(value):
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    if value.get("action") == "no_change":
        return {"action": "no_change", "reason": value.get("reason", "")}
    if value.get("action") == "adjust" and value.get("subject") == "target":
        value = {**value, "rotation_local_degrees": value.get("rotation_local_degrees", [0, 0, 0])}
    if value.get("action") == "stop":
        return {"action": "stop", "reason": value.get("reason", "")}
    if value.get("action") == "adjust" and value.get("subject") == "both":
        for part in action_components(value):
            validate_action(part)
        return value
    if value.get("action") != "adjust" or value.get("subject") not in ("camera", "target"):
        raise ValueError("Expected stop or adjust camera/target")
    for key in ("translation_local", "rotation_local_degrees"):
        vector = value.get(key)
        if (not isinstance(vector, list) or len(vector) != 3 or
                any(type(x) not in (int, float) or not math.isfinite(x) for x in vector)):
            raise ValueError("Invalid vector: " + key)
    limit = .5 if value["subject"] == "camera" else .25
    if math.dist(value["translation_local"], [0, 0, 0]) > limit + 1e-9:
        raise ValueError("Translation exceeds the per-step bound")
    ratio = value.get("scale_ratio")
    if type(ratio) not in (int, float) or not math.isfinite(ratio) or not .8 <= ratio <= 1.25:
        raise ValueError("Scale must be within [0.8, 1.25]")
    if any(abs(x) > 8 for x in value["rotation_local_degrees"]):
        raise ValueError("Camera rotation exceeds eight degrees per axis")
    if value["subject"] == "camera" and ratio != 1:
        raise ValueError("Camera actions must have scale_ratio=1")
    if value["subject"] == "target" and any(value["rotation_local_degrees"]):
        raise ValueError("Target actions support translation and scale")
    return value
