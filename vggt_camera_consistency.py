"""Check VGGT camera motion against paired cameras without inspecting images."""

import math
import re

import numpy as np


_FRAME_PATTERN = re.compile(r"frame_(\d+)")
_CV_TO_BLENDER_CAMERA = np.diag([1.0, -1.0, -1.0])


def _rotation_angle(matrix):
    cosine = np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def _project_rotation(matrix):
    left, _, right = np.linalg.svd(matrix)
    result = left @ right
    if np.linalg.det(result) < 0:
        left[:, -1] *= -1
        result = left @ right
    return result


def _slerp(first, second, weight):
    """Interpolate two rotation matrices using a short-arc quaternion slerp."""
    relative = first.T @ second
    angle = math.acos(float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)))
    if angle < 1e-8:
        return first.copy()
    axis = np.array([
        relative[2, 1] - relative[1, 2],
        relative[0, 2] - relative[2, 0],
        relative[1, 0] - relative[0, 1],
    ]) / (2.0 * math.sin(angle))
    theta = angle * weight
    skew = np.array([[0.0, -axis[2], axis[1]],
                     [axis[2], 0.0, -axis[0]],
                     [-axis[1], axis[0], 0.0]])
    step = np.eye(3) + math.sin(theta) * skew + (1.0 - math.cos(theta)) * (skew @ skew)
    return first @ step


def _sample_pose(frame, indices, centers, rotations):
    if frame < indices[0] or frame > indices[-1]:
        raise ValueError("Source frame lies outside the VGGT trajectory")
    right = int(np.searchsorted(indices, frame, side="right"))
    left = max(0, min(right - 1, len(indices) - 2))
    right = left + 1
    span = indices[right] - indices[left]
    weight = (frame - indices[left]) / span
    center = centers[left] * (1.0 - weight) + centers[right] * weight
    rotation = _slerp(rotations[left], rotations[right], weight)
    return center, rotation


def _fit_similarity(vggt_centers, blender_centers, vggt_rotations, blender_rotations):
    world_rotation = _project_rotation(sum(
        blender @ vggt.T for vggt, blender in zip(vggt_rotations, blender_rotations)
    ))
    rotated = np.array([world_rotation @ center for center in vggt_centers])
    source_mean = rotated.mean(axis=0)
    target_mean = blender_centers.mean(axis=0)
    source_centered = rotated - source_mean
    target_centered = blender_centers - target_mean
    denominator = float(np.sum(source_centered * source_centered))
    if denominator < 1e-10:
        return world_rotation, None, None, None
    scale = float(np.sum(source_centered * target_centered) / denominator)
    translation = target_mean - scale * source_mean
    predicted = scale * rotated + translation
    errors = np.linalg.norm(predicted - blender_centers, axis=1)
    return world_rotation, scale, translation, errors


def check_vggt_camera_consistency(summary, pairs, max_rotation_error_deg=15.0,
                                  max_position_error_m=0.2):
    """Return a numeric gate report from frame indices and camera matrices only."""
    indices = np.asarray(summary["source_frame_indices_zero_based"], dtype=float)
    extrinsics = np.asarray(summary["world_to_camera_extrinsics"], dtype=float)
    if extrinsics.shape != (len(indices), 4, 4) or len(indices) < 2:
        raise ValueError("VGGT summary must contain one 4x4 extrinsic per frame")
    if np.any(np.diff(indices) <= 0):
        raise ValueError("VGGT frame indices must be strictly increasing")
    camera_to_world = np.linalg.inv(extrinsics)
    centers = camera_to_world[:, :3, 3]
    rotations = camera_to_world[:, :3, :3] @ _CV_TO_BLENDER_CAMERA

    sampled_centers, sampled_rotations, blender_centers, blender_rotations = [], [], [], []
    pair_frames = []
    for pair in pairs:
        match = _FRAME_PATTERN.search(pair["source"]["file"])
        if not match:
            raise ValueError("Source frame path must contain frame_<index>")
        frame = int(match.group(1))
        center, rotation = _sample_pose(frame, indices, centers, rotations)
        matrix = np.asarray(pair["camera_matrix"], dtype=float)
        if matrix.shape != (4, 4):
            raise ValueError("Each pair camera_matrix must be 4x4")
        pair_frames.append(frame)
        sampled_centers.append(center)
        sampled_rotations.append(rotation)
        blender_centers.append(matrix[:3, 3])
        blender_rotations.append(matrix[:3, :3])

    if len(pairs) < 2:
        raise ValueError("At least two paired camera poses are required")
    sampled_centers = np.asarray(sampled_centers)
    blender_centers = np.asarray(blender_centers)
    fit = _fit_similarity(sampled_centers, blender_centers,
                          sampled_rotations, blender_rotations)
    _, scale, _, in_sample_errors = fit
    relative_rotation_errors = []
    for first in range(len(pairs)):
        for second in range(first + 1, len(pairs)):
            vggt_relative = sampled_rotations[first].T @ sampled_rotations[second]
            blender_relative = blender_rotations[first].T @ blender_rotations[second]
            relative_rotation_errors.append(_rotation_angle(vggt_relative.T @ blender_relative))

    report = {
        "protocol": "vggt_camera_pair_consistency_v1",
        "pair_frames": pair_frames,
        "pair_count": len(pairs),
        "registration_anchors": "provided camera_matrix values only",
        "scan_pose_fit_used": False,
        "relative_rotation_error_deg": relative_rotation_errors,
        "scale_m_per_vggt_unit_from_pair_cameras": scale,
        "in_sample_position_errors_m": (in_sample_errors.tolist()
                                          if in_sample_errors is not None else None),
    }

    if len(pairs) >= 3:
        loo = []
        for held_out in range(len(pairs)):
            keep = [index for index in range(len(pairs)) if index != held_out]
            q, loo_scale, translation, _ = _fit_similarity(
                sampled_centers[keep], blender_centers[keep],
                [sampled_rotations[index] for index in keep],
                [blender_rotations[index] for index in keep],
            )
            if loo_scale is None:
                position_error = None
            else:
                predicted_center = loo_scale * (q @ sampled_centers[held_out]) + translation
                position_error = float(np.linalg.norm(predicted_center - blender_centers[held_out]))
            rotation_error = _rotation_angle(
                (q @ sampled_rotations[held_out]).T @ blender_rotations[held_out]
            )
            loo.append({"pair_frame": pair_frames[held_out],
                        "rotation_error_deg": rotation_error,
                        "position_error_m": position_error,
                        "scale_m_per_vggt_unit": loo_scale})
        report["leave_one_out"] = loo
        rotation_errors = [row["rotation_error_deg"] for row in loo]
        position_errors = [row["position_error_m"] for row in loo]
        report["max_rotation_error_deg"] = max(rotation_errors)
        report["max_position_error_m"] = (max(position_errors)
                                           if all(value is not None for value in position_errors)
                                           else None)
        report["evaluation"] = "leave_one_camera_out"
    else:
        report["max_rotation_error_deg"] = max(relative_rotation_errors)
        report["max_position_error_m"] = (float(np.max(in_sample_errors))
                                           if in_sample_errors is not None else None)
        report["evaluation"] = "pairwise_check_only_insufficient_for_independent_holdout"

    reasons = []
    if len(pairs) < 3:
        reasons.append("insufficient_independent_camera_holdout")
    if report["max_rotation_error_deg"] > max_rotation_error_deg:
        reasons.append("rotation_inconsistency")
    if scale is None or scale <= 0:
        reasons.append("nonpositive_or_unobservable_scale")
    position_error = report["max_position_error_m"]
    if position_error is None or position_error > max_position_error_m:
        reasons.append("position_inconsistency_or_insufficient_holdout")
    report["consistent"] = not reasons
    report["stop_reasons"] = reasons
    report["thresholds"] = {"max_rotation_error_deg": max_rotation_error_deg,
                            "max_position_error_m": max_position_error_m}
    return report


def check_geometry_registered_vggt_cameras(summary, pairs, registration,
                                           max_rotation_error_deg=20.0,
                                           max_position_error_m=0.5):
    """Validate VGGT camera poses after a point-cloud-to-Blender geometry fit."""
    indices = np.asarray(summary["source_frame_indices_zero_based"], dtype=float)
    extrinsics = np.asarray(summary["world_to_camera_extrinsics"], dtype=float)
    if extrinsics.shape != (len(indices), 4, 4) or len(indices) < 2:
        raise ValueError("VGGT summary must contain one 4x4 extrinsic per frame")
    camera_to_world = np.linalg.inv(extrinsics)
    centers = camera_to_world[:, :3, 3]
    rotations = camera_to_world[:, :3, :3] @ _CV_TO_BLENDER_CAMERA
    world_rotation = np.asarray(registration["vggt_to_blender_rotation"], dtype=float)
    scale = float(registration["scale_blender_units_per_vggt_unit"])
    translation = np.asarray(registration["vggt_to_blender_translation"], dtype=float)
    if world_rotation.shape != (3, 3) or translation.shape != (3,) or scale <= 0:
        raise ValueError("Invalid VGGT-to-Blender similarity transform")

    records = []
    for pair in pairs:
        match = _FRAME_PATTERN.search(pair["source"]["file"])
        if not match:
            raise ValueError("Source frame path must contain frame_<index>")
        frame = int(match.group(1))
        center, rotation = _sample_pose(frame, indices, centers, rotations)
        matrix = np.asarray(pair["camera_matrix"], dtype=float)
        predicted_center = scale * (world_rotation @ center) + translation
        predicted_rotation = world_rotation @ rotation
        records.append({
            "pair_id": pair.get("pair_id"), "frame": frame,
            "position_error_m": float(np.linalg.norm(predicted_center - matrix[:3, 3])),
            "rotation_error_deg": _rotation_angle(predicted_rotation.T @ matrix[:3, :3]),
        })
    if not records:
        raise ValueError("At least one paired camera pose is required")
    max_position = max(record["position_error_m"] for record in records)
    max_rotation = max(record["rotation_error_deg"] for record in records)
    reasons = []
    if max_position > max_position_error_m:
        reasons.append("registered_camera_position_error")
    if max_rotation > max_rotation_error_deg:
        reasons.append("registered_camera_orientation_error")
    return {
        "protocol": "vggt_geometry_registered_camera_check_v1",
        "registration_source": "VGGT depth points fitted to Blender scene vertices",
        "scan_pose_fit_used": False,
        "camera_pose_count": len(records), "camera_errors": records,
        "max_position_error_m": max_position,
        "max_rotation_error_deg": max_rotation,
        "consistent": not reasons, "stop_reasons": reasons,
        "thresholds": {"max_position_error_m": max_position_error_m,
                       "max_rotation_error_deg": max_rotation_error_deg},
    }
