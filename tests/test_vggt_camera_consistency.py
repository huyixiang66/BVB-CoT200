import unittest

import numpy as np

from vggt_camera_consistency import (
    check_geometry_registered_vggt_cameras,
    check_vggt_camera_consistency,
)


def rotation_z(degrees):
    angle = np.deg2rad(degrees)
    return np.array([[np.cos(angle), -np.sin(angle), 0.0],
                     [np.sin(angle), np.cos(angle), 0.0],
                     [0.0, 0.0, 1.0]])


def make_case(outlier=False):
    indices = [0, 10, 20]
    centers_cv = [np.array([0.0, 0.0, 0.0]),
                  np.array([1.0, 0.0, 0.0]),
                  np.array([1.0, 1.0, 0.0])]
    rotations_cv = [rotation_z(degree) for degree in (0, 10, 25)]
    world_rotation = rotation_z(35)
    scale = 2.0
    offset = np.array([3.0, -2.0, 1.0])
    extrinsics = []
    pairs = []
    basis = np.diag([1.0, -1.0, -1.0])
    for index, center, rotation in zip(indices, centers_cv, rotations_cv):
        c2w = np.eye(4)
        c2w[:3, :3] = rotation
        c2w[:3, 3] = center
        extrinsics.append(np.linalg.inv(c2w).tolist())
        pair_camera = np.eye(4)
        pair_camera[:3, :3] = world_rotation @ rotation @ basis
        pair_camera[:3, 3] = scale * world_rotation @ center + offset
        if outlier and index == 20:
            pair_camera[:3, 3] += np.array([0.5, 0.0, 0.0])
        pairs.append({"source": {"file": f"frame_{index:06d}.jpg"},
                      "camera_matrix": pair_camera.tolist()})
    return {"source_frame_indices_zero_based": indices,
            "world_to_camera_extrinsics": extrinsics}, pairs


def make_registration_case():
    summary, pairs = make_case()
    rotation = rotation_z(35)
    registration = {
        "vggt_to_blender_rotation": rotation.tolist(),
        "scale_blender_units_per_vggt_unit": 2.0,
        "vggt_to_blender_translation": [3.0, -2.0, 1.0],
    }
    return summary, pairs, registration


class VggtCameraConsistencyTests(unittest.TestCase):
    def test_consistent_similarity_passes_leave_one_out(self):
        summary, pairs = make_case()
        report = check_vggt_camera_consistency(summary, pairs)
        self.assertTrue(report["consistent"])
        self.assertEqual(report["evaluation"], "leave_one_camera_out")
        self.assertAlmostEqual(report["scale_m_per_vggt_unit_from_pair_cameras"], 2.0)
        self.assertLess(report["max_rotation_error_deg"], 1e-5)
        self.assertLess(report["max_position_error_m"], 1e-5)
        self.assertFalse(report["scan_pose_fit_used"])

    def test_inconsistent_holdout_fails(self):
        summary, pairs = make_case(outlier=True)
        report = check_vggt_camera_consistency(summary, pairs)
        self.assertFalse(report["consistent"])
        self.assertIn("position_inconsistency_or_insufficient_holdout", report["stop_reasons"])

    def test_two_views_use_pairwise_check_without_claiming_holdout(self):
        summary, pairs = make_case()
        report = check_vggt_camera_consistency(summary, pairs[:2])
        self.assertIn("insufficient_for_independent_holdout", report["evaluation"])

    def test_frame_outside_trajectory_is_rejected(self):
        summary, pairs = make_case()
        pairs[0]["source"]["file"] = "frame_000030.jpg"
        with self.assertRaisesRegex(ValueError, "outside"):
            check_vggt_camera_consistency(summary, pairs)

    def test_geometry_registered_cameras_pass_known_similarity(self):
        summary, pairs, registration = make_registration_case()
        report = check_geometry_registered_vggt_cameras(summary, pairs, registration)
        self.assertTrue(report["consistent"])
        self.assertFalse(report["scan_pose_fit_used"])
        self.assertLess(report["max_position_error_m"], 1e-5)
        self.assertLess(report["max_rotation_error_deg"], 1e-5)

    def test_geometry_registered_camera_error_fails(self):
        summary, pairs, registration = make_registration_case()
        pairs[0]["camera_matrix"][0][3] += 1.0
        report = check_geometry_registered_vggt_cameras(summary, pairs, registration)
        self.assertFalse(report["consistent"])
        self.assertIn("registered_camera_position_error", report["stop_reasons"])


if __name__ == "__main__":
    unittest.main()
