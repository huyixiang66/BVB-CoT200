"""Background-only registration must not absorb moving-object error."""

import unittest

import refine_alignment as a
import refine_measurement as m
from tests.test_partial_features import bound_features, partial_observation


def mapped(point):
    return [0.96 * point[0] - 0.08 * point[1] + 0.09,
            0.08 * point[0] + 0.96 * point[1] + 0.12]


def aligned_row():
    row = partial_observation()
    row["landmarks"] = [
        {"id": f"stationary-{i}", "baseline": xy, "original": mapped(xy)}
        for i, xy in enumerate(([0.1, 0.1], [0.7, 0.1], [0.1, 0.65],
                                [0.7, 0.65], [0.4, 0.15], [0.4, 0.65]))
    ]
    for f in row["features"]:
        f["original"] = mapped([f["baseline"][0] + 0.08, f["baseline"][1]])
    return row


class BackgroundAlignmentTests(unittest.TestCase):
    def test_large_shared_shift_is_validated_instead_of_blindly_rejected(self):
        row = aligned_row()
        result = a.background_alignment(row)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["model"], "similarity")
        self.assertLess(result["max_heldout_error"], 1e-8)
        recovered = a.inverse_point(mapped([0.4, 0.3]), result)
        self.assertAlmostEqual(recovered[0], 0.4)
        self.assertAlmostEqual(recovered[1], 0.3)

    def test_target_coordinates_cannot_change_background_fit(self):
        row = aligned_row()
        before = a.background_alignment(row)
        row["features"][0]["original"] = [0.99, 0.99]
        self.assertEqual(a.background_alignment(row), before)

    def test_target_residual_survives_alignment_and_can_select_a_change(self):
        row = aligned_row()
        evidence = m.partial_measurement(row, bound_features(), align_background=True)
        self.assertIsNotNone(evidence)
        baseline = m.partial_loss(bound_features(), evidence)
        better = [{**f, "xy": [f["xy"][0] + 0.08, f["xy"][1]]} for f in bound_features()]
        improved = m.partial_loss(better, evidence)
        self.assertGreater(baseline, 0.03)
        self.assertAlmostEqual(improved, 0)
        self.assertEqual(m.choose({"no_change": [baseline] * 2, "move": [improved] * 2}), "move")

    def test_shared_motion_alone_never_creates_object_error(self):
        row = aligned_row()
        for f in row["features"]:
            f["original"] = mapped(f["baseline"])
        evidence = m.partial_measurement(row, bound_features(), align_background=True)
        self.assertAlmostEqual(m.partial_loss(bound_features(), evidence), 0)

    def test_holdout_inconsistency_and_parallax_abstain(self):
        for corruption in ("outlier", "shear"):
            row = aligned_row()
            if corruption == "outlier":
                row["landmarks"][-1]["original"] = [0.9, 0.9]
            else:
                for p in row["landmarks"]:
                    x, y = p["baseline"]
                    p["original"] = [x + 0.3 * y, y]
            result = a.background_alignment(row)
            self.assertFalse(result["accepted"])
            self.assertEqual(result["reason"], "heldout_landmark_mismatch")

    def test_too_few_duplicate_or_collinear_landmarks_abstain(self):
        row = aligned_row()
        for landmarks in (row["landmarks"][:3], [row["landmarks"][0]] * 6,
                          [{"id": str(i), "baseline": [0.1 * i, 0.2],
                            "original": [0.1 * i, 0.3]} for i in range(6)]):
            self.assertFalse(a.background_alignment({"landmarks": landmarks})["accepted"])

    def test_nearest_extrapolation_is_limited_without_complete_target_boxes(self):
        row = aligned_row()
        result = a.background_alignment(row)
        self.assertTrue(a.supports_point([0.4, 0.3], result))
        self.assertFalse(a.supports_point([0.99, 0.99], result))

    def test_large_scale_change_is_not_accepted_as_camera_calibration(self):
        row = aligned_row()
        for p in row["landmarks"]:
            p["original"] = [0.4 * x + 0.1 for x in p["baseline"]]
        self.assertFalse(a.background_alignment(row)["accepted"])

    def test_translation_is_preferred_when_it_predicts_holdouts(self):
        row = aligned_row()
        for p in row["landmarks"]:
            p["original"] = [p["baseline"][0] + 0.08, p["baseline"][1] + 0.15]
        result = a.background_alignment(row)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["model"], "translation")
        self.assertEqual(len(result["attempts"]), 1)

    def test_invalid_coordinates_and_reflection_are_rejected(self):
        for bad in (True, float("nan"), float("inf"), -1):
            row = aligned_row()
            row["landmarks"][0]["original"][0] = bad
            self.assertFalse(a.background_alignment(row)["accepted"])
        row = aligned_row()
        for p in row["landmarks"]:
            p["original"] = [1 - p["baseline"][0], p["baseline"][1]]
        self.assertFalse(a.background_alignment(row)["accepted"])


if __name__ == "__main__":
    unittest.main()
