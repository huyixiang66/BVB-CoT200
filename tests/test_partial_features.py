"""Partial observations constrain a frozen transform group, not a complete bbox."""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import refine_measurement as m
import refine_measured as runner
import refinement as r
from tests.test_refine_measurement import observation


def partial_observation():
    return {
        "same_physical_target": True,
        "features": [
            {"id": "left top corner", "original": [0.4, 0.3], "baseline": [0.3, 0.3],
             "same_physical_point": True},
            {"id": "right top corner", "original": [0.6, 0.3], "baseline": [0.5, 0.3],
             "same_physical_point": True},
        ],
        "landmarks": observation()["landmarks"],
        "target_original": None, "target_baseline": None,
    }


def bound_features():
    return [{"id": f["id"], "xy": f["baseline"], "visible": True,
             "object_name": "top", "local_point": [i, 0, 0]}
            for i, f in enumerate(partial_observation()["features"])]


class PartialFeatureTests(unittest.TestCase):
    def test_partial_runner_scores_anchors_and_validates_before_applying(self):
        from tests.test_refinement import fixture_case
        for winner, aligned in (("proposal", False), ("tie", False), ("proposal", True), ("tie", True)):
            with self.subTest(winner=winner, aligned=aligned), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                parent, run = root / "parent", root / "run"
                case = fixture_case(root)
                opaque = root / "opaque.bin"
                opaque.write_bytes(b"synthetic-not-image")
                pairs = [{"pair_id": i, "scene_frame": i + 1, "normalized_time": i / 3,
                          "original_frame_index_0based": i} for i in range(3)]
                records = [{**p, "file": str(opaque), "sha256": r.digest(opaque)} for p in pairs]
                r.save_json(parent / "config.json", {})
                r.save_json(parent / "preflight.json", {"after_gap_m": 2.0})
                r.save_json(parent / "proposed_pairs.json", pairs)
                searched_ids = []

                def project(case, directory, blender, selected, candidates, observations=None, anchors=None):
                    if anchors is not None:
                        searched_ids.append([p["pair_id"] for p in selected])
                        self.assertTrue(all(str(p["scene_frame"]) in anchors for p in selected))
                    return {c["id"]: {p["pair_id"]: {
                        "features": [{**f, "xy": [f["xy"][0] + (0.1 if c["id"] == "toward_030" else 0),
                                                 f["xy"][1]]} for f in bound_features()],
                    } for p in selected} for c in candidates}

                validation = {"pairs": [{"pair_id": 2, "winner": winner, "moving_visible_both": True,
                    "same_physical_target": True, "background_consistent": True,
                    "position_or_size_better": True, "no_new_artifacts": True}]}
                annotation = partial_observation()
                if aligned:
                    annotation["landmarks"] = [
                        {"id": str(i), "original": xy, "baseline": xy} for i, xy in enumerate(
                            ([0.1, 0.1], [0.7, 0.1], [0.1, 0.65], [0.7, 0.65], [0.4, 0.15], [0.4, 0.65]))]
                with patch.object(runner, "verify_parent", return_value=case), \
                        patch.object(r, "extract_frames", return_value=records), \
                        patch.object(runner.it, "render_pair_frames", side_effect=lambda c, d, b, n, ps, ca:
                                     [records[p["pair_id"]] for p in ps]), \
                        patch.object(runner, "project", side_effect=project), \
                        patch.object(runner, "measure_pair", return_value=annotation) as measure, \
                        patch.object(runner.it, "judge", return_value=validation), \
                        patch.object(runner.it, "geometry_stage", return_value={
                            "after_state": {"surface_distance_m": 1.7}, "scene_after_blend": "accepted.blend"}) as apply, \
                        patch.object(r, "answer") as answer:
                    outcome = runner.run_case(parent, run, "unused", "model", "https://example.test",
                                              True, True, aligned)
                self.assertEqual(outcome["accepted_edits"], int(winner == "proposal"))
                self.assertEqual(apply.call_count, int(winner == "proposal"))
                self.assertTrue(all(call.kwargs["partial"] for call in measure.call_args_list))
                self.assertEqual(searched_ids, [[0, 1], [2]])
                if aligned:
                    self.assertEqual(outcome["protocol"], "partial_background_alignment_v2")
                    self.assertTrue(all(call.kwargs["align_background"] for call in measure.call_args_list))
                    self.assertTrue(r.read_json(run / "background_0.json")["accepted"])
                answer.assert_not_called()

    def test_cropped_object_without_boxes_can_constrain_local_features(self):
        evidence = m.partial_measurement(partial_observation(), bound_features())
        self.assertIsNotNone(evidence)
        improved = [{**f, "xy": [f["xy"][0] + 0.1, f["xy"][1]]} for f in bound_features()]
        self.assertAlmostEqual(m.partial_loss(improved, evidence), 0)
        self.assertGreater(m.partial_loss(bound_features(), evidence), 0.04)

    def test_unmatched_or_occluded_features_do_not_become_zero_loss(self):
        for key, value in (("same_physical_target", False), ("same_physical_target", "true"),
                           ("features", [])):
            row = {**partial_observation(), key: value}
            self.assertIsNone(m.partial_measurement(row, bound_features()))
        evidence = m.partial_measurement(partial_observation(), bound_features())
        self.assertEqual(m.partial_loss([], evidence), 1.0)
        hidden = [{**f, "visible": False} for f in bound_features()]
        self.assertEqual(m.partial_loss(hidden, evidence), 1.0)

    def test_duplicate_ids_wrong_binding_and_background_motion_are_rejected(self):
        row = partial_observation()
        row["features"][1]["id"] = row["features"][0]["id"]
        self.assertIsNone(m.partial_measurement(row, bound_features()))
        wrong = [{**f, "xy": [0.8, 0.8]} for f in bound_features()]
        self.assertIsNone(m.partial_measurement(partial_observation(), wrong))
        row = partial_observation()
        row["landmarks"][0]["original"] = [0.9, 0.9]
        self.assertIsNone(m.partial_measurement(row, bound_features()))

    def test_small_background_shift_is_not_target_motion(self):
        row = partial_observation()
        for point in row["landmarks"]:
            point["original"][0] += 0.02
        for feature in row["features"]:
            feature["original"] = [feature["baseline"][0] + 0.02, feature["baseline"][1]]
        evidence = m.partial_measurement(row, bound_features())
        self.assertAlmostEqual(m.partial_loss(bound_features(), evidence), 0)


if __name__ == "__main__":
    unittest.main()
