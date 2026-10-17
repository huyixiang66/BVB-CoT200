"""Numeric evidence gates, without pixels, model calls, or benchmark labels."""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import refine_measurement as m
import refine_measured as runner
import refinement as r


def observation():
    return {
        "target_original": [0.4, 0.3, 0.6, 0.5],
        "target_baseline": [0.3, 0.3, 0.5, 0.5],
        "landmarks": [
            {"id": "a", "original": [0.1, 0.1], "baseline": [0.1, 0.1]},
            {"id": "b", "original": [0.8, 0.1], "baseline": [0.8, 0.1]},
            {"id": "c", "original": [0.1, 0.8], "baseline": [0.1, 0.8]},
        ],
    }


class MeasurementTests(unittest.TestCase):
    def test_upfront_selection_skips_bad_first_view_without_using_error_size(self):
        pairs = [{"pair_id": i, "normalized_time": i * 0.2} for i in range(5)]
        measurements = {0: None, 1: {"error": 100}, 2: {"error": 0},
                        3: {"error": 10}, 4: {"error": 1}}
        split = runner.split_measured(pairs, measurements)
        self.assertEqual(split["train"], pairs[1:3])
        self.assertEqual(split["holdouts"], pairs[3:])
        self.assertEqual(runner.split_measured(pairs, {i: None for i in range(5)}),
                         {"train": [], "holdouts": []})

    def test_upfront_selection_rejects_missing_observations(self):
        with self.assertRaises(ValueError):
            runner.split_measured([{"pair_id": 0, "normalized_time": 0}], {})

    def test_target_error_is_not_removed_as_background_motion(self):
        evidence = m.measurement(observation(), [0.3, 0.5, 0.5, 0.7])
        self.assertIsNotNone(evidence)
        self.assertAlmostEqual(m.loss([0.4, 0.5, 0.6, 0.7], evidence), 0)
        self.assertGreater(m.loss([0.3, 0.5, 0.5, 0.7], evidence), 0.04)

    def test_background_mismatch_and_invalid_coordinates_abstain(self):
        row = observation()
        row["landmarks"][2]["original"] = [0.6, 0.8]
        self.assertIsNone(m.measurement(row, [0.3, 0.5, 0.5, 0.7]))
        for bad in (None, [0, 0, 0, 1], [0, 0, float("nan"), 1], [0, 0, True, 1]):
            row = observation()
            row["target_original"] = bad
            self.assertIsNone(m.measurement(row, [0.3, 0.5, 0.5, 0.7]))

    def test_wrong_geometry_instance_or_degenerate_landmarks_abstain(self):
        self.assertIsNone(m.measurement(observation(), [0.7, 0.5, 0.9, 0.7]))
        row = observation()
        row["landmarks"] = [row["landmarks"][0]] * 3
        self.assertIsNone(m.measurement(row, [0.3, 0.5, 0.5, 0.7]))

    def test_candidate_must_improve_each_view_beyond_uncertainty(self):
        base = [0.08, 0.08]
        self.assertEqual(m.choose({"no_change": base, "good": [0.01, 0.02]}), "good")
        self.assertEqual(m.choose({"no_change": base, "tradeoff": [0, 0.09]}), "no_change")
        self.assertEqual(m.choose({"no_change": base, "noise": [0.075, 0.075]}), "no_change")
        self.assertEqual(m.choose({"no_change": base, "missing": [0]}), "no_change")
        self.assertEqual(m.choose({"no_change": base, "invalid": [float("nan"), 0]}), "no_change")

    def test_upfront_runner_checks_all_views_before_candidate_search(self):
        from tests.test_refinement import fixture_case
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent, run = root / "parent", root / "measured"
            case = fixture_case(root)
            opaque = root / "opaque.bin"
            opaque.write_bytes(b"not-image-content")
            pairs = [{"pair_id": i, "scene_frame": i + 1, "normalized_time": i / 4,
                      "original_frame_index_0based": i} for i in range(4)]
            records = [{**p, "file": str(opaque), "sha256": r.digest(opaque)} for p in pairs]
            r.save_json(parent / "config.json", {})
            r.save_json(parent / "preflight.json", {"after_gap_m": 2.0})
            r.save_json(parent / "proposed_pairs.json", pairs)
            events = []

            def measure(case, run, model, url, pair, original, baseline):
                events.append(("measure", pair["pair_id"]))
                return {**observation(), "target_original": None} if pair["pair_id"] == 0 else observation()

            def project(case, run, blender, selected, candidates):
                events.append(("project", [p["pair_id"] for p in selected]))
                return {c["id"]: {p["pair_id"]: {"first_hit_group_samples": 3,
                        "projected_bbox_xyxy": [0.3, 0.5, 0.5, 0.7]} for p in selected}
                        for c in candidates}

            with patch.object(runner, "verify_parent", return_value=case), \
                    patch.object(r, "extract_frames", return_value=records), \
                    patch.object(runner.it, "render_pair_frames", return_value=records), \
                    patch.object(runner, "project", side_effect=project), \
                    patch.object(runner, "measure_pair", side_effect=measure), \
                    patch.object(r, "answer") as answer:
                outcome = runner.run_case(parent, run, "unused", "model", "https://example.test",
                                          measure_before_split=True)
            self.assertEqual(events, [("project", [0, 1, 2, 3]), ("measure", 0), ("measure", 1),
                                      ("measure", 2), ("measure", 3), ("project", [1, 2])])
            split = r.read_json(run / "evidence_split.json")
            self.assertEqual(split["train"], pairs[1:3])
            self.assertEqual(split["holdouts"], pairs[3:])
            self.assertEqual(outcome["measurable_pairs"], 3)
            self.assertEqual(outcome["stop_reason"], "no_measurable_training_improvement")
            answer.assert_not_called()

    def test_blind_runner_accepts_only_after_numeric_and_artifact_holdouts(self):
        from tests.test_refinement import fixture_case
        for heldout_winner in ("proposal", "tie"):
            with self.subTest(winner=heldout_winner), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                parent, run = root / "parent", root / "measured"
                case = fixture_case(root)
                image = root / "opaque.bin"
                image.write_bytes(b"opaque-test-input-not-an-image")
                pairs = [{"pair_id": i, "scene_frame": i + 1} for i in range(3)]
                records = [{"file": str(image), "sha256": r.digest(image), "scene_frame": i + 1}
                           for i in range(3)]
                r.save_json(parent / "config.json", {})
                r.save_json(parent / "outcome.json", {"after_gap_m": 2.0})
                r.save_json(parent / "proposed_pairs.json", pairs)
                r.save_json(parent / "evidence_split.json", {"train": pairs[:2], "holdouts": pairs[2:]})
                r.save_json(parent / "original/manifest.json", {"frames": records})
                r.save_json(parent / "alignment_baseline/result.json", {"frames": records})

                def projection(case, directory, blender, selected_pairs, candidates):
                    return {c["id"]: {p["pair_id"]: {
                        "first_hit_group_samples": 3,
                        "projected_bbox_xyxy": ([0.4, 0.5, 0.6, 0.7] if c["id"] == "toward_030"
                                                else [0.3, 0.5, 0.5, 0.7]),
                    } for p in selected_pairs} for c in candidates}

                validated = {"pairs": [{"pair_id": 2, "winner": heldout_winner,
                    "moving_visible_both": True, "same_physical_target": True,
                    "background_consistent": True, "position_or_size_better": True,
                    "no_new_artifacts": True}]}
                with patch.object(runner, "verify_parent", return_value=case), \
                        patch.object(runner, "project", side_effect=projection), \
                        patch.object(runner, "measure_pair", return_value=observation()) as measured, \
                        patch.object(runner.it, "render_pair_frames", return_value=[records[2]]), \
                        patch.object(runner.it, "judge", return_value=validated), \
                        patch.object(runner.it, "geometry_stage", return_value={
                            "after_state": {"surface_distance_m": 1.7},
                            "scene_after_blend": "accepted.blend"}) as apply, \
                        patch.object(r, "answer") as answer:
                    outcome = runner.run_case(parent, run, "blender", "model", "https://example.test")
                accepted = heldout_winner == "proposal"
                self.assertEqual(outcome["accepted_edits"], int(accepted))
                self.assertEqual(apply.call_count, int(accepted))
                self.assertEqual(outcome["after_gap_m"], 1.7 if accepted else 2.0)
                self.assertEqual([call.args[4]["pair_id"] for call in measured.call_args_list], [0, 1, 2])
                answer.assert_not_called()
                self.assertFalse(outcome["gt_used"])


if __name__ == "__main__":
    unittest.main()
