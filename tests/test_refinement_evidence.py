"""Evidence gates must reject the failure mode observed in q750 and q898."""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import refinement as r
import refine_iterative as it


class EvidenceGateTests(unittest.TestCase):
    def test_context_only_retrieval_cannot_authorize_geometry_selection(self):
        observations = [
            {"pair_id": 3, "visible_target": "none", "alignment_quality": "high"},
            {"pair_id": 1, "visible_target": "reference", "alignment_quality": "medium"},
        ]
        self.assertEqual(r.usable_evidence_pairs(observations), [])

    def test_only_visible_moving_target_with_alignment_is_usable(self):
        observations = [
            {"pair_id": 1, "visible_target": "moving", "alignment_quality": "low"},
            {"pair_id": 2, "visible_target": "both", "alignment_quality": "high"},
            {"pair_id": 3, "visible_target": "moving", "alignment_quality": "medium"},
        ]
        self.assertEqual(r.usable_evidence_pairs(observations), [2, 3])


class IterativePolicyTests(unittest.TestCase):
    def test_missing_reference_is_not_reported_as_ambiguous_moving(self):
        from tests.test_refinement import fixture_case
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = fixture_case(root)
            for value in case["inputs"].values():
                Path(value).write_bytes(b"synthetic-source")
            source = root / "input.json"
            r.save_json(source, r.blind_case(case))
            case["objects"]["reference"] = {"label": "window", "auto_match": True}
            probe = {"dynamic_members": [], "components": {"moving": [], "reference": []},
                     "initial_state": {"status": "unresolved", "unresolved_roles": ["reference"],
                                       "surface_distance_m": None}}
            with patch.object(r, "doctor"), patch.object(r, "resolve_case_objects", return_value=case), \
                    patch.object(r, "inspect", return_value={"timeline": [0, 100]}), \
                    patch.object(it, "geometry_stage", return_value=probe), \
                    patch.object(it, "judge") as judge:
                outcome = it.run_case(source, root / "run", "unused", "test", "https://example.test")
            self.assertEqual(outcome["stop_reason"], "unresolved_object_names")
            self.assertEqual(outcome["unresolved_roles"], ["reference"])
            self.assertNotIn("before_gap_m", outcome)
            judge.assert_not_called()

    def test_q750_disconnected_oven_is_not_merged_with_literal_stove(self):
        groups = [{"members": ["Stove hob", "Burner"]}, {"members": ["Built in oven glass"]}]
        self.assertEqual(it.pick_component("stove", groups), groups[0])

    def test_multiple_literal_instances_are_ambiguous(self):
        self.assertIsNone(it.pick_component("chair", [
            {"members": ["Chair left"]}, {"members": ["Chair right"]}]))

    def test_single_group_can_use_synonym_but_not_multiple_unanchored_groups(self):
        self.assertEqual(it.pick_component("sofa", [{"members": ["Couch"]}]), {"members": ["Couch"]})
        self.assertIsNone(it.pick_component("sofa", [{"members": ["Couch"]}, {"members": ["Settee"]}]))

    def test_no_target_no_pairs_even_with_large_bbox(self):
        frames = [{"scene_frame": i, "moving": {
            "first_hit_group_samples": 0, "projected_area_fraction": 1.0,
        }} for i in range(10)]
        self.assertEqual(it.propose_pairs({"frames": frames}, 100, [0, 9]), [])

    def test_visibility_sampling_is_separated_and_deterministic(self):
        frames = [{"scene_frame": i, "moving": {
            "first_hit_group_samples": 10, "projected_area_fraction": 0.1,
        }} for i in range(101)]
        pairs = it.propose_pairs({"frames": frames}, 201, [0, 100])
        self.assertLessEqual(len(pairs), 8)
        self.assertEqual(pairs[0]["original_frame_index_0based"], 0)
        self.assertTrue(all(abs(a["normalized_time"] - b["normalized_time"]) >= it.POLICY["time_separation"]
                            for i, a in enumerate(pairs) for b in pairs[i + 1:]))

    def test_alignment_cannot_pass_string_booleans_or_missing_target(self):
        pairs = [{"pair_id": i, "normalized_time": i / 3} for i in range(4)]
        rows = [{"pair_id": i, "moving_visible_original": True,
                 "moving_visible_reconstruction": True, "same_physical_target": True,
                 "background_consistent": True} for i in range(4)]
        rows[0]["moving_visible_original"] = "true"
        evidence = it.split_evidence({"pairs": rows}, pairs)
        self.assertEqual(evidence, {"train": pairs[1:3], "holdouts": pairs[3:]})

    def test_alignment_rejects_duplicate_ids(self):
        with self.assertRaises(ValueError):
            it.split_evidence({"pairs": [{"pair_id": 0}, {"pair_id": 0}]},
                              [{"pair_id": 0}, {"pair_id": 1}])

    def test_independent_validation_rejects_tie_artifacts_and_missing_pairs(self):
        row = {"pair_id": 3, "winner": "proposal", "moving_visible_both": True,
               "same_physical_target": True, "background_consistent": True,
               "position_or_size_better": True, "no_new_artifacts": True}
        self.assertTrue(it.accept_validation({"pairs": [row]}, [{"pair_id": 3}]))
        for key, value in (("winner", "tie"), ("no_new_artifacts", False),
                           ("moving_visible_both", "true"), ("pair_id", 2)):
            self.assertFalse(it.accept_validation({"pairs": [{**row, key: value}]}, [{"pair_id": 3}]))

    def test_steps_shrink_without_gt_and_scale_preserves_support(self):
        first, second = it.round_candidates(0), it.round_candidates(1)
        self.assertEqual(first[1]["distance_m"], 2 * second[1]["distance_m"])
        self.assertTrue(all(c["pivot"] == "bottom" for c in first if c["operation"] == "scale"))

    def test_end_to_end_accept_then_reject_keeps_last_scene_without_qa_or_gt(self):
        from tests.test_refinement import fixture_case
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = fixture_case(root)
            for value in case["inputs"].values():
                Path(value).write_bytes(b"synthetic-source")
            source = root / "input.json"
            r.save_json(source, r.blind_case(case))
            run = root / "run"
            pairs = [{"pair_id": i, "scene_frame": i * 10,
                      "original_frame_index_0based": i * 10, "normalized_time": i / 3} for i in range(4)]
            probe = {
                "dynamic_members": [], "initial_state": {"surface_distance_m": 2.0},
                "components": {"moving": [{"members": ["table"]}],
                               "reference": [{"members": ["chair"]}]},
            }
            applied_path = root / "accepted.blend"
            applied_path.write_bytes(b"synthetic-accepted-scene")
            mutations = []

            def geometry(current_case, current_run, blender, name, spec):
                if spec.get("action") == "apply":
                    mutations.append(current_case["inputs"]["blend"])
                    return {"before_state": {"surface_distance_m": 2.0},
                            "after_state": {"surface_distance_m": 1.7},
                            "scene_after_blend": str(applied_path)}
                return probe

            alignment = {"pairs": [{"pair_id": i, "moving_visible_original": True,
                                   "moving_visible_reconstruction": True, "same_physical_target": True,
                                   "background_consistent": True} for i in range(4)]}

            def validation(pair_id, winner):
                return {"pairs": [{"pair_id": pair_id, "winner": winner, "moving_visible_both": True,
                                   "same_physical_target": True, "background_consistent": True,
                                   "position_or_size_better": True, "no_new_artifacts": True}]}

            responses = [
                alignment,
                {"candidate_id": "toward_030", "supporting_pair_ids": [0, 1]},
                validation(2, "proposal"),
                {"candidate_id": "toward_015", "supporting_pair_ids": [0, 1]},
                validation(3, "tie"),
            ]
            with patch.object(r, "doctor"), patch.object(r, "resolve_case_objects", return_value=case), \
                    patch.object(r, "inspect", return_value={"timeline": [0, 100]}), \
                    patch.object(it, "geometry_stage", side_effect=geometry), \
                    patch.object(r, "video_metadata", return_value={"frame_count": 101}), \
                    patch.object(it, "propose_pairs", return_value=pairs), \
                    patch.object(r, "extract_frames", return_value=[{"file": "unused"}] * 4), \
                    patch.object(it, "render_pair_frames", side_effect=lambda c, run, b, n, ps, cand:
                                 [{"pair_id": p["pair_id"], "scene_frame": p["scene_frame"]}
                                  for p in ps]), \
                    patch.object(it, "judge", side_effect=responses) as judge, \
                    patch.object(r, "answer") as answer:
                outcome = it.run_case(source, run, "unused", "test", "https://example.test")
            self.assertEqual(outcome["accepted_edits"], 1)
            self.assertEqual(outcome["after_gap_m"], 1.7)
            self.assertEqual(outcome["final_blend"], str(applied_path))
            self.assertEqual(outcome["stop_reason"], "heldout_rejected")
            self.assertEqual(len(mutations), 1)
            self.assertEqual(judge.call_count, 5)
            answer.assert_not_called()
            self.assertNotIn("ground_truth_m", r.read_json(run / "resolved_case.json"))
            heldouts = [call.args[5][0]["pair_id"] for call in judge.call_args_list
                        if call.args[1] == "validate"]
            self.assertEqual(heldouts, [2, 3])

    def test_adjacent_time_frames_cannot_be_independent_holdouts(self):
        pairs = [{"pair_id": i, "normalized_time": value}
                 for i, value in enumerate((0.1, 0.2, 0.24, 0.7, 0.74))]
        rows = [{"pair_id": p["pair_id"], "moving_visible_original": True,
                 "moving_visible_reconstruction": True, "same_physical_target": True,
                 "background_consistent": True} for p in pairs]
        split = it.split_evidence({"pairs": rows}, pairs)
        self.assertEqual([p["pair_id"] for p in split["holdouts"]], [3])

    def test_no_change_baseline_reuses_exact_frozen_frame_records(self):
        records = [{"pair_id": 2, "scene_frame": 42, "file": "frozen.png", "sha256": "same"}]
        result = it.baseline_pair_frames(records, [{"pair_id": 2, "scene_frame": 42}])
        self.assertIs(result[0], records[0])
        with self.assertRaises(ValueError):
            it.baseline_pair_frames(records, [{"pair_id": 2, "scene_frame": 43}])


if __name__ == "__main__":
    unittest.main()
