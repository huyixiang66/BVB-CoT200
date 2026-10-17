import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import refinement as r
from scripts import diagnose_box_multiview as runner
from tests.test_refinement import fixture_case


class JointRunnerTests(unittest.TestCase):
    def test_fit_three_hold_out_fourth_without_render_api_or_applying(self):
        self.run_objective("mean")

    def test_worst_gain_preserves_holdout_and_no_render(self):
        self.run_objective("worst_gain")

    def test_translation_ablation_never_changes_scale(self):
        self.run_objective("worst_gain", translation_only=True)

    def test_censored_edges_preserves_holdout_and_no_render(self):
        self.run_objective("worst_gain", metric="censored_edges")

    def test_default_restores_border_only_worst_gain_without_annotations(self):
        self.run_objective("worst_gain", metric="censored_edges", use_defaults=True)

    def test_edge_types_are_frozen_and_holdout_not_used_for_search(self):
        self.run_objective("worst_gain", metric="censored_edges", annotated=True)

    def run_objective(self, objective, translation_only=False, metric="center_size", annotated=False,
                      use_defaults=False):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case = fixture_case(root)
            Path(case["inputs"]["blend"]).write_bytes(b"opaque blend")
            source, parent, run = root / "source", root / "parent", root / "run"
            parts = root / "parts.json"
            r.save_json(parts, {"case_id": case["case_id"],
                               "blend_sha256": r.digest(case["inputs"]["blend"]),
                               "body": ["body"], "carried": ["lamp"]})
            pairs = [{"pair_id": i, "scene_frame": i+1} for i in range(4)]
            r.save_json(parent / "resolved_case.json", case)
            r.save_json(parent / "proposed_pairs.json", pairs)
            r.save_json(source / "config.json", {"parent": str(parent)})
            for i in range(4):
                r.save_json(source / f"before_{i}.json", {
                    "same_physical_target": True, "original_box": [.3, .4, .5, .6],
                    "baseline_box": [.2, .4, .4, .6]})
                r.save_json(source / "calls" / f"before_{i}" / "request.json", {"frames": []})
            stages = []

            def geometry(case, run, blender, views, candidates, save=False):
                self.assertFalse(save)
                if translation_only:
                    self.assertTrue(all(c["parameters"][2] == 1 for c in candidates))
                stages.append((run.name, [p["pair_id"] for p in views]))
                return {
                    "before": {"surface_distance_m": 2},
                    "candidate_states": {c["id"]: {"surface_distance_m": 1.9} for c in candidates},
                    "candidates": {c["id"]: [
                        {"candidate": c, "projected_bbox_xyxy": [
                            .2+c["parameters"][0]/3, .4, .4+c["parameters"][0]/3, .6],
                         "unclipped_bbox_top_left": [
                             .2+c["parameters"][0]/3, .4, .4+c["parameters"][0]/3, .6],
                         "first_hit_group_samples": 1} for p in views] for c in candidates}}

            argv = ["run", "--source", str(source), "--parts", str(parts), "--run", str(run),
                    "--blender", "unused"]
            if not use_defaults:
                argv += ["--objective", objective, "--metric", metric]
            if annotated:
                annotation = root / "annotations.json"
                r.save_json(annotation, {
                    "source": str(source), "frame_sha256": {},
                    "observation_sha256": {str(i): r.digest(source / f"before_{i}.json") for i in range(4)},
                    "pairs": {str(i): {"original": ["silhouette"] * 4,
                                      "baseline": ["silhouette"] * 4} for i in range(4)}})
                argv += ["--edge-annotations", str(annotation)]
            if translation_only:
                argv.append("--translation-only")
            with patch("sys.argv", argv), patch.object(runner.measured, "verify_parent", return_value=case), \
                    patch.object(runner, "geometry", side_effect=geometry), \
                    patch.object(r, "call_model") as model:
                runner.main()
            model.assert_not_called()
            outcome = r.read_json(run / "outcome.json")
            self.assertEqual(outcome["objective"], objective)
            self.assertEqual(outcome["translation_only"], translation_only)
            self.assertEqual(outcome["metric"], metric)
            self.assertEqual(bool(outcome["edge_annotations"]), annotated)
            self.assertEqual(outcome["accepted_edits"], 0)
            self.assertEqual(outcome["renders"], 0)
            self.assertEqual(outcome["final_blend"], case["inputs"]["blend"])
            self.assertEqual(stages, [("initial", [0, 1, 2, 3]), ("search_0", [0, 1, 2]),
                                      ("search_1", [0, 1, 2]), ("check", [0, 1, 2, 3])])
            self.assertEqual(outcome["checks"][-1]["role"], "holdout")
            self.assertEqual(len(outcome["checks"][0]["after_components"]), 4)
