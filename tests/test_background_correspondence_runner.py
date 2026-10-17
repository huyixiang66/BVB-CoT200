"""The background ablation reuses targets/frames and never edits or answers QA."""

import tempfile
import copy
import unittest
from pathlib import Path
from unittest.mock import patch

import refinement as r
from scripts import measure_background_correspondences as runner


class BackgroundRunnerTests(unittest.TestCase):
    def test_background_replacement_preserves_every_non_background_field(self):
        old = {"pairs": [{"pair_id": 1, "scene_frame": 8, "normalized_time": 0.4,
                         "features": [{"id": "target"}], "transform_members": ["target"],
                         "landmarks": [{"id": "old"}], "background_excluded_members": ["target"]}]}
        new = copy.deepcopy(old)
        new["background_exclusion"] = "shared_target_collections_v1"
        new["pairs"][0]["landmarks"] = [{"id": "new", "object_name": "wall"}]
        new["pairs"][0]["background_excluded_members"].append("accessory")
        self.assertEqual(runner.replace_background(old, new), new)
        for field, value in (("features", []), ("scene_frame", 9), ("pair_id", 2),
                             ("transform_members", ["other"]), ("normalized_time", 0.5)):
            changed = copy.deepcopy(new)
            changed["pairs"][0][field] = value
            with self.assertRaises(ValueError):
                runner.replace_background(old, changed)

    def test_background_replacement_rejects_contamination_and_pair_loss(self):
        old = {"pairs": [{"pair_id": 1, "features": [], "transform_members": ["target"],
                         "landmarks": [], "background_excluded_members": ["target"]}]}
        for changes in (
            {"pairs": []},
            {"pairs": [{**old["pairs"][0], "landmarks": [{"id": "bad", "object_name": "target"}]}]},
        ):
            new = {**old, "background_exclusion": "shared_target_collections_v1", **changes}
            with self.assertRaises(ValueError):
                runner.replace_background(old, new)

    def fixture(self, root):
        source, parent, run = root / "source", root / "parent", root / "run"
        opaque = root / "opaque.bin"
        opaque.write_bytes(b"not-an-image")
        frames = [{"file": str(opaque), "sha256": r.digest(opaque), "source": side}
                  for side in ("original", "baseline")]
        feature = {"id": "target:1", "object_name": "target", "xy": [0.3, 0.3],
                   "local_point": [0, 0, 0], "visible": True}
        proposal = {"pair_id": 7, "features": [feature], "landmarks": [
            {"id": "wall:1", "object_name": "wall", "xy": [0.5, 0.5]}]}
        annotation = {"same_physical_target": True, "features": [
            {"id": "target:1", "baseline": [0.3, 0.3], "original": [0.4, 0.3],
             "same_physical_point": True}], "landmarks": [], "reason": "frozen target"}
        r.save_json(source / "proposals.json", {"pairs": [proposal]})
        r.save_json(source / "config.json", {"selected_pair_ids": [7], "model": "mock",
                                            "proposal_sha256": r.digest(source / "proposals.json")})
        r.save_json(source / "measurement_7.json", {"annotation": annotation})
        r.save_json(source / "calls/grounded_7/request.json", {"frames": frames})
        argv = ["probe", "--source", str(source), "--parent", str(parent), "--run", str(run),
                "--base-url", "https://example.test"]
        case = {"objects": {"moving": {"label": "target"}}}
        return source, run, argv, case, frames, annotation

    def test_explicit_background_decisions_preserve_frozen_targets_and_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            source, run, argv, case, frames, annotation = self.fixture(Path(directory))
            response = {"finish_reason": "stop", "text":
                        '{"matches":[{"id":"wall:1","status":"unmatchable","original":null,"reason":"hidden"}]}'}
            with patch("sys.argv", argv), patch.object(runner.measured, "verify_parent", return_value=case), \
                    patch.object(r, "call_model", return_value=response) as call, \
                    patch.object(r, "answer") as answer, \
                    patch.object(runner.measured.it, "geometry_stage") as edit:
                runner.main()
            self.assertEqual(call.call_count, 1)
            self.assertEqual(call.call_args.args[5], frames)
            actual = r.read_json(run / "measurement_7.json")["annotation"]
            self.assertEqual(actual, annotation)
            self.assertEqual(r.read_json(source / "measurement_7.json")["annotation"], annotation)
            answer.assert_not_called()
            edit.assert_not_called()

    def test_frame_hash_failure_prevents_model_call(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, argv, case, frames, _ = self.fixture(Path(directory))
            Path(frames[0]["file"]).write_bytes(b"changed")
            with patch("sys.argv", argv), patch.object(runner.measured, "verify_parent", return_value=case), \
                    patch.object(r, "call_model") as call:
                with self.assertRaisesRegex(ValueError, "Source frame changed"):
                    runner.main()
            call.assert_not_called()

    def test_structural_source_shortage_skips_without_model_call(self):
        with tempfile.TemporaryDirectory() as directory:
            source, run, argv, case, _, _ = self.fixture(Path(directory))
            proposal = r.read_json(source / "proposals.json")
            proposal["anchor_policy"] = {"structural": True}
            r.save_json(source / "proposals.json", proposal)
            config = r.read_json(source / "config.json")
            config["proposal_sha256"] = r.digest(source / "proposals.json")
            r.save_json(source / "config.json", config)
            with patch("sys.argv", argv), patch.object(runner.measured, "verify_parent", return_value=case), \
                    patch.object(r, "call_model") as call:
                runner.main()
            call.assert_not_called()
            report = r.read_json(run / "report.json")
            self.assertEqual(report["pairs"][0]["status"], "skipped_insufficient_source_landmarks")
            self.assertEqual(report["successful_calls"], 0)

    def test_filtered_catalog_reaches_model_without_changing_target_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, run, argv, case, frames, annotation = self.fixture(root)
            old = r.read_json(source / "proposals.json")
            old["pairs"][0]["background_excluded_members"] = ["target"]
            r.save_json(source / "proposals.json", old)
            config = r.read_json(source / "config.json")
            config["proposal_sha256"] = r.digest(source / "proposals.json")
            r.save_json(source / "config.json", config)
            new = copy.deepcopy(old)
            new["background_exclusion"] = "shared_target_collections_v1"
            new["pairs"][0]["landmarks"] = [
                {"id": "wall:2", "object_name": "wall", "xy": [0.6, 0.5]}]
            replacement = root / "filtered/proposals.json"
            r.save_json(replacement, new)
            r.save_json(replacement.parent / "report.json", {
                "proposal_sha256": r.digest(replacement),
                "code_sha256": {"refine_grounding.py": r.digest(r.ROOT / "refine_grounding.py")}})
            argv += ["--background-proposals", str(replacement)]
            response = {"finish_reason": "stop", "text":
                        '{"matches":[{"id":"wall:2","status":"matched","original":[0.7,0.5],"reason":"junction"}]}'}
            with patch("sys.argv", argv), patch.object(runner.measured, "verify_parent", return_value=case), \
                    patch.object(r, "call_model", return_value=response) as call:
                runner.main()
            self.assertIn("wall:2", call.call_args.args[4])
            self.assertNotIn("wall:1", call.call_args.args[4])
            self.assertEqual(call.call_args.args[5], frames)
            actual = r.read_json(run / "measurement_7.json")["annotation"]
            self.assertEqual(actual["features"], annotation["features"])
            self.assertEqual(actual["landmarks"][0]["baseline"], [0.6, 0.5])
            config = r.read_json(run / "config.json")
            self.assertEqual(config["input_sha256"][str(replacement.resolve())], r.digest(replacement))
            self.assertEqual(config["background_variant"], "shared_target_collections_v1")


if __name__ == "__main__":
    unittest.main()
