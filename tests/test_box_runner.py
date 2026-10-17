"""Check that only the training frame selects edits; heldout boxes verify afterwards."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import refinement as r
from scripts import run_box_refinement as runner
from tests.test_refinement import fixture_case


class BoxRunnerTests(unittest.TestCase):
    def test_validation_prompt_freezes_original_region(self):
        original = [.3, .4, .5, .6]
        with patch.object(runner.it, "judge", return_value={}) as judge:
            runner.locate(Path("."), "after_0", "model", "url", "table", [],
                          fixed_original=original)
        self.assertIn(str(original), judge.call_args.args[4])
        self.assertIn("Do not relocalize or reject", judge.call_args.args[4])

    def test_heldout_does_not_steer_search_and_failed_holdout_keeps_source(self):
        for improves, reuse in ((True, False), (False, False), (True, True), (False, True)):
            with self.subTest(improves=improves, reuse=reuse), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                case = fixture_case(root)
                parent, frames, run = root / "parent", root / "frames", root / "run"
                opaque = root / "opaque.bin"
                opaque.write_bytes(b"synthetic-no-pixels")
                pairs = [{"pair_id": i, "scene_frame": i+1, "original_frame_index_0based": i}
                         for i in range(3)]
                originals = [{"file": str(opaque), "sha256": r.digest(opaque),
                              "source_frame_index_0based": i} for i in range(3)]
                baselines = [{**originals[i], "scene_frame": i+1} for i in range(3)]
                r.save_json(parent / "proposed_pairs.json", pairs)
                r.save_json(parent / "config.json", {})
                r.save_json(parent / "resolved_case.json", case)
                r.save_json(frames / "original/manifest.json", {"frames": originals})
                r.save_json(frames / "alignment_baseline/result.json", {"frames": baselines})
                proposal = root / "proposal.blend"
                proposal.write_bytes(b"synthetic blend")
                calls = []

                def geometry(c, directory, blender, selected, candidates, save=False):
                    calls.append((directory.name, [p["pair_id"] for p in selected]))
                    records = {c["id"]: [{"candidate": c, "projected_bbox_xyxy": [
                        .2+c["parameters"][0]/3, .4, .4+c["parameters"][0]/3, .6],
                        "first_hit_group_samples": 2}] for c in candidates}
                    if save:
                        return {"proposal_blend": str(proposal), "frames": baselines,
                                "before": {"surface_distance_m": 2}, "after": {"surface_distance_m": 1.9}}
                    return {"candidates": records}

                def locate(run, name, model, url, label, records, fixed_original=None):
                    if name.startswith("after"):
                        self.assertEqual(fixed_original, [.3, .4, .5, .6])
                    improved = name.startswith("after") and (improves or name == "after_0")
                    return {"same_physical_target": True, "original_box": [.3, .4, .5, .6],
                            "baseline_box": [.3, .4, .5, .6] if improved else [.2, .4, .4, .6]}

                argv = ["run", "--parent", str(parent), "--frames", str(frames), "--run", str(run),
                        "--blender", "unused", "--base-url", "https://example.test"]
                if reuse:
                    source = root / "observations"
                    r.save_json(source / "config.json", {
                        "parent": str(parent), "model": "gpt-6-luna",
                        "input_sha256": {str(parent / "resolved_case.json"):
                                         r.digest(parent / "resolved_case.json")}})
                    r.save_json(source / "split.json", {"train": pairs[:1], "holdouts": pairs[1:]})
                    for pid in range(3):
                        r.save_json(source / f"before_{pid}.json", locate(
                            source, f"before_{pid}", "", "", "", []))
                        r.save_json(source / "calls" / f"before_{pid}" / "request.json", {
                            "frames": [originals[pid], baselines[pid]]})
                    argv += ["--observations", str(source)]
                with patch("sys.argv", argv), patch.object(runner.measured, "verify_parent", return_value=case), \
                        patch.object(runner, "geometry", side_effect=geometry), \
                        patch.object(runner, "locate", side_effect=locate) as localization, \
                        patch.object(r, "answer") as qa:
                    runner.main()
                outcome = r.read_json(run / "outcome.json")
                self.assertEqual(outcome["accepted_edits"], int(improves))
                self.assertEqual(outcome["final_blend"], str(proposal) if improves else case["inputs"]["blend"])
                self.assertEqual(calls, [("initial", [0]), ("search_0", [0]),
                                         ("search_1", [0]), ("preview", [0, 1, 2])])
                self.assertEqual(localization.call_count, 3 if reuse else 6)
                qa.assert_not_called()


if __name__ == "__main__":
    unittest.main()
