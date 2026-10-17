import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import refine_box as b
import refinement as r
from scripts import measure_box_edges as runner
from scripts.measure_box_edges import validate


class EdgeTypeTests(unittest.TestCase):
    def test_internal_occlusion_is_a_bound_not_an_exact_extent(self):
        target = [.3, .3, .6, .7]
        kinds = ["occluded", "unknown", "silhouette", "silhouette"]
        self.assertEqual(b.edge_error([.1, .1, .6, .7], target, kinds), 0)
        self.assertGreater(b.edge_error([.4, .4, .6, .7], target, kinds), 0)
        self.assertGreater(b.edge_error([.1, .1, .6, .7], target), 0)

    def test_occluded_baseline_does_not_calibrate_a_hidden_contour(self):
        result = b.predict_edges([.1, .1, .6, .8], [.1, .1, .6, .8],
                                 [.3, .3, .6, .7],
                                 ["occluded", "unknown", "silhouette", "silhouette"])
        self.assertEqual(result, [.1, .1, .6, .7])

    def test_unknown_edges_do_not_impose_containment_even_at_border(self):
        target = [0, .3, .6, .7]
        kinds = ["unknown", "unknown", "silhouette", "silhouette"]
        self.assertEqual(b.edge_error([.4, .4, .6, .7], target, kinds), 0)
        self.assertGreater(b.edge_error([.4, .4, .6, .7], target,
                                       ["cropped", "occluded", "silhouette", "silhouette"]), 0)

    def test_edge_annotation_validation(self):
        valid = {"original": ["silhouette"] * 4, "baseline": ["unknown"] * 4}
        self.assertEqual(validate(valid), valid)
        for invalid in ({}, {**valid, "original": ["silhouette"] * 3},
                        {**valid, "baseline": ["likely"] * 4}):
            with self.assertRaises(ValueError):
                validate(invalid)

    def test_measurement_only_classifies_frozen_boxes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, parent, run = root / "source", root / "parent", root / "run"
            r.save_json(source / "config.json", {"parent": str(parent), "model": "mock"})
            r.save_json(parent / "resolved_case.json", {"objects": {"moving": {"label": "table"}}})
            r.save_json(parent / "proposed_pairs.json", [{"pair_id": i} for i in range(4)])
            opaque = root / "opaque.bin"
            opaque.write_bytes(b"no pixels")
            frozen = {}
            for i in range(4):
                path = source / f"before_{i}.json"
                r.save_json(path, {"same_physical_target": True, "original_box": [.1, .2, .4, .6],
                                  "baseline_box": [.2, .2, .5, .6]})
                frozen[str(path)] = r.digest(path)
                r.save_json(source / "calls" / f"before_{i}" / "request.json", {
                    "frames": [{"file": str(opaque), "sha256": r.digest(opaque)}]})
            reply = {"original": ["silhouette", "occluded", "silhouette", "unknown"],
                     "baseline": ["silhouette"] * 4}
            with patch("sys.argv", ["run", "--source", str(source), "--run", str(run),
                                     "--base-url", "https://example.test"]), \
                    patch.object(runner.it, "judge", return_value=reply) as judge:
                runner.main()
            self.assertEqual(judge.call_count, 4)
            self.assertTrue(all(r.digest(p) == h for p, h in frozen.items()))
            annotation = r.read_json(run / "annotations.json")
            self.assertEqual(len(annotation["pairs"]), 4)
            self.assertEqual(annotation["qa_calls"], 0)
            self.assertIn("Do not relocate", judge.call_args.args[4])
