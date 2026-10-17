import tempfile
import unittest
from pathlib import Path

import refine_batch as batch


class BatchCaseTests(unittest.TestCase):
    def test_builds_all_abs_distance_cases_without_ground_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = batch.build_cases(root / "original", root / "bvb", root / "blends")
        self.assertEqual(len(cases), 25)
        self.assertEqual(len({case["case_id"] for case in cases}), 25)
        self.assertTrue(all(case["question_type"] == "object_abs_distance" for case in cases))
        self.assertTrue(all("ground_truth_m" not in case for case in cases))
        self.assertTrue(all("historical_answers" not in case for case in cases))
        self.assertTrue(all("diagnostic_candidate_id" not in case for case in cases))
        self.assertTrue(all(case["objects"]["moving"]["auto_match"] for case in cases))
        self.assertTrue(all(case["objects"]["reference"]["auto_match"] for case in cases))

    def test_candidate_grid_is_fixed_symmetric_and_includes_no_change(self):
        candidates = batch.default_candidates()
        self.assertEqual(candidates[0], {"id": "no_change", "operation": "no_change"})
        translations = [candidate for candidate in candidates if candidate["operation"] == "translate"]
        self.assertEqual(
            {(candidate["direction"], candidate["distance_m"]) for candidate in translations},
            {(direction, distance) for direction in ("toward", "away")
             for distance in (0.15, 0.30, 0.45)},
        )

    def test_asset_templates_support_scene_and_question_id(self):
        question = {"dataset": "scannet", "scene_name": "scene001", "qa_id": "42"}
        path = batch.asset_path(Path("assets"), "{dataset}/orig_{scene}_q{qa_id}.mp4", question)
        self.assertTrue(path.endswith(str(Path("assets") / "scannet" / "orig_scene001_q42.mp4")))

    def test_report_targets_only_parse_abs_distance_labels(self):
        targets = batch.targets()
        self.assertEqual(len(targets), 25)
        self.assertEqual(targets["819"], 2.3)


if __name__ == "__main__":
    unittest.main()
