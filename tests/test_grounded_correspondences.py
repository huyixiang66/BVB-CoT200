import copy
import unittest

import refine_grounding as g


class GroundedCorrespondenceTests(unittest.TestCase):
    def setUp(self):
        self.proposal = {"features": [{"id": "top:1", "xy": [0.2, 0.3], "object_name": "top"}],
                         "landmarks": [{"id": "wall:2", "xy": [0.4, 0.5], "object_name": "wall"}]}
        self.response = {"same_physical_target": True, "matches": [
            {"id": "top:1", "original": [0.3, 0.3], "same_physical_point": True},
            {"id": "wall:2", "original": [0.5, 0.5], "same_physical_point": True}]}

    def test_model_cannot_change_source_coordinate_or_role(self):
        self.response["matches"][0].update(baseline=[0.99, 0.99], role="landmarks")
        row = g.observation(self.response, self.proposal)
        self.assertEqual(row["features"][0]["baseline"], [0.2, 0.3])
        self.assertEqual(len(row["landmarks"]), 1)

    def test_unknown_and_duplicate_ids_rejected(self):
        for matches in ([{"id": "invented"}], [self.response["matches"][0]] * 2):
            with self.assertRaises(ValueError):
                g.observation({**self.response, "matches": matches}, self.proposal)

    def test_null_or_unmatched_points_are_not_fabricated(self):
        self.response["matches"][0]["original"] = None
        self.response["matches"][1]["same_physical_point"] = False
        row = g.observation(self.response, self.proposal)
        self.assertEqual(row["features"], [])
        self.assertEqual(row["landmarks"], [])

    def test_partial_matches_do_not_require_full_object(self):
        self.response["matches"] = self.response["matches"][:1]
        row = g.observation(self.response, self.proposal)
        self.assertTrue(row["same_physical_target"])
        self.assertEqual(len(row["features"]), 1)

    def test_invalid_coordinates_rejected(self):
        for value in (True, float("nan"), 2):
            row = copy.deepcopy(self.response)
            row["matches"][0]["original"][0] = value
            with self.assertRaises(ValueError):
                g.observation(row, self.proposal)

    def test_duplicate_source_ids_rejected(self):
        self.proposal["landmarks"][0]["id"] = "top:1"
        with self.assertRaises(ValueError):
            g.observation(self.response, self.proposal)

    def test_prompt_does_not_request_qa_or_baseline_inference(self):
        text = g.prompt("nightstand", self.proposal)
        self.assertIn("Only return ORIGINAL", text)
        self.assertIn("Partial visibility is allowed", text)
        self.assertIn("top:1", text)

    def test_background_requires_explicit_decision_for_every_anchor(self):
        with self.assertRaises(ValueError):
            g.background_observation({"matches": []}, self.proposal)

    def test_background_rejects_target_ids_and_duplicate_records(self):
        for matches in ([{"id": "top:1", "status": "unmatchable", "reason": "hidden"}],
                        [{"id": "wall:2", "status": "unmatchable", "reason": "hidden"}] * 2):
            with self.assertRaises(ValueError):
                g.background_observation({"matches": matches}, self.proposal)

    def test_background_freezes_coordinates_and_excludes_accessories(self):
        response = {"matches": [{"id": "wall:2", "status": "matched",
                                "original": [0.6, 0.7], "baseline": [0.9, 0.9],
                                "reason": "same wall junction"}]}
        result = g.background_observation(response, self.proposal)
        self.assertEqual(result[0]["baseline"], [0.4, 0.5])
        for status in ("unmatchable", "target_or_accessory"):
            response["matches"][0]["status"] = status
            self.assertEqual(g.background_observation(response, self.proposal), [])

    def test_background_rejects_malformed_decisions(self):
        for changes in ({"status": "maybe"}, {"reason": ""}, {"original": [True, 0.5]}):
            row = {"id": "wall:2", "status": "matched", "original": [0.5, 0.5], "reason": "wall"}
            row.update(changes)
            with self.assertRaises(ValueError):
                g.background_observation({"matches": [row]}, self.proposal)

    def test_background_prompt_contains_no_target_catalog(self):
        text = g.background_prompt("nightstand", self.proposal)
        self.assertIn("wall:2", text)
        self.assertNotIn("top:1", text)
        self.assertIn("every catalog ID", text)


if __name__ == "__main__":
    unittest.main()
