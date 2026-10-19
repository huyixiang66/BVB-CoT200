import unittest

from scripts.run_render_rank_refinement import (blinded_presentation, candidate_grid,
                                                choose_training_winner,
                                                presentation_order_bias)


class RenderRankRefinementTests(unittest.TestCase):
    def test_candidate_grid_is_bounded_and_scene_independent(self):
        candidates = candidate_grid()
        self.assertEqual(len(candidates), 7)
        by_id = {item["id"]: item["parameters"] for item in candidates}
        self.assertEqual(by_id["identity"], [0.0, 0.0, 1.0])
        self.assertTrue(all(abs(item["parameters"][0]) <= 0.05 and
                            abs(item["parameters"][1]) <= 0.05 and
                            0.95 <= item["parameters"][2] <= 1.05 for item in candidates))

    def test_training_consensus_selects_candidate_supported_by_both_views(self):
        ids = [item["id"] for item in candidate_grid()]
        a = {"ranked_candidate_ids": ["x_plus_005", "x_minus_005", "identity"] +
             [item for item in ids if item not in {"x_plus_005", "x_minus_005", "identity"}],
             "confidence": 0.8, "same_target": True}
        b = {"ranked_candidate_ids": ["x_plus_005", "x_minus_005", "identity"] +
             [item for item in ids if item not in {"x_plus_005", "x_minus_005", "identity"}],
             "confidence": 0.75, "same_target": True}
        result = choose_training_winner([a, b], ids)
        self.assertEqual(result["candidate_id"], "x_plus_005")
        self.assertIsNone(result["reason"])

    def test_candidate_display_order_is_blind_and_changes_between_views(self):
        candidates = candidate_grid()
        first, first_labels = blinded_presentation(candidates, 3)
        second, second_labels = blinded_presentation(candidates, 4)
        self.assertEqual(set(first_labels.values()), {f"option_{i}" for i in range(1, 8)})
        self.assertEqual(set(second_labels.values()), {f"option_{i}" for i in range(1, 8)})
        self.assertNotEqual([item["id"] for item in first], [item["id"] for item in second])

    def test_training_consensus_rejects_identity_or_disagreement(self):
        ids = [item["id"] for item in candidate_grid()]
        identity = {"ranked_candidate_ids": ids, "confidence": 0.9, "same_target": True}
        self.assertEqual(choose_training_winner([identity, identity], ids)["reason"],
                         "identity_ranked_best")
        opposite = {"ranked_candidate_ids": list(reversed(ids)), "confidence": 0.9,
                    "same_target": True}
        self.assertEqual(choose_training_winner([identity, opposite], ids)["reason"],
                         "training_views_disagree")

    def test_presentation_order_copy_is_rejected(self):
        rankings = [
            {"ranked_candidate_ids": ["b", "a"], "presented_candidate_order": ["b", "a"]},
            {"ranked_candidate_ids": ["a", "b"], "presented_candidate_order": ["a", "b"]},
        ]
        self.assertTrue(presentation_order_bias(rankings))
        rankings[1]["ranked_candidate_ids"] = ["b", "a"]
        self.assertFalse(presentation_order_bias(rankings))


if __name__ == "__main__":
    unittest.main()
