import unittest

import refine_box as b


class BoxTests(unittest.TestCase):
    def test_partial_box_allowed_without_background(self):
        self.assertTrue(b.usable({"same_physical_target": True,
                                 "original_box": [0, .3, .2, .8],
                                 "baseline_box": [.3, .2, .6, 1]}))

    def test_absent_identity_or_invalid_box_rejected(self):
        self.assertFalse(b.usable({"original_box": [0, 0, 1, 1], "baseline_box": [0, 0, 1, 1]}))
        self.assertEqual(b.error(None, [0, 0, 1, 1]), 1)

    def test_baseline_calibration_preserves_measured_box(self):
        result = b.predicted_box([.1, .2, .5, .6], [.1, .2, .5, .6], [.2, .3, .6, .7])
        for actual, expected in zip(result, [.2, .3, .6, .7]):
            self.assertAlmostEqual(actual, expected)

    def test_projected_motion_predicts_observed_motion(self):
        actual = b.predicted_box([.2, .2, .6, .6], [.1, .2, .5, .6], [.2, .3, .6, .7])
        self.assertAlmostEqual(b.error(actual, [.3, .3, .7, .7]), 0)

    def test_grid_is_symmetric_bounded_and_has_identity(self):
        for round_index, center in ((0, [0, 0, 1]), (1, [.6, .3, 1.2])):
            rows = b.candidates(center, round_index)
            self.assertIn([0, 0, 1], [r["parameters"] for r in rows])
            for r in rows:
                x, y, scale = r["parameters"]
                self.assertLessEqual((x*x+y*y)**.5, .900001)
                self.assertTrue(.7 <= scale <= 1.3)

    def test_search_can_choose_move_without_background(self):
        def row(identity, x):
            return {"candidate": {"id": identity, "parameters": [x, 0, 1]},
                    "projected_bbox_xyxy": [.2+x, .4, .4+x, .6], "first_hit_group_samples": 2}
        best, _ = b.select([row("base", 0), row("move", .1)],
                           {"baseline_box": [.2, .4, .4, .6], "original_box": [.3, .4, .5, .6]},
                           [.2, .4, .4, .6])
        self.assertEqual(best["candidate"]["id"], "move")

    def test_joint_views_reject_single_view_overfit(self):
        observations = [{"baseline_box": [.2, .4, .4, .6], "original_box": [.3, .4, .5, .6]},
                        {"baseline_box": [.2, .4, .4, .6], "original_box": [.2, .4, .4, .6]}]
        initial = [[.2, .4, .4, .6]] * 2
        def rows(name, offsets):
            return [{"candidate": {"id": name, "parameters": [offsets[0], 0, 1]},
                     "projected_bbox_xyxy": [.2+x, .4, .4+x, .6],
                     "first_hit_group_samples": 1} for x in offsets]
        records = [rows("identity", [0, 0]), rows("overfit", [.1, .3]), rows("joint", [.05, 0])]
        self.assertEqual(b.select([r[0] for r in records], observations[0], initial[0])[0]["candidate"]["id"],
                         "overfit")
        best, scores = b.select_multiview(records, observations, initial)
        self.assertEqual(best["candidate"]["id"], "joint")
        self.assertEqual(len(best["view_losses"]), 2)
        self.assertEqual(len(scores), 3)

    def test_worst_gain_does_not_trade_one_view_for_average(self):
        observations = [{"baseline_box": [.1, .4, .3, .6],
                         "original_box": [.3, .4, .5, .6]}] * 3
        initial = [[.1, .4, .3, .6]] * 3
        def rows(name, offsets):
            return [{"candidate": {"id": name, "parameters": [offsets[0], 0, 1]},
                     "projected_bbox_xyxy": [.1+x, .4, .3+x, .6],
                     "first_hit_group_samples": 1} for x in offsets]
        records = [rows("identity", [0, 0, 0]), rows("average", [.2, .2, -.05]),
                   rows("balanced", [.05, .05, .05])]
        mean, _ = b.select_multiview(records, observations, initial)
        self.assertEqual(mean["candidate"]["id"], "average")
        worst, scores = b.select_multiview(records, observations, initial, objective="worst_gain")
        self.assertEqual(worst["candidate"]["id"], "balanced")
        self.assertGreater(worst["worst_gain"], 0)
        self.assertEqual(len(scores), 3)

    def test_worst_gain_keeps_identity_when_every_move_hurts_a_view(self):
        observations = [{"baseline_box": [.2, .4, .4, .6],
                         "original_box": [.3, .4, .5, .6]}] * 2
        initial = [[.2, .4, .4, .6]] * 2
        records = [[{"candidate": {"id": name, "parameters": [step, 0, 1]},
                     "projected_bbox_xyxy": [.2+x, .4, .4+x, .6],
                     "first_hit_group_samples": 1} for x in offsets]
                   for name, step, offsets in (("identity", 0, [0, 0]), ("bad", .1, [.1, -.1]))]
        best, _ = b.select_multiview(records, observations, initial, objective="worst_gain")
        self.assertEqual(best["candidate"]["id"], "identity")

    def test_censored_edges_allow_hidden_extent_but_penalize_inside_border(self):
        target = [0, .2, .4, 1]
        self.assertEqual(b.edge_error([-.3, .2, .4, 1.2], target), 0)
        self.assertAlmostEqual(b.edge_error([.1, .2, .4, .9], target), .05)
        self.assertEqual(b.edge_error(None, target), 1)

    def test_censored_baseline_is_not_calibrated_to_image_border(self):
        initial = [-.3, .2, .4, .6]
        observed = [0, .21, .41, .61]
        predicted = b.predict_edges([-.2, .3, .5, .7], initial, observed)
        for actual, expected in zip(predicted, [-.2, .31, .51, .71]):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(b.edge_error(b.predict_edges(initial, initial, observed), observed), 0)

    def test_edge_loss_distinguishes_two_cropped_boxes_with_different_hidden_extent(self):
        target = [0, .2, .4, .6]
        self.assertEqual(b.edge_error([-.1, .2, .4, .6], target),
                         b.edge_error([-.8, .2, .4, .6], target))
        self.assertGreater(b.edge_error([.1, .2, .4, .6], target), 0)

    def test_edge_selector_preserves_unclipped_candidate_projection(self):
        initial = [-.3, .2, .4, .6]
        observation = {"baseline_box": [0, .2, .4, .6], "original_box": [0, .2, .5, .6]}
        records = [
            {"candidate": {"id": "identity", "parameters": [0, 0, 1]},
             "unclipped_bbox_top_left": initial, "first_hit_group_samples": 2},
            {"candidate": {"id": "move", "parameters": [.1, 0, 1]},
             "unclipped_bbox_top_left": [-.2, .2, .5, .6], "first_hit_group_samples": 2}]
        winner, _ = b.select(records, observation, initial, metric="censored_edges")
        self.assertEqual(winner["candidate"]["id"], "move")
        self.assertEqual(winner["predicted_box"][0], -.2)
        self.assertAlmostEqual(winner["loss"], 0)

    def test_missing_unclipped_projection_is_not_silently_substituted(self):
        row = {"candidate": {"id": "invalid", "parameters": [.1, 0, 1]},
               "unclipped_bbox_top_left": None, "first_hit_group_samples": 2}
        observation = {"baseline_box": [0, .2, .4, .6], "original_box": [0, .2, .5, .6]}
        best, _ = b.select([row], observation, [-.3, .2, .4, .6], metric="censored_edges")
        self.assertEqual(best["loss"], 1)

    def test_edge_identity_has_zero_gain_when_baseline_is_cropped_but_target_is_not(self):
        initial = [-.3, .2, .4, .6]
        observation = {"baseline_box": [0, .2, .4, .6], "original_box": [.2, .2, .5, .6]}
        row = {"candidate": {"id": "identity", "parameters": [0, 0, 1]},
               "unclipped_bbox_top_left": initial, "first_hit_group_samples": 2}
        best, _ = b.select_multiview([[row]], [observation], [initial],
                                     objective="worst_gain", metric="censored_edges")
        self.assertAlmostEqual(best["worst_gain"], 0)


if __name__ == "__main__":
    unittest.main()
