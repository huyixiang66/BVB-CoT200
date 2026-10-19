import math
import unittest

from scripts.run_direct_view_refinement import parse_observation, residual, transform_delta


class DirectViewTests(unittest.TestCase):
    def test_parses_no_box_response_and_scores_position_and_scale(self):
        value = parse_observation({"text": '{"same_target":true,"visible_both":true,'
                                    '"confidence":0.9,"screen_delta":[0.03,-0.03],'
                                    '"scale_ratio":1.1}',
                                   "finish_reason": "stop"})
        self.assertAlmostEqual(residual(value), (2 ** 0.5) * 0.03 + 0.5 * math.log(1.1))

    def test_uncertain_numeric_correction_has_no_residual(self):
        value = parse_observation({"text": '{"same_target":true,"visible_both":true,'
                                   '"confidence":0.8,"screen_delta":null,"scale_ratio":1.0}',
                                  "finish_reason": "stop"})
        self.assertEqual(value["confidence"], 0.5)
        self.assertIsNone(residual(value))

    def test_rejects_unbounded_screen_correction(self):
        with self.assertRaisesRegex(ValueError, "5% per-axis bound"):
            parse_observation({"text": '{"same_target":true,"visible_both":true,'
                               '"confidence":0.9,"screen_delta":[0.06,0.0],'
                               '"scale_ratio":1.0}', "finish_reason": "stop"})

    def test_maps_screen_correction_to_bounded_world_xy_and_scale(self):
        pair = {"camera_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                "target_center_world": [0, 0, -2], "width": 500, "height": 500,
                "fx_px": 500, "fy_px": 500}
        observation = {"screen_delta": [0.1, -0.05], "apparent_scale_ratio": 1.2}
        self.assertEqual(transform_delta(pair, observation), [0.1, 0.05, 1.05])

    def test_missing_target_does_not_receive_a_residual(self):
        self.assertIsNone(residual({"same_target": False, "visible_both": False}))

    def test_uncertain_action_does_not_receive_a_residual(self):
        value = parse_observation({"text": '{"same_target":true,"visible_both":true,'
                                   '"confidence":0.5,"screen_delta":null,"scale_ratio":1.0}',
                                  "finish_reason": "stop"})
        self.assertIsNone(residual(value))


if __name__ == "__main__":
    unittest.main()
