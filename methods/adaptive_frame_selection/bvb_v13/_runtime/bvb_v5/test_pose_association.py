import itertools
import unittest

from bvb_v5.dinov2_appearance import crop_bounds
from bvb_v5.instance_sampling import representative_support
from bvb_v5.pose_association import enhanced_support, refine_groups


class PoseAssociationTests(unittest.TestCase):
    def test_crop_policy_keeps_box_coordinates_and_clips_context(self):
        box = [2, 3, 22, 23]
        self.assertEqual(crop_bounds(box, 640, 480, "tight"), (box, 0.))
        self.assertEqual(crop_bounds(box, 640, 480, "adaptive-context"), ([0, 0, 37, 38], .75))
        self.assertEqual(crop_bounds([100, 100, 300, 300], 640, 480, "adaptive-context"),
                         ([60, 60, 340, 340], .20))

    def test_used_pose_edge_cannot_absorb_unrelated_group(self):
        pairs = {}
        for a, b in itertools.combinations("abcd", 2):
            pose = (a, b) == ("b", "c")
            appearance = (a, b) in {("a", "b"), ("b", "d")}
            pairs[a, b] = dict(peak_a=a, peak_b=b, pose_depth=pose, pose_keypoints=False,
                               same_frame_distinct_evidence=False, matching=None,
                               merge_supported=appearance,
                               multi_view_score=.65 if (a, b) == ("a", "b") else .42 if appearance else None)

        def support(names, records):
            return enhanced_support(names, records, {}, representative_support)

        result, history = refine_groups([["a", "b"], ["c"], ["d"]], [], pairs, support)
        self.assertEqual(result, [["a", "b", "c"], ["d"]])
        self.assertEqual(history[0]["cross_pose_edges"], [["b", "c"]])
