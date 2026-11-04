import unittest
from unittest.mock import patch

from bvb_v5.instance_sampling import main, representative_support
from bvb_v5.instance_report import representative_caption
from bvb_v5.pose_association import enhanced_support


class V5IntegrationTests(unittest.TestCase):
    def test_cli_defaults_enable_pose_and_keep_threshold_point_four(self):
        argv = ['v5']
        for name in ['v3', 'v4', 'appearance', 'cache', 'lingbot', 'out']:
            argv.extend(['--' + name, '/tmp/example'])
        with patch('sys.argv', argv), patch('bvb_v5.instance_sampling.run') as run:
            main()
        args = run.call_args.args[0]
        self.assertEqual(args.threshold, .40)
        self.assertEqual(args.mode, 'aggressive')
        self.assertTrue(args.pose_association)

    def test_geometry_cannot_override_same_frame_separation(self):
        pair = dict(peak_a='a', peak_b='b', pose_depth=True, pose_keypoints=True,
                    same_frame_distinct_evidence=True, matching=None,
                    merge_supported=False, multi_view_score=None)
        self.assertIsNone(enhanced_support(['a', 'b'], {('a', 'b'): pair}, {},
                                          representative_support))

    def test_pose_only_support_does_not_report_priority_as_similarity(self):
        pair = dict(peak_a='a', peak_b='b', pose_depth=True, pose_keypoints=False,
                    same_frame_distinct_evidence=False, matching=None,
                    merge_supported=False, multi_view_score=None)
        proof = enhanced_support(['a', 'b'], {('a', 'b'): pair}, {}, representative_support)
        self.assertTrue(proof['priority_only'])
        caption = representative_caption(proof)
        self.assertIn('位姿证据', caption)
        self.assertNotIn('0.600', caption)


if __name__ == '__main__':
    unittest.main()
