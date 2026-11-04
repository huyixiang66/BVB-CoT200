import unittest
import numpy as np
from bvb_v5.global_context import select_context


class ContextTests(unittest.TestCase):
    def test_static_camera_still_covers_time_and_deduplicates(self):
        candidates = [dict(source_frame_index=i*30, timestamp_s=float(i), sharpness=100.) for i in range(80)]
        poses = np.repeat(np.eye(4)[None], 80, axis=0)
        rows = select_context(candidates, poses)
        ids = [r['source_frame_index'] for r in rows]
        self.assertEqual(len(ids), 16)
        self.assertEqual(ids, sorted(set(ids)))
        self.assertEqual({int(r['timestamp_s'])//10 for r in rows}, set(range(8)))

    def test_short_input_and_alignment(self):
        candidates = [dict(source_frame_index=i, timestamp_s=i/30, sharpness=1.) for i in range(3)]
        poses = np.repeat(np.eye(4)[None], 3, axis=0)
        self.assertEqual(len(select_context(candidates, poses)), 3)
        with self.assertRaises(ValueError):
            select_context(candidates, poses[:2])
