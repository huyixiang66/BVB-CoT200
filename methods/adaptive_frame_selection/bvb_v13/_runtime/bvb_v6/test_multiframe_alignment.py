import math
import unittest

import numpy as np
from scipy.optimize import least_squares

from .multiframe_alignment import edge_residual, fit_group, projection, transformed


class MultiframeAlignmentTests(unittest.TestCase):
    def test_blender_camera_projection(self):
        points = np.array([[-1, -1, -4], [1, -1, -4], [1, 1, -4], [-1, 1, -4]], dtype=float)
        k = [[400, 0, 320], [0, 400, 240], [0, 0, 1]]
        np.testing.assert_allclose(projection(points, np.eye(4), k), [220, 140, 420, 340])

    def test_frame_clipped_edges_are_one_sided(self):
        np.testing.assert_allclose(edge_residual(np.array([-20, 25, 680, 400]), [0, 25, 640, 400]), [0, 0, 0, 0])
        self.assertGreater(edge_residual(np.array([10, 25, 630, 400]), [0, 25, 640, 400])[0], 0)

    def test_multiple_views_recover_synthetic_translation(self):
        points = np.array([[x, y, z] for x in (-.5, .5) for y in (-.2, .2) for z in (-5., -4.5)])
        pivot = np.mean(points, axis=0); k = [[400, 0, 320], [0, 400, 240], [0, 0, 1]]
        cameras = []
        for x, y in ((-.5, 0), (.4, 0), (0, .4)):
            camera = np.eye(4); camera[:3, 3] = [x, y, 0]; cameras.append(camera)
        truth = np.array([.15, -.08, .12, 0.])
        boxes = [projection(transformed(points, pivot, truth), c, k) for c in cameras]
        def residual(x):
            return np.concatenate([edge_residual(projection(transformed(points, pivot, x), c, k), b)
                                   for c, b in zip(cameras, boxes)])
        result = least_squares(residual, np.zeros(4), bounds=([-.75, -.75, -.35, -math.radians(20)],
                                                             [.75, .75, .35, math.radians(20)]))
        np.testing.assert_allclose(result.x[:3], truth[:3], atol=.005)

    def test_insufficient_frames_do_not_produce_transform(self):
        target = {'group_id': 'target-01', 'category': 'chair'}
        points = np.array([[x, y, z] for x in (-.5, .5) for y in (-.2, .2) for z in (-5., -4.5)])
        result = fit_group(target, points, [])
        self.assertEqual(result['status'], 'insufficient')
        self.assertNotIn('translation', result)


if __name__ == '__main__':
    unittest.main()
