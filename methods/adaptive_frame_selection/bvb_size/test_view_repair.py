import unittest
from .view_repair import wall_retry_candidates


class ViewRepairTests(unittest.TestCase):
    def test_only_unresolved_occlusion_triggers_wall_retry(self):
        occluded = {'blockers': ['object_0:external_occlusion'], 'views': [{'wall_occluders': ['wall-b', 'wall-a']}]}
        small = {'blockers': ['object_1:too_small'], 'views': [{'wall_occluders': ['wall-c']}]}
        self.assertEqual(wall_retry_candidates('unresolved', occluded, small), ['wall-a', 'wall-b', 'wall-c'])
        self.assertEqual(wall_retry_candidates('single', occluded, None), [])
        self.assertEqual(wall_retry_candidates('unresolved', small, None), [])
        self.assertEqual(wall_retry_candidates('unresolved', {'blockers': ['object_0:external_occlusion'], 'views': [{}]}, None), [])
