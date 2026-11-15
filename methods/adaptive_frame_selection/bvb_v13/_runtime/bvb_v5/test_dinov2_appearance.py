"""Identity safeguards must hold even for arbitrarily high appearance similarity."""
import unittest
from bvb_v5.dinov2_appearance import pair_evidence, roi_key


class AppearanceEvidenceTest(unittest.TestCase):
    def test_same_frame_distinct_cannot_merge_despite_identical_embedding(self):
        peaks = {
            'a': {'object':'chair','observations':[{'frame':1,'score':.9,'box_xyxy':[0,0,20,20]}]},
            'b': {'object':'chair','observations':[{'frame':1,'score':.9,'box_xyxy':[50,0,70,20]}]}}
        x = pair_evidence('a','b',1.,1.,peaks,{})
        self.assertTrue(x['same_frame_distinct_evidence'])
        self.assertEqual(x['decision'],'do_not_merge_same_frame_distinct')
        self.assertFalse(x['automatic_merge'])

    def test_missing_geometry_does_not_confirm_identity(self):
        peaks={k:{'object':'chair','observations':[]} for k in ['a','b']}
        x=pair_evidence('a','b',.99,.99,peaks,{})
        self.assertIsNone(x['sparse_geometry_separation_ratio'])
        self.assertEqual(x['decision'],'appearance_evidence_only')
        self.assertFalse(x['automatic_merge'])

    def test_sparse_geometry_conflict_is_review_not_ground_truth(self):
        peaks={k:{'object':'chair','observations':[]} for k in ['a','b']}
        geometry={'a':{'center':[0,0,0],'radius':.1},'b':{'center':[1,0,0],'radius':.1}}
        x=pair_evidence('a','b',.99,.99,peaks,geometry)
        self.assertTrue(x['geometry_conflict_heuristic'])
        self.assertEqual(x['decision'],'review_geometry_conflict')

    def test_same_frame_boxes_have_different_cache_keys(self):
        self.assertNotEqual(roi_key('chair',1,[0,0,20,20]),roi_key('chair',1,[50,0,70,20]))


if __name__=='__main__':unittest.main()
