"""Reference-mode regression checks: cardinality, scale integrity, old contract."""
import copy
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from bvb_v6.reconstruction import expand


class ReferenceModeTests(unittest.TestCase):
    def setUp(self):
        self.q={'id':1,'question_type':'object_size_estimation','objects':['monitor'],
                'roles':{'target':'monitor'},'object_size_reference_mode':True}
        self.data={'structure':[],'targets':[['a','monitor',[0,0,1],[2,.2,1]]],'context':[],
            'scale_reference':{'object':['r','paper',[1,0,1],[1,.1,1.4]],'axis':2,
                'assumed_length_m':.297,'scale_basis':'estimated_visual_prior',
                'rationale':'A visible rectangular sheet beside the monitor.',
                'uncertainty':'Assumed A4; no printed scale is visible.',
                'source_timestamps_s':[1.5],'confidence':'low'}}

    def test_scale_and_target_are_separate(self):
        scene,audit=expand(self.data,self.q)
        self.assertEqual(len(audit['targets']),1)
        self.assertEqual(len(scene['objects']),2)
        self.assertAlmostEqual(audit['scale_reference']['meters_per_scene_unit'],.297/1.4)
        self.assertFalse(audit['scale_reference']['metric_scale_verified'])
        self.assertFalse(audit['ground_truth_used'])

    def test_extra_target_rejected(self):
        self.data['targets'].append(['b','monitor',[1,0,1],[2,.2,1]])
        with self.assertRaisesRegex(ValueError,'exactly one'):expand(self.data,self.q)

    def test_invalid_scale_rejected(self):
        for value in (0,-1,float('nan'),True):
            self.data['scale_reference']['assumed_length_m']=value
            with self.assertRaises(ValueError):expand(self.data,self.q)

    def test_target_cannot_calibrate_itself(self):
        self.data['scale_reference']['object'][1]='monitor'
        with self.assertRaisesRegex(ValueError,'differ'):expand(self.data,self.q)

    def test_missing_reference_is_explicit(self):
        self.data['scale_reference']=None
        with self.assertRaises(ValueError):expand(self.data,self.q)
        self.data['reference_unavailable_reason']='Only the target is visible.'
        _,audit=expand(self.data,self.q)
        self.assertIsNone(audit['scale_reference'])

    def test_default_preserves_legacy_scope(self):
        self.q.pop('object_size_reference_mode');self.data.pop('scale_reference')
        self.data['targets'].append(['b','monitor',[1,0,1],[2,.2,1]])
        _,audit=expand(self.data,self.q)
        self.assertEqual(len(audit['targets']),2)
        self.assertNotIn('scale_reference',audit)

    def test_cannot_claim_measured_scale(self):
        self.data['scale_reference']['scale_basis']='measured'
        with self.assertRaisesRegex(ValueError,'estimate'):expand(self.data,self.q)

    def test_missing_wall_style_placeholder_preserves_opening(self):
        opening=[[0,0,1,1]]
        self.data['structure']=[['wall','wall',[0,2,1],[3,.1,2],None,'white',opening]]
        original=copy.deepcopy(self.data)
        scene,audit=expand(self.data,self.q)
        wall=next(o for o in scene['objects'] if o['id']=='structure_wall')
        self.assertEqual(wall['openings'],opening)
        self.assertEqual(self.data,original)
        self.assertFalse(audit['transport_normalizations']['geometry_changed'])


if __name__=='__main__':unittest.main()
