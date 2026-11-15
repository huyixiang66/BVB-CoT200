import unittest
import numpy as np
from .select_frames import pose_distance,choose


def candidate(i,angle=0,context=False):
    r=np.radians(angle);center=np.array([np.sin(r),0,np.cos(r)])
    return {'source_frame_index':i,'camera_center':center.tolist(),'view_direction':(-center).tolist(),'quality_score':.1 if context else .8,'status':'context_fallback' if context else 'locally_verified'}


class Selection(unittest.TestCase):
    def test_pure_rotation_no_geometric_gain(self):
        a=candidate(1);b=candidate(2,90);b['camera_center']=a['camera_center']
        self.assertEqual(pose_distance(a,b,1)['novelty'],0)
    def test_four_when_every_view_repeats(self):
        rows=[candidate(i) for i in range(20)];chosen,extra=choose(rows,1)
        self.assertEqual(len(chosen),4);self.assertFalse(extra)
    def test_real_evidence_preferred_to_context(self):
        rows=[candidate(i,i*5) for i in range(4)]+[candidate(8,180,True)]
        chosen,_=choose(rows,1);self.assertEqual(len(chosen),4);self.assertTrue(all(r['status']!='context_fallback' for r in chosen))
    def test_minimum_even_when_context_required(self):
        rows=[candidate(i,i*10,i>1) for i in range(4)];chosen,_=choose(rows,1)
        self.assertEqual(len(chosen),4);self.assertEqual(sum(r['status']=='context_fallback' for r in chosen),2)
    def test_budget_and_scale_invariance(self):
        rows=[candidate(i,i*20) for i in range(18)];chosen,_=choose(rows,1)
        self.assertTrue(4<=len(chosen)<=8)
        a,b=candidate(1),candidate(2,40);x=pose_distance(a,b,1)
        a['camera_center']=(np.array(a['camera_center'])*7).tolist();b['camera_center']=(np.array(b['camera_center'])*7).tolist()
        self.assertAlmostEqual(x['novelty'],pose_distance(a,b,7)['novelty'])
    def test_shared_frame_can_cover_two_instance_reservations(self):
        rows=[candidate(i,i*3) for i in range(9)]
        chosen,_=choose(rows,1,required_frame_sets=[[0,1,2,3],[0,4,5,6]])
        ids={r['source_frame_index'] for r in chosen}
        self.assertTrue({0,1,2,3,4,5,6}<=ids);self.assertLessEqual(len(ids),8)
    def test_reservation_minimum_overrides_default_cap(self):
        rows=[candidate(i) for i in range(12)]
        chosen,_=choose(rows,1,required_frame_sets=[list(range(4)),list(range(4,8)),list(range(8,12))])
        self.assertEqual(len(chosen),12)


if __name__=='__main__':unittest.main()
