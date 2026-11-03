import copy
import unittest
import numpy as np
from .questions import extract_question,prepare
from .select_frames import select

def fixture(n=60):
    candidates=[{'timestamp_s':float(i),'source_frame_index':i*30,'sharpness':100.,'frame':f'{i}.jpg'} for i in range(n)]
    detections=[{'detections':[]} for _ in candidates]
    flow=[{'global_motion_since_previous_candidate':0.,'object_roi_motion_per_second':{}} for _ in candidates]
    return candidates,detections,flow

class SamplingTests(unittest.TestCase):
    def test_parser_answer_independence_and_roles(self):
        q={'id':1,'question_type':'object_rel_direction_medium','question':'If I am standing by the refrigerator and facing the sofa, is the kettle to my left, right, or back?\nAn object is to my back if I would have to turn at least 135 degrees in order to face it.','ground_truth':'anything'}
        a=extract_question(q);q['ground_truth']='door bed false';b=extract_question(q)
        self.assertEqual(a,b);self.assertEqual(a['objects'],['refrigerator','sofa','kettle']);self.assertNotIn('ground_truth',a)
    def test_static_object_covered_despite_unrelated_motion(self):
        c,d,f=fixture();d[20]['detections']=[{'object':'kettle','score':.9}]
        for i in range(40,59):f[i]['global_motion_since_previous_candidate']=100.
        q={'objects':['kettle'],'question_type':'object_size_estimation'}
        r=select(q,c,d,f,10);self.assertIn(20,r['selected_indices']);self.assertTrue(r['selected_object_coverage']['kettle'])
        self.assertEqual(len(set(r['selected_indices'])),10)
    def test_roi_change_affects_density_but_not_budget(self):
        c,d,f=fixture()
        for row in d:row['detections']=[{'object':'chair','score':.8}]
        q={'objects':['chair'],'question_type':'object_size_estimation'};baseline=select(q,c,d,f,12)
        for i in range(20,30):f[i]['object_roi_motion_per_second']['chair']=.2
        changed=select(q,c,d,f,12)
        self.assertGreater(sum(changed['density'][20:30]),sum(baseline['density'][20:30]));self.assertEqual(changed['selected_count'],12)
    def test_missing_objects_and_small_budget(self):
        c,d,f=fixture();q={'objects':['pan'],'question_type':'object_counting'}
        r=select(q,c,d,f,1);self.assertEqual(r['objects_not_detected'],['pan']);self.assertTrue(r['fallback']);self.assertEqual(r['selected_count'],1)
        r=select(q,c,d,f,100);self.assertEqual(r['selected_count'],60)
    def test_multiple_objects_order_and_determinism(self):
        c,d,f=fixture()
        for i,obj in [(9,'bed'),(32,'sofa')]:d[i]['detections']=[{'object':obj,'score':.8}]
        q={'objects':['bed','sofa'],'question_type':'obj_appearance_order'}
        r=select(q,c,d,f,12);self.assertTrue({8,9,31,32}<=set(r['selected_indices']));self.assertEqual(r,select(q,c,d,f,12))
    def test_unknown_template_fails_loudly(self):
        with self.assertRaises(ValueError):extract_question({'question':'Unknown format','question_type':'object_counting'})
    def test_route_variants_and_landmarks(self):
        q={'question_type':'route_planning','question':'You are a robot beginning by the chair and facing the window. You want to navigate to the door. 1. Go forward until past the bed 2. Go forward until the wall 3. Go forward until the door.'}
        self.assertEqual(set(extract_question(q)['objects']),{'chair','window','door','bed','wall'})

if __name__=='__main__':unittest.main()
