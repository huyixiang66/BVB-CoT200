import unittest
import numpy as np
from .object_clustering import PARAMETERS,roi_surface,compare_roi,cluster,supported_pair_score,simultaneous_conflict

class ObjectGroupingTests(unittest.TestCase):
    def test_roi_excludes_central_background(self):
        x,y=np.meshgrid(np.linspace(-.8,-.4,12),np.linspace(-.25,.25,12))
        obj=np.c_[x.ravel(),y.ravel(),np.ones(x.size)]
        bg=np.c_[np.linspace(-.1,.1,100),np.zeros(100),np.full(100,3.)]
        cloud={'points':np.r_[obj,bg],'frame_ids':np.zeros(len(obj)+len(bg),int)}
        k=np.array([[100,0,100],[0,100,100],[0,0,1.]])
        d=roi_surface(0,np.eye(4),k,cloud,[200,200],[15,70,65,130])
        self.assertIsNotNone(d);self.assertLess(d['anchor'][0],-.3);self.assertAlmostEqual(d['depth'],1.)
        self.assertIsNone(roi_surface(0,np.eye(4),k,cloud,[200,200],None))

    def test_unknown_geometry_uses_available_appearance(self):
        r=compare_roi(None,None,1.,.73)
        self.assertEqual(r['score'],.73);self.assertIsNone(r['geometry']);self.assertFalse(r['far'])

    def test_support_needs_distinct_views_on_both_sides(self):
        pairs=[dict(score=.9,frames=[1,3]),dict(score=.8,frames=[1,4]),dict(score=.6,frames=[2,4])]
        self.assertEqual(supported_pair_score(pairs),(.6,2))

    def evidence(self,values,conflict=None):
        return {k:dict(score=v,independent_frame_pairs=2,cannot_link=k==conflict) for k,v in values.items()}

    def test_single_valid_frame_merges_but_conflicts_still_block(self):
        pairs=[dict(score=.99,frames=[1,3],valid=False),dict(score=.65,frames=[2,3],valid=True)]
        score,count=supported_pair_score(pairs,minimum=1)
        self.assertEqual((score,count),(.65,1))
        e={('a','b'):dict(score=score,independent_frame_pairs=count,cannot_link=False)}
        params={**PARAMETERS,'minimum_frame_pairs':1}
        self.assertEqual(len(cluster(['a','b'],e,params=params)[0]),1)
        self.assertEqual(len(cluster(['a','b'],e)[0]),2)
        e[('a','b')]['cannot_link']=True
        self.assertEqual(len(cluster(['a','b'],e,params=params)[0]),2)
        self.assertEqual(supported_pair_score([pairs[0]],minimum=1),(0.,0))

    def test_one_strong_bridge_does_not_join_clusters(self):
        e=self.evidence({('a','b'):.95,('a','c'):.1,('b','c'):.9})
        groups,_=cluster(['a','b','c'],e);self.assertEqual(len(groups),2)

    def test_multiple_links_allow_a_weak_pair(self):
        e=self.evidence({('a','b'):.95,('a','c'):.7,('a','d'):.3,('b','c'):.8,('b','d'):.7,('c','d'):.96})
        self.assertEqual(len(cluster(list('abcd'),e)[0]),1)
        self.assertGreater(len(cluster(list('abcd'),e,'complete')[0]),1)
        e[('a','d')]['cannot_link']=True
        self.assertGreater(len(cluster(list('abcd'),e)[0]),1)

    def test_simultaneous_distinct_boxes(self):
        a=[dict(source_frame_index=3,box_xyxy=[0,0,10,10],dino_score=.7)]
        b=[dict(source_frame_index=3,box_xyxy=[20,0,30,10],dino_score=.7)]
        self.assertTrue(simultaneous_conflict(a,b))
        b[0]['source_frame_index']=4;self.assertFalse(simultaneous_conflict(a,b))

if __name__=='__main__':unittest.main()
