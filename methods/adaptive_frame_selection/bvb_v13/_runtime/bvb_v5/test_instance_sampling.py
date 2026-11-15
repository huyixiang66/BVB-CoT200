import unittest
import numpy as np
from bvb_v5.instance_sampling import constrained_clusters, match_observations, representative_clusters


def obs(i,t,anchor=False):
    return {'roi_id':str(i),'feature_index':i,'timestamp_s':t,'source_frame_index':round(t*30),'anchor':anchor}


class InstanceSamplingTest(unittest.TestCase):
    def test_representative_can_merge_without_complete_link(self):
        p={('a','b'):{'merge_supported':True,'multi_view_score':.8,'same_frame_distinct_evidence':False},
           ('b','c'):{'merge_supported':True,'multi_view_score':.8,'same_frame_distinct_evidence':False},
           ('a','c'):{'merge_supported':False,'multi_view_score':.55,'same_frame_distinct_evidence':False}}
        self.assertEqual(representative_clusters(['a','b','c'],p)[0],[['a','b','c']])
        p[('a','c')]['same_frame_distinct_evidence']=True
        self.assertEqual(len(representative_clusters(['a','b','c'],p)[0]),2)

    def test_long_chain_without_common_representative_cannot_collapse(self):
        import itertools
        p={pair:{'merge_supported':pair in [('a','b'),('b','c'),('c','d')],
                 'multi_view_score':.8 if pair in [('a','b'),('b','c'),('c','d')] else .55,
                 'same_frame_distinct_evidence':False} for pair in itertools.combinations('abcd',2)}
        self.assertGreater(len(representative_clusters(list('abcd'),p)[0]),1)

    def test_relaxed_matching_does_not_require_anchors(self):
        f=np.ones((4,1))
        self.assertIsNone(match_observations([obs(0,0),obs(1,.25)],[obs(2,3),obs(3,3.25)],f))
        self.assertIsNotNone(match_observations([obs(0,0),obs(1,.25)],[obs(2,3),obs(3,3.25)],f,.25,False))

    def test_bridge_does_not_merge_three_groups(self):
        p={('a','b'):{'merge_supported':True,'multi_view_score':.9},
           ('b','c'):{'merge_supported':True,'multi_view_score':.89},
           ('a','c'):{'merge_supported':False,'multi_view_score':.3}}
        groups,_=constrained_clusters(['a','b','c'],p)
        self.assertEqual(groups,[['a','b'],['c']])

    def test_all_cross_pairs_supported_can_merge(self):
        p={pair:{'merge_supported':True,'multi_view_score':.9} for pair in [('a','b'),('a','c'),('b','c')]}
        self.assertEqual(constrained_clusters(['a','b','c'],p)[0],[['a','b','c']])

    def test_one_good_match_cannot_pass_as_two(self):
        f=np.array([[1.,0],[0,1],[1,0],[0,-1]])
        x=match_observations([obs(0,0,True),obs(1,1)],[obs(2,5,True),obs(3,6)],f)
        self.assertLess(x['bottleneck'],.82)

    def test_one_frame_per_side_insufficient(self):
        self.assertIsNone(match_observations([obs(0,0,True)],[obs(1,5,True)],np.ones((2,1))))

    def test_shared_source_frame_is_not_cross_time_evidence(self):
        f=np.eye(2)
        x=match_observations([obs(0,0,True),obs(1,1)],[obs(0,0,True),obs(1,1)],f)
        self.assertEqual(x['bottleneck'],0.)

    def test_adjacent_near_duplicates_not_independent_support(self):
        self.assertIsNone(match_observations([obs(0,0,True),obs(1,.25)],
                          [obs(2,5,True),obs(3,5.25)],np.ones((4,1))))


if __name__=='__main__':unittest.main()
