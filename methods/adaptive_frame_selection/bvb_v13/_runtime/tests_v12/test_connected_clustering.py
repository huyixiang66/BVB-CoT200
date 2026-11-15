import unittest
from bvb_v13.connected_clustering import connected_groups,reweight

def edge(a,b,score,conflict=False):
    return ((a,b),dict(peaks=[a,b],score=score,valid=True,cannot_link=conflict))

class ConnectedPolicyTests(unittest.TestCase):
    def test_transitive_chain_without_all_pairs_support(self):
        es=dict([edge('a','b',.8),edge('b','c',.7),edge('a','c',.1)])
        groups,history=connected_groups(['a','b','c'],es)
        self.assertEqual(groups,[['a','b','c']]);self.assertEqual(len(history['merges']),2)
    def test_indirect_conflict_is_not_bypassed(self):
        es=dict([edge('a','b',.8),edge('b','c',.7),edge('a','c',.1,True)])
        groups,h=connected_groups(['a','b','c'],es)
        self.assertEqual(groups,[['a','b'],['c']]);self.assertEqual(len(h['blocked']),1)
        raw,h=connected_groups(['a','b','c'],es,protect=False)
        self.assertEqual(raw,[['a','b','c']]);self.assertEqual(h['internal_conflicts'],[['a','c']])
    def test_surface_dominates_and_unknown_geometry_uses_appearance(self):
        p=dict(geometry=.5,nearest_surface=0.,semantic=0.,reliability=1.)
        self.assertAlmostEqual(reweight(p)['score'],.8)
        p.update(geometry=None,semantic=.7,reliability=0.)
        self.assertEqual(reweight(p)['score'],.7)
    def test_invalid_frame_is_not_a_connection(self):
        es=dict([edge('a','b',.9)]);es[('a','b')]['valid']=False
        self.assertEqual(connected_groups(['a','b'],es)[0],[['a'],['b']])

if __name__=='__main__':unittest.main()
