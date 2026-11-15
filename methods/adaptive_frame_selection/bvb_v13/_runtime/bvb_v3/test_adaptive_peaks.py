"""Small policy regressions; visual quality is evaluated separately."""
import unittest
from .adaptive_peaks import associate, peak_tracks, merge_peak_fragments


class Policies(unittest.TestCase):
    def box(self,x,score=.7):return {'object':'chair','score':score,'box_xyxy':[x,0,x+10,10]}
    def tracks(self,boxes):
        rows=[{'source_frame_index':i,'timestamp_s':float(i),'detections':b} for i,b in enumerate(boxes)]
        return peak_tracks(rows,'chair',{i:i for i in range(len(rows))},lambda u,v,b:(b,0))[0]
    def test_simultaneous_instances(self):
        p=self.tracks([[self.box(0),self.box(30)] for _ in range(3)])
        self.assertEqual(len(p),2)
        self.assertTrue(all(len(t['observations'])==3 for t in p))
    def test_unobserved_gap_is_not_identity_match(self):
        self.assertEqual(len(self.tracks([[self.box(0)],[],[],[],[],[self.box(0)]])),2)
    def test_fallback_only_if_whole_category_below_cut(self):
        p=self.tracks([[self.box(0,.4)],[self.box(0,.45)]])
        self.assertTrue(p[0]['fallback']);self.assertEqual(p[0]['anchor']['frame'],1)
        p=self.tracks([[self.box(0,.4),self.box(30,.7)]])
        self.assertEqual(len(p),1);self.assertFalse(p[0]['fallback'])
    def test_no_double_assignment(self):
        self.assertEqual(len(associate({0:[0,0,10,10],1:[1,0,11,10]},[self.box(0)])),1)
    def test_peak_merge_preserves_same_frame_disjoint_boxes(self):
        def p(k,x,s):return {'anchor':{'frame':k,'box_xyxy':[x,0,x+10,10],'score':s},'tracklet':k*100+x}
        kept,suppressed=merge_peak_fragments([p(0,0,.8),p(1,0,.7),p(0,30,.6),p(7,0,.7)],list(range(8)))
        self.assertEqual((len(kept),len(suppressed)),(3,1))


if __name__=='__main__':unittest.main()
