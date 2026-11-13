import unittest
from bvb_v11.counting_mode import enabled,peak_threshold,assign_colors
from bvb_v11.reliability import PALETTE
from bvb_v11.coverage_check import validate


class CountingModeTest(unittest.TestCase):
    def test_opt_in_and_question_gate(self):
        q={'question_type':'object_counting_single'}
        self.assertFalse(enabled(q))
        self.assertTrue(enabled(q,True))
        self.assertFalse(enabled({'question_type':'object_rel_distance_closest'},True))
        self.assertEqual(peak_threshold([q],True),.25)
        self.assertEqual(peak_threshold([q]),.5)
        with self.assertRaises(ValueError):peak_threshold([{'question_type':'route_planning'}],True)

    def test_dense_layout_neighbors_and_stability(self):
        rows=[dict(instance_id=f'{cat}-{i:02}',category=cat,center=[i%10,i//10,0]) for cat in ['chair','table'] for i in range(50)]
        a=assign_colors(rows,PALETTE);b=assign_colors(list(reversed(rows)),PALETTE)
        self.assertEqual(a,b)
        self.assertTrue(all(a['instances'][x]['rgba']!=a['instances'][y]['rgba'] for x,y in a['neighbour_edges']))
        self.assertFalse({tuple(c) for c in a['category_palette']['chair']}&{tuple(c) for c in a['category_palette']['table']})

    def test_coverage_checks_instance_colors_and_saved_parts(self):
        data=dict(question={'objects':['table']},colors={'table':PALETTE[0]},counting_mode=True,instance_colors={'t1':PALETTE[2]})
        records=[dict(name=n,role='target',category='table',instance_id='t1',dimensions=[1,1,1],color=PALETTE[2],hide_render=False) for n in ['top','leg']]
        self.assertEqual(validate(data,records,None)['instances_by_category'],{'table':['t1']})
        records[1]['color']=PALETTE[0]
        with self.assertRaises(ValueError):validate(data,records,None)

if __name__=='__main__':unittest.main()
