"""Acceptance regressions: utility cannot compensate for failed geometry."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/direct_views'))
from acceptance import assess_view
from fit_camera import center, union_box, dot, fit_camera
import continuous_fit


def item(name, position=(0., 0., 0.)):
    return dict(key=name+'#0', semantic=name,
                box={'min':list(position),'max':[v+.2 for v in position]})


def view(items):
    return dict(forward=[0.,1.,0.], instance_metrics={i['key']:{
        'edge_margin':.1,'area_fraction':.04,'visible_fraction':.8,
        'projected_box':[.2,.2,.4,.4],'readable':True} for i in items})


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.target=item('target')

    def test_readable_target_passes(self):
        self.assertTrue(assess_view(view([self.target]),[self.target])['passed'])

    def test_room_footprint_can_explicitly_omit_bare_floor_visibility(self):
        camera=view([self.target]);camera['instance_metrics'][self.target['key']]['visible_fraction']=0.
        self.assertFalse(assess_view(camera,[self.target])['passed'])
        self.assertTrue(assess_view(camera,[self.target],min_visible=0.)['passed'])

    def test_large_but_occluded_target_fails(self):
        camera=view([self.target]);metric=camera['instance_metrics'][self.target['key']]
        metric.update(area_fraction=.8,visible_fraction=.1)
        self.assertIn('target#0:occluded',assess_view(camera,[self.target])['failures'])

    def test_clipped_target_fails(self):
        camera=view([self.target]);camera['instance_metrics'][self.target['key']]['edge_margin']=-.01
        self.assertFalse(assess_view(camera,[self.target])['passed'])

    def test_edge_on_object_fails_even_with_large_box_area(self):
        camera=view([self.target]);camera['instance_metrics'][self.target['key']]['projected_box']=[.2,.2,.21,.9]
        self.assertIn('target#0:short_side_too_small',assess_view(camera,[self.target])['failures'])

    def test_each_relation_must_pass_instead_of_average_retention(self):
        objects=[item('a'),item('b',(0.,1.,0.)),item('c',(1.,0.,0.)),item('d',(1.,0.,1.))]
        gate=assess_view(view(objects),objects,[('a','b'),('a','c'),('a','d')])
        self.assertFalse(gate['passed'])
        self.assertIn('a#0 -> b#0:compressed',gate['failures'])

    def test_failed_silhouette_cannot_be_overridden_by_surface_visibility(self):
        camera=view([self.target]);camera['instance_metrics'][self.target['key']]['silhouette']={
            'clear':False,'silhouette_visibility_fraction':.4}
        self.assertFalse(assess_view(camera,[self.target])['passed'])

    def test_missing_relation_endpoint_fails(self):
        self.assertFalse(assess_view(view([self.target]),[self.target],[('target','missing')])['passed'])

    def test_facing_reference_does_not_claim_behind_object_is_visible(self):
        facing,behind=item('facing'),item('behind')
        camera=view([facing,behind]);camera['instance_metrics'][behind['key']]['visible_fraction']=0.
        self.assertTrue(assess_view(camera,[facing])['passed'])
        self.assertFalse(assess_view(camera,[facing,behind])['passed'])


class SolverGateTests(unittest.TestCase):
    def run_solver(self, include_accepted):
        target=item('target')
        g=SimpleNamespace(floor={'min':[0.,0.,0.],'max':[5.,5.,0.]},room_span=5.,
                          union_box=union_box,center=center,dot=dot,fit_camera=fit_camera,
                          instances={'target':[target]})
        def camera(boxes,location,**kwargs):
            return dict(location=list(location),target=[2.,2.,1.],forward=[0.,0.,-1.],lens_mm=35.)
        def finish(cam,*args):
            metric={'edge_margin':.1,'area_fraction':.1 if cam['location'][0]==0 else .0035,
                    'visible_fraction':.19 if cam['location'][0]==0 else .21,
                    'projected_box':[.2,.2,.4,.4]}
            cam.update(instance_metrics={target['key']:metric},metadata={},
                       relation_check={'projection_retention':1.})
            cam['acceptance']=assess_view(cam,[target])
        def minimize(objective,*args,**kwargs):
            objective([0.,1.,2.,0.,0.])
            if include_accepted:objective([1.,1.,2.,0.,0.])
        with patch.object(continuous_fit,'minimize',minimize),patch.object(continuous_fit,'from_position',camera):
            return continuous_fit.solve(g,[target],[target],[target['box']],
                                        'distance',[],'test',finish,lambda loc:True)

    def test_accepted_solution_wins_over_lower_loss_occluded_solution(self):
        main,solutions=self.run_solver(True)
        self.assertEqual(main['location'][0],1.)
        self.assertTrue(all(c['acceptance']['passed'] for c in solutions))

    def test_no_passed_solution_returns_only_rejected_diagnostic(self):
        main,diagnostics=self.run_solver(False)
        self.assertIsNone(main)
        self.assertEqual(len(diagnostics),1)
        self.assertFalse(diagnostics[0]['acceptance']['passed'])


if __name__=='__main__':unittest.main()
