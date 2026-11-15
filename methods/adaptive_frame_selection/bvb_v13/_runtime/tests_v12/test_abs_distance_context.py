import unittest
import numpy as np
from bvb_v13.abs_distance_context import appearance_times,bridge_intervals,select_rows,plan,validate_question_binding,prepare
from unittest.mock import patch
from pathlib import Path


class ContextTests(unittest.TestCase):
    def test_retention_metadata_does_not_change_question_identity(self):
        q=dict(id=2990,scene_name='scene',question_type='object_abs_distance',question='Distance?',objects=['chair','table'],roles=dict(first='chair',second='table'),options=None)
        validate_question_binding(dict(q,retention_policy=dict(category='single',decision_file='/audit/result.json')),q)
        for key,value in [('id',2991),('scene_name','other'),('question_type','object_size'),('question','Other?'),('objects',['chair','bed']),('roles',dict(first='table',second='chair')),('options',['A','B'])]:
            with self.subTest(field=key),self.assertRaisesRegex(ValueError,key):
                validate_question_binding(q,dict(q,**{key:value}))

    def test_wrong_question_fails_before_creating_partial_input(self):
        with patch('bvb_v13.abs_distance_context.read',side_effect=[dict(public_question=dict(id=2)),dict(public_question=dict(id=1))]),patch('bvb_v13.abs_distance_context.multi.prepare') as build:
            with self.assertRaisesRegex(ValueError,'wrong question'):
                prepare(Path('/source'),Path('/out'),Path('/poses'),Path('/context'))
            build.assert_not_called()

    def test_bridge_excludes_unrelated_before_and_after_sightings(self):
        self.assertEqual(bridge_intervals(dict(first=[0,4],second=[36,40])),[[4.,36.]])
        self.assertEqual(bridge_intervals(dict(first=[0,20],second=[10,30])),[[0.,30.]])

    def test_uniform_spacing_without_position_motion(self):
        rows=[dict(source_frame_index=i,timestamp_s=float(i),sharpness=1) for i in range(41)]
        cameras=np.repeat(np.eye(4)[None],41,axis=0)
        selected,report=select_rows(rows,cameras,dict(first=[0,4],second=[36,40]),interval_seconds=8.)
        self.assertEqual(selected,[4,12,20,28,36])
        self.assertEqual(report['max_temporal_gap_s'],8)
        self.assertEqual(report['position_guard_count'],0)

    def test_position_guard_preserves_brief_excursion_missed_by_uniform(self):
        rows=[dict(source_frame_index=i,timestamp_s=float(i),sharpness=1) for i in range(17)]
        cameras=np.repeat(np.eye(4)[None],17,axis=0);cameras[3,0,3]=10
        selected,report=select_rows(rows,cameras,dict(first=[0],second=[16]))
        self.assertIn(3,selected)
        self.assertEqual(report['represented_position_groups'],report['position_group_count'])
        self.assertEqual(report['position_guard_count'],1)

    def test_two_second_sampling_keeps_the_same_bridge_scope(self):
        rows=[dict(source_frame_index=i,timestamp_s=float(i),sharpness=1) for i in range(41)]
        cameras=np.repeat(np.eye(4)[None],41,axis=0)
        selected,report=select_rows(rows,cameras,dict(first=[0,4],second=[36,40]))
        self.assertEqual(selected,list(range(4,37,2)))
        self.assertEqual(report['max_temporal_gap_s'],2.)
        self.assertEqual(report['interval_seconds'],2.)

    def test_colocated_sightings_need_no_invented_interval(self):
        rows=[dict(source_frame_index=10,timestamp_s=1.,sharpness=1)]
        selected,report=select_rows(rows,np.eye(4)[None],dict(first=[1],second=[1]))
        self.assertEqual(selected,[0]);self.assertEqual(report['bridge_intervals_s'],[[1.,1.]])

    def test_unlocalized_fillers_do_not_expand_endpoint_span(self):
        q=dict(roles=dict(first='chair',second='table'))
        groups=[dict(object=c,selected=[dict(timestamp_s=t,observations=[dict(object=c,box_xyxy=[1,2,3,4])]),dict(timestamp_s=1000,observations=[dict(object=c,box_xyxy=None)])]) for c,t in [('chair',1),('table',5)]]
        self.assertEqual(appearance_times(q,groups),dict(first=[1.],second=[5.]))

    def test_other_tasks_are_not_changed(self):
        with self.assertRaisesRegex(ValueError,'abs-distance only'):plan(dict(question_type='room_size_estimation'),[],None,None)


if __name__=='__main__':unittest.main()
