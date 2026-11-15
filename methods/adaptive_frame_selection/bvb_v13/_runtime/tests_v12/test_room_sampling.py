import unittest
import numpy as np
from bvb_v13.room_sampling import select_room_frames


class RoomSamplingTest(unittest.TestCase):
    def sample(self, positions, directions, sharpness=None, **kw):
        rows=[dict(source_frame_index=i*10,timestamp_s=float(i),sharpness=(sharpness or [1]*len(positions))[i]) for i in range(len(positions))]
        poses=[dict(frame=i*10,time=float(i),position=p,forward=d) for i,(p,d) in enumerate(zip(positions,directions))]
        return select_room_frames({'question_type':'room_size_estimation_multiple'},rows,poses,**kw)

    def test_stationary_repeats_choose_sharpest(self):
        group, report=self.sample([[0,0,0]]*3,[[1,0,0]]*3,[1,9,2])
        self.assertEqual(report['selected_source_frame_indices'],[10])
        self.assertEqual(len(group['candidate_frames']),3)

    def test_each_position_and_opposite_direction_retained(self):
        group, report=self.sample([[0,0,0],[0,0,0],[10,0,0]],[[1,0,0],[-1,0,0],[1,0,0]])
        self.assertEqual(report['position_group_count'],2)
        self.assertEqual(len(group['selected']),3)
        self.assertEqual(report['group_count'],1)

    def test_coverage_precedes_sharpness(self):
        angles=np.radians([0,40,80])
        _,report=self.sample([[0,0,0]]*3,[[np.cos(a),np.sin(a),0] for a in angles],[100,1,100])
        self.assertEqual(report['selected_source_frame_indices'],[10])

    def test_scale_invariance(self):
        p=np.array([[0,0,0],[.1,0,0],[1,0,0],[1.1,0,0]])
        a=self.sample(p,[[1,0,0]]*4)[1]
        b=self.sample(p*100,[[1,0,0]]*4)[1]
        self.assertEqual(a['position_groups'],b['position_groups'])

    def test_invalid_poses_excluded_and_other_types_rejected(self):
        group,report=self.sample([[0,0,0],[float('nan'),0,0]],[[1,0,0]]*2)
        self.assertEqual(report['excluded_pose_frame_indices'],[10])
        self.assertEqual(len(group['selected']),1)
        with self.assertRaises(ValueError):
            select_room_frames({'question_type':'object_abs_distance'},[],[])


if __name__=='__main__':unittest.main()
