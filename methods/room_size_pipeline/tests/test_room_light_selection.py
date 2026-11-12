import math
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from room_size_pipeline.light_selection import choose_light_frames


def row(t, yaw=0, x=0, sharpness=10):
    a=math.radians(yaw)
    return {'timestamp_s':t, 'sharpness':sharpness,
            'c2w_blender':[[math.cos(a),0,-math.sin(a),x],
                           [math.sin(a),0,math.cos(a),0],
                           [0,-1,0,0],[0,0,0,1]]}


class LightSelectionTests(unittest.TestCase):
    def test_repeated_view_prefers_clear_frame(self):
        times,_=choose_light_frames(40,[row(0,sharpness=1),row(8,sharpness=20),row(32,sharpness=2)])
        self.assertEqual(times,[8])

    def test_turn_in_place_is_preserved(self):
        times,_=choose_light_frames(24,[row(0),row(8,180)])
        self.assertEqual(times,[0,8])

    def test_scale_invariance(self):
        a=[row(i*2,x=i*.2) for i in range(8)]
        b=[row(i*2,x=i*20) for i in range(8)]
        self.assertEqual(choose_light_frames(16,a)[0],choose_light_frames(16,b)[0])

    def test_budget_is_soft(self):
        times,report=choose_light_frames(4,[row(0),row(1,180),row(2,90)])
        self.assertGreater(len(times),report['reference_count'])

    def test_bad_sharpness(self):
        with self.assertRaises(ValueError):
            choose_light_frames(5,[row(0,sharpness=float('nan'))])


if __name__=='__main__':unittest.main()
