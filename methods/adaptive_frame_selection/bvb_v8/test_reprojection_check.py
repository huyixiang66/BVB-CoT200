import json
import tempfile
import unittest
from pathlib import Path

from .reprojection_check import check_group, project


class ReprojectionTests(unittest.TestCase):
    def test_blender_camera_looks_down_negative_z(self):
        c2w = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        k = [[100,0,320],[0,100,240],[0,0,1]]
        self.assertEqual(project([0,0,-2], c2w, k), [320.0, 240.0])
        self.assertEqual(project([1,1,-2], c2w, k), [370.0, 190.0])
        self.assertIsNone(project([0,0,2], c2w, k))

    def test_only_matching_category_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); (p/'assets').mkdir()
            def write(name, data):
                (p/name).write_text(json.dumps(data), encoding='utf-8')
            write('input-frames.json', [{'source_frame_index':1,'timestamp_s':1,
                  'observations':[{'object':'chair','box_xyxy':[300,200,340,280]}]}])
            write('pose-reference.json', {'rows':[{'source_frame_index':1,
                  'c2w_blender':[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
                  'intrinsics_640x480':[[100,0,320],[0,100,240],[0,0,1]]}]})
            write('assets/scene-spec.json', {'objects':[{'id':'target_bed','p':[0,0,-2]}]})
            write('assets/question-audit.json', {'targets':[{'id':'target_bed','label':'bed'}]})
            result=check_group(p)
            self.assertEqual(result['observations_with_box'],1)
            self.assertEqual(result['any_center_inside'],0)
            self.assertEqual(result['rows'][0]['candidates'],[])


if __name__ == '__main__':
    unittest.main()
