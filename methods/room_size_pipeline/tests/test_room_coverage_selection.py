import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from room_size_pipeline.coverage_selection import choose_coverage_frames
from room_size_pipeline.measurement import measurement_exclusions


class CoverageTests(unittest.TestCase):
    def fixture(self,root):
        samples=[dict(filename=f'{i}.jpg',sha256='hash',source_frame_index=i,timestamp_s=i) for i in range(8)]
        cameras=np.tile(np.eye(4),(8,1,1));cameras[4:,0,3]=10
        detections=root/'detections.jsonl'
        detections.write_text(json.dumps({'source_frame_index':6,'detections':[{'object':'bed','score':.9,'box_xyxy':[100,100,400,350]}]}))
        config={'frames_root':str(root),'poses':str(root),'detections':str(detections)}
        visible=np.zeros((8,40),dtype=bool);visible[:4,:20]=True;visible[4:,20:]=True
        allowed=np.ones(8,dtype=bool);allowed[0]=False
        return config,samples,cameras,(visible,np.ones(40),allowed,np.zeros(8),.1)

    def test_separate_positions_and_reference_survive_without_budget_cap(self):
        with tempfile.TemporaryDirectory() as t:
            config,samples,cameras,geometry=self.fixture(Path(t))
            with patch('room_size_pipeline.coverage_selection.geometry',return_value=geometry),patch('room_size_pipeline.coverage_selection.sha256',return_value='hash'),patch('room_size_pipeline.coverage_selection.cv2.imread',return_value=np.ones((16,16,3),dtype=np.uint8)):
                times,report=choose_coverage_frames(config,samples,cameras)
            self.assertNotIn(0,times)
            self.assertIn(6,times)
            self.assertTrue(any(t<4 for t in times));self.assertTrue(any(t>=4 for t in times))
            self.assertEqual(report['weighted_coverage'],1)
            self.assertTrue(report['groups_are_not_room_labels'])

    def test_missing_detector_does_not_silently_switch_method(self):
        with self.assertRaisesRegex(ValueError,'cached detections'):
            choose_coverage_frames({},[],[])


class MeasurementTests(unittest.TestCase):
    def test_rim_height_excludes_tap_only_when_scope_explicit(self):
        obj={'name':'tap pillar'}
        self.assertEqual(measurement_exclusions({'label':'washbasin','source':'floor-to-rim height'},obj),[2])
        self.assertEqual(measurement_exclusions({'label':'washbasin','source':'overall height'},obj),[])
        self.assertEqual(measurement_exclusions({'label':'washbasin','source':'floor-to-rim height'},{'name':'basin body'}),[])

    def test_leaf_thickness_excludes_accessories_not_body(self):
        ref={'label':'door','source':'leaf width, thickness, and height'}
        self.assertEqual(measurement_exclusions(ref,{'name':'entrance pull'}),[1])
        self.assertEqual(measurement_exclusions(ref,{'name':'entrance door leaf'}),[])
        self.assertEqual(measurement_exclusions(ref,{'name':'custom','component_role':'door_leaf'}),[])
        self.assertEqual(measurement_exclusions({'label':'door','source':'overall assembly'}, {'name':'entrance pull'}),[])

    def test_invalid_exclusion_rejected(self):
        with self.assertRaises(ValueError):measurement_exclusions({}, {'measurement_exclude_axes':[3]})


if __name__=='__main__':unittest.main()
