import tempfile
import unittest
from pathlib import Path
import json
import numpy as np

from .selection import score_pair, cluster_rows, PRESET
from .reconstruction import shared_pose, pick


class RegionTests(unittest.TestCase):
    def setUp(self):
        self.params=json.loads(PRESET.read_text())['parameters']

    def test_preset_and_gate(self):
        self.assertEqual(self.params,dict(appearance_weight=.75,geometry_gate=0.,minimum_pairs=1,threshold=.468))
        self.assertAlmostEqual(score_pair(.4,.8,self.params),.76)
        self.assertAlmostEqual(score_pair(0,1,self.params),.75)
        self.assertEqual(score_pair(.2,1,{**self.params,'geometry_gate':.35}),.2)

    def test_complete_link_blocks_chain(self):
        rows={i:dict(source_frame_index=i,observations=[dict(object='door',peak_id=f'p{i}',box_xyxy=[0,0,1,1],dino_score=.8,status='anchor')]) for i in range(3)}
        pairs=[dict(frames=[a,b],geometry=g,semantic=0) for a,b,g in [(0,1,.9),(1,2,.8),(0,2,.2)]]
        groups,evidence=cluster_rows(rows,pairs,self.params)
        self.assertEqual(sorted(len(g['peak_ids']) for g in groups),[1,2])
        self.assertEqual(evidence['p0']['p2'],.2)
        self.assertEqual({r['source_frame_index'] for g in groups for r in g['selected']},set(rows))

    def test_empty_and_single_peak(self):
        self.assertEqual(cluster_rows({},[],self.params),([],{}))
        rows={0:dict(source_frame_index=0,observations=[dict(object='door',peak_id='p',box_xyxy=None,dino_score=None,status='context')])}
        self.assertEqual(len(cluster_rows(rows,[],self.params)[0]),1)

    def test_pose_pick_covers_categories_then_baseline(self):
        poses=np.repeat(np.eye(4)[None],5,axis=0);poses[:,0,3]=[0,.01,.02,2,3]
        rows=[dict(source_frame_index=i,observations=[dict(object='door' if i<4 else 'chair',box_xyxy=[0,0,1,1])]) for i in range(5)]
        candidates={i:dict(index=i,sharpness=10-i) for i in range(5)}
        chosen=pick(rows,candidates,poses,3)
        self.assertEqual([r['source_frame_index'] for r in chosen],[0,3,4])

    def test_shared_world_preserves_camera_distances(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);m=np.repeat(np.eye(4)[None],3,axis=0);m[:,0,3]=[1,2,3]
            k=np.repeat(np.array([[500,0,320],[0,500,240],[0,0,1]])[None],3,axis=0)
            np.savez(p/'camera-poses.npz',c2w=m,intrinsics=k)
            (p/'sampling.json').write_text(json.dumps({'frames':[]}))
            (p/'geometry-info.json').write_text(json.dumps({'image_size_wh':[640,480]}))
            out, intrinsics, source=shared_pose(p,None)
            np.testing.assert_allclose(out[0,:3,3],0)
            np.testing.assert_allclose(np.linalg.norm(np.diff(out[:,:3,3],axis=0),axis=1),[1,1])
            np.testing.assert_allclose(np.linalg.det(out[:,:3,:3]),1)
            np.testing.assert_allclose(intrinsics,k)
            self.assertIn('native scale',source['coordinate_policy'])


if __name__=='__main__':unittest.main()
