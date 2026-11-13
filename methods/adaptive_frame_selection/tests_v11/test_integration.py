import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from bvb_v10.pipeline import read,save
from bvb_v11.driver import kind,prepare
from bvb_v11.common import video_path
from bvb_v11 import room

class Integration(unittest.TestCase):
    def test_public_template_dispatch(self):
        cases={'object_rel_direction_backward_easy':'rel_dir','object_rel_direction_medium':'rel_dir',
            'object_rel_distance_farthest':'rel_dist','object_rel_distance_closest':'rel_dist',
            'object_abs_distance':'abs_dist','object_counting_multiple':'counting',
            'object_counting_single':'counting','object_size_estimation':'object_size',
            'route_planning':'route','room_size_estimation_multiple':'room_size','room_size_estimation':'room_size'}
        for source,expected in cases.items():self.assertEqual(kind(dict(question_type=source)),expected)
        with self.assertRaises(ValueError):kind(dict(question_type='unknown'))

    def test_abs_requires_context_before_any_prepare(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);save(root/'source/manifest.json',dict(question=dict(question_type='object_abs_distance')))
            with self.assertRaisesRegex(ValueError,'context plan'):prepare(root/'source',root/'out',root/'poses')
            self.assertFalse((root/'out').exists())

    def test_explicit_video_path_no_host_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);video=root/'scene.mp4';video.write_bytes(b'fixture')
            save(root/'run-config.json',dict(videos={'scene':str(video)}))
            self.assertEqual(video_path(root,'scene'),video.resolve())
            with self.assertRaisesRegex(ValueError,'videos'):video_path(root,'other')

    def test_direction_renderer_receives_occlusion_pass_only_for_v11(self):
        from bvb_v10 import pipeline
        for version in ['bvb_v10','bvb_v11']:
            with tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp);save(out/'experiment.json',dict(version=version))
                with patch.object(pipeline.subprocess,'run') as process:
                    process.return_value.returncode=0;pipeline.execute(out,'/unused/blender',True)
                script=(out/'render.py').read_text();compile(script,'render.py','exec')
                self.assertEqual('occlusion_reports.append' in script,version=='bvb_v11')
                self.assertNotIn('elevated-',script)

    def test_room_diagnostic_view_never_sent_to_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);blend=root/'a.blend';video=root/'a.mp4'
            blend.write_bytes(b'blend');video.write_bytes(b'video')
            question=root/'q.json';save(question,dict(qa_id='1',question_type='room_size_estimation',question='Area?'))
            out=root/'out';save(out/'geometry-selector/render_manifest.json',dict(questions=[dict(qa_id='1',qa_ready=False,views=[])]))
            with self.assertRaisesRegex(RuntimeError,'No accepted room view'):room.run(blend,video,question,out,'/unused/blender')
            self.assertFalse((out/'qa/request.json').exists())
            self.assertEqual(blend.read_bytes(),b'blend')

if __name__=='__main__':unittest.main()
