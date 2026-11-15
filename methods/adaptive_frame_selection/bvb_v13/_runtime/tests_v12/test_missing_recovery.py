import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from bvb_v10.pipeline import save, read, sha
from bvb_v13 import missing_recovery as mr
from bvb_v13.candidate_recovery import canonical_frames
from bvb_v13.semantic_recovery import ensure_question_selection
from bvb_v13.video_fallback import locate


def group(name, verdict='accept'):
    return dict(object=name, group_id=name+'-'+verdict, selected=[],
                semantic_review=dict(status=verdict))


class MissingRecovery(unittest.TestCase):
    def setup_source(self, root, groups, objects=('chair', 'table')):
        q=dict(id=1, scene_name='scene', question_type='object_counting', objects=list(objects))
        data=dict(questions={'1':dict(question=q,arms=dict(semantic_selected=groups))})
        save(root/'run-config.json',{})
        save(root/'source.json',data)
        return root/'source.json',root/'result.json'

    def test_rejected_extra_candidate_does_not_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,dest=self.setup_source(root,[group('chair'),group('chair','reject')],['chair'])
            with patch('bvb_v13.candidate_recovery.canonical_frames',side_effect=AssertionError('No scan needed')):
                mr.recover_selection(root,'scene',source,dest)
            row=read(dest)['questions']['1']
            self.assertEqual(row['arms']['semantic_selected'],[group('chair')])
            self.assertFalse(row['semantic_recovery']['search_performed'])

    def test_missing_only_empty_locator_is_not_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,dest=self.setup_source(root,[group('chair'),group('table','uncertain')])
            save(root/'packets/scene/cache/candidates.json',[])
            with patch('bvb_v13.candidate_recovery.canonical_frames',return_value=([],dict(frames=[]))), \
                 patch('bvb_v13.video_fallback.locate',return_value=dict(categories=[dict(category='table',instances=[])])) as search, \
                 patch('bvb_v13.missing_recovery.video_path',return_value=root/'video'):
                mr.recover_selection(root,'scene',source,dest)
                self.assertEqual(search.call_args.args[2],['table'])
                self.assertEqual(search.call_args.kwargs['fps'],.5)
                self.assertTrue(search.call_args.kwargs['missing_only'])
            row=read(dest)['questions']['1']
            self.assertEqual(row['coverage_missing_categories'],['table'])
            self.assertEqual(row['arms']['semantic_selected'],[group('chair')])
            self.assertNotIn('answer',row)

    def test_unreviewed_proposals_cannot_be_selected(self):
        for verdict in ['uncertain','reject','pending']:
            with self.assertRaisesRegex(ValueError,'Unreviewed'):
                mr.choose_for_question({},'chair',[group('chair',verdict)],Path('/unused'),lambda _:None)

    def test_public_preflight_uses_scene_and_half_fps_handler(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);src,dst=self.setup_source(root,[group('chair')],['chair'])
            q=read(src)['questions']['1']['question']
            save(root/'questions.public.json',[q])
            save(root/'scenes/scene/input/review/reviewed-selection.json',read(src))
            with patch.object(mr,'recover_selection',return_value=dst) as recover:
                self.assertEqual(ensure_question_selection(root,'1'),dst)
                self.assertEqual(recover.call_args.args[1],'scene')

    def test_half_fps_has_91_exact_frames_for_181_6_seconds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cache=root/'packets/scene/cache';rows=[]
            for n in range(364):
                path=cache/f'frames/{n}.jpg';path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(str(n).encode())
                rows.append(dict(timestamp_s=n/2,source_frame_index=n*15,frame=f'frames/{n}.jpg',frame_sha256=sha(path)))
            save(cache/'candidates.json',rows)
            poses=root/'poses';save(poses/'sampling.json',dict(frames=[dict(source_frame_index=r['source_frame_index'],sha256=r['frame_sha256']) for r in rows]))
            video=root/'video.mp4';video.write_bytes(b'test')
            save(root/'run-config.json',dict(poses={'scene':str(poses)},videos={'scene':str(video)}))
            with patch('subprocess.check_output',return_value=json.dumps(dict(format=dict(duration='181.6')))):
                _,manifest=canonical_frames(root,'scene',fps=.5)
            self.assertEqual(manifest['fps'],.5)
            self.assertEqual([f['timestamp_s'] for f in manifest['frames']],list(range(0,182,2)))

    def test_one_fps_cache_is_not_reused_as_half_fps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);video=root/'video';video.write_bytes(b'test')
            save(root/'frames.json',dict(video_sha256=sha(video),fps=1,frames=[]))
            with patch('shutil.which',return_value='/bin/tool'),patch('subprocess.check_output',return_value='{"format":{"duration":"2"}}'):
                with self.assertRaisesRegex(ValueError,'fps differs'):
                    locate(video,{},['chair'],root,enabled=True,fps=.5,missing_only=True)

if __name__=='__main__':unittest.main()
