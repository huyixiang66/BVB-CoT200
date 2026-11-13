from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from bvb_v10.pipeline import read
from bvb_v11.retrieval_policy import admitted_only
from bvb_v11.video_fallback import locate
from bvb_v11.candidate_recovery import recover_question,search_category

class DisabledRetrieval(unittest.TestCase):
    def test_positive_evidence_retained_uncertainty_not_promoted(self):
        groups=[dict(object=c,group_id=c+v,selected=[],semantic_review=dict(status=v)) for c,v in [('chair','accept'),('chair','reject'),('table','uncertain')]]
        source=dict(questions={'1':dict(question=dict(objects=['chair','table']),arms=dict(semantic_selected=groups))})
        row=admitted_only(source)['questions']['1']
        self.assertEqual([g['object'] for g in row['arms']['semantic_selected']],['chair'])
        self.assertEqual(row['coverage_missing_categories'],['table'])
        self.assertFalse(row['semantic_recovery']['search_performed'])
        self.assertFalse(row['semantic_recovery']['completed'])
        self.assertEqual(len(source['questions']['1']['arms']['semantic_selected']),3)

    def test_old_enabled_flag_cannot_launch_video_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'locator'
            with patch('subprocess.check_output',side_effect=AssertionError('Must not probe or send video')):
                with self.assertRaisesRegex(RuntimeError,'disabled'):locate(Path('/missing/video'),{},['chair'],out,enabled=True)
            self.assertEqual(read(out/'status.json')['stage'],'disabled')
            self.assertFalse((out/'frames').exists())
            self.assertFalse((out/'locate-request').exists())

    def test_missing_counting_evidence_does_not_become_zero(self):
        with self.assertRaisesRegex(RuntimeError,'disabled'):recover_question(Path('/missing'),'1',['chair'])
        with self.assertRaisesRegex(RuntimeError,'zero count'):search_category(Path('/missing'),'scene','chair')

if __name__=='__main__':unittest.main()
