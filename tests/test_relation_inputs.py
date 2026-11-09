import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/direct_views'))
from relation_inputs import prepare_input, role_palette


class RelationInputTests(unittest.TestCase):
    def setUp(self):
        self.q = dict(qa_id='test', question_type='object_rel_direction_hard',
                      question='Where is the table?', options=['A. left', 'B. right'],
                      roles=dict(standing=['bed'], facing=['chair'], query=['table']),
                      ground_truth='DO_NOT_LEAK', baseline_answer='DO_NOT_LEAK')
        self.view = lambda role: dict(role=role, file=role+'.png', sha256='hash', geometry_accepted=True)

    def test_role_order_and_no_gt(self):
        row=dict(qa_ready=True, views=[self.view('direction_layout'), self.view('standing_reference')])
        result=prepare_input(self.q,row)
        self.assertEqual([x['role'] for x in result['images']], ['standing_reference','direction_layout'])
        self.assertNotIn('DO_NOT_LEAK',str(result))
        self.assertEqual(len(role_palette(self.q)),3)

    def test_missing_standing_requires_explicit_fallback(self):
        row=dict(qa_ready=False, views=[self.view('direction_layout')])
        with self.assertRaises(ValueError):prepare_input(self.q,row)
        result=prepare_input(self.q,row,allow_overview_only=True)
        self.assertFalse(result['geometry_qa_ready'])
        self.assertIn('No first-person',result['prompt'])
        self.assertNotIn('Image 2',result['prompt'])

    def test_diagnostic_rejected_even_in_fallback(self):
        v=self.view('direction_layout');v['geometry_accepted']=False
        with self.assertRaises(ValueError):prepare_input(self.q,dict(qa_ready=False,views=[v]),allow_overview_only=True)

    def test_distance_preserves_question_units(self):
        q=dict(qa_id='d',question_type='object_abs_distance',question='Distance in centimeters?',roles=dict(endpoints=['bed','chair']))
        result=prepare_input(q,dict(qa_ready=True,views=[self.view('distance')]))
        self.assertIn('centimeters',result['prompt'])
        self.assertEqual(len(role_palette(q)),2)


if __name__=='__main__':unittest.main()
