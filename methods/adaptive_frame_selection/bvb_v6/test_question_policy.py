import copy
import tempfile
import unittest
from pathlib import Path

from .coverage import admitted, apply_coverage, comparison_key
from .question_policy import question_policy
from .semantic_review import write
from .test_coverage import group, selection


def enable(root):
    write(Path(root) / 'coverage-policy.json', {'policy': 'question_aware_category_coverage'})


class QuestionPolicyTests(unittest.TestCase):
    def test_representative_questions(self):
        examples = [
            ({'question_type': 'object_rel_direction_medium'}, 'single', []),
            ({'question_type': 'object_abs_distance'}, 'single', ['metric_scale']),
            ({'question_type': 'object_rel_distance'}, 'multiple', []),
            ({'question_type': 'object_counting'}, 'multiple', []),
            ({'question_type': 'obj_appearance_order'}, 'supplemental', ['appearance_time']),
            ({'task': '3D Video Grounding'}, 'single', ['camera_pose_at_query', 'camera_coordinates', 'metric_scale']),
            ({'task': 'Dimensional Measurement', 'question': 'Distance between camera and table?'},
             'single', ['metric_scale', 'camera_pose_at_query']),
            ({'task': 'Speed & Acceleration'}, 'supplemental', ['trajectory', 'timestamps']),
            ({'type': '(Cross-Video) Memoery Update'}, 'supplemental', ['cross_video_alignment', 'multiple_time_states']),
            ({'type': 'Prediction'}, 'supplemental', ['motion_history', 'timestamps', 'prediction_assumptions']),
            ({'type': '(Spatial Construction) Instance/Scene Attribute', 'ori_question': 'How many chairs?'},
             'multiple', []),
            ({'question_type': 'unseen_type'}, 'unknown', []),
        ]
        for question, category, evidence in examples:
            with self.subTest(question=question):
                policy = question_policy(question)
                self.assertEqual(policy['category'], category)
                self.assertEqual(policy['additional_evidence'], evidence)

    def test_repeated_roles_and_explicit_override(self):
        q = {'question_type': 'object_abs_distance', 'roles': {'first': 'chair', 'second': 'chair'}}
        self.assertEqual(question_policy(q)['category'], 'unknown')
        q['retention_policy'] = {'category': 'multiple', 'reason': 'two chairs'}
        self.assertEqual(question_policy(q)['category'], 'multiple')
        with self.assertRaises(ValueError):
            question_policy({'retention_policy': {'category': 'missing'}})

    def test_single_waits_for_question_specific_comparison(self):
        groups = [group('a', 'accept'), group('b', 'accept'), group('c', 'uncertain')]
        source = selection(groups)
        source['questions']['q']['question']['question_type'] = 'object_rel_direction_medium'
        with tempfile.TemporaryDirectory() as root:
            enable(root)
            q = apply_coverage(copy.deepcopy(source), root)['questions']['q']
            self.assertFalse(q['coverage_complete'])
            self.assertEqual(q['coverage_blockers']['fridge'], 'single_instance_comparison_pending')
            self.assertEqual(q['arms']['semantic_selected'], [])
            cid = comparison_key('fridge', ['a', 'b'], 'q')
            result = {'selected_id': 'b', 'confidence': 'low', 'reason': 'relative best',
                      'uncertainty': 'not verified', 'ranking': [
                          {'candidate_id': 'b', 'rank': 1, 'reason': 'first'},
                          {'candidate_id': 'a', 'rank': 2, 'reason': 'second'}]}
            path = Path(root) / 'coverage' / cid / 'result.json'
            write(path, {**result, 'status': 'compared', 'request_hash': 'same'})
            write(path.parent / 'input.json', {'request_hash': 'same'})
            q = apply_coverage(copy.deepcopy(source), root)['questions']['q']
            chosen = q['arms']['semantic_selected']
            self.assertTrue(q['coverage_complete'])
            self.assertEqual([g['group_id'] for g in chosen], ['b'])
            self.assertTrue(admitted(chosen[0]['semantic_review']))
            self.assertEqual(chosen[0]['semantic_review']['status'], 'accept')
            self.assertEqual(len(q['coverage_excluded']), 2)

    def test_multiple_keeps_confirmed_and_uncertain_without_forcing_one(self):
        source = selection([group('a', 'accept'), group('b', 'uncertain'), group('c', 'reject')])
        source['questions']['q']['question']['question_type'] = 'object_counting'
        with tempfile.TemporaryDirectory() as root:
            enable(root)
            q = apply_coverage(source, root)['questions']['q']
            self.assertEqual([g['group_id'] for g in q['arms']['semantic_selected']], ['a', 'b'])
            self.assertEqual([g['group_id'] for g in q['coverage_excluded']], ['c'])
            self.assertTrue(all(admitted(g['semantic_review']) for g in q['arms']['semantic_selected']))
            self.assertTrue(q['coverage_complete'])

    def test_single_never_restores_a_rejected_only_candidate(self):
        source = selection([group('a', 'reject')])
        source['questions']['q']['question']['question_type'] = 'object_abs_distance'
        with tempfile.TemporaryDirectory() as root:
            enable(root)
            q = apply_coverage(source, root)['questions']['q']
            self.assertFalse(q['coverage_complete'])
            self.assertEqual(q['arms']['semantic_selected'], [])
            self.assertEqual(q['coverage_blockers']['fridge'], 'all_candidates_rejected_or_unavailable')

    def test_pending_review_blocks_and_legacy_policy_is_unchanged(self):
        source = selection([group('a', 'accept'), group('b', 'pending')])
        source['questions']['q']['question']['question_type'] = 'object_counting'
        with tempfile.TemporaryDirectory() as root:
            legacy = apply_coverage(copy.deepcopy(source), root)['questions']['q']
            self.assertEqual([g['group_id'] for g in legacy['arms']['semantic_selected']], ['a'])
            enable(root)
            q = apply_coverage(copy.deepcopy(source), root)['questions']['q']
            self.assertEqual(q['coverage_blockers'], {'fridge': 'semantic_review_pending'})
            self.assertFalse(q['coverage_complete'])


if __name__ == '__main__':
    unittest.main()
