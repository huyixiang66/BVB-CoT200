import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from bvb_v13.question_policy import policy_for, decide_policies


class FixedPolicyTest(unittest.TestCase):
    def test_task_rules_and_room_group(self):
        cases={'object_counting_single':'multiple','object_counting_multiple':'multiple',
               'object_rel_distance_closest':'multiple','object_rel_distance_farthest':'multiple',
               'object_rel_direction_backward_hard':'single','object_abs_distance':'single',
               'object_size_estimation':'single','route_planning':'single',
               'room_size_estimation_multiple':'single','room_size_estimation':'single'}
        for kind, expected in cases.items():
            result=policy_for(dict(id=1,question_type=kind))
            self.assertEqual(result['category'],expected)
            self.assertEqual(result['model_calls'],0)
        self.assertEqual(policy_for(dict(id=1,question_type='room_size_estimation'))['evidence_group'],'whole_scene')

    def test_distinct_roles_same_category_do_not_collapse(self):
        q=dict(id=1,question_type='object_abs_distance',roles={'first':'chair','second':'chair'})
        self.assertEqual(policy_for(q)['category'],'multiple')

    def test_old_model_override_does_not_override_fixed_policy(self):
        q=dict(id=1,question_type='object_counting_single',retention_policy={'category':'single'})
        self.assertEqual(policy_for(q)['category'],'multiple')
        with self.assertRaises(ValueError):policy_for(dict(id=1,question_type='unknown'))

    def test_local_receipt_resume_and_input_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);d={'questions':{'1':{'question':dict(id=1,question_type='object_abs_distance')}}}
            with patch('socket.socket.connect',side_effect=AssertionError('Policy must not use network')):
                a=decide_policies(d,folder);self.assertEqual(a,decide_policies(d,folder))
            self.assertFalse((folder/'request.json').exists())
            self.assertEqual(json.loads((folder/'result.json').read_text())['input_tokens'],0)
            d['questions']['1']['question']['question_type']='object_counting_single'
            with self.assertRaises(ValueError):decide_policies(d,folder)

    def test_audit_entry_skips_policy_api_but_keeps_reviews_and_comparison(self):
        root=Path(__file__).resolve().parents[1]
        sys.path.insert(0,str(root/'scripts'))
        import review_revsi_first10_astra as ar
        from bvb_v13.frontend import audit
        from bvb_v6 import semantic_review as sr
        from bvb_v6 import coverage
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);source=base/'source.json';out=base/'out'
            q=dict(id=1,question_type='object_abs_distance',question='Distance?',objects=['chair'])
            groups=[{'group_id':'a','object':'chair'},{'group_id':'b','object':'chair'}]
            sr.write(source,dict(questions={'1':dict(question=q,arms={'roi_robust':groups})}))
            def prepare(src,review):
                sr.write(review/'manifest.json',dict(jobs=['a','b']))
            def request(folder,*args):
                sr.write(folder/'request.json',dict(status='reviewed'))
                return dict(status='reviewed')
            def export(review):
                d=sr.read(out/'input/selection.json');row=d['questions']['1']
                row['retention_policy']=row['question']['retention_policy']
                row['arms']['semantic_selected']=[groups[0]]
                return d
            def compare_prepare(review):sr.write(review/'coverage/manifest.json',dict(comparisons=['chair']))
            with patch.object(sr,'prepare',side_effect=prepare),patch.object(sr,'request_job',side_effect=request) as reviews,\
                 patch.object(sr,'export',side_effect=export),patch.object(coverage,'prepare',side_effect=compare_prepare),\
                 patch.object(coverage,'submit') as comparison,patch.object(ar,'resolve',return_value=SimpleNamespace(key='unused')),\
                 patch('socket.socket.connect',side_effect=AssertionError('Unexpected model call')):
                audit(source,out)
            self.assertEqual(reviews.call_count,2)
            comparison.assert_called_once()
            receipt=sr.read(out/'question-policy/result.json')
            self.assertEqual(receipt['model_calls'],0)
            self.assertFalse((out/'question-policy/request.json').exists())
            self.assertEqual(sr.read(out/'input/selection.json')['questions']['1']['question']['retention_policy']['decision_source'],'fixed_question_type_v12')


if __name__=='__main__':unittest.main()
