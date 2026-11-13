import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
from bvb_v11 import abs_distance
from bvb_v10.pipeline import save, read, sha
from bvb_v11.abs_distance import validate_choices, validate_answer


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.q = [dict(id=1, objects=['chair', 'table'])]
        self.catalog = [dict(candidate_id='ref', category='washing machine')]
        self.choice = dict(question_id='1', candidate_id='ref', distinct_from_question_objects=True,
                           dimension='width', dimension_description='机身水平宽度', size_m=.6,
                           size_range_m=[.5, .7], reason='横向尺寸稳定', uncertainty='未经标定')

    def test_reject_endpoint_alias_vertical_and_invalid_priors(self):
        validate_choices(dict(choices=[self.choice]), self.q, self.catalog)
        cases = [dict(distinct_from_question_objects=False), dict(dimension='height'),
                 dict(size_m=float('nan')), dict(size_range_m=[.7, .5]), dict(size_m=True)]
        for change in cases:
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_choices(dict(choices=[dict(self.choice, **change)]), self.q, self.catalog)
        with self.assertRaises(ValueError):
            validate_choices(dict(choices=[self.choice]), self.q, [dict(candidate_id='ref', category='chair')])

    def test_complete_unique_question_coverage(self):
        for rows in [[], [self.choice, self.choice], [dict(self.choice, question_id='2')]]:
            with self.assertRaises(ValueError): validate_choices(dict(choices=rows), self.q, self.catalog)

    def test_endpoint_reference_is_bound_to_exact_question_and_role(self):
        q = dict(self.q[0], roles={'first':'chair', 'second':'table'})
        c = dict(candidate_id='endpoint', category='chair', endpoint_question_id='1', endpoint_role='first')
        choice = dict(self.choice, candidate_id='endpoint', reference_role='first', distinct_from_question_objects=False)
        validate_choices(dict(choices=[choice]), [q], [c], allow_endpoints=True)
        for changed in [dict(c, endpoint_question_id='2'), dict(c, endpoint_role='second'), dict(c, category='table')]:
            with self.assertRaises(ValueError):
                validate_choices(dict(choices=[choice]), [q], [changed], allow_endpoints=True)
        with self.assertRaises(ValueError):
            validate_choices(dict(choices=[dict(choice, reference_role='third', distinct_from_question_objects=True)]), [q], [c], allow_endpoints=True)

    def test_endpoint_prepare_keeps_two_geometry_categories(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'source'; out = Path(tmp)/'out'
            public = dict(self.q[0], question_type='object_abs_distance')
            save(source/'scale-reference.json', dict(public_question=public, category='chair', group_id='g1',
                 reference_role='first', choice=dict(self.choice, reference_role='first')))
            def existing_prepare(*args):
                save(out/'input.json', dict(question=public)); save(out/'experiment.json', {})
                (out/'system-prompt.txt').write_text('existing prompt')
            with patch.object(abs_distance.reliability, 'prepare', side_effect=existing_prepare):
                abs_distance.prepare(source, out, Path(tmp)/'poses')
            self.assertEqual(read(out/'input.json')['question']['objects'], ['chair', 'table'])
            self.assertFalse(read(out/'experiment.json')['third_object_reference'])
            self.assertEqual(read(out/'experiment.json')['reference_role'], 'first')

    def test_answer_needs_honest_visibility_and_units(self):
        a = dict(answer=2., unit='m', sufficient_evidence=False, reference_category='washing machine',
                 reference_visible=False, reason='估计', scale_reasoning='尺度不确定', vertical_limitations='俯视图无法确认高度', limitations='未标定')
        validate_answer(a, 'washing machine')
        for change in [dict(sufficient_evidence=True), dict(unit='cm'), dict(answer=float('inf')), dict(reference_category='chair')]:
            with self.assertRaises(ValueError): validate_answer(dict(a, **change), 'washing machine')

    def test_geometry_scope_includes_reference_but_public_question_does_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'source'; out = Path(tmp)/'out'
            public = dict(self.q[0], question_type='object_abs_distance', question='distance?')
            info = dict(public_question=public, category='washing machine', group_id='ref', choice=self.choice)
            save(source/'scale-reference.json', info)
            geometry = dict(public, objects=public['objects']+['washing machine'])
            def existing_prepare(*args):
                save(out/'input.json', dict(question=geometry, frames=[], colors={}))
                save(out/'experiment.json', dict(qa_view_count=3))
                (out/'system-prompt.txt').write_text('existing V10_2 prompt')
                return {'soft_reference': True}
            with patch.object(abs_distance.reliability, 'prepare', side_effect=existing_prepare):
                abs_distance.prepare(source, out, Path(tmp)/'poses')
            self.assertEqual(read(out/'question-public.json'), public)
            self.assertEqual(read(out/'input.json')['question'], geometry)
            self.assertEqual(read(out/'input.json')['public_question'], public)
            self.assertEqual(read(out/'experiment.json')['qa_view_count'], 1)
            self.assertTrue((out/'system-prompt.txt').read_text().startswith('existing V10_2 prompt'))

    def test_qa_sends_only_top_and_two_endpoint_question(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            public = dict(self.q[0], question_type='object_abs_distance', question='distance?')
            save(out/'question-public.json', public)
            save(out/'input.json', dict(colors={'chair': [1,0,0,1], 'table': [0,1,0,1], 'washing machine': [0,0,1,1]}))
            save(out/'scale-reference.json', dict(category='washing machine', choice=self.choice))
            save(out/'experiment.json', dict(scale_reference_sha256=sha(out/'scale-reference.json')))
            save(out/'coverage-validation.json', dict(valid=True))
            (out/'views').mkdir(); (out/'views/top.png').write_bytes(b'top'); (out/'views/side.png').write_bytes(b'side')
            save(out/'views/views.json', dict(views=[dict(id=n, image=n+'.png', sha256=sha(out/'views'/(n+'.png'))) for n in ['top', 'side']]))
            answer = dict(answer=2, unit='m', reference_category='washing machine', reference_visible=True,
                          sufficient_evidence=False, reason='估计', scale_reasoning='约3倍宽度', vertical_limitations='高度不明', limitations='先验不确定')
            captured = []
            def request(folder, content, **kwargs):
                captured.extend(content); return dict(answer=json.dumps(answer))
            direct = types.SimpleNamespace(request=request, image_item=lambda p: dict(type='input_image', path=str(p)))
            runner = types.SimpleNamespace(parse=json.loads, state=lambda *args: None)
            with patch.dict('sys.modules', {'subscription_direct_client':direct,'bvb_v11.common':runner}):
                abs_distance.qa(out)
            images = [c for c in captured if c['type'] == 'input_image']
            self.assertEqual([c['path'] for c in images], [str(out/'views/top.png')])
            self.assertEqual(read(out/'qa/answer.json')['view_ids'], ['top'])
            self.assertIn('PUBLIC QUESTION: '+json.dumps(public, ensure_ascii=False), captured[0]['text'])


if __name__ == '__main__': unittest.main()
