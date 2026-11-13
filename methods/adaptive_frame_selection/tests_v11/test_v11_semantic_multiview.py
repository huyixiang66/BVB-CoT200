import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
from bvb_v10.pipeline import read,save,sha
from bvb_v11 import semantic_recovery as recovery,abs_distance_multiview as multi
from bvb_v11.abs_distance import validate_choices


def group(category,status,gid):
    return dict(object=category,group_id=gid,selected=[],semantic_review=dict(status=status))


class SemanticTests(unittest.TestCase):
    def test_standalone_prepare_checks_excluded_audits_from_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'prepared';review=root/'review';cache=root/'cache'
            q=dict(id=1,scene_name='scene',objects=['chair'])
            save(source/'manifest.json',dict(question=q,groups=[dict(group_id='good',folder='good',objects=['chair'])]))
            save(source/'good/semantic-review.json',dict(status='accept',review_file=str(review/'jobs/good/review.json')))
            save(review/'reviewed-selection.json',dict(questions={'1':dict(question=q,arms=dict(semantic_selected=[group('chair','accept','good')]),semantic_excluded=[group('chair','reject','bad')])}))
            save(cache/'candidates.json',[])
            save(source/'good/input-frames.json',[dict(path=str(cache/'frames/a.jpg'),source_frame_index=1,timestamp_s=0,sha256='hash',observations=[])])
            def recovered(base,sid,selection,destination):
                row=read(selection)['questions']['1']
                self.assertEqual(recovery.triggered_categories(row),['chair'])
                raise RuntimeError('recovery invoked')
            with patch.object(recovery,'recover_selection',side_effect=recovered), self.assertRaisesRegex(RuntimeError,'recovery invoked'):
                recovery.ensure_prepared(source,root/'poses')

    def test_rejected_excluded_candidate_triggers_even_with_positive_selected(self):
        for task in ['object_counting','object_abs_distance','object_rel_distance','object_size','room_size','route']:
            row=dict(question=dict(objects=['chair','table'],question_type=task),
                arms=dict(semantic_selected=[group('chair','accept','old'),group('table','accept','table')]),
                semantic_excluded=[group('chair','reject','bad')])
            self.assertEqual(recovery.triggered_categories(row),['chair'])
            data=dict(questions={'1':row})
            result=recovery.apply(data,{'chair':[group('chair','accept','new')]})['questions']['1']
            self.assertEqual([g['group_id'] for g in result['arms']['semantic_selected']],['table','new'])
            self.assertTrue(result['coverage_complete'])
            self.assertEqual(data['questions']['1']['arms']['semantic_selected'][0]['group_id'],'old')

    def test_uncertain_never_retained_and_empty_search_stays_missing(self):
        row=dict(question=dict(objects=['chair']),arms=dict(semantic_selected=[group('chair','uncertain','old')]))
        result=recovery.apply(dict(questions={'1':row}),{'chair':[]})['questions']['1']
        self.assertEqual(result['coverage_missing_categories'],['chair'])
        self.assertEqual(result['arms']['semantic_selected'],[])
        with self.assertRaises(ValueError):recovery.apply(dict(questions={'1':row}),{'chair':[group('chair','uncertain','new')]})

    def test_stable_vertical_dimension_allowed_but_variable_rejected(self):
        q=[dict(id=1,objects=['chair','table'])];catalog=[dict(candidate_id='r',category='bottle')]
        c=dict(question_id='1',candidate_id='r',distinct_from_question_objects=True,dimension='height',dimension_stability='stable',
            stability_reason='identifiable fixed subtype',size_m=.3,size_range_m=[.28,.32],reason='stable',dimension_description='base to cap',uncertainty='not measured')
        validate_choices(dict(choices=[c]),q,catalog,stable_only=True)
        for bad in ['variable','unavailable',None]:
            with self.assertRaises(ValueError):validate_choices(dict(choices=[dict(c,dimension_stability=bad)]),q,catalog,stable_only=True)

    def test_three_chosen_images_jointly_sent_without_fixed_top(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);cats=['chair','table','bottle'];public=dict(id=1,objects=cats[:2],question='distance?')
            save(out/'question-public.json',public);save(out/'input.json',dict(question=dict(objects=cats),colors={c:[1,0,0,1] for c in cats}))
            prior=dict(dimension='height',dimension_description='height',size_m=.3,size_range_m=[.28,.32],stability_reason='stable subtype',uncertainty='prior')
            save(out/'scale-reference.json',dict(category='bottle',reference_role='third',choice=prior))
            save(out/'experiment.json',dict(scale_reference_sha256=sha(out/'scale-reference.json')));save(out/'coverage-validation.json',dict(valid=True))
            views=[]
            for vid in ['top','oblique-0','oblique-90','elevated-180']:
                p=out/'views'/(vid+'.png');p.parent.mkdir(exist_ok=True);p.write_bytes(vid.encode());views.append(dict(id=vid,image=p.name,sha256=sha(p)))
            save(out/'views/views.json',dict(views=views,overhead_occlusion_removed=True))
            selected=['oblique-0','oblique-90','elevated-180']
            selection=dict(view_ids=selected,coverage=[dict(category=c,visible_in=selected) for c in cats])
            answer=dict(answer=1,unit='m',reference_category='bottle',reference_visible=True,reference_visible_in=[selected[0]],
                sufficient_evidence=False,reason='estimate',scale_reasoning='prior',vertical_limitations='height uncertain',limitations='prior',cross_view_consistency='joint')
            calls=[]
            def request(folder,content,**kwargs):
                calls.append(content);return dict(answer=json.dumps(selection if len(calls)==1 else answer))
            direct=types.SimpleNamespace(request=request,image_item=lambda p:dict(type='input_image',path=str(p)))
            runner=types.SimpleNamespace(parse=json.loads,state=lambda *a:None)
            with patch.dict('sys.modules',{'subscription_direct_client':direct,'bvb_v11.common':runner}):multi.qa(out)
            self.assertEqual(len(calls),2)
            self.assertEqual([Path(c['path']).stem for c in calls[1] if c['type']=='input_image'],selected)
            self.assertEqual(read(out/'qa/answer.json')['view_ids'],selected)
            self.assertIn('PUBLIC QUESTION: '+json.dumps(public,ensure_ascii=False),calls[1][0]['text'])


if __name__=='__main__':unittest.main()
