import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from bvb_v10.pipeline import save,read,sha
from bvb_v13.progressive_qa import answer,order_views,InsufficientEvidence,completion_stage
from bvb_v13.model_images import prepare
from bvb_v13.pipeline import prepare_reference_images

class Progressive(unittest.TestCase):
    def fixture(self,root,count=4):
        views=[]
        for i in range(count):
            p=root/f'view-{i}.png';Image.new('RGB',(900,900),(i*20,30,40)).save(p)
            views.append(dict(id='top' if i==0 else f'oblique-{i*90:03d}',path=str(p),sha256=sha(p)))
        return views

    def run_case(self,responses,count=4):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);root=Path(tmp.name);views=self.fixture(root,count);calls=[];save(root/'question-public.json',dict(question='Count three chairs',question_type='object_counting_single',objects=['chair']))
        def request(folder,content,**kwargs):
            calls.append(content)
            return dict(answer=responses[len(calls)-1])
        good=lambda a,ids:self.assertEqual(a['answer'],1)
        with patch('subscription_direct_client.image_item',side_effect=lambda p:dict(type='input_image',path=str(p))):
            try:result=answer(root,'Count three chairs using these THREE rendered views.',good,views=views,task='counting',request_fn=request)
            except InsufficientEvidence as exc:result=exc.result
        return root,views,calls,result

    def result(self,**kwargs):return '1'

    def test_clear_first_view_stops_and_halves_without_changing_original(self):
        root,views,calls,result=self.run_case([self.result()])
        self.assertEqual(len(calls),1);self.assertEqual(len(result['view_ids']),1)
        images=[c for c in calls[0] if c['type']=='input_image'];self.assertEqual(len(images),1)
        self.assertEqual(Image.open(images[0]['path']).size,(450,450))
        self.assertEqual(Image.open(views[0]['path']).size,(900,900))
        self.assertIn('Count three chairs',calls[0][0]['text'])
        self.assertNotIn('THREE rendered views',calls[0][0]['text'])

    def test_uncertain_adds_exactly_one_view_then_stops(self):
        a='NEED_MORE_VIEWS'
        _,_,calls,result=self.run_case([a,self.result()])
        self.assertEqual([sum(c['type']=='input_image' for c in call) for call in calls],[1,2])
        self.assertEqual(result['qa_steps'],2);self.assertFalse(result['evidence_exhausted'])

    def test_all_candidates_used_without_inventing_certainty(self):
        a='NEED_MORE_VIEWS'
        _,_,calls,result=self.run_case([a]*4)
        self.assertEqual(len(calls),4);self.assertTrue(result['evidence_exhausted']);self.assertFalse(result['sufficient_evidence'])
        self.assertEqual(len(set(result['view_ids'])),4)

    def test_short_more_response_does_not_validate_or_save_provisional_answer(self):
        root,_,calls,result=self.run_case(['NEED_MORE_VIEWS',self.result()])
        self.assertEqual(result['qa_steps'],2)
        self.assertEqual(read(root/'qa/step-01/answer.json'),dict(need_more_views=True,view_ids=['top']))
        self.assertTrue(read(root/'qa/progressive-summary.json')['steps'][0]['short_response'])
        self.assertIn('output exactly NEED_MORE_VIEWS',calls[0][0]['text'])

    def test_exhausted_short_responses_are_terminal_without_fabricated_answer(self):
        root,views,calls,result=self.run_case(['NEED_MORE_VIEWS']*3,count=3)
        self.assertIsNone(result['answer']);self.assertEqual(result['status'],'insufficient_evidence')
        self.assertEqual(completion_stage(root),'insufficient-evidence')
        self.assertNotIn('option',read(root/'qa/answer.json'))
        with self.assertRaises(InsufficientEvidence):
            answer(root,'Count',lambda *a:self.fail('should not validate'),views=views,task='counting',request_fn=lambda *a,**k:self.fail('must not re-request on resume'))

    def test_json_or_prose_is_not_silently_accepted(self):
        for raw in ['{"answer":1}', '1 because I see one', 'NEED_MORE_VIEWS because unclear']:
            with self.subTest(raw=raw),self.assertRaises(ValueError):self.run_case([raw])

    def test_stale_plan_and_changed_image_are_rejected(self):
        root,views,_,_=self.run_case([self.result()])
        with self.assertRaisesRegex(ValueError,'inputs changed'):
            answer(root,'changed prompt',lambda a,ids:None,views=views,task='counting')
        Image.new('RGB',(800,800)).save(views[0]['path'])
        with self.assertRaisesRegex(ValueError,'candidate image changed'):
            answer(root,'prompt',lambda a,ids:None,views=views,task='counting')

    def test_saved_answer_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=dict(answer=17,legacy=True);save(root/'qa/answer.json',old)
            self.assertEqual(answer(root,'new',None),old)

    def test_route_starts_from_top_even_if_other_view_preferred(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);views=self.fixture(root)
            save(root/'view-selection/selection.json',dict(view_ids=['oblique-090','top','oblique-180']))
            self.assertEqual(order_views(root,views,'route')[0]['id'],'top')

    def test_reference_images_halve_previous_transport_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'reference-top.png';Image.new('RGB',(1600,1000)).save(p);before=sha(p)
            records=prepare_reference_images(root,dict(reference_images={p.name:before}),'attempt-01')
            self.assertEqual(records[0]['previous_sent_size'],[1024,640])
            self.assertEqual(records[0]['sent_size'],[512,320]);self.assertEqual(sha(p),before)

    def test_frozen_reference_request_keeps_original_transport(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'reference-top.png';Image.new('RGB',(1600,1000)).save(p)
            exp=dict(reference_images={p.name:sha(p)})
            first=prepare_reference_images(root,exp,'attempt-01')
            save(root/'attempt-01/request.json',dict(status='completed'))
            second=prepare_reference_images(root,exp,'attempt-01',max_side=256)
            self.assertEqual(first,second)
            self.assertEqual(Image.open(second[0]['path']).size,(512,320))

if __name__=='__main__':unittest.main()
