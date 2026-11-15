import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from bvb_v10.pipeline import read,save,sha
from bvb_v13.size_originals import candidates
from bvb_v13.progressive_qa import answer

class SizeOriginals(unittest.TestCase):
    def fixture(self,out):
        q=dict(question='What is the longest dimension of the suitcase in centimeters?',question_type='object_size_estimation',objects=['suitcase'])
        frames=[]
        for i,t in enumerate([0,.5,2,4]):
            p=out/f'frame-{i}.png';Image.new('RGB',(640,480),(i*30,20,30)).save(p)
            frames.append(dict(path=str(p),sha256=sha(p),timestamp_s=t,groups=['g'],observations=[dict(object='suitcase',box_xyxy=[0,0,50*(i+1),100],dino_score=.9)]))
        packet=dict(question=q,colors={'suitcase':[1,0,0,1]},groups=[dict(group_id='g')],frames=frames)
        save(out/'input.json',packet);save(out/'question-public.json',q)
        return packet,q
    def test_selection_is_bounded_temporal_filtered_and_source_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);p,q=self.fixture(out);before=sha(out/'input.json');rows=candidates(out)
            self.assertEqual(len(rows),3);self.assertEqual(rows[0]['source_packet_index'],3)
            self.assertNotIn(1,[r['source_packet_index'] for r in rows]);self.assertEqual(sha(out/'input.json'),before)
            for r in rows:self.assertEqual(r['scale'],1);self.assertEqual(r['sent_size'],[640,480]);self.assertEqual(sha(r['path']),r['sha256'])
    def test_first_request_includes_original_and_render_then_adds_one_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);p,q=self.fixture(out);originals=candidates(out);image=out/'render.png';Image.new('RGB',(900,900)).save(image)
            calls=[]
            def request(folder,content,**kwargs):
                calls.append(content);return dict(answer='NEED_MORE_VIEWS' if len(calls)==1 else '65')
            with patch('subscription_direct_client.image_item',side_effect=lambda p:dict(type='input_image',path=str(p))):
                result=answer(out,'legacy render only prompt',None,views=[dict(id='top',path=str(image),sha256=sha(image))],task='object_size',question=q,original_images=originals,request_fn=request)
            self.assertEqual([sum(c['type']=='input_image' for c in r) for r in calls],[2,3])
            manifest=read(out/'qa/step-01/image-inputs.json')['images']
            self.assertEqual(manifest[0]['sent_size'],[450,450]);self.assertEqual(manifest[1]['sent_size'],[640,480])
            self.assertIn('category colors apply ONLY to renders',calls[0][0]['text'])
            self.assertEqual(result['answer'],65);self.assertEqual(len(result['original_frame_ids']),2)
            self.assertEqual(len(read(out/'qa/progressive-plan.json')['original_candidates']),3)
    def test_changed_original_and_other_task_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);p,q=self.fixture(out);rows=candidates(out);image=out/'render.png';Image.new('RGB',(900,900)).save(image);views=[dict(id='top',path=str(image),sha256=sha(image))]
            with self.assertRaisesRegex(ValueError,'only enabled for object size'):
                answer(out,'prompt',None,views=views,task='counting',original_images=rows)
            Path(rows[0]['path']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'image changed'):candidates(out)
    def test_original_selection_rejects_stale_reconstruction_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);self.fixture(out)
            save(out/'reconstruction-frame-selection.json',dict(source_input_sha256=sha(out/'input.json'),kept_source_packet_indices=[0,1,2,3]))
            with self.assertRaisesRegex(ValueError,'differs from reconstruction'):candidates(out)

if __name__=='__main__':unittest.main()
