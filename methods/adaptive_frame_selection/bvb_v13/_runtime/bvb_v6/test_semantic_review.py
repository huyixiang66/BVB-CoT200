import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from .semantic_review import aggregate,validate_reply,representatives,reconstruction_gate,sha,write,export,digest,PROMPT,SCHEMA,MODEL,clean_images
from PIL import Image, ImageOps
from .reconstruction import submit

class SemanticReviewTests(unittest.TestCase):
    def test_diagnostic_images_have_no_annotation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);src=p/'original.jpg'
            im=Image.new('RGB',(64,48),'#526456');im.save(src)
            e=clean_images(src,[12,12,23,20],p,'e001')
            self.assertEqual((p/e['full']).read_bytes(),src.read_bytes())
            expected=ImageOps.pad(Image.open(src).convert('RGB').crop(e['crop_bounds_xyxy']),(384,384),method=Image.Resampling.BICUBIC,color='#ededed')
            self.assertEqual(Image.open(p/e['crop']).tobytes(),expected.tobytes())
            self.assertFalse(e['annotations'])
    def test_any_confirmed_target_accepts(self):
        def decision(*vs):return aggregate([{'verdict':v} for v in vs])
        self.assertEqual(decision('false_positive'),'reject')
        self.assertEqual(decision('false_positive','uncertain'),'uncertain')
        self.assertEqual(decision('match','false_positive'),'accept')
        self.assertEqual(decision('uncertain','false_positive','match'),'accept')
        self.assertEqual(decision('match','uncertain'),'accept')
        self.assertEqual(decision(),'uncertain')

    def test_incomplete_or_fabricated_evidence_never_accepted(self):
        e={'evidence_id':'e1','verdict':'match','observed_object':'mouse','reason':'visible wheel'}
        self.assertEqual(validate_reply({'evidence':[e]},['e1'])['status'],'accept')
        for evidence in [[e,e],[],[{**e,'evidence_id':'unknown'}]]:
            with self.assertRaises(ValueError):validate_reply({'evidence':evidence},['e1'])

    def test_representatives_cover_each_peak_not_only_final_frames(self):
        g={'object':'tv','peak_ids':['p1','p2'],'selected':[], 'candidate_frames':[]}
        for fid,peak,score in [(1,'p1',.5),(2,'p1',.8),(3,'p2',.6)]:
            g['candidate_frames'].append(dict(source_frame_index=fid,timestamp_s=fid,frame_sha256='hash',
                 observations=[dict(object='tv',peak_id=peak,box_xyxy=[0,0,10,10],dino_score=score)]))
        self.assertEqual([r['source_frame_index'] for r in representatives(g)],[2,3])

    def test_rejected_or_missing_receipt_blocks_before_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'tv';folder.mkdir();write(root/'manifest.json',{'semantic_review_required':True})
            for status in [None,'reject','mixed','uncertain']:
                if status:write(folder/'semantic-review.json',{'status':status})
                with patch('openai.OpenAI') as client:
                    with self.assertRaises(ValueError):submit(folder,'unused',100,'','LLM_API_KEY','high',base_url='https://api.openai.com/v1',model='gpt-6-astra')
                    client.assert_not_called()
            write(folder/'input-frames.json',[])
            write(folder/'semantic-review.json',{'status':'accept','prepared_frames_sha256':sha(folder/'input-frames.json')})
            reconstruction_gate(folder)
            write(folder/'input-frames.json',[{'changed':True}])
            with self.assertRaises(ValueError):reconstruction_gate(folder)

    def test_export_preserves_hypotheses_but_filters_reconstruction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.json';out=root/'review';out.mkdir()
            groups=[dict(group_id=x,selected=[{}]) for x in ['a','b','c']]
            write(source,dict(questions={'q':dict(arms={'roi_robust':groups},summary={})}))
            write(out/'manifest.json',dict(selection=str(source),selection_sha256=sha(source),arm='roi_robust',jobs=['a','b','c'],references={'q':{x:x for x in ['a','b','c']}}))
            for gid,status in [('a','accept'),('b','reject'),('c','uncertain')]:
                job={'object':'tv'};job['request_hash']=digest(dict(job=job,prompt=PROMPT,schema=SCHEMA,model=MODEL))
                write(out/'jobs'/gid/'input.json',job)
                verdict={'accept':'match','reject':'false_positive','uncertain':'uncertain'}[status]
                write(out/'jobs'/gid/'review.json',dict(status=status,evidence=[{'verdict':verdict}],request_hash=job['request_hash']))
            s=export(out);q=s['questions']['q']
            self.assertEqual(len(q['arms']['roi_robust']),3)
            self.assertEqual([g['group_id'] for g in q['arms']['semantic_accepted']],['a'])
            self.assertEqual(len(q['semantic_excluded']),2)
if __name__=='__main__':unittest.main()
