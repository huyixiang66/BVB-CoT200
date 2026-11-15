import json,tempfile,unittest,sys
from .pipeline import ROOT
sys.path.insert(0,str(ROOT/"scripts"))
from pathlib import Path
from unittest.mock import patch
from .entities import public_questions,prepare
from .features import MODEL,REVISION,flow

class PortableTests(unittest.TestCase):
 def test_public_fields_and_single_scene(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'q.json';q=dict(id=1,scene_name='scene',question='Where is the kettle?',options=['A. left','B. right'],objects=['kettle'],roles={'query':['kettle']},answer='B',gt='B',queried_object_ids=[4])
   p.write_text(json.dumps([q]));clean=public_questions(p)[0]
   for k in ['answer','gt','queried_object_ids']:self.assertNotIn(k,clean)
   with patch('subscription_direct_client.request',side_effect=AssertionError('Must not call API')):
    out=prepare(p,Path(d)/'out');self.assertEqual(json.loads(out.read_text())['objects'],['kettle'])
   p.write_text(json.dumps([q,dict(q,id=2,scene_name='other')]))
   with self.assertRaises(ValueError):public_questions(p)
 def test_invalid_id_and_duplicates(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'q.json'
   for rows in [[dict(id='../outside',scene_name='s',question='q')],[dict(id=1,scene_name='s',question='q')]*2]:
    p.write_text(json.dumps(rows))
    with self.assertRaises(ValueError):public_questions(p)
 def test_feature_pins(self):
  self.assertEqual(MODEL,'openmmlab-community/mm_grounding_dino_base_all');self.assertEqual(REVISION,'b32293f50a56f7218d9f73ff940ffb9eba84d63c')
 def test_flow_adjacent_frame_count(self):
  import cv2,numpy as np,csv
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);video=root/'v.avi';w=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),10,(640,480));self.assertTrue(w.isOpened())
   for i in range(10):
    im=np.zeros((480,640,3),np.uint8);cv2.rectangle(im,(100+i*3,100),(180+i*3,180),(255,255,255),-1);w.write(im)
   w.release();flow(video,root/'flow.csv');rows=list(csv.DictReader((root/'flow.csv').open()));self.assertEqual(len(rows),9);self.assertEqual([int(x['current_source_frame']) for x in rows],list(range(1,10)))
 def test_qa_replay_no_network(self):
  from . import qa as module
  with tempfile.TemporaryDirectory() as d:
   out=Path(d);(out/'views').mkdir();im=out/'views/top.png';im.write_bytes(b'test-image')
   q=dict(id=1,question='Which direction?',options=['A. left','B. right'],objects=['chair']);module.save(out/'question-public.json',q);module.save(out/'input.json',{'colors':{'chair':[1,0,0,1]}});module.save(out/'views/views.json',{'views':[dict(id='top',image='top.png',sha256=module.sha(im))]})
   replies=[{'answer':json.dumps(dict(view_id='top',reason='test',limitations='test'))},{'answer':json.dumps(dict(option='A',answer='left',reason='test',confidence='low',limitations='test'))}]
   with patch.object(module,'request',side_effect=replies) as mock:
    module.qa(out);self.assertEqual(mock.call_count,2)
   with patch.object(module,'request',side_effect=AssertionError('Completed QA must resume without request')):module.qa(out)
   self.assertEqual(module.read(out/'qa/answer.json')['option'],'A')
if __name__=='__main__':unittest.main()
