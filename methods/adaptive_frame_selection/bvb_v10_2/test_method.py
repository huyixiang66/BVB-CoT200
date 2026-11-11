import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from .pose_exporter import resolve_interval
from . import pipeline
from .report import render

class MethodTests(unittest.TestCase):
 def test_interval_keeps_keyframes_bounded_without_changing_samples(self):
  for n,k in [(180,2),(475,2),(640,2),(641,3),(868,3)]:self.assertEqual(resolve_interval(n),k)
  self.assertEqual(resolve_interval(868,'2'),2)
  for n,k in [(0,'auto'),(3,0),(3,-1)]:
   with self.assertRaises(ValueError):resolve_interval(n,k)
 def test_run_uses_portable_qa_and_requested_blender(self):
  with tempfile.TemporaryDirectory() as tmp:
   out=Path(tmp)
   with patch.object(pipeline,'infer') as infer,patch.object(pipeline,'execute') as execute,patch('bvb_v10.qa.qa') as qa:
    pipeline.run(out,'/custom/blender')
   infer.assert_called_once_with(out)
   self.assertEqual(execute.call_args_list[0].args,(out,'/custom/blender'))
   self.assertEqual(execute.call_args_list[1].args,(out,'/custom/blender',True))
   qa.assert_called_once_with(out)
   self.assertEqual(json.loads((out/'status.json').read_text())['stage'],'completed')
 def test_failed_qa_records_failure_and_raises(self):
  with tempfile.TemporaryDirectory() as tmp:
   out=Path(tmp)
   with patch.object(pipeline,'infer'),patch.object(pipeline,'execute'),patch('bvb_v10.qa.qa',side_effect=RuntimeError('transport interrupted')):
    with self.assertRaises(RuntimeError):pipeline.run(out)
   self.assertEqual(json.loads((out/'status.json').read_text())['stage'],'failed')
 def test_failed_answer_counts_zero_in_report(self):
  with tempfile.TemporaryDirectory() as tmp:
   out=Path(tmp);audit=out/'audit/input/review';audit.mkdir(parents=True)
   (audit/'reviewed-selection.json').write_text(json.dumps({'questions':{'1':{'question':{'question':'Where?','objects':[],'options':['A. left','B. right']},'arms':{}}}}))
   q=out/'questions/1';q.mkdir(parents=True);(q/'status.json').write_text(json.dumps({'stage':'failed'}))
   ref=out/'display.json';ref.write_text(json.dumps({'1':'A'}));render(out,ref)
   page=(out/'report/index.html').read_text();self.assertIn('V10：0/1',page);self.assertIn('得分：0',page)
if __name__=='__main__':unittest.main()
