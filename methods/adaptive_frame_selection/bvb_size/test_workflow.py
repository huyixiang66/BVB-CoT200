import json,tempfile,unittest
from pathlib import Path
from .workflow import batch,prepare_reviewed_qa
class WorkflowTests(unittest.TestCase):
 def test_batch_keeps_failures_and_continues(self):
  def f(x):
   if x['id']=='bad':raise ValueError('missing')
   return 'ok'
  self.assertEqual([x['status'] for x in batch([{'id':'bad'},{'id':'good'}],f)],['failed','completed'])
 def test_qa_requires_explicit_matched_visual_review(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'review.json').write_text(json.dumps({'usable':False}));(p/'views.json').write_text('{}')
   with self.assertRaisesRegex(ValueError,'visually approved'):
    prepare_reviewed_qa(p/'joint.json',p/'views.json',p/'review.json',p/'out',lambda x:None)

 def test_build_never_falls_back_to_unsandboxed(self):
  from unittest.mock import patch
  from .workflow import build_server_case
  with patch('bvb_size.workflow.shutil.which',return_value=None):
   with self.assertRaisesRegex(RuntimeError,'bubblewrap'):
    build_server_case('case','/work','blender','validator')

 def test_visual_receipt_must_match_image(self):
  import hashlib
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'view.png').write_bytes(b'frame');(p/'views.json').write_text(json.dumps({'images':['view.png']}))
   h=hashlib.sha256((p/'views.json').read_bytes()).hexdigest()
   (p/'review.json').write_text(json.dumps({'usable':True,'views_sha256':h,'image_sha256':{'view.png':'wrong'}}))
   with self.assertRaisesRegex(ValueError,'rendered image'):
    prepare_reviewed_qa(p/'joint.json',p/'views.json',p/'review.json',p/'out',lambda x:None)

 def test_build_uses_isolated_adapter(self):
  import types
  from unittest.mock import patch
  from .workflow import build_server_case
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);case=p/'case';joint=case/'joint';joint.mkdir(parents=True)
   (joint/'input.json').write_text('{}');c=case/'inference/reconstruct';c.mkdir(parents=True);(c/'generated.py').write_text('pass')
   v=p/'validator';v.mkdir();(v/'validate.py').write_text('pass');(v/'v10_contract.py').write_text('pass')
   def fake(cmd,check,timeout):
    self.assertIn('linux_build.py',cmd[4]);self.assertEqual(timeout,660)
    (joint/'scene.blend').write_bytes(b'blend')
   with patch('bvb_size.workflow.sys.platform','linux'),patch('bvb_size.workflow.shutil.which',return_value='/usr/bin/bwrap'),patch('bvb_size.workflow.subprocess.run',side_effect=fake):
    result=build_server_case(case,'/workcase',p/'blender',v)
   self.assertEqual(result['status'],'built_needs_view_review')


 def test_view_review_records_image_hashes_and_rejects_unresolved(self):
  from .workflow import record_visual_review
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'a.png').write_bytes(b'img');(p/'views.json').write_text(json.dumps({'mode':'single','images':['a.png']}))
   record=record_visual_review(p/'views.json',p/'review.json',True,'target visible')
   self.assertTrue(record['usable']);self.assertEqual(len(record['image_sha256']['a.png']),64)
   (p/'views.json').write_text(json.dumps({'mode':'unresolved','images':['a.png']}))
   with self.assertRaisesRegex(ValueError,'Unresolved'):record_visual_review(p/'views.json',p/'bad.json',True,'bad')

 def test_views_requires_verified_single_mesh_and_sandbox(self):
  from unittest.mock import patch
  from .workflow import render_server_views
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'scene.blend').write_bytes(b'blend');(p/'blender').write_bytes(b'bin')
   with patch('bvb_size.workflow.sys.platform','linux'),patch('bvb_size.workflow.shutil.which',return_value='/bin/bwrap'):
    with self.assertRaisesRegex(ValueError,'reviewed complete mesh'):
     render_server_views(p/'scene.blend','bed',None,p/'out',p/'blender')

 def test_batch_stages_require_backend_and_isolate_missing_inputs(self):
  from .batch_workflow import advance
  with self.assertRaisesRegex(ValueError,'backend'):advance([], 'qa')
  rows=advance([{'id':'missing','work':'does-not-exist'}],'build')
  self.assertEqual(rows[0]['status'],'failed')
