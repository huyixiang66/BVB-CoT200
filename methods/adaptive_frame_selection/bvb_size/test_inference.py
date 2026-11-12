import json
import tempfile
import unittest
from pathlib import Path
from .handoff import digest
from .inference import execute, execute_batch

class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.img=self.root/'image.jpg';self.img.write_bytes(b'frame')
        self.req=self.root/'request.json';self.req.write_text(json.dumps({'model':'gpt-6-astra','effort':'high','prompt':'question','images':[{'path':str(self.img),'sha256':digest(self.img)}]}))
    def test_qa_resume_and_hash_guard(self):
        calls=[]
        def backend(req):
            calls.append(req);return {'status':'completed','answer':'{"answer":12,"reason":"test","sufficient_evidence":true}'}
        path=execute(self.req,self.root/'out',backend,'qa');self.assertEqual(json.loads(path.read_text())['answer'],12)
        execute(self.req,self.root/'out',backend,'qa');self.assertEqual(len(calls),1)
        self.img.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'changed'):execute(self.req,self.root/'out',backend,'qa')
    def test_failed_calls_are_not_retried(self):
        calls=[]
        def backend(req):calls.append(req);raise RuntimeError('connection')
        for _ in range(2):
            with self.assertRaises(RuntimeError):execute(self.req,self.root/'out',backend,'qa')
        self.assertEqual(len(calls),1)
    def test_batch_isolates_a_failed_request(self):
        missing=self.root/'missing.json'
        def backend(req):return {'status':'completed','answer':'{"answer":12}'}
        results=execute_batch([{'id':'bad','kind':'qa','request':str(missing),'output':str(self.root/'bad')},
                               {'id':'ok','kind':'qa','request':str(self.req),'output':str(self.root/'ok')}],backend)
        self.assertEqual([r['status'] for r in results],['failed','completed'])

    def test_reconstruction_validates_python(self):
        def backend(req):return {'status':'completed','answer':'not valid python ?'}
        with self.assertRaises(SyntaxError):execute(self.req,self.root/'out',backend,'reconstruct')
        self.assertFalse((self.root/'out/reconstruct/generated.py').exists())
