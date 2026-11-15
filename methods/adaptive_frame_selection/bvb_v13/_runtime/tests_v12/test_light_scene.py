import ast,copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from bvb_v10.pipeline import read,save,sha
from bvb_v13.scene_spec import validate,compile_script
from bvb_v13.light_semantic import select_one,request_with_cache

class CompactScene(unittest.TestCase):
    def fixture(self):
        packet=dict(question=dict(objects=['chair']),instance_limits={'chair':1},colors={'chair':[1,0,0,1]},frames=[dict(timestamp_s=0)],architecture_inventory_required=True)
        spec=dict(o=[[0,[1,2,1],90,[['b',[0,0,0],[1,.8,.2]],['c',[0,0,-.4],[.1,.1,.7]]]]],a=[['f',[0,0,-.1],[6,6,.2],0,[0]],['w',[-3,0,1],[.1,6,2],0,[0]],['w',[3,0,1],[.1,6,2],0,[0]]],c=[['s',[0],[1,2,.4],[1,1,.2],0]],g=[[[1,2],[0]]])
        return packet,spec
    def test_compiler_preserves_instances_parts_scope_and_no_model_code(self):
        p,s=self.fixture();r=validate(s,p)
        self.assertEqual(len(r['objects']),1);self.assertEqual(len(r['objects'][0]['parts']),2)
        self.assertEqual(r['context'][0]['targets'],['object_000']);self.assertEqual(r['objects'][0]['rotation'],[0,0,90])
        code=compile_script(s,p,'/tmp/scene.blend');ast.parse(code);self.assertIn('def build(spec,destination)',code)
    def test_malformed_or_unbounded_data_rejected(self):
        p,s=self.fixture()
        mutations=[lambda x:x.update(code='print(1)'),lambda x:x['o'][0].__setitem__(0,True),lambda x:x['o'][0][1].__setitem__(0,float('nan')),lambda x:x['a'][0][2].__setitem__(0,-1),lambda x:x['c'][0].__setitem__(1,[99]),lambda x:x['a'][0].__setitem__(4,[22]),lambda x:x['o'].append(copy.deepcopy(x['o'][0])),lambda x:x['g'][0].__setitem__(0,[0,1])]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                v=copy.deepcopy(s);mutate(v)
                with self.assertRaises((ValueError,TypeError)):validate(v,p)
    def test_room_labels_are_data_not_python(self):
        p,s=self.fixture();p['room_mode']=True;s['o'][0][0]="chair'); raise RuntimeError('injection"
        ast.parse(compile_script(s,p,'/tmp/scene.blend'))
        self.assertEqual(validate(s,p)['objects'][0]['category'],s['o'][0][0])
    def test_one_representative_prefers_detection_quality(self):
        group=dict(object='chair',candidate_frames=[dict(source_frame_index=i,observations=[dict(object='chair',peak_id=str(i),box_xyxy=[0,0,100,100],dino_score=i/10)]) for i in range(3)])
        candidates=[dict(source_frame_index=i,peak_id=str(i)) for i in range(3)]
        self.assertEqual(select_one(group,candidates),[candidates[2]])
    def test_identical_identity_review_reused_without_duplicate_usage(self):
        from bvb_v6 import semantic_review as sr
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cache=root/'cache';calls=[]
            evidence=[dict(evidence_id='e001',full='f.jpg',crop='c.png')]
            job=dict(request_hash='same',evidence=evidence)
            for name in ['a','b']:
                folder=root/name;folder.mkdir();(folder/'f.jpg').write_bytes(b'full');(folder/'c.png').write_bytes(b'crop')
                evidence[0].update(full_sha256=sha(folder/'f.jpg'),crop_sha256=sha(folder/'c.png'));save(folder/'input.json',job)
            def invoke(folder,*args):
                calls.append(folder);verdict=dict(evidence=[dict(evidence_id='e001',verdict='match',observed_object='chair',reason='visible')],status='accept',request_hash='same')
                save(folder/'review.json',verdict);r=dict(status='reviewed',usage=dict(prompt_tokens=100),request_hash='same',provider='openai');save(folder/'request.json',r);return r
            with patch.dict('os.environ',{'BVB_LIGHT_REVIEW_CACHE':str(cache)}),patch.object(sr,'verify_input',return_value='prompt'):
                request_with_cache(invoke,root/'a','key');r=request_with_cache(invoke,root/'b','key')
            self.assertEqual(len(calls),1);self.assertTrue(r['cache_hit']);self.assertIsNone(r['usage'])

if __name__=='__main__':unittest.main()
