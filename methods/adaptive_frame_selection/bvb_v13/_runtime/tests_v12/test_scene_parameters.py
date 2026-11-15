import ast,copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from bvb_v10.pipeline import save,read,sha
from bvb_v13.scene_spec import validate,compile_script
from bvb_v13.scene_prompt import build,task_for,TASK_RULES

class SceneParameters(unittest.TestCase):
    def fixture(self):
        p=dict(question=dict(objects=['chair'],question_type='object_rel_distance_closest'),instance_limits={'chair':4},colors={'chair':[1,0,0,1]},frames=[dict(timestamp_s=7)],architecture_inventory_required=True)
        s=dict(v=2,o=[[0,[1,2,3],[1,2,3]],[0,[2,3,4],['c',1,2,3],90],[0,[0,0,1],{'p':[['b',[0,0,0],[1,1,.2]],['c',[0,0,-.4],[.1,.1,.7],20]]},[10,20,30]]],a=[['f',[0,0,-.1],[6,6,.2],[0]],['w',[-3,0,1],[.1,6,2],[0],90],['w',[3,0,1],[.1,6,2],[0]]],c=[['s',[0],[1,2,.4],[1,1,.2]]],g=[[[1,2],[0]]])
        return p,s

    def test_same_geometry_and_metadata_as_explicit_contract(self):
        p,s=self.fixture()
        old=dict(o=[[0,[1,2,3],0,[['b',[0,0,0],[1,2,3]]]],[0,[2,3,4],90,[['c',[0,0,0],[1,2,3]]]],[0,[0,0,1],[10,20,30],s['o'][2][2]['p']]],a=[['f',[0,0,-.1],[6,6,.2],0,[0]],['w',[-3,0,1],[.1,6,2],90,[0]],['w',[3,0,1],[.1,6,2],0,[0]]],c=[['s',[0],[1,2,.4],[1,1,.2],0]],g=s['g'])
        original=copy.deepcopy(s)
        self.assertEqual(validate(s,p),validate(old,p));self.assertEqual(s,original)
        out=validate(s,p)
        self.assertEqual(out['objects'][1]['id'],'object_001')
        self.assertEqual(out['objects'][1]['color'],[1,0,0,1])
        self.assertEqual(out['architecture'][0]['times'],[7])
        self.assertLess(len(json.dumps(s)),len(json.dumps(old)))
        ast.parse(compile_script(s,p,'/tmp/scene.blend'))

    def test_nonbox_primitives_and_rotation_defaults(self):
        p,s=self.fixture()
        for shape in ['b','c','s','t']:
            s['o']=[[0,[0,0,1],[shape,1,2,3]]]
            r=validate(s,p)['objects'][0]
            self.assertEqual(r['parts'][0]['size'],[1.,2.,3.]);self.assertEqual(r['rotation'],[0.,0.,0.])

    def test_bad_parameters_do_not_bypass_existing_validation(self):
        p,s=self.fixture()
        mutations=[lambda x:x.update(v=True),lambda x:x.update(v=1),lambda x:x.update(code='bad'),lambda x:x['o'][0].__setitem__(2,[1,0,3]),lambda x:x['o'][0].__setitem__(0,True),lambda x:x['o'][0].__setitem__(2,['bad',1,2,3]),lambda x:x['o'][0].__setitem__(2,{'p':[]}),lambda x:x['o'][0].append(float('nan')),lambda x:x['a'][0].__setitem__(3,[9]),lambda x:x['c'][0].__setitem__(1,[99]),lambda x:x['g'][0].__setitem__(0,[0,1])]
        for mutate in mutations:
            bad=copy.deepcopy(s);mutate(bad)
            with self.assertRaises((ValueError,TypeError)):validate(bad,p)
        p['instance_limits']['chair']=1
        with self.assertRaisesRegex(ValueError,'Instance limit'):validate(s,p)

    def test_seven_tasks_have_geometry_policies(self):
        types=['object_rel_direction_easy','object_rel_distance_farthest','object_abs_distance','object_counting_multiple','object_size_estimation','route_planning','room_size_estimation']
        self.assertEqual(len({task_for({'question':{'question_type':q}}) for q in types}),7)
        for q in types:
            p={'question':{'question_type':q}}
            prompt=build(p)
            self.assertIn(TASK_RULES[task_for(p)],prompt)
            self.assertIn('Never merge distinct instances',prompt)
            self.assertIn('"v":2',prompt)

    def test_room_and_zero_count_cases(self):
        p,s=self.fixture();p['room_mode']=True
        for row in s['o']:row[0]='chair'
        self.assertEqual(validate(s,p)['objects'][0]['category'],'chair')
        self.assertIn('short STRINGS',build(p))
        p.pop('room_mode');p.update(counting_allow_all_rejected=True,counting_zero_evidence_categories=['chair'])
        s['o']=[];s.pop('c')
        self.assertEqual(validate(s,p)['objects'],[])
        self.assertIn('never fabricate them',build(p))

    def test_inference_compiles_parameters_and_logs_local_expansion(self):
        from bvb_v13.compact_inference import infer
        import subscription_direct_client as client
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);frame=out/'f.png';frame.write_bytes(b'fixture')
            packet=dict(room_mode=True,question=dict(question_type='room_size_estimation',objects=[]),frames=[dict(path=str(frame),sha256=sha(frame),timestamp_s=0,groups=[],observations=[])],groups=[])
            save(out/'input.json',packet);save(out/'experiment.json',dict(input_sha256=sha(out/'input.json')))
            response=dict(v=2,o=[['table',[0,0,1],[2,1,.1]]],a=[['f',[0,0,-.1],[6,6,.2],[0]]])
            with patch.object(client,'image_item',return_value={'type':'input_image'}),patch.object(client,'request',return_value={'answer':json.dumps(response)}) as request:
                infer(out)
            self.assertEqual(request.call_count,1)
            self.assertIn('"v":2',request.call_args.kwargs['instructions'])
            self.assertEqual(read(out/'scene-spec.json'),response)
            self.assertEqual(read(out/'attempt-01/scene-validation.json')['single_primitive_objects'],1)
            self.assertEqual(read(out/'attempt-01/scene-normalized.json')['objects'][0]['rotation'],[0,0,0])
            ast.parse((out/'generated.py').read_text())

if __name__=='__main__':unittest.main()
