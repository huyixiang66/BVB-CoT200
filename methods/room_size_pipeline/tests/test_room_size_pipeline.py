import copy
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from room_size_pipeline.selection import choose_frames,sha256
from room_size_pipeline.contracts import validate_scene,validate_reference_size,validate_bounds
from room_size_pipeline.pipeline import run,write


def fixtures():
    spec={'regions':[{'id':'room','space_id':'room','size_m':[4,5],'evidence_times_s':[0]}],
          'connections':[], 'missing_evidence':[],
          'references':[{'id':'bed','region_id':'room','label':'bed','size_m':[2,1.5,.5],
                         'size_ranges_m':[[1.8,2.2],[1.4,1.6],[.4,.6]],'qa_axes':[0,1],
                         'source':'original frames','evidence_times_s':[0]}]}
    scene={'layout':{'room':{'origin':[0,0],'swap':False}},'openings':[],
           'objects':[{'id':'bed','region_id':'room','label':'bed','position':[1.5,2,.25],
                       'size_m':[2,1.5,.5],'yaw':0,'parts':[{'shape':'box','center':[0,0,0],
                       'size':[1,1,1],'color':[.8,.7,.6]}]}]}
    return spec,scene


def pose(t,x=0):
    return {'timestamp_s':t,'c2w_blender':[[1,0,0,x],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}


class SelectionTests(unittest.TestCase):
    def test_uniform_reference(self):
        times,report=choose_frames(173)
        self.assertEqual(len(times),22);self.assertEqual(times[-1],168)

    def test_duplicates_reduce(self):
        times,report=choose_frames(8,[pose(i) for i in range(8)])
        self.assertEqual(len(times),1)

    def test_soft_budget_not_cap(self):
        rows=[{**pose(i,i),'visible_cells':[f'{i}-{j}' for j in range(10)]} for i in range(8)]
        times,report=choose_frames(8,rows)
        self.assertGreater(len(times),report['reference_count'])

    def test_turn_in_place(self):
        rows=[pose(0),pose(1)]
        rows[1]['c2w_blender']=[[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]]
        self.assertEqual(len(choose_frames(16,rows)[0]),2)

    def test_invalid_pose(self):
        r=pose(0);r['c2w_blender'][0][0]=float('nan')
        with self.assertRaises(ValueError):choose_frames(8,[r])

    def test_upstream_symlink_frame_is_valid(self):
        from room_size_pipeline.v10_adapter import select
        class Matrix:
            def tolist(self):return pose(0)['c2w_blender']
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);frames=root/'frames';frames.mkdir();original=root/'original.jpg';original.write_bytes(b'original')
            try:(frames/'000000.jpg').symlink_to(original)
            except OSError:self.skipTest('OS disallows unprivileged symlinks')
            poses=root/'poses';poses.mkdir();video=root/'video.mp4';video.write_bytes(b'video')
            write(poses/'sampling.json',{'frames':[{'filename':'000000.jpg','timestamp_s':0,'source_frame_index':0,'sha256':sha256(original)}]})
            modules={'bvb_v6.reconstruction':SimpleNamespace(shared_pose=lambda *a:([Matrix()],[Matrix()],{}))}
            with patch('room_size_pipeline.v10_adapter.load',return_value=modules),patch('room_size_pipeline.v10_adapter.subprocess.check_output',return_value='{"format":{"duration":8}}'):
                result=select({'v10_root':str(root),'poses':str(poses),'frames_root':str(frames),'video':str(video),'selection_mode':'camera_diversity'},root/'out')
            self.assertEqual(len(result['frames']),1)
            self.assertEqual(result['frames'][0]['sha256'],sha256(original))


class GeometryTests(unittest.TestCase):
    def test_valid_scene(self):
        spec,scene=fixtures();self.assertEqual(validate_scene(spec,scene)['room'],(0,4,0,5))

    def test_reference_outside_range_rejected(self):
        spec,_=fixtures()
        with self.assertRaisesRegex(ValueError,'outside original-frame'):validate_reference_size(spec['references'][0],[3,1.5,.5])

    def test_actual_reference_inside_range(self):
        spec,_=fixtures();validate_reference_size(spec['references'][0],[2.05,1.5,.5])

    def test_outside_floor_rejected(self):
        with self.assertRaisesRegex(ValueError,'outside fixed'):validate_bounds((0,4,0,5),[[-1,0,0],[1,2,1]])

    def test_uncertainty_notes_do_not_block_best_estimate(self):
        spec,scene=fixtures();spec['missing_evidence']=['unseen boundary']
        validate_scene(spec,scene)
        spec['regions'][0]['size_m'][0]=None
        with self.assertRaisesRegex(ValueError,'Invalid numeric'):validate_scene(spec,scene)

    def test_disconnected_room_rejected(self):
        spec,scene=fixtures();r=copy.deepcopy(spec['regions'][0]);r['id']='annex'
        spec['regions'].append(r);scene['layout']['annex']={'origin':[4,0],'swap':False}
        with self.assertRaisesRegex(ValueError,'Disconnected'):validate_scene(spec,scene)

    def test_accounting_subdivision_open(self):
        spec,scene=fixtures();r=copy.deepcopy(spec['regions'][0]);r['id']='annex'
        spec['regions'].append(r);scene['layout']['annex']={'origin':[4,0],'swap':False}
        spec['connections']=[{'regions':['room','annex'],'kind':'open'}]
        scene['openings']=[{'region':'room','side':'east','start':0,'end':5},
                           {'region':'annex','side':'west','start':0,'end':5}]
        validate_scene(spec,scene)
        scene['openings'][0]['end']=1
        with self.assertRaisesRegex(ValueError,'Missing aligned'):validate_scene(spec,scene)


    def test_distinct_spaces_allow_uncased_opening_but_not_narrow_gap(self):
        spec,scene=fixtures()
        annex=copy.deepcopy(spec['regions'][0]);annex.update(id='annex',space_id='second')
        spec['regions'].append(annex)
        ref=copy.deepcopy(spec['references'][0]);ref.update(id='second-bed',region_id='annex')
        spec['references'].append(ref)
        scene['layout']['annex']={'origin':[4,0],'swap':False}
        spec['connections']=[{'regions':['room','annex'],'kind':'open'}]
        scene['openings']=[{'region':'room','side':'east','start':1,'end':2},
                           {'region':'annex','side':'west','start':1,'end':2}]
        validate_scene(spec,scene)
        scene['openings'][1]['end']=1.6
        with self.assertRaisesRegex(ValueError,'Missing aligned'):validate_scene(spec,scene)
        scene['openings'][1]['end']=2
        annex['space_id']='room'
        with self.assertRaisesRegex(ValueError,'Missing aligned'):validate_scene(spec,scene)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.video=self.root/'video.mp4';self.video.write_bytes(b'fixture')
        self.config={'video':str(self.video),'out':str(self.root/'run'),'question':'Estimate total floor area in square meters.'}
        self.calls=[]
        upstream=self.root/'upstream'
        for name in ['bvb_v6','bvb_v10','bvb_v10_2']:
            d=upstream/name;d.mkdir(parents=True);(d/'__init__.py').write_text('# fixture')
        poses=self.root/'poses';poses.mkdir()
        for name in ['camera-poses.npz','sampling.json','sparse-geometry-corrected.npz']:(poses/name).write_bytes(b'fixture')
        self.config.update(v10_root=str(upstream),poses=str(poses),frames_root=str(self.root))

    def select(self,config,out):
        out.mkdir(parents=True);p=out/'image.jpg';p.write_bytes(b'fake image for orchestration test')
        return {'frames':[{'file':str(p),'sha256':sha256(p),'timestamp_s':0}]}

    def render(self,config,out):
        d=out/'render';(d/'top.png').write_bytes(b'fake render');(d/'scene.blend').write_bytes(b'fixture');(d/'joint-original.blend').write_bytes(b'original')
        write(d/'reference.json',{'accepted':True,'text':'Central bed length 2 m, width 1.5 m'})
        write(d/'geometry.json',{'diagnostic_area':20})

    def prepare(self,config,out,manifest,spec):
        write(out/'v10_input/prepared.json',{'prompt':'Original V10_2 Python contract',
              'images':[manifest['frames'][0]['file']],
              'content':[{'type':'text','text':'camera after image'},{'type':'image','index':0}]})

    def backend(self,request):
        self.calls.append(request)
        value={'estimate':json.dumps(fixtures()[0]),'reconstruct':'import bpy\n# upstream Python fixture','qa':json.dumps({'answer_m2':20})}[request['stage']]
        return {'status':'completed','answer':value,'usage':{'fixture':True}}

    def test_full_chain_and_resume_without_calls(self):
        run(self.config,self.backend,selector=self.select,renderer=self.render,preparer=self.prepare)
        self.assertEqual(len(self.calls),3)
        self.assertEqual(self.calls[1]['stage'],'reconstruct')
        self.assertIn('content',self.calls[1])
        self.assertIn('import bpy',(Path(self.config['out'])/'reconstruct/generated.py').read_text())
        qa=self.calls[-1];self.assertEqual(len(qa['images']),1)
        self.assertIn('Estimate total floor area',qa['prompt'])
        self.assertNotIn('diagnostic_area',qa['prompt']);self.assertNotIn('20',qa['prompt'])
        run(self.config,None,selector=self.select,renderer=self.render,preparer=self.prepare)
        self.assertEqual(len(self.calls),3)

    def test_failed_attempt_not_retried(self):
        def fail(request):raise RuntimeError('fixture')
        with self.assertRaises(RuntimeError):run(self.config,fail,selector=self.select)
        with self.assertRaisesRegex(RuntimeError,'no automatic retry'):run(self.config,self.backend,selector=self.select)
        self.assertEqual(self.calls,[])

    def test_changed_video_rejected(self):
        run(self.config,until='select',selector=self.select);self.video.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'inputs/code changed'):run(self.config,until='select')

    def test_scoring_fields_rejected(self):
        with self.assertRaises(ValueError):run({**self.config,'gt':20},until='select')

    def test_changed_upstream_rejected(self):
        run(self.config,until='select',selector=self.select)
        (Path(self.config['v10_root'])/'bvb_v10/__init__.py').write_text('# changed')
        with self.assertRaisesRegex(ValueError,'inputs/code changed'):run(self.config,until='select')

    def test_python_not_shape_json(self):
        from room_size_pipeline.v10_adapter import python_answer
        self.assertEqual(python_answer('```python\nimport bpy\n```').strip(),'import bpy')
        with self.assertRaises(SyntaxError):python_answer('invalid python !!!')
        with self.assertRaises(ValueError):python_answer('{"objects": []}')

    def test_engine_compatibility_only_changes_engine_assignment(self):
        from room_size_pipeline.v10_adapter import python_answer
        code='import bpy\nlabel="BLENDER_EEVEE_NEXT"\nbpy.context.scene.render.engine="BLENDER_EEVEE_NEXT"\n'
        result=python_answer(code)
        self.assertIn('label="BLENDER_EEVEE_NEXT"',result)
        self.assertIn("render.engine='BLENDER_EEVEE'",result)
        self.assertEqual(python_answer(result),result)

    def test_default_renderer_uses_integrated_postprocessor(self):
        from room_size_pipeline import v10_adapter
        out=self.root/'render-test';dest=out/'render';dest.mkdir(parents=True)
        (dest/'scene.blend').write_bytes(b'fixture')
        with patch.object(v10_adapter.sys,'platform','linux'), \
             patch.object(v10_adapter.shutil,'which',return_value='/usr/bin/bwrap'), \
             patch.object(v10_adapter.subprocess,'run') as process:
            v10_adapter.render({'blender':'blender'},out)
        self.assertEqual(process.call_count,2)
        script=Path(process.call_args_list[1].args[0][process.call_args_list[1].args[0].index('--python')+1])
        self.assertEqual(script.name,'blender_scene.py')
        text=script.read_text(encoding='utf8')
        self.assertIn('measurement_exclusions(ref,o)',text)
        self.assertIn('measurement_matrix',text)
        self.assertNotIn('repair_contracts',text)
        self.assertIn('ratio>=.8 and area>=.005 and complete',text)


if __name__=='__main__':unittest.main()
