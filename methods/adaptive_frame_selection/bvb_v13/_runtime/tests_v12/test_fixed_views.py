from PIL import Image
import itertools
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from bvb_v10.pipeline import save,read,sha
from bvb_v13 import driver
from bvb_v13.view_selection import configure,score_candidates,choose,ensure_selection,camera


def cube(name,center,role='target',category='chair',ident=None,size=1):
    return dict(name=name,role=role,category=category,instance_id=ident or name,
                world_bbox=[[center[i]+d[i]*size/2 for i in range(3)] for d in itertools.product([-1,1],repeat=3)])


class FixedViews(unittest.TestCase):
    def fixture(self,out,qtype='object_rel_direction_easy'):
        q=dict(question_type=qtype,question='Where is the chair?',objects=['chair'],options=['A. left'])
        save(out/'input.json',dict(question=q,colors={'chair':[1,0,0,1]}))
        save(out/'question-public.json',q)
        save(out/'experiment.json',dict(version='bvb_v13',view_selection_mode='fixed'))
        save(out/'geometry-inspection.json',[cube('chair',(0,0,0))])
        views=[]
        for vid in ['top','oblique-000','oblique-090','oblique-180']:
            f=out/'views'/(vid+'.png');f.parent.mkdir(exist_ok=True);Image.new('RGB',(900,900)).save(f)
            views.append(dict(id=vid,image=f.name,sha256=sha(f)))
        save(out/'views/views.json',dict(views=views,qa_walls_hidden=False))

    def test_envelope_fit_in_frame(self):
        points=np.array(cube('c',(10,-5,3),size=12)['world_bbox'])
        for vid in ['top','oblique-045','elevated-270']:
            r,p,s=camera({'id':vid},points);xy=((points-p)@r)[:,:2]/s+.5
            self.assertTrue(np.all(xy>=0));self.assertTrue(np.all(xy<=1))

    def test_occluder_changes_geometric_visibility(self):
        objects=[cube('chair',(0,0,0)),cube('cover',(0,0,2),'context','support',size=2)]
        units,scores=score_candidates(objects,[{'id':'top'}],['chair'])
        self.assertEqual(scores[0]['visibility'],[0.])
        _,scores=score_candidates(objects,[{'id':'top'}],['chair'],occlusion={'views':[{'view_id':'top','hidden_objects':[{'name':'cover'}]}]})
        self.assertEqual(scores[0]['visibility'],[1.])

    def test_same_instance_parts_do_not_hide_each_other(self):
        objs=[cube('base',(0,0,0),ident='one'),cube('top',(0,0,2),ident='one')]
        units,scores=score_candidates(objs,[{'id':'top'}],['chair'])
        self.assertEqual(len(units),1);self.assertEqual(scores[0]['visibility'],[1.])

    def test_complementarity_and_route_constraint(self):
        scores=[dict(id=vid,quality=q,camera_direction=d) for vid,q,d in [
            ('top',[.8,.8],[0,0,-1]),('oblique-000',[1,0],[1,0,0]),
            ('oblique-090',[0,1],[0,1,0]),('elevated-000',[1,0],[1,0,0])]]
        for task in ['rel_dist','abs_dist','counting','object_size','route']:
            _,selected=choose(scores,task)
            self.assertIn('oblique-090',[v['id'] for v in selected])
            self.assertEqual(len(selected),3)
            if task=='route':self.assertIn('top',[v['id'] for v in selected])
        self.assertEqual(len(choose(scores,'rel_dir')[1]),1)

    def test_fixed_qa_calls_only_final_answer(self):
        from bvb_v10 import qa as direction
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);self.fixture(out);calls=[]
            def request(path,*args,**kwargs):
                calls.append(path.name)
                self.assertEqual(path.name,'step-01')
                return dict(answer='A')
            with patch.object(direction,'request',side_effect=request),patch.object(direction,'image_item',return_value={}):
                driver.qa(out)
            self.assertEqual(calls,['step-01']);self.assertEqual(read(out/'view-selection/selection.json')['model_calls'],0)
            self.assertFalse((out/'view-selection/request.json').exists())
            with self.assertRaisesRegex(ValueError,'fresh output'):configure(out,'astra')

    def test_short_gate_then_complete_direction_answer(self):
        from bvb_v10 import qa as direction
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);self.fixture(out);calls=[]
            def request(path,*args,**kwargs):
                calls.append(path.name)
                return dict(answer='NEED_MORE_VIEWS' if len(calls)==1 else 'A')
            with patch.object(direction,'request',side_effect=request):driver.qa(out)
            self.assertEqual(calls,['step-01','step-02']);self.assertEqual(read(out/'qa/answer.json')['answer'],'left')

    def test_driver_records_exhausted_views_as_unanswered(self):
        from bvb_v10 import qa as direction
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);self.fixture(out)
            with patch.object(direction,'request',return_value=dict(answer='NEED_MORE_VIEWS')) as request:
                result=driver.qa(out)
                self.assertEqual(request.call_count,4)
            self.assertIsNone(result['answer']);self.assertEqual(read(out/'status.json')['stage'],'insufficient-evidence')
            with patch.object(direction,'request',side_effect=AssertionError('No new call on resume')):
                driver.run(out,'unused')
            self.assertEqual(read(out/'status.json')['stage'],'insufficient-evidence')

    def test_astra_mode_preserves_selection_and_final_answer_calls(self):
        from bvb_v10 import qa as direction
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);self.fixture(out)
            save(out/'experiment.json',dict(version='bvb_v13'))
            configure(out,'astra')
            calls=[]
            def request(path,*args,**kwargs):
                calls.append(path.name)
                return dict(answer='{"view_id":"top"}' if path.name=='view-selection' else 'A')
            with patch.object(direction,'request',side_effect=request),patch.object(direction,'image_item',return_value={}):
                driver.qa(out)
            self.assertEqual(calls,['view-selection','step-01'])

    def test_cache_identity_and_image_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);self.fixture(out)
            first=ensure_selection(out);self.assertEqual(first,ensure_selection(out))
            data=read(out/'geometry-inspection.json');data[0]['name']='changed';save(out/'geometry-inspection.json',data)
            with self.assertRaisesRegex(ValueError,'sources/policy'):ensure_selection(out)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);self.fixture(out);(out/'views/top.png').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):ensure_selection(out)

    def test_fixed_default_and_existing_receipts_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);save(out/'experiment.json',{})
            self.assertEqual(configure(out),'fixed')
            configure(out,'astra');self.assertIsNone(ensure_selection(out))
            save(out/'view-selection/selection.json',dict(view_id='top'))
            with self.assertRaisesRegex(ValueError,'fresh output'):configure(out,'fixed')

    def test_route_requires_walls(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);self.fixture(out,'route_planning')
            d=read(out/'views/views.json');d['qa_walls_hidden']=True;save(out/'views/views.json',d)
            with self.assertRaisesRegex(ValueError,'retain walls'):ensure_selection(out)

    def test_missing_target_fails_without_fallback(self):
        with self.assertRaisesRegex(ValueError,'Missing target'):score_candidates([cube('a',(0,0,0))],[{'id':'top'}],['chair','table'])

if __name__=='__main__': unittest.main()
