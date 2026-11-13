import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from bvb_v11.route import route_steps,validate_answer,normalize_selection
from bvb_v11 import reliability

class Route(unittest.TestCase):
    def test_extra_route_context_not_a_missing_target(self):
        ids=['top','oblique-045','elevated-180'];a=dict(view_ids=ids,coverage=[dict(category=c,visible_in=ids) for c in ['chair','floor connectivity']])
        normalized=normalize_selection(a,dict.fromkeys(ids),['chair'])
        self.assertEqual(len(normalized['coverage']),1);self.assertEqual(len(normalized['context_coverage']),1)
        with self.assertRaises(ValueError):normalize_selection(a,dict.fromkeys(ids),['chair','desk'])
    def test_exact_category_after_role_prefix(self):
        ids=['top','oblique-045','elevated-180']
        a=dict(view_ids=ids,coverage=[dict(category=c,visible_in=ids,uncertainty='') for c in ['start: chair','waypoint: chair']])
        self.assertEqual(normalize_selection(a,dict.fromkeys(ids),['chair'])['coverage'][0]['category'],'chair')
    def test_duplicate_source_number_preserves_actions(self):
        rows=route_steps('1. [please fill in]. 2. Go forward until the door. 2. [please fill in]. 3. Go forward until the bed. You have reached the final destination.')
        self.assertEqual([r['step'] for r in rows],[1,2,3,4])
        self.assertEqual([r['printed_step'] for r in rows],[1,2,2,3])
    def test_option_sequence_must_match(self):
        q=dict(question='1. [please fill in]. 2. Go forward until the door. 3. [please fill in].',options=['A. Turn Left, Turn Right'])
        a=dict(option='A',answer='Turn Left, Turn Right',turns=[dict(step=1,action='Turn Left'),dict(step=3,action='Turn Right')],sufficient_evidence=True)
        validate_answer(a,q)
        a['turns'][1]['action']='Turn Left'
        with self.assertRaises(ValueError):validate_answer(a,q)
    def test_route_keeps_walls_counting_still_hides(self):
        for hidden in [False,True]:
            with tempfile.TemporaryDirectory() as folder:
                out=Path(folder);(out/'coverage-validation.json').write_text('{"valid":true}');(out/'experiment.json').write_text(json.dumps(dict(qa_walls_hidden=hidden)))
                with patch.object(reliability.subprocess,'run') as run:
                    run.return_value.returncode=0;reliability.execute(out,'/unused/blender',render=True)
                script=(out/'render.py').read_text()
                self.assertEqual('for wall_name in wall_states:bpy.data.objects[wall_name].hide_render=True' in script,hidden)
                self.assertIn('qa_walls_hidden='+str(hidden),script)
                compile(script,'render.py','exec')

if __name__=='__main__':unittest.main()
