"""Offline contract tests for question-wide reconstruction and final instance caps."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT
sys.path.insert(0,str(SOURCE))
from bvb_v6 import instance_resolution as ir
from bvb_v6 import coverage
from bvb_v6.reconstruction import expand, prepare


def question(kind='object_abs_distance'):
    return {'id':1,'question_type':kind,'question':'What is the distance between the pan and basket?',
            'objects':['pan','basket'],'roles':{}}


def instance(rep, members=None, label='pan'):
    return {'label':label,'representative':rep,'members':members or [rep],
            'status':'confirmed','reason':'Same visible handle and counter in the source frames'}


class FinalInstances(unittest.TestCase):
    def setUp(self):
        self.plan={'policy':ir.instance_policy(question()),'candidates':[
            {'candidate_id':'a::p','label':'pan'}, {'candidate_id':'b::p','label':'pan'},
            {'candidate_id':'b::basket','label':'basket'}]}
        self.good={'instances':[instance('a::p',['a::p','b::p']),instance('b::basket',label='basket')], 'excluded':[]}

    def test_single_and_multiple_policy(self):
        self.assertEqual(ir.instance_policy(question())['limits'],{'pan':1,'basket':1})
        for kind in ['object_counting','object_rel_distance']:
            self.assertEqual(ir.instance_policy(question(kind))['limits'],{'pan':None,'basket':None})
        self.assertEqual(ir.instance_policy(question('unknown sti question'))['limits']['pan'],None)

    def test_plural_roles_and_explicit_override(self):
        q=question();q['question']='How far apart are the two pans?'
        self.assertIsNone(ir.instance_policy(q)['limits']['pan'])
        q=question();q['roles']={'left':'pan','right':'pan'}
        self.assertIsNone(ir.instance_policy(q)['limits']['pan'])
        q['instance_limits']={'pan':2}
        self.assertEqual(ir.instance_policy(q)['limits']['pan'],2)
        q['instance_limits']={'pan':0}
        with self.assertRaises(ValueError):ir.instance_policy(q)

    def test_preserve_views_one_final_representative(self):
        self.assertEqual(ir.validate_decision(self.plan,self.good),{'a::p','b::basket'})
        self.assertEqual(len(self.good['instances'][0]['members']),2)

    def test_duplicate_instance_rejected_for_single(self):
        bad=copy.deepcopy(self.good)
        bad['instances']=[instance('a::p'),instance('b::p'),instance('b::basket',label='basket')]
        with self.assertRaises(ValueError):ir.validate_decision(self.plan,bad)
        self.plan['policy']=ir.instance_policy(question('object_counting'))
        self.assertEqual(len(ir.validate_decision(self.plan,bad)),3)

    def test_no_drop_all_no_cross_category_no_invention(self):
        bad=copy.deepcopy(self.good);bad['instances']=bad['instances'][1:]
        bad['excluded']=[{'candidate_id':x,'reason':'uncertain'} for x in ['a::p','b::p']]
        with self.assertRaises(ValueError):ir.validate_decision(self.plan,bad)
        for member in ['b::basket','invented']:
            bad=copy.deepcopy(self.good);bad['instances'][0]['members'].append(member)
            with self.assertRaises(ValueError):ir.validate_decision(self.plan,bad)

    def test_incomplete_accounting_rejected(self):
        bad=copy.deepcopy(self.good);bad['instances'][0]['members']=['a::p']
        with self.assertRaises(ValueError):ir.validate_decision(self.plan,bad)

    def test_local_schema_accepts_secondary_targets(self):
        data={'structure':[], 'targets':[['p','pan',[0,0,1],[.3,.2,.1]],['b','basket',[.5,0,1],[.2,.2,.2]]],
              'context':[['support',['p','b'],'support',[0,0,.5],[1,1,1]]]}
        spec,audit=expand(data,question(),walls_only=True)
        self.assertEqual(set(t['label'] for t in audit['targets']),{'pan','basket'})
        before=copy.deepcopy((spec,audit))
        s,a=ir.filter_region(spec,audit,'g',{'g::target_b'})
        self.assertEqual([t['label'] for t in a['targets']],['basket'])
        self.assertEqual(a['context_relations'][0]['targets'],['target_b'])
        self.assertEqual((spec,audit),before)
        s,a=ir.filter_region(spec,audit,'g',set())
        self.assertEqual(s['objects'],[])

    def test_context_rejects_landmarks_containers_obstacles_and_chains(self):
        data={'structure':[], 'targets':[['p','pan',[0,0,1],[.3,.2,.1]]],
              'context':[['stove',['p'],'support',[0,0,.5],[1,1,1]]]}
        for relation in ['anchor','container','attached','obstacle']:
            bad=copy.deepcopy(data);bad['context'][0][2]=relation
            with self.subTest(relation=relation),self.assertRaises(ValueError):
                expand(bad,question(),walls_only=True)
        bad=copy.deepcopy(data)
        bad['context'].append(['cabinet',['stove'],'support',[0,0,0],[1,1,1]])
        with self.assertRaises(ValueError):expand(bad,question(),walls_only=True)

    def test_floor_structure_preserved_and_roof_rejected(self):
        from local_context_scene import prompt_for
        data={'structure':[['f','floor',[0,0,-.05],[3,3,.1]]], 'targets':[], 'context':[]}
        spec,audit=expand(data,question(),walls_and_floor=True)
        self.assertEqual(spec['objects'][0]['kind'],'floor')
        self.assertEqual(spec['objects'][0]['color'],'white')
        filtered,_=ir.filter_region(spec,audit,'g',set())
        self.assertEqual(filtered['objects'],spec['objects'])
        for kind in ['roof','ceiling']:
            bad=copy.deepcopy(data);bad['structure'][0][1]=kind
            with self.assertRaises(ValueError):expand(bad,question(),walls_and_floor=True)
        prompt=prompt_for(question(),walls_and_floor=True)
        self.assertIn('kind is ONLY wall or floor',prompt)
        self.assertNotIn('do not create floor geometry',prompt)
        bad=copy.deepcopy(data);bad['structure']=[['floor','floor',[0,0,0],[3,3,.1]]]
        with self.assertRaises(ValueError):expand(bad,question(),walls_only=True)

    def test_single_cluster_waits_for_comparison_then_retains_winner(self):
        groups=[{'group_id':str(i),'object':'pan','selected':[{}],
                 'semantic_review':{'status':'accept','review_id':str(i)}} for i in range(3)]
        q=question();q['objects']=['pan']
        selection={'questions':{'1':{'question':q,'arms':{'roi_robust':groups,'semantic_accepted':copy.deepcopy(groups)},
                   'semantic_excluded':[],'summary':{}}},'semantic_review':{'source_arm':'roi_robust'}}
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);ir.save(p/'coverage-policy.json',{'policy':'question_aware_category_coverage'})
            out=coverage.apply_coverage(selection,p)
            self.assertEqual(out['questions']['1']['arms']['semantic_selected'],[])
            self.assertEqual(out['questions']['1']['coverage_blockers']['pan'],'coverage_comparison_pending')
            cid=coverage.comparison_key('pan',['0','1','2'],'1')
            (p/'coverage'/cid).mkdir(parents=True)
            ir.save(p/'coverage'/cid/'input.json',{'request_hash':'test-comparison'})
            ir.save(p/'coverage'/cid/'result.json',{
                'status':'compared','request_hash':'test-comparison','selected_id':'1',
                'confidence':'medium','reason':'Visible whole target','uncertainty':'Scale uncertain',
                'ranking':[{'candidate_id':i,'rank':rank,'reason':'Compared visible extent'}
                           for rank,i in enumerate(['1','0','2'],1)]})
            out=coverage.apply_coverage(selection,p)
            kept=out['questions']['1']['arms']['semantic_selected']
            self.assertEqual([g['group_id'] for g in kept],['1'])
            self.assertTrue(all(coverage.admitted(g['semantic_review']) for g in kept))

    def test_plan_hash_rejects_stale_decision_and_missing_target_stays_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);g=root/'g';(g/'assets').mkdir(parents=True)
            image=root/'frame.jpg';image.write_bytes(b'test image bytes; no API invoked')
            frame={'source_frame_index':0,'timestamp_s':0,'path':str(image),'sha256':ir.file_sha(image)}
            ir.save(root/'manifest.json',{'question':question(),'groups':[{'group_id':'g','folder':'g'}]})
            ir.save(g/'assets/scene-spec.json',{'objects':[{'id':'p','kind':'pan','p':[0,0,0],'s':[1,1,1]}]})
            ir.save(g/'assets/question-audit.json',{'targets':[{'id':'p','label':'pan'}]})
            ir.save(g/'input-frames.json',[frame]);ir.save(g/'pose-reference.json',{})
            ir.save(g/'pose-reference-vlm.json',{});ir.save(g/'question.json',question())
            (g/'system-prompt.txt').write_text('full question objects')
            plan=ir.make_plan(root)
            self.assertEqual({c['label'] for c in plan['candidates']},{'pan'})
            (root/'instance-resolution').mkdir()
            decision={'instances':[instance('g::p')],'excluded':[]}
            ir.save(root/'instance-resolution/result.json',{'status':'validated','request_hash':plan['request_hash'],'decision':decision})
            self.assertEqual(ir.resolve_for_build(root)[2],{'g::p'})
            (g/'system-prompt.txt').write_text('changed prompt')
            with self.assertRaises(ValueError):ir.resolve_for_build(root)


if __name__=='__main__':unittest.main()
