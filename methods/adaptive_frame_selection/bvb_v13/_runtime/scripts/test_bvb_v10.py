"""Meaningful scope regression checks; no API requests or GPU work."""
import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bvb_v10.contract import validate,validate_architecture
from bvb_v10.prompt import build

class ContractTests(unittest.TestCase):
    def setUp(self):
        self.q=dict(id=1,question='Where is the kettle?',objects=['kettle'])
        self.t=dict(name='kettle body',role='target',category='kettle',instance_id='k1')
        self.c=dict(name='counter',role='context',relation='support',supports='["k1"]',color=[1,1,1,1],vertices=8,faces=6,modifiers=[])
        self.f=dict(name='floor',role='structure',category='floor',color=[1,1,1,1],vertices=8,faces=6,modifiers=[])
    def test_target_counter_floor(self):
        r=validate([self.t,self.c,self.f],self.q,{'kettle':1});self.assertEqual(r['context_blocks'],1);self.assertFalse(r['physical_support_verified'])
    def test_support_chain_rejected(self):
        extra=dict(self.c,name='cabinet',supports='["counter"]')
        with self.assertRaises(ValueError):validate([self.t,self.c,extra],self.q,{'kettle':1})
    def test_proximity_not_support(self):
        self.c['relation']='near'
        with self.assertRaises(ValueError):validate([self.t,self.c],self.q,{})
    def test_direct_contact(self):
        c=dict(self.c,relation='contact',supports=None,contacts='["k1"]')
        self.assertEqual(validate([self.t,c],self.q,{})['context_blocks'],1)
        c['contacts']='["counter"]'
        with self.assertRaises(ValueError):validate([self.t,c],self.q,{})
    def test_contact_requires_own_references(self):
        with self.assertRaises(ValueError):validate([self.t,dict(self.c,relation='contact')],self.q,{})
    def test_architecture_inventory(self):
        w1=dict(self.f,name='wall1',category='wall');w2=dict(self.f,name='wall2',category='wall')
        records=[self.f,w1,w2]
        inventory=[dict(id='floor',kind='floor',mesh_names=['floor'],evidence_times_s=[.5]),
                   dict(id='opening',kind='opening',mesh_names=['wall1','wall2'],evidence_times_s=[1.])]
        r=validate_architecture(inventory,records,[.5,1.]);self.assertFalse(r['visual_completeness_verified'])
        inventory[1]['mesh_names']=['wall1']
        with self.assertRaises(ValueError):validate_architecture(inventory,records,[.5,1.])
    def test_inventory_missing_geometry_or_evidence(self):
        inventory=[dict(id='floor',kind='floor',mesh_names=['floor'],evidence_times_s=[.5])]
        with self.assertRaises(ValueError):validate_architecture(inventory,[self.f],[1.])
        with self.assertRaises(ValueError):validate_architecture(inventory,[self.f,dict(self.f,name='other floor')],[.5])
    def test_unrelated_target(self):
        self.t['category']='television'
        with self.assertRaises(ValueError):validate([self.t],self.q,{})
    def test_no_roof_or_solid_opening(self):
        for label in ['roof','ceiling','door_opening']:
            with self.subTest(label=label),self.assertRaises(ValueError):validate([self.t,dict(self.f,category=label)],self.q,{})
    def test_white_cuboid(self):
        for change in [{'color':[.4,.4,.4,1]},{'vertices':20},{'modifiers':['BEVEL']}]:
            with self.subTest(change=change),self.assertRaises(ValueError):validate([self.t,dict(self.c,**change)],self.q,{})
    def test_cap_and_multiple(self):
        other=dict(self.t,name='second kettle',instance_id='k2')
        with self.assertRaises(ValueError):validate([self.t,other],self.q,{'kettle':1})
        self.assertEqual(validate([self.t,other],self.q,{'kettle':None})['instance_count'],2)
    def test_parts_not_instances(self):
        self.assertEqual(validate([self.t,dict(self.t,name='handle')],self.q,{'kettle':1})['instance_count'],1)
    def test_absence_not_fabricated(self):
        self.assertEqual(validate([self.f],self.q,{'kettle':1})['missing_categories'],['kettle'])
    def test_prompt_contract(self):
        p=build(self.q,{'kettle':1},{},Path('/tmp/example'),'camera')
        for s in ['DIRECTLY physically supporting','No recursive support chains','Door openings are EMPTY SPACE','ONE coherent scene','All meshes are static','DIRECTLY TOUCHING','ARCHITECTURE IS MANDATORY','architecture_inventory']:
            self.assertIn(s,p)
        self.assertNotIn('ALL other visible objects should be represented',p)

if __name__=='__main__':unittest.main()
