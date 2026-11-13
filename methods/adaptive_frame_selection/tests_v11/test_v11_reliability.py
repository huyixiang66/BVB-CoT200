import copy,unittest
from bvb_v11.coverage_check import validate
from bvb_v11.reliability import validate_instances,validate_views
class Coverage(unittest.TestCase):
 def setUp(self):
  self.d={'question':{'objects':['chair','table']},'colors':{'chair':[1,0,0,1],'table':[0,1,0,1]},'groups':[{'group_id':'g1','primary_objects':['chair'],'timestamps_s':[1]},{'group_id':'g2','primary_objects':['table'],'timestamps_s':[2]}]}
  self.rs=[dict(name=c,role='target',category=c,instance_id=c+'1',dimensions=[1,1,1],color=self.d['colors'][c],hide_render=False) for c in ['chair','table']]
  self.inv=[dict(group_id='g'+str(i+1),category=c,instance_ids=[c+'1'],evidence_timestamps_s=[i+1],description=c) for i,c in enumerate(['chair','table'])]
 def test_missing_approved_target_blocks(self):
  with self.assertRaisesRegex(ValueError,'missing'):validate(self.d,self.rs[:1],self.inv)
 def test_no_duplicate_colors(self):
  self.d['colors']['table']=[1,0,0,1]
  with self.assertRaisesRegex(ValueError,'unique'):validate(self.d,self.rs,self.inv)
 def test_partial_or_missing_inventory_allowed(self):
  for inventory in [None,[],self.inv[:1],'bad json']:
   self.assertTrue(validate(self.d,self.rs,inventory)['valid'])
 def test_false_positive_group_can_be_unmapped(self):
  self.inv[0]['instance_ids']=[]
  self.assertTrue(validate(self.d,self.rs,self.inv)['valid'])
 def test_additional_instance_without_group_allowed(self):
  self.rs.append(dict(self.rs[0],instance_id='chair2'))
  self.assertEqual(validate(self.d,self.rs,self.inv)['instances_by_category']['chair'],['chair1','chair2'])
 def test_hidden_or_invalid_geometry_still_blocks(self):
  for updates in [dict(hide_render=True),dict(dimensions=[0,0,0]),dict(dimensions=[float('nan'),1,1])]:
   rs=copy.deepcopy(self.rs);rs[0].update(updates)
   with self.assertRaises(ValueError):validate(self.d,rs,self.inv)
 def test_multiple_groups_same_instance_allowed(self):
  self.d['groups'].append(dict(group_id='g3',primary_objects=['chair'],timestamps_s=[3]))
  self.inv.append(dict(self.inv[0],group_id='g3',evidence_timestamps_s=[3]))
  self.assertTrue(validate(self.d,self.rs,self.inv)['valid'])
 def test_qa_cannot_skip_instance(self):
  with self.assertRaises(ValueError):validate_instances({'instance_consideration':[dict(category='chair',considered_count=1,unresolved_count=0)]},{'chair':['c1','c2']})
 def test_exactly_three_complementary_views(self):
  a={'view_ids':['top','oblique-000','oblique-090'],'coverage':[{'category':'chair','visible_in':['top']} ]}
  validate_views(a,dict.fromkeys(a['view_ids']),['chair'])
  a['view_ids']=['top','oblique-000','elevated-000']
  with self.assertRaises(ValueError):validate_views(a,dict.fromkeys(a['view_ids']),['chair'])
if __name__=='__main__':unittest.main()
