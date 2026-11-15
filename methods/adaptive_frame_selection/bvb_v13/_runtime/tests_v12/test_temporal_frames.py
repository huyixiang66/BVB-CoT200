import copy,unittest
from bvb_v13.temporal_frames import select
from bvb_v13.scene_spec import validate
from bvb_v13.scene_evidence import pack

class TemporalFrames(unittest.TestCase):
    def packet(self,times):
        return dict(question=dict(objects=['chair']),frames=[dict(timestamp_s=t,path=str(i),sha256=str(i),groups=['g0'],observations=[dict(box_xyxy=[i,0,i+1,1])]) for i,t in enumerate(times)],groups=[dict(group_id='g0',timestamps_s=times[:])])
    def test_boundary_and_last_kept_not_transitive(self):
        p=self.packet([0,.5,1,1.5,2]);original=copy.deepcopy(p)
        d,r=select(p)
        self.assertEqual(r['kept_source_packet_indices'],[0,2,4]);self.assertEqual(p,original)
        self.assertEqual([f['timestamp_s'] for f in d['frames']],[0,1,2])
        self.assertEqual(select(self.packet([0,.4,.8]))[1]['kept_source_packet_indices'],[0,2])
        self.assertEqual(select(self.packet([0,.50001]))[1]['after'],2)
    def test_sorted_stable_and_no_observation_transfer(self):
        p=self.packet([2,0,.5,1,1]);d,r=select(p)
        self.assertEqual(r['kept_source_packet_indices'],[1,3,0])
        self.assertEqual(d['frames'][0]['observations'],p['frames'][1]['observations'])
        self.assertEqual(d['groups'][0]['timestamps_s'],[0,1,2])
        self.assertEqual([f[0] for f in pack(d)['frames']],[0,1,2])
    def test_group_with_no_remaining_frame_is_explicit(self):
        p=self.packet([0,.2]);p['groups'].append(dict(group_id='g1',timestamps_s=[.2]));p['frames'][1]['groups']=['g1']
        d,r=select(p)
        self.assertEqual(d['frames'][0]['groups'],['g0'])
        self.assertIn('soft evidence only',d['groups'][1]['supplied_image_status'])
        self.assertEqual(r['groups'][1]['after'],0)
    def test_selected_frame_indices_resolve_to_selected_times(self):
        p=self.packet([0,.5,1]);d,r=select(p)
        spec=dict(v=2,o=[[0,[0,0,1],[1,1,1]]],a=[['f',[0,0,0],[3,3,.1],[1]]])
        self.assertEqual(validate(spec,d)['architecture'][0]['times'],[1])
    def test_invalid_timestamps_and_sparse_frames(self):
        for t in [True,-1,float('nan'),float('inf'),'0']:
            with self.assertRaises(ValueError):select(self.packet([t]))
        d,r=select(self.packet([0,1,2]));self.assertEqual(r['removed'],0)
        d,r=select(self.packet([]));self.assertEqual(r['after'],0)

if __name__=='__main__':unittest.main()
