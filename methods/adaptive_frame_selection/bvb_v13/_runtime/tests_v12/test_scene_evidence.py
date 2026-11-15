import copy
import unittest
from bvb_v13.scene_evidence import pack


def decode(t):
    return [dict(zip(t['fields'], r)) for r in t['rows']]


class SceneEvidence(unittest.TestCase):
    def fixture(self):
        return dict(question=dict(id=9, scene_name='secret', question='distance?', objects=['chair']),
                    groups=[dict(group_id='long-name', timestamps_s=[1], primary_objects=['chair'],
                                 evidence_role='structure_only', allowed_geometry=['wall'])],
                    frames=[dict(timestamp_s=1, groups=['long-name'], compact_camera=[1.234567, 2, 3],
                                 observations=[dict(object='chair', peak_id='detector-key',
                                                    box_xyxy=[.12345, 2.3987, 10, 20],
                                                    dino_score=.123456, status='uncertain', extra='keep')])])

    def test_geometry_permissions_and_unknown_fields_survive(self):
        p=self.fixture();before=copy.deepcopy(p);x=pack(p)
        self.assertEqual(p,before)
        self.assertEqual(x['frames'][0][3],p['frames'][0]['compact_camera'])
        fields=x['header']['observation_fields']
        o=dict(zip(fields,x['frames'][0][4][0]))
        self.assertEqual(o['status'],'uncertain');self.assertEqual(o['extra'],'keep')
        self.assertEqual(o['dino_score'],.123456);self.assertNotIn('peak_id',o)
        for old,new in zip(p['frames'][0]['observations'][0]['box_xyxy'],o['box_xyxy']):
            self.assertLessEqual(abs(old-new),.05000001)
        g=decode(x['header']['groups'])[0]
        self.assertEqual(g['allowed_geometry'],['wall']);self.assertEqual(g['evidence_role'],'structure_only')
        self.assertNotIn('timestamps_s',g);self.assertEqual(x['frames'][0][2],[g['group_id']])

    def test_unrepresented_timestamps_not_dropped_and_bad_links_rejected(self):
        p=self.fixture();p['groups'][0]['timestamps_s']=[1,2]
        self.assertEqual(decode(pack(p)['header']['groups'])[0]['timestamps_s'],[1,2])
        p['frames'][0]['groups']=['missing']
        with self.assertRaises(ValueError):pack(p)

    def test_anchors_keep_notes_and_uncertainty(self):
        anchors=[dict(group_id='long-name',center=[.123456,2,3],confidence='low',note='not a center'),
                 dict(group_id='long-name',confidence='high',note='different caution')]
        x=pack(self.fixture(),anchors)['anchors'];a=decode(x['soft_surface_anchors'])
        self.assertEqual(a[0]['center'],[.1235,2,3]);self.assertEqual(a[0]['confidence'],'low')
        for old,new in zip(anchors,a):self.assertEqual(x['notes'][new['note_index']],old['note'])

    def test_room_without_observations_or_anchors(self):
        p=self.fixture();p['room_mode']=True;p['question']['objects']=[]
        p['frames'][0].pop('observations');x=pack(p)
        self.assertEqual(x['header']['observation_fields'],[])
        self.assertEqual(x['frames'][0][4],[]);self.assertIsNone(x['anchors'])

if __name__=='__main__':unittest.main()
