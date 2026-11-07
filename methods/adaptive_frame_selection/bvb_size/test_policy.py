import unittest
from bvb_size.policy import choose_reference, load_priors, needs_detail, reference_prompt
from bvb_size.evaluation import vsi_mean_relative_accuracy


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.priors=load_priors()
        self.target={'id':'bowl','category':'bowl','extent':[.25,.25,.1],'verified':True}
        self.pan={'id':'pan','category':'pan','extent':[.4,.3,.1],'verified':True}

    def test_reliable_target_does_not_request_reference(self):
        target=dict(self.target, category='printer')
        self.assertEqual(choose_reference(target,[self.pan],self.priors),(None,'target_prior_reliable'))

    def test_excluded_and_unverified_candidates_are_not_selected(self):
        printer=dict(self.pan,id='printer',category='printer')
        kettle=dict(self.pan,id='kettle',category='kettle')
        unverified=dict(self.pan,verified=False)
        self.assertEqual(choose_reference(self.target,[printer,kettle,unverified],self.priors),(None,'no_eligible_reference'))
        self.assertEqual(choose_reference(self.target,[printer,self.pan],self.priors)[0]['id'],'pan')

    def test_size_matching_and_unknown_categories(self):
        giant=dict(self.pan,id='large',extent=[5,3,2])
        self.assertEqual(choose_reference(self.target,[giant,self.pan],self.priors)[0]['id'],'pan')
        self.assertEqual(choose_reference(dict(self.target,category='unknown'),[self.pan],self.priors)[1],'unknown_target_category')

    def test_view_trigger_and_prompt(self):
        self.assertTrue(needs_detail(150,.9))
        self.assertTrue(needs_detail(500,.7))
        self.assertFalse(needs_detail(500,.9))
        self.assertIn('right is the pan',reference_prompt('Size?', 'bowl', 'pan'))
        self.assertNotIn('reference object',reference_prompt('Size?', 'bed', None))

    def test_vsi_metric(self):
        self.assertEqual(vsi_mean_relative_accuracy(29,29),1)
        self.assertEqual(vsi_mean_relative_accuracy(5,29),0)
        self.assertAlmostEqual(vsi_mean_relative_accuracy(20,29),5/11)
        with self.assertRaises(ValueError): vsi_mean_relative_accuracy(1,0)

if __name__=='__main__':
    unittest.main()
