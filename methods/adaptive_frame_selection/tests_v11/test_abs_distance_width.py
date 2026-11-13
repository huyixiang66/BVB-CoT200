import copy
import unittest
from bvb_v11.abs_distance_width import validate_choices


class WidthTests(unittest.TestCase):
    def setUp(self):
        self.questions=[dict(id='1',objects=['chair','table'],roles=dict(first='chair',second='table'))]
        self.catalog=[dict(candidate_id='r',category='oven')]
        self.result=dict(choices=[dict(question_id='1',candidate_id='r',reference_role='third',
            distinct_from_question_objects=True,dimension='width',dimension_description='front left to right',
            dimension_stability='stable',stability_reason='standard cabinet module',size_m=.595,
            size_range_m=[.59,.6],reason='supported appliance subtype',uncertainty='estimated')])

    def test_accepts_stable_width(self):
        self.assertEqual(validate_choices(self.result,self.questions,self.catalog)[0]['dimension'],'width')

    def test_rejects_other_dimensions_even_when_stable(self):
        for dimension in ['height','length','diameter','depth','diagonal']:
            with self.subTest(dimension=dimension):
                r=copy.deepcopy(self.result);r['choices'][0]['dimension']=dimension
                with self.assertRaises(ValueError):validate_choices(r,self.questions,self.catalog)

    def test_unstable_width_is_not_eligible(self):
        self.result['choices'][0]['dimension_stability']='uncertain'
        with self.assertRaises(ValueError):validate_choices(self.result,self.questions,self.catalog)


if __name__=='__main__':unittest.main()
