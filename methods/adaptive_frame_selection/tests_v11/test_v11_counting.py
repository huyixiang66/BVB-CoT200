import unittest
from bvb_v11.counting import validate_answer

class Counting(unittest.TestCase):
    def test_multi_category_total(self):
        validate_answer(dict(answer=7,counts_by_category=[dict(category='chair',count=5),dict(category='table',count=2)],sufficient_evidence=True),['chair','table'])
    def test_total_must_match(self):
        with self.assertRaises(ValueError):
            validate_answer(dict(answer=3,counts_by_category=[dict(category='chair',count=5)],sufficient_evidence=True),['chair'])
    def test_no_fractional_boolean_or_missing_category(self):
        for value in [1.5,True,-1]:
            with self.assertRaises(ValueError):
                validate_answer(dict(answer=value,counts_by_category=[dict(category='chair',count=value)],sufficient_evidence=True),['chair'])
        with self.assertRaises(ValueError):
            validate_answer(dict(answer=3,counts_by_category=[dict(category='chair',count=3)],sufficient_evidence=True),['chair','table'])

if __name__=='__main__':unittest.main()
