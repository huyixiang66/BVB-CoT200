import copy
import tempfile
import unittest
from pathlib import Path
from .coverage import apply_coverage, admitted, eligible_ids, comparison_key, validate_result
from .semantic_review import write

def group(gid, status, obj='fridge'):
    return dict(group_id=gid, object=obj, selected=[{}], semantic_review=dict(review_id=gid,status=status))

def selection(groups, objects=None):
    return dict(semantic_review={'source_arm':'roi_robust'}, questions={'q':dict(
        question={'objects':objects or ['fridge']}, summary={}, arms={
            'roi_robust':copy.deepcopy(groups),
            'semantic_accepted':[g for g in groups if g['semantic_review']['status']=='accept']},
        semantic_excluded=[g for g in groups if g['semantic_review']['status']!='accept'])})

class CoverageTests(unittest.TestCase):
    def test_single_rejected_or_uncertain_is_kept_without_relabel(self):
        for status in ['reject','uncertain']:
            with tempfile.TemporaryDirectory() as tmp:
                q=apply_coverage(selection([group('a',status)]),tmp)['questions']['q']
                r=q['arms']['semantic_selected'][0]['semantic_review']
                self.assertTrue(q['coverage_complete']);self.assertTrue(admitted(r))
                self.assertEqual(r['status'],status)
                self.assertEqual(r['admission']['mode'],'only_candidate')

    def test_all_accepts_kept_without_extra_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            q=apply_coverage(selection([group('a','accept'),group('b','accept'),group('c','uncertain')]),tmp)['questions']['q']
            self.assertEqual([g['group_id'] for g in q['arms']['semantic_selected']],['a','b'])
            self.assertTrue(q['coverage_complete'])

    def test_unfinished_comparison_and_no_candidates_do_not_invent_winners(self):
        with tempfile.TemporaryDirectory() as tmp:
            q=apply_coverage(selection([group('a','uncertain'),group('b','reject')],['fridge','bowl']),tmp)['questions']['q']
            self.assertEqual(q['coverage_missing_categories'],['fridge','bowl'])
            self.assertEqual(q['arms']['semantic_selected'],[])

    def test_comparison_winner_admitted_and_tampering_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            gs=[group('a','uncertain'),group('b','uncertain'),group('c','reject')]
            ids=eligible_ids(gs);self.assertEqual(ids,['a','b'])
            p=Path(tmp)/'coverage'/comparison_key('fridge',ids)/'result.json'
            r=dict(selected_id='b',confidence='low',reason='best available',uncertainty='not verified',
                ranking=[dict(candidate_id='b',rank=1,reason='better'),dict(candidate_id='a',rank=2,reason='worse')])
            write(p,{**r,'status':'compared','request_hash':'h'});write(p.parent/'input.json',{'request_hash':'h'})
            q=apply_coverage(selection(gs),tmp)['questions']['q'];g=q['arms']['semantic_selected'][0]
            self.assertEqual(g['group_id'],'b');self.assertTrue(admitted(g['semantic_review']))
            self.assertEqual(g['semantic_review']['status'],'uncertain')
            write(p,{**r,'status':'compared','request_hash':'changed'})
            self.assertFalse(admitted(g['semantic_review']))

    def test_all_rejected_candidates_remain_eligible_for_relative_comparison(self):
        self.assertEqual(eligible_ids([group('b','reject'),group('a','reject')]),['a','b'])

    def test_rankings_cannot_invent_or_duplicate_candidates(self):
        r=dict(selected_id='a',confidence='low',reason='reason',uncertainty='',
               ranking=[dict(candidate_id='a',rank=1,reason='reason'),dict(candidate_id='b',rank=2,reason='reason')])
        validate_result(r,['a','b'])
        for bad in [{**r,'selected_id':'b'},{**r,'ranking':r['ranking'][:1]},
                    {**r,'ranking':[r['ranking'][0],r['ranking'][0]]}]:
            with self.assertRaises(ValueError):validate_result(bad,['a','b'])

if __name__=='__main__':unittest.main()
