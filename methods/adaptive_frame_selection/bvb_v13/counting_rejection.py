"""Counting-only admission: never rescue explicitly rejected clusters for coverage."""
import copy
from .counting import TYPES

def apply(selection):
    data=copy.deepcopy(selection)
    for row in data['questions'].values():
        if row['question']['question_type'] not in TYPES:
            raise ValueError('Empty admission policy is counting-only')
        row['retention_policy']=dict(category='multiple',reason='Counting retains distinct instances; zero candidates triggers video recovery.')
        groups=row['arms']['semantic_accepted']+row['semantic_excluded']
        byid={g['group_id']:g for g in groups}
        groups=[byid[g['group_id']] for g in row['arms']['roi_robust']]
        if any(g['semantic_review']['status'] not in {'accept','reject','uncertain'} for g in groups):
            raise ValueError('Unfinished semantic review cannot become an empty result')
        selected=[];excluded=[]
        for g in groups:
            keep=g['semantic_review']['status'] in {'accept','uncertain'}
            g['semantic_review']['admission']=dict(selected=keep,
                mode='multiple_candidates' if keep else 'not_selected',object=g['object'],
                candidate_count=sum(x['object']==g['object'] for x in groups),
                reason='Counting: no singleton or all-rejected forced retention; uncertain remains uncertain.')
            (selected if keep else excluded).append(g)
        missing=sorted(set(row['question']['objects'])-{o for g in selected for o in g.get('objects',[g['object']])})
        row['arms']['semantic_selected']=selected;row['coverage_excluded']=excluded
        row.update(coverage_missing_categories=missing,coverage_complete=not missing,
                   coverage_blockers={o:'no_candidates' for o in missing},counting_allow_all_rejected=True)
        row['summary']['semantic_selected']=dict(groups=len(selected),image_slots=sum(len(g['selected']) for g in selected))
    data['semantic_review'].update(selection_arm='semantic_selected',
        coverage_policy='counting_allow_all_rejected_then_full_video_fallback')
    return data

def install_review_policy():
    """Override the counting runner's coverage layer without editing frozen V10."""
    from pathlib import Path
    from bvb_v6 import coverage,semantic_review
    def prepare(root):
        root=Path(root)
        semantic_review.write(root/'coverage/manifest.json',dict(comparisons=[]))
        semantic_review.write(root/'coverage-policy.json',dict(enabled=True,policy='counting_allow_all_rejected'))
        semantic_review.export(root)
    coverage.prepare=prepare
    coverage.apply_coverage=lambda selection,root:apply(selection)
