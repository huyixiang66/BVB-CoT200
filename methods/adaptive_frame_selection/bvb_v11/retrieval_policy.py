"""Full-video retrieval is disabled by the user's current inference policy."""
import copy

FULL_VIDEO_RETRIEVAL_ENABLED=False
POLICY='full-video-retrieval-disabled-20261113'

def admitted_only(data):
    """Keep existing positive evidence; never promote uncertain groups or infer zero."""
    result=copy.deepcopy(data)
    for row in result['questions'].values():
        selected=row['arms'].get('semantic_selected',[])
        kept=[g for g in selected if g.get('semantic_review',{}).get('status')=='accept']
        row['arms']['semantic_selected']=kept
        missing=sorted(set(row['question']['objects'])-{g['object'] for g in kept})
        row.update(coverage_missing_categories=missing,missing_objects=missing,coverage_complete=not missing,
            coverage_blockers={c:'no_confirmed_evidence_retrieval_disabled' for c in missing},
            semantic_recovery=dict(policy=POLICY,enabled=False,search_performed=False,completed=False,unresolved_categories=missing))
        row.setdefault('summary',{})['semantic_selected']=dict(groups=len(kept),image_slots=sum(len(g['selected']) for g in kept))
    result.setdefault('semantic_review',{}).update(required=True,selection_arm='semantic_selected',coverage_required=True,recovery_policy=POLICY)
    return result
