"""Select only reported structural-wall blockers for an unresolved view."""


def wall_retry_candidates(mode, common, pair):
    if mode != 'unresolved':
        return []
    attempts = [common] + ([pair] if pair else [])
    if not any('external_occlusion' in reason for attempt in attempts for reason in attempt['blockers']):
        return []
    return sorted({name for attempt in attempts for view in attempt['views']
                   for name in view.get('wall_occluders', [])})
