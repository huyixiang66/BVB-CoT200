"""Shared, explicit geometry gates for solved and analytic views.

These are geometric proxies, not guarantees of object recognition. Failed
views remain diagnostics; a high utility score cannot make them pass.
"""
import math

MARGIN = .07
MIN_RELATION = .55
MIN_SHORT_SIDE = 10 / 512


def relation_retention(camera, first, second):
    delta = [(second['box']['min'][i] + second['box']['max'][i]
              - first['box']['min'][i] - first['box']['max'][i]) / 2
             for i in range(3)]
    length = math.sqrt(sum(v*v for v in delta))
    if length < 1e-8:
        return 0.
    alignment = sum(camera['forward'][i]*delta[i]/length for i in range(3))
    return math.sqrt(max(0., 1.-alignment*alignment))


def assess_view(camera, required, pairs=(), min_area=.003, min_visible=.20,
                margin=MARGIN, min_relation=MIN_RELATION):
    failures = []
    violation = 0.
    metrics = camera.get('instance_metrics', {})

    def check(ok, reason, deficit=1.):
        nonlocal violation
        if not ok:
            failures.append(reason)
            violation += max(0., deficit)

    check(bool(required), 'no_required_objects')
    for item in required:
        key = item['key']
        metric = metrics.get(key)
        if metric is None:
            check(False, key+':missing_metrics')
            continue
        edge = metric['edge_margin']
        area = metric['area_fraction']
        visible = metric['visible_fraction']
        box = metric['projected_box']
        short_side = min(box[2]-box[0], box[3]-box[1])
        check(edge >= margin-1e-5, key+':cropped_or_behind',
              max(0., margin-edge)/max(margin, .01))
        check(area >= min_area, key+':too_small',
              max(0., 1.-area/min_area))
        check(short_side >= MIN_SHORT_SIDE, key+':short_side_too_small',
              max(0., 1.-short_side/MIN_SHORT_SIDE))
        check(visible >= min_visible, key+':occluded',
              max(0., 1.-visible/min_visible) if min_visible else 0.)
        silhouette = metric.get('silhouette')
        if silhouette is not None:
            check(silhouette['clear'], key+':silhouette_failed',
                  max(.1, 1.-silhouette['silhouette_visibility_fraction']/.55))
    relations = []
    for a, b in pairs:
        left = [item for item in required if item['semantic']==a]
        right = [item for item in required if item['semantic']==b]
        check(bool(left and right), a+' -> '+b+':missing_endpoint')
        for first in left:
            for second in right:
                retention = relation_retention(camera, first, second)
                relations.append(dict(first=first['key'], second=second['key'],
                                      projection_retention=retention))
                check(retention >= min_relation, first['key']+' -> '+second['key']+':compressed',
                      max(0., 1.-retention/min_relation))
    return dict(passed=not failures, failures=failures, violation=violation,
                required_keys=[item['key'] for item in required],
                pairs=[list(pair) for pair in pairs], relations=relations,
                thresholds=dict(margin=margin, min_area=min_area,
                                min_visible=min_visible, min_short_side=MIN_SHORT_SIDE,
                                min_relation=min_relation))
