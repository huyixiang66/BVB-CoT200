"""Bounded continuous camera optimization with geometry-only evidence.

The small set of starting points initializes Powell optimization; it is not a
fixed view catalogue. Each solve moves location and horizontal look-at jointly.
"""
import math
import os
import sys

site = os.environ.get('CONTINUOUS_SCIPY_SITE')
if site:
    sys.path.append(site)
from scipy.optimize import minimize
from position_fit import from_position
from acceptance import assess_view


def solve_standing(g, q, all_items, finish, safe_location, standing_free, diagnostics=None):
    roles = q['roles']
    stand = next(i for i in all_items if i['semantic'] == roles['standing'][0])
    facing = next(i for i in all_items if i['semantic'] == roles['facing'][0])
    query = next(i for i in all_items if i['semantic'] == roles['query'][0])
    box, face = stand['box'], g.center(facing['box'])
    centre = g.center(box)
    heading = g.unit_delta((centre[0],centre[1],0), (face[0],face[1],0))
    floor_z = g.floor['max'][2]
    bounds = [(max(g.floor['min'][i]+.05,box['min'][i]-.7),
               min(g.floor['max'][i]-.05,box['max'][i]+.7)) for i in (0,1)]
    if any(a >= b for a,b in bounds):
        return None, 0
    starts = [[max(bounds[i][0],min(bounds[i][1],centre[i]+direction[i]*
                   ((box['max'][i]-box['min'][i])/2+.45))) for i in (0,1)]
              for direction in (heading,(-heading[1],heading[0],0),(-heading[0],-heading[1],0))]
    best = None
    best_failed = None
    evaluations = 0

    def objective(x):
        nonlocal best, best_failed, evaluations
        evaluations += 1
        loc = [float(x[0]),float(x[1]),floor_z+1.55]
        nearest = [max(box['min'][i],min(box['max'][i],loc[i])) for i in (0,1)]
        dist = math.dist(loc[:2], nearest)
        if not (.25 <= dist <= .75):
            return 100.+abs(dist-.5)*20
        if not safe_location(loc) or not standing_free(loc):
            return 1000.
        current = g.unit_delta((loc[0],loc[1],0),(face[0],face[1],0))
        if g.dot(current,heading) < math.cos(math.radians(25)):
            return 100.+25*(1.-g.dot(current,heading))
        cam = from_position([facing['box']],loc,target=face,min_lens=18.,max_lens=35.,margin=.07)
        if cam is None:
            return 100.
        cam.update(id='standing_continuous',method='continuous_standing_reference',ortho_scale=None)
        finish(cam,[facing,query],all_items,'standing_reference')
        cam['joint_acceptance'] = cam['acceptance']
        cam['acceptance'] = assess_view(cam, [facing], min_area=g.MIN_AREA,
                                        min_visible=g.MIN_VISIBLE)
        cam['metadata'].update(standing_distance=dist, frame_keys=[facing['key']],
            evidence_scope='standing_orientation_reference',
            projection_check_keys=[facing['key'],query['key']],
            heading_deviation_deg=math.degrees(math.acos(max(-1.,min(1.,g.dot(current,heading))))))
        fm=cam['instance_metrics'][facing['key']]; qm=cam['instance_metrics'][query['key']]
        loss=3*(1-fm['visible_fraction'])+2*(1-qm['visible_fraction'])
        gate = cam['acceptance']
        if gate['passed']:
            if best is None or loss < best[0]:
                best=(loss,cam)
        elif best_failed is None or (gate['violation'], loss) < best_failed[:2]:
            best_failed=(gate['violation'],loss,cam)
        return loss + 20.*gate['violation']

    for start in starts:
        minimize(objective,start,method='Powell',bounds=bounds,options={'maxfev':120,'xtol':.02,'ftol':.01})
    if best:
        best[1]['metadata'].update(optimizer='Powell', evaluations=evaluations, loss=best[0])
    elif best_failed is not None and diagnostics is not None:
        diagnostics.append(best_failed[2])
    return (best[1] if best else None), evaluations


def solve(g, required, all_items, boxes, role, pairs, prefix, finish, safe_location, prior_views=()):
    target_box = g.union_box(boxes)
    target_center = g.center(target_box)
    spans = [max(.1, target_box['max'][i]-target_box['min'][i]) for i in range(3)]
    floor = g.floor
    bounds = [(floor['min'][i]+.05, floor['max'][i]-.05) for i in (0, 1)]
    bounds += [(floor['max'][2]+.25, floor['max'][2]+min(5., 2*g.room_span-.05))]
    bounds += [(-.25*spans[i], .25*spans[i]) for i in (0, 1)]
    center = [(lo+hi)/2 for lo,hi in bounds[:2]]
    height = min(bounds[2][1], max(bounds[2][0], floor['max'][2]+2.2))
    # Analytic framing gives two task-derived initial poses. Powell may move
    # freely afterwards; these are not a finite output camera catalogue.
    azimuth = 90. if spans[0]>=spans[1] else 0.
    for a,b in pairs:
        first=next((i for i in required if i['semantic']==a),None)
        second=next((i for i in required if i['semantic']==b),None)
        if first and second:
            ca,cb=g.center(first['box']),g.center(second['box'])
            if math.hypot(cb[0]-ca[0],cb[1]-ca[1])>1e-8:
                azimuth=math.degrees(math.atan2(cb[1]-ca[1],cb[0]-ca[0]))+90.
                break
    starts=[]
    for angle in (azimuth,azimuth+180.):
        fitted=g.fit_camera(boxes,angle,35.,margin=.07,lens_mm=35.)
        location=[max(bounds[i][0],min(bounds[i][1],fitted['location'][i])) for i in range(3)]
        starts.append(location+[0.,0.])
    starts.append(center+[height,0.,0.])
    best = None
    best_failed = None
    start_best = None
    solutions = []
    evaluations = 0
    valid = 0

    def objective(x):
        nonlocal best, best_failed, start_best, evaluations, valid
        evaluations += 1
        location = [float(v) for v in x[:3]]
        target = [target_center[0]+float(x[3]), target_center[1]+float(x[4]), target_center[2]]
        camera = from_position(boxes, location, target=target, min_lens=8., max_lens=35., margin=.07)
        if camera is None:
            return 1e5
        if not safe_location(location):
            return 1000. + max(0.,24.-camera['lens_mm'])
        if camera['lens_mm'] < 24.:
            return 100. + 10*(24.-camera['lens_mm'])
        camera.update(id=prefix+'_continuous', method='continuous_powell_camera_solve', ortho_scale=None)
        finish(camera, required, all_items, role, pairs)
        metrics = [camera['instance_metrics'][i['key']] for i in required]
        visible = [m['visible_fraction'] for m in metrics]
        # AABB screen area times the surface-ray fraction is only a proxy for
        # visible pixels, not a rendered segmentation silhouette.
        pixel_proxy = [m['area_fraction']*m['visible_fraction'] for m in metrics]
        separation = camera['relation_check']['projection_retention'] if pairs else 1.
        overlap = 0.
        for item, metric in zip(required, metrics):
            a = metric['projected_box']
            area = max(1e-9, (a[2]-a[0])*(a[3]-a[1]))
            for other in g.instances.get(item['semantic'], []):
                if other['key'] == item['key']:
                    continue
                points = [g.project(p, camera) for p in g.bbox_corners(other['box'])]
                if any(p[2] <= .03 for p in points):
                    continue
                b = [min(p[0] for p in points), min(p[1] for p in points),
                     max(p[0] for p in points), max(p[1] for p in points)]
                intersection = max(0., min(a[2],b[2])-max(a[0],b[0])) * max(0., min(a[3],b[3])-max(a[1],b[1]))
                overlap = max(overlap, min(1., intersection/area))
        context_penalty = 0.
        for previous in prior_views:
            if previous.get('ortho_scale') is not None:
                continue
            alignment = g.dot(camera['forward'], previous['forward'])
            distance = math.dist(camera['location'], previous['location'])
            # Similar orientation aids correspondence, but near-identical camera
            # positions and headings provide little new evidence.
            context_penalty += .4*max(0., .5-alignment)
            context_penalty += .7*max(0., 1.-distance/.8)*max(0., (alignment-.90)/.10)
            anchors = {a for a,b in pairs}
            for item, metric in zip(required, metrics):
                if item['semantic'] not in anchors:
                    continue
                prior_metric = previous.get('instance_metrics', {}).get(item['key'])
                if prior_metric and prior_metric['area_fraction'] > 1e-6 and metric['area_fraction'] > 1e-6:
                    context_penalty += .12*min(3., abs(math.log(metric['area_fraction']/prior_metric['area_fraction'])))
        loss = (4.*(1.-min(visible, default=0.))
                + 3.*(1.-min(1., min(pixel_proxy, default=0.)/.003))
                + 1.5*(1.-separation) + .6*overlap + context_penalty)
        # A preference, not a hard tilt: overhead and lower views may win.
        elevation = math.degrees(math.asin(max(-1., min(1., -camera['forward'][2]))))
        loss += .5 * (1.-max(0., 1.-abs(elevation-35.)/55.))
        if role == 'single_size_context':
            dimensions = [required[0]['box']['max'][i]-required[0]['box']['min'][i] for i in range(3)]
            axis = max(range(3),key=lambda i: dimensions[i])
            axis_retention = math.sqrt(max(0., 1.-camera['forward'][axis]**2))
            loss += 5.*(1.-min(1., pixel_proxy[0]/.08)) + 2.*(1.-axis_retention)
        valid += 1
        gate = camera['acceptance']
        if gate['passed']:
            if best is None or loss < best[0]:
                best = (loss, camera)
            if start_best is None or loss < start_best[0]:
                start_best = (loss, camera)
        elif best_failed is None or (gate['violation'], loss) < best_failed[:2]:
            best_failed = (gate['violation'], loss, camera)
        return loss + 20.*gate['violation']

    for start in starts:
        start_best = None
        minimize(objective, start, method='Powell', bounds=bounds,
                 options={'maxfev': 180, 'xtol': .025, 'ftol': .01})
        if start_best is not None:
            solutions.append(start_best)
    if best is None:
        if best_failed is not None:
            best_failed[2]['metadata'].update(optimizer='Powell', evaluations=evaluations,
                                              rejection='no geometry-accepted solution')
        return None, [best_failed[2]] if best_failed is not None else []
    camera = best[1]
    camera['metadata'].update(optimizer='Powell', loss=best[0],
        evaluations=evaluations, valid_evaluations=valid, starts=len(starts),
        pixel_proxy='AABB area times visible-surface sample fraction; not silhouette',
        distractor_proxy='maximum projected AABB overlap with same-semantic instances',
        previous_views=len(prior_views))
    accepted = []
    seen = set()
    for loss, solution in sorted(solutions, key=lambda item: item[0]):
        key = tuple(round(v, 4) for v in solution['location']+solution['target'])
        if key not in seen:
            seen.add(key)
            solution['metadata'].update(optimizer='Powell', loss=loss,
                                        evaluations=evaluations)
            accepted.append(solution)
    return camera, accepted
