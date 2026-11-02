"""Continuous camera solving with shared geometry acceptance and relation groups.

Only role-accepted views enter chosen_readable. Unaccepted best attempts stay
diagnostic; geometry acceptance does not resolve semantic-instance ambiguity.
"""
import json
import math
from pathlib import Path
import sys
import time

START = time.monotonic()
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import geometry_core as g
from relation_layout import comparison_groups, direction_points
from position_fit import from_position, perimeter_positions
from silhouette import make_target_bvh, measure as measure_silhouette
from continuous_fit import solve as continuous_solve
from continuous_fit import solve_standing
from acceptance import assess_view, MIN_RELATION, MIN_SHORT_SIDE

SCENE = g.SCENE
OUT = Path(g.cli_arg('--out', str(HERE / f'geometry_{SCENE}.json')))
MARGIN = .07
AREA = .0008
VISIBLE = .20
SIZE_VISIBLE = .55
MAX_VIEWS = 6
TARGET_BVH = {}
REJECTED_VIEWS = []
FLOOR_PARTS = [g.bbox(g.bpy.data.objects[name]) for name in g.groups['room']['floor']]
POSITIONS = [p for p in perimeter_positions(g.floor)
             if any(all(box['min'][i]+.04 <= p[i] <= box['max'][i]-.04 for i in (0, 1))
                    for box in FLOOR_PARTS)]


def box_gap(a, b):
    return sum(max(0., max(a['min'][i]-b['max'][i], b['min'][i]-a['max'][i]))**2
               for i in range(3))


def union_span(items):
    if not items:
        return float('inf')
    box = g.union_box([item['box'] for item in items])
    return math.sqrt(sum((box['max'][i]-box['min'][i])**2 for i in range(3)))


def semantic_items(semantic):
    return [dict(item, semantic=semantic) for item in g.instances.get(semantic, [])]


def nearest_to(anchor, semantic):
    choices = semantic_items(semantic)
    return min(choices, key=lambda item: box_gap(anchor['box'], item['box'])) if choices else None


def nearest_pair(left_semantic, right_semantic):
    left, right = semantic_items(left_semantic), semantic_items(right_semantic)
    if not left or not right:
        return None, None
    return min(((a, b) for a in left for b in right),
               key=lambda pair: box_gap(pair[0]['box'], pair[1]['box']))


CURRENT_RESOLUTION = {}


def resolve_question(q):
    """Resolve repeated names to physical instances without using answers."""
    qtype = q['question_type']
    targets = [s for s in q['targets'] if s != '__room__']
    missing = [s for s in targets if s not in g.instances]
    record = {'items': [], 'missing': missing, 'mode': qtype, 'instance_keys': [], 'notes': []}
    if qtype == 'object_counting':
        record['items'] = [dict(item, semantic=s) for s in targets
                           for item in g.instances.get(s, [])]
        record['instance_keys'] = [item['key'] for item in record['items']]
        record['mode'] = 'counting_all_instances'
        return record
    if qtype == 'room_size_estimation':
        record['items'] = [dict(item, semantic='__room__')
                           for item in g.instances['__room__']]
        record['instance_keys'] = [item['key'] for item in record['items']]
        record['mode'] = 'room_floor'
        return record
    if qtype == 'object_abs_distance':
        a, b = q['roles'].get('endpoints', [None, None])[:2]
        if a in g.instances and b in g.instances:
            left, right = nearest_pair(a, b)
            record.update(items=[left, right], mode='nearest_endpoint_pair',
                          instance_keys=[left['key'], right['key']])
        else:
            record['notes'].append('距离端点缺失，使用房间级回退图。')
        return record
    if qtype == 'object_rel_distance':
        anchor_semantic = q['roles'].get('anchor', [None])[0]
        candidates = [s for s in q['roles'].get('candidates', [])
                      if s != anchor_semantic and s in g.instances]
        if anchor_semantic not in g.instances or not candidates:
            record['notes'].append('相对距离锚点或候选缺失，使用房间级回退图。')
            return record
        best = None
        for anchor in semantic_items(anchor_semantic):
            chosen, gaps = [anchor], []
            for semantic in candidates:
                item = nearest_to(anchor, semantic)
                if item is not None:
                    chosen.append(item)
                    gaps.append(box_gap(anchor['box'], item['box']))
            key = (union_span(chosen), sum(gaps), len(chosen))
            if best is None or key < best[0]:
                best = key, chosen
        record.update(items=best[1] if best else [], mode='nearest_candidate_star')
        record['instance_keys'] = [item['key'] for item in record['items']]
        return record
    if qtype == 'object_size_estimation':
        semantic = q['roles'].get('target', [None])[0]
        choices = semantic_items(semantic) if semantic in g.instances else []
        if choices:
            item = max(choices, key=lambda value: math.prod(
                max(0., value['box']['max'][i]-value['box']['min'][i]) for i in range(3)))
            record.update(items=[item], mode='single_size_representative',
                          instance_keys=[item['key']])
        else:
            record['notes'].append('尺寸目标缺失，使用房间级回退图。')
        return record
    if qtype.startswith('object_rel_direction'):
        roles = q['roles']
        standing, facing, query = (roles.get(key, [None])[0]
                                   for key in ('standing', 'facing', 'query'))
        if standing in g.instances and facing in g.instances and query in g.instances:
            triplets = []
            for anchor in semantic_items(standing):
                f, item = nearest_to(anchor, facing), nearest_to(anchor, query)
                if f and item:
                    triplets.append([anchor, f, item])
            if triplets:
                record.update(items=min(triplets, key=union_span),
                              mode='nearest_direction_triplet')
                record['instance_keys'] = [item['key'] for item in record['items']]
                return record
        record['notes'].append('方向题目标缺失，使用房间级回退图。')
    return record


def selected_items(q, semantics=None):
    wanted = set(semantics or [item['semantic'] for item in CURRENT_RESOLUTION['items']])
    return [item for item in CURRENT_RESOLUTION['items'] if item['semantic'] in wanted]


def safe_location(loc):
    inside = all(g.floor['min'][i]+.04<=loc[i]<=g.floor['max'][i]-.04 for i in (0,1))
    if not inside or not (g.floor['max'][2]+.2<=loc[2]<=g.floor['max'][2]+2*g.room_span):
        return False
    if not g.camera_free(loc):
        return False
    origin=g.Vector((loc[0],loc[1],g.floor['max'][2]+.3))
    hit,normal,_,_=g.bvh.ray_cast(origin,g.Vector((0,0,-1)),.7)
    return hit is not None and abs(normal.z)>.5 and abs(hit.z-g.floor['max'][2])<.25


def relation_score(cam, required, pairs=()):
    values = [cam['instance_metrics'][i['key']] for i in required]
    readable = sum(m['readable'] for m in values)
    visible = [min(1,m['visible_fraction']/.35) for m in values]
    area = min((m['area_fraction'] for m in values),default=0)
    retained = []
    for a,b in pairs:
        for i in required:
            for j in required:
                if i['semantic']==a and j['semantic']==b:
                    v=g.unit_delta(g.center(i['box']),g.center(j['box']))
                    retained.append(math.sqrt(max(0,1-g.dot(cam['forward'],v)**2)))
    preserve=sum(retained)/len(retained) if retained else 0
    anchors={a for a,b in pairs}
    anchor_readable=all(cam['instance_metrics'][i['key']]['readable'] for i in required if i['semantic'] in anchors)
    elevation=math.degrees(math.asin(max(-1,min(1,-cam['forward'][2]))))
    recognition=max(0,1-abs(elevation-35)/55)
    axis_quality=0.
    if cam.get('metadata',{}).get('role')=='single_size_context':
        dims=[required[0]['box']['max'][i]-required[0]['box']['min'][i] for i in range(3)]
        axis=max(range(3),key=lambda i:dims[i])
        axis_quality=math.sqrt(max(0,1-cam['forward'][axis]**2))
    board_visibility=cam.get('cutting_board_visibility',{}).get('silhouette_visibility_fraction',0.)
    score=100*anchor_readable+20*readable+min(visible,default=0)+.4*preserve+2*recognition+.4*min(1,math.sqrt(area/.006))+.8*axis_quality+4*board_visibility
    return {'readable_count':readable,'required_count':len(values),
        'complete':readable==len(values),'anchor_readable':anchor_readable,'projection_retention':preserve,
        'min_area':area,'recognition_elevation':elevation,'score':score}


def has_metrics(cam, required):
    return all(item['key'] in cam['instance_metrics'] for item in required)


def finish(cam, required, all_items, role, pairs=()):
    cam['metadata']=dict(cam.get('metadata',{}),role=role,
        required_semantics=list(dict.fromkeys(i['semantic'] for i in required)),
        frame_keys=[i['key'] for i in required])
    cam['instance_metrics']=g.evaluate(cam,all_items)
    for item in required:
        if item['semantic']!='cutting board':
            continue
        if item['key'] not in TARGET_BVH:
            TARGET_BVH[item['key']]=make_target_bvh(item,g.triangles)
        mask=measure_silhouette(cam,item['box'],TARGET_BVH[item['key']],
            g.bvh,g.face_names,item['names'],grid=14)
        clear=(mask['silhouette_samples']>=8 and mask['silhouette_visibility_fraction']>=.55
               and mask['visible_image_fraction']>=.0006
               and min(mask['visible_width_px'],mask['visible_height_px'])>=10)
        mask['clear']=clear
        cam['cutting_board_visibility']=mask
        cam['instance_metrics'][item['key']]['silhouette']=mask
        cam['instance_metrics'][item['key']]['readable'] &= clear
    room_layout = bool(required) and all(item['semantic']=='__room__' for item in required)
    visible_gate = 0. if room_layout else g.MIN_VISIBLE
    cam['relation_check']=relation_score(cam,required,pairs)
    cam['geometry_safe'] = safe_location(cam['location'])
    cam['acceptance'] = assess_view(cam, required, pairs, min_area=g.MIN_AREA,
                                     min_visible=visible_gate)
    if not cam['geometry_safe']:
        cam['acceptance']['passed'] = False
        cam['acceptance']['failures'].append('unsafe_camera_location')
        cam['acceptance']['violation'] += 1.
    cam['relation_check']['complete'] = cam['acceptance']['passed']
    if room_layout:
        cam['metadata']['evidence_scope']='room_footprint_layout'
        for item in required:
            cam['instance_metrics'][item['key']]['readable']=cam['acceptance']['passed']
            cam['instance_metrics'][item['key']]['readability_scope']='room_footprint_layout'
    return cam


def overhead(required, scale=None, ident='top'):
    box=g.union_box([i['box'] for i in required]); target=g.center(box)
    span=max(box['max'][j]-box['min'][j] for j in (0,1))/(1-2*MARGIN)
    return {'id':ident,'method':'relation_group_overhead','location':[target[0],target[1],g.ceiling['max'][2]+.7],
        'target':list(target),'right':[1,0,0],'up':[0,1,0],'forward':[0,0,-1],
        'ortho_scale':scale or span,'lens_mm':None,'elevation_deg':90.,
        'metadata':{'shared_scale':'exact orthographic scale' if scale else 'fitted group scale'}}


def angles(required,pairs):
    result={0.,90.,180.,270.}
    for a,b in pairs:
        for i in required:
            for j in required:
                if i['semantic']==a and j['semantic']==b:
                    v=g.unit_delta(g.center(i['box']),g.center(j['box']))
                    angle=math.degrees(math.atan2(v[1],v[0]))
                    result.update(((angle+90)%360,(angle+270)%360))
    return sorted(result)


def fit_group(required,all_items,role,pairs=(),prefix='group',context=False,prior_views=()):
    boxes=[i['box'] for i in required]
    if context:
        box=g.union_box(boxes); target=g.center(box)
        spans=[max((box['max'][i]-box['min'][i])*1.45,.7 if i<2 else .4) for i in range(3)]
        boxes=[{'min':[target[i]-spans[i]/2 for i in range(3)],
                'max':[target[i]+spans[i]/2 for i in range(3)]}]
    main, attempts = continuous_solve(g,required,all_items,boxes,role,pairs,prefix,
                                     finish,safe_location,prior_views)
    REJECTED_VIEWS.extend(c for c in attempts if not c['acceptance']['passed'])
    return main, [c for c in attempts if c['acceptance']['passed']]


def pair_covered(view, required, pair):
    if (not has_metrics(view, required) or not view.get('geometry_safe')
            or not set(pair).issubset({item['semantic'] for item in required})):
        return False
    return assess_view(view, required, [pair], min_area=g.MIN_AREA,
                       min_visible=g.MIN_VISIBLE)['passed']


def paired_comparisons(q,all_items,prior_views=()):
    available={i['semantic'] for i in all_items}
    qq=dict(q,roles=dict(q.get('roles',{})))
    qq['roles']['candidates']=[s for s in qq['roles'].get('candidates',[]) if s in available]
    groups=comparison_groups(qq,all_items)
    frames=[[i for i in all_items if i['semantic'] in s] for s in groups]
    anchor=q['roles']['anchor'][0]
    pairsets=[[(anchor,s) for s in group if s!=anchor] for group in groups]
    pools=[]
    for n,frame in enumerate(frames):
        _,candidates=fit_group(frame,all_items,f'anchor_comparison_{n+1}',pairsets[n],f'compare{n+1}',prior_views=prior_views)
        valid=[c for c in candidates if c['acceptance']['passed']]
        ordered=sorted(valid,key=lambda c:c['relation_check']['score'],reverse=True)
        low=[c for c in ordered if c['relation_check']['recognition_elevation']<=50]
        pools.append(list({c['id']:c for c in ordered[:8]+low[:8]}.values()))
    if len(pools) >= 2 and all(pools):
        def rank(cams):
            checks=[c['relation_check'] for c in cams]
            ratio=min(r['readable_count']/r['required_count'] for r in checks)
            low=sum(r['recognition_elevation']<=50 for r in checks)
            similar=g.dot(cams[0]['forward'],cams[1]['forward'])>math.cos(math.radians(30))
            return ratio,low,similar,sum(r['score'] for r in checks)
        chosen=list(max(((a,b) for a in pools[0] for b in pools[1]),key=rank))
        for cam in chosen:cam['metadata']['shared_scale']='shared anchor; camera scale can differ'
        return chosen,groups
    return [max(pool,key=lambda c:c['relation_check']['score']) for pool in pools if pool],groups


def standing_free(loc):
    floor_z=g.floor['max'][2]
    for box in g.solid_boxes:
        if box['max'][2]<=floor_z+.20:continue
        if box['min'][2]>=loc[2]:continue
        if all(box['min'][i]-.06<=loc[i]<=box['max'][i]+.06 for i in (0,1)):
            return False
    origin=g.Vector((loc[0],loc[1],floor_z+.14))
    hit,normal,_,_=g.bvh.ray_cast(origin,g.Vector((0,0,-1)),.4)
    return hit is not None and abs(normal.z)>.5 and abs(hit.z-floor_z)<.2


def standing_reference(q,all_items):
    s=q['roles']['standing'][0]; f=q['roles']['facing'][0]; query=q['roles']['query'][0]
    standing=next(i for i in all_items if i['semantic']==s)['box']
    facing=g.center(next(i for i in all_items if i['semantic']==f)['box'])
    required=[i for i in all_items if i['semantic'] in (f,query)]
    center_heading=g.unit_delta((g.center(standing)[0],g.center(standing)[1],0.),(facing[0],facing[1],0.))
    candidates=[]
    for height in (1.45,1.65):
        for point in direction_points(standing,facing,g.floor,height=height):
            loc=point['location']
            if not safe_location(loc) or not standing_free(loc):continue
            heading=g.unit_delta((loc[0],loc[1],0.),(facing[0],facing[1],0.))
            deviation=math.degrees(math.acos(max(-1,min(1,g.dot(heading,center_heading)))))
            if deviation>25:continue
            for pitch in (10.,20.,30.,40.):
                rad=math.radians(pitch)
                forward=[heading[0]*math.cos(rad),heading[1]*math.cos(rad),-math.sin(rad)]
                right,up,_=g.basis([-v for v in forward])
                for lens in (35.,24.):
                    cam={'id':f'standing_{len(candidates):03d}','method':'standing_neighborhood',
                        'location':list(loc),'target':[loc[j]+forward[j]*2 for j in range(3)],
                        'right':right,'up':up,'forward':forward,'lens_mm':lens,
                        'tan_half_fov':36/(2*lens),'ortho_scale':None,
                        'metadata':{'standing_semantic':s,'standing_distance':point['stand_distance'],
                            'heading_deviation_deg':deviation,'pitch_deg':pitch,'height_above_floor':height,
                            'yaw_facing_error_deg':0.,'frame_keys':[]}}
                    # Framing is tested rather than guaranteed for a fixed
                    # standing position: behind-camera targets remain missing.
                    finish(cam,required,all_items,'standing_reference')
                    cam['metadata']['frame_keys']=[]
                    cam['metadata']['projection_check_keys']=[i['key'] for i in required]
                    candidates.append(cam)
    def score(cam):
        check=cam['relation_check']; m=cam['instance_metrics']
        q_vis=max((v['visible_fraction'] for k,v in m.items() if k.startswith(query+'#')),default=0)
        f_vis=max((v['visible_fraction'] for k,v in m.items() if k.startswith(f+'#')),default=0)
        # Include a legitimate facing-reference image even when Q is behind.
        return (check['readable_count'],min(q_vis,f_vis),f_vis,
            -cam['metadata']['heading_deviation_deg'],cam['lens_mm'],
            -cam['metadata']['standing_distance'])
    usable=[c for c in candidates if any(v['visible_fraction']>=VISIBLE for k,v in c['instance_metrics'].items() if k.startswith(f+'#'))]
    return max(usable,key=score) if usable else None,len(candidates)


def top_view(required,role='top'):
    box=g.union_box([i['box'] for i in required]); target=g.center(box)
    span=max(box['max'][i]-box['min'][i] for i in (0,1))/(1-2*MARGIN)
    cam={'id':role,'method':'room_or_count_top',
         'location':[target[0],target[1],g.ceiling['max'][2]+.8],
         'target':list(target),'right':[1.,0.,0.],'up':[0.,1.,0.],
         'forward':[0.,0.,-1.],'ortho_scale':span,'lens_mm':None,'elevation_deg':90.}
    return finish(cam,required,required,role)


def unique_views(views):
    result=[]; seen=set()
    for cam in views:
        key=(tuple(round(v,4) for v in cam['location']),
             tuple(round(v,4) for v in cam.get('target',[])),
             tuple(round(v,4) for v in cam.get('forward',[])),
             None if cam.get('ortho_scale') is None else round(cam['ortho_scale'],4),
             None if cam.get('lens_mm') is None else round(cam['lens_mm'],4))
        if key in seen:continue
        seen.add(key); result.append(cam)
    return result


def candidate_locations(required):
    locations=perimeter_positions(g.floor,heights=(1.45,2.25),inset=.20)
    for height in (1.45,2.25):
        for x in (.2,.5,.8):
            for y in (.2,.5,.8):
                locations.append([g.floor['min'][0]+x*(g.floor['max'][0]-g.floor['min'][0]),
                                  g.floor['min'][1]+y*(g.floor['max'][1]-g.floor['min'][1]),
                                  g.floor['max'][2]+height])
    for center in [g.center(i['box']) for i in required[:6]]:
        for radius in (.75,1.5,2.5):
            for angle in range(0,360,45):
                rad=math.radians(angle)
                locations.append([center[0]+radius*math.cos(rad),
                                  center[1]+radius*math.sin(rad),g.floor['max'][2]+1.65])
    unique=[]; seen=set()
    for location in locations:
        key=tuple(round(v,4) for v in location)
        if key not in seen:
            seen.add(key); unique.append(location)
    return unique


def choose_view(required,min_lens=18.):
    boxes=[i['box'] for i in required]
    target=g.center(g.union_box(boxes)); candidates=[]
    for location in candidate_locations(required):
        if not safe_location(location):continue
        cam=from_position(boxes,location,target=target,min_lens=min_lens,max_lens=35.,margin=MARGIN)
        if cam is not None:
            cam['id']='context'; cam['method']='room_fov_fit'
            candidates.append(finish(cam,required,required,'target_context'))
    if not candidates:return None
    def score(cam):
        metrics=list(cam['instance_metrics'].values())
        visible=[m['visible_fraction'] for m in metrics]
        area=[m['area_fraction'] for m in metrics]
        return (sum(m['readable'] for m in metrics),min(visible,default=0.),
                sum(visible),min(area,default=0.),-cam['elevation_deg'])
    return max(candidates,key=score)


def room_overview():
    room=[dict(item, semantic='__room__') for item in g.instances['__room__']]
    cam,_=fit_group(room,room,'room_perspective',prefix='room_context')
    if cam is not None:
        cam['id']='room_perspective'; cam['metadata']['role']='room_perspective'
        return cam
    return top_view(room,'room_top_fallback')


def fallback_room_views():
    room=[dict(item, semantic='__room__') for item in g.instances['__room__']]
    return [room_overview(),top_view(room,'room_top')]


def build_special(q,items,resolution):
    """Counting, room-size and size views; no relationship camera needed."""
    qtype=q['question_type']; notes=list(resolution['notes'])
    if qtype=='object_counting':
        if not items:return [],notes,'reconstruction_missing_semantic'
        top=top_view(items,'count_top')
        if top['acceptance']['passed']:
            return [top],notes,None
        REJECTED_VIEWS.append(top)
        main,_=fit_group(items,items,'count_group_context',prefix='count_group',context=True)
        views=[main] if main else []
        for index,item in enumerate(items):
            if len(views)>=MAX_VIEWS:break
            if any(v['instance_metrics'][item['key']]['readable'] for v in views):continue
            support,_=fit_group([item],items,'count_instance_context',prefix='count_'+str(index),context=True,prior_views=views)
            if support:views.append(support)
        return views,notes,None
    if qtype=='room_size_estimation':
        room=[dict(item, semantic='__room__') for item in g.instances['__room__']]
        top=top_view(room,'room_top')
        if top['acceptance']['passed']:return [top],notes,None
        REJECTED_VIEWS.append(top)
        return [],notes,'no_accepted_view'
    if qtype=='object_size_estimation':
        g.MIN_AREA=.006; g.MIN_VISIBLE=SIZE_VISIBLE
        if not items:return [],notes,'reconstruction_missing_semantic'
        main,_=fit_group(items,items,'single_size_context',prefix='size',context=True)
        notes.append('Main view includes local scale context; no automatic second context image')
        if main and main['relation_check']['complete']:
            return [main],notes,'complete'
        return [],notes,'no_accepted_view'
    raise RuntimeError('Not a special question type: '+qtype)


records=[]
for q in g.questions:
    qtype=q['question_type']; print('Q',q['qa_id'],qtype,flush=True)
    REJECTED_VIEWS=[]
    CURRENT_RESOLUTION=resolve_question(q)
    items=CURRENT_RESOLUTION['items']
    missing=sorted(set(CURRENT_RESOLUTION['missing']))
    ambiguous=([s for s in q['targets'] if len(g.instances.get(s, [])) > 1]
               if qtype not in {'object_counting','room_size_estimation'} else [])
    g.MIN_AREA=AREA; g.MIN_VISIBLE=VISIBLE
    views=[]; notes=list(CURRENT_RESOLUTION['notes']); reference=None
    special_status=None

    if qtype not in {'object_abs_distance','object_rel_distance'} and not qtype.startswith('object_rel_direction'):
        views,notes,special_status=build_special(q,items,CURRENT_RESOLUTION)
    elif qtype=='object_rel_distance':
        anchor=q['roles']['anchor'][0]
        available={i['semantic'] for i in items}
        pairs=[(anchor,s) for s in q['roles']['candidates'] if s!=anchor and s in available]
        if items and anchor in available:
            layout,_=fit_group(items,items,'layout_overview',pairs,'overview')
            if layout:views.append(layout)
            comparisons,groups=paired_comparisons(q,items,prior_views=views)
            views.extend(comparisons)
            notes.append('Continuous shared-anchor comparison groups: '+str(groups))
            for a,b in pairs:
                if len(views)>=MAX_VIEWS:break
                required=selected_items(q,[a,b])
                if any(pair_covered(v,required,(a,b)) for v in views):continue
                prefix='pair_'+a.replace(' ','_')+'_'+b.replace(' ','_')
                extra,_=fit_group(required,items,'anchor_pair_support',[(a,b)],prefix,prior_views=views)
                if extra:views.append(extra)
        else:
            special_status='reconstruction_missing_semantic'
    elif qtype=='object_abs_distance':
        if items:
            pair=tuple(q['roles']['endpoints'][:2])
            main,_=fit_group(items,items,'distance_pair',[pair],'distance')
            if main:views.append(main)
        else:
            special_status='reconstruction_missing_semantic'
    else:
        if items:
            reference,evaluations=solve_standing(g,q,items,finish,safe_location,standing_free,
                                                 diagnostics=REJECTED_VIEWS)
            if reference:views.append(reference)
            notes.append('Standing continuous evaluations: '+str(evaluations))
            layout,_=fit_group(items,items,'direction_layout',prefix='direction_layout')
            if layout:views.append(layout)
            pair=(q['roles']['facing'][0],q['roles']['query'][0])
            required=selected_items(q,list(pair))
            if not any(pair_covered(v,required,pair) for v in views):
                extra,_=fit_group(required,items,'direction_relation_support',[pair],
                                  prefix='direction_pair',prior_views=views)
                if extra:views.append(extra)
        else:
            special_status='reconstruction_missing_semantic'

    # This is the only route to render/QA inputs: every emitted view passed its
    # declared role's geometry gates. Failed best attempts stay diagnostic.
    accepted=[]
    for view in unique_views(views):
        if view.get('geometry_safe') and view['acceptance']['passed']:
            accepted.append(view)
        else:
            REJECTED_VIEWS.append(view)
    views=accepted[:MAX_VIEWS]
    readable=set().union(*({k for k,v in c['instance_metrics'].items() if v['readable']} for c in views)) if views else set()
    needed=[i for i in items if not qtype.startswith('object_rel_direction')
            or i['semantic'] in (q['roles']['facing'][0],q['roles']['query'][0])]
    missing_readable=sorted({i['key'] for i in needed}-readable)
    pairs=[]
    if qtype=='object_abs_distance':pairs=[tuple(q['roles']['endpoints'][:2])]
    elif qtype=='object_rel_distance':pairs=[(q['roles']['anchor'][0],s) for s in q['roles']['candidates']]
    elif qtype.startswith('object_rel_direction'):pairs=[(q['roles']['facing'][0],q['roles']['query'][0])]
    missing_pairs=[]
    for pair in pairs:
        required=selected_items(q,list(pair))
        if not any(pair_covered(v,required,pair) for v in views):
            missing_pairs.append(pair[0]+' -> '+pair[1])
    direction=qtype.startswith('object_rel_direction')
    standing_ok=not direction or (reference is not None and reference['acceptance']['passed'])
    standing_joint=not direction or (reference is not None and reference['joint_acceptance']['passed'])
    if missing or special_status=='reconstruction_missing_semantic':
        selection_status='incomplete_missing_semantic'
    elif not views:
        selection_status='no_accepted_view'
    elif missing_pairs:
        selection_status='incomplete_relation'
    elif direction and not standing_ok:
        selection_status='incomplete_standing_reference'
    elif missing_readable:
        selection_status='incomplete_readability'
    elif ambiguous:
        selection_status='incomplete_instance_ambiguity'
    else:
        selection_status='complete'
    object_boxes={s:[i['box'] for i in g.instances[s]]
                  for s in q['targets'] if s!='__room__' and s in g.instances}
    records.append(dict(q,chosen_readable=views,diagnostic_views=unique_views(REJECTED_VIEWS),
        qa_ready=selection_status=='complete',missing_semantics=missing,
        resolution_mode=CURRENT_RESOLUTION['mode'],instance_keys=CURRENT_RESOLUTION['instance_keys'],
        ambiguous_semantics=ambiguous,missing_readable=missing_readable,missing_pairs=missing_pairs,
        selection_status=selection_status,standing_reference_accepted=standing_ok,
        standing_joint_view_complete=standing_joint,notes=notes,object_boxes=object_boxes))
    print('VIEWS',q['qa_id'],len(views),selection_status,'missing_pairs',missing_pairs,flush=True)

payload={'scene':SCENE,'blend':g.bpy.data.filepath,'question_count':len(records),
    'selection_method':'continuous_solve_with_role_geometry_acceptance',
    'qa_status':'NOT_RUN','api_calls':0,
    'parameters':{'margin':MARGIN,'min_area':AREA,'min_visible':VISIBLE,'max_views':MAX_VIEWS,
        'min_relation_projection':MIN_RELATION,'min_projected_short_side':MIN_SHORT_SIDE,
        'standing_offset_range':[.25,.75],'standing_eye_height':1.55,
        'standing_lens_range':[18,35],'max_heading_deviation_deg':25,
        'relative_strategy':'accepted overview + two shared-anchor groups + missing-pair support',
        'failed_views_are_diagnostics_only':True},
    'elapsed_seconds':time.monotonic()-START,'questions':records}
OUT.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf8')
print('VIEW_SELECTION_OK',len(records),OUT,round(payload['elapsed_seconds'],2),flush=True)
