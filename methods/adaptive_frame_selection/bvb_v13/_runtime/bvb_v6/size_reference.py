"""Opt-in object-size reconstruction: one cluster, one target, visible scale reference.

The meter scale is an explicit visual assumption, never benchmark ground truth.
Existing reconstruction contracts remain unchanged when the flag is absent.
"""
import math
import copy

FLAG = 'object_size_reference_mode'


def enabled(question):
    active = question.get(FLAG, False)
    if type(active) is not bool:
        raise ValueError(FLAG+' must be boolean')
    if active and (question.get('question_type') != 'object_size_estimation' or len(question['objects']) != 1):
        raise ValueError('Reference mode requires one object-size target category')
    return active


def instruction():
    return '''
OBJECT SIZE REFERENCE MODE (overrides conflicting scope/schema instructions):
Astra has selected ONE target cluster. Reconstruct exactly ONE visible target
instance from that cluster. Do not retain alternative target instances. All
geometry remains in the supplied shared native-scale coordinates.
Actively find ONE separate visible scale-reference object in these images.
It may be a direct support or another familiar nearby object with a defensible
size assumption. It must be a DIFFERENT physical object from the target.
Use its actual visible bounds, orientation and location. Never invent a ruler,
reference object, exact dimensions, or a calibration observation in the photo.
Prefer references whose real-world size is more constrained (paper, sockets,
tiles, standard furniture parts), and explain uncertainty. A generic door or
furniture dimension is an ASSUMPTION, not a measurement. Do not infer the
reference size from the target's expected size or from any benchmark answer.
Do not duplicate this same reference object in context or structure.

Return exactly structure, targets, context, scale_reference. Existing compact
array contracts apply to the first three keys. scale_reference is an object:
{"object":["ref1","visible reference label",[x,y,z],[sx,sy,sz],[rx,ry,rz]],
 "axis":0, "assumed_length_m":0.30,
 "scale_basis":"estimated_visual_prior",
 "rationale":"what is visible and why this dimension is a reasonable assumption",
 "uncertainty":"why the real dimension might differ",
 "source_timestamps_s":[12.5], "confidence":"low|medium|high"}.
axis is 0/1/2 for the reference LOCAL X/Y/Z full bounding-box dimension.
assumed_length_m is the positive estimated real length of that dimension.
Keep s in native pose units; the host computes meters per scene unit and draws
a numbered meter ruler ALONG this reference. Do not emit tick/text geometry.
If no defensible separate reference is visible, use scale_reference=null and
reference_unavailable_reason as an extra key explaining the evidence gap.
Never fabricate a reference merely to satisfy the requested annotation.
'''


def expand(data, question, legacy_expand, **kwargs):
    if not enabled(question):
        return legacy_expand(data, question, **kwargs)
    if not isinstance(data, dict) or not {'structure','targets','context','scale_reference'} <= set(data):
        raise ValueError('Reference mode requires structure, targets, context, scale_reference')
    if set(data)-{'structure','targets','context','scale_reference','reference_unavailable_reason'}:
        raise ValueError('Unexpected reference-mode fields')
    if len(data['targets']) != 1:
        raise ValueError('Reference mode requires exactly one target instance')
    transport=copy.deepcopy({k:data[k] for k in ('structure','targets','context')})
    normalizations=[]
    # An array of wall openings in the optional style slot is unambiguous:
    # insert the omitted default placeholder, without changing any geometry.
    for row in transport['structure']:
        if isinstance(row,list) and len(row)==7 and row[1]=='wall' and isinstance(row[6],list):
            if all(isinstance(opening,list) and len(opening)==4 for opening in row[6]):
                row.insert(6,None)
                normalizations.append({'id':row[0],'operation':'insert null style before supplied openings'})
    scene, audit = legacy_expand(transport, question, **kwargs)
    if normalizations:
        audit['transport_normalizations']={'changes':normalizations,'geometry_changed':False}
    reference = data['scale_reference']
    audit[FLAG] = True
    if reference is None:
        reason = data.get('reference_unavailable_reason')
        if not isinstance(reason,str) or not reason.strip():
            raise ValueError('Missing reference requires an evidence-gap reason')
        audit['scale_reference'] = None
        audit['reference_unavailable_reason'] = reason
        return scene, audit
    required={'object','axis','assumed_length_m','scale_basis','rationale','uncertainty','source_timestamps_s','confidence'}
    if not isinstance(reference,dict) or set(reference)!=required:
        raise ValueError('Invalid scale_reference fields')
    row=reference['object']
    if not isinstance(row,list) or not 4 <= len(row) <= 5 or not isinstance(row[1],str):
        raise ValueError('Reference object must be [id,label,p,s,r?]')
    if row[1].lower().strip() in {s.lower() for s in question['objects']}:
        raise ValueError('Reference must differ from the target category')
    axis, length = reference['axis'], reference['assumed_length_m']
    if type(axis) is not int or axis not in (0,1,2):
        raise ValueError('Reference axis must be 0, 1 or 2')
    if type(length) not in (int,float) or not math.isfinite(length) or length <= 0:
        raise ValueError('Assumed reference length must be positive and finite')
    if reference['scale_basis'] != 'estimated_visual_prior':
        raise ValueError('Metric scale must be disclosed as an estimate')
    if reference['confidence'] not in ('low','medium','high'):
        raise ValueError('Invalid reference confidence')
    for key in ('rationale','uncertainty'):
        if not isinstance(reference[key],str) or not reference[key].strip():
            raise ValueError('Reference requires '+key)
    times=reference['source_timestamps_s']
    if not isinstance(times,list) or not times or any(type(t) not in (int,float) or not math.isfinite(t) or t<0 for t in times):
        raise ValueError('Reference requires visible source timestamps')
    ref_question=dict(question,objects=[row[1]],roles={'reference':row[1]})
    ref_scene,_=legacy_expand({'structure':[],'targets':[row],'context':[]}, ref_question, **kwargs)
    obj=ref_scene['objects'][0]
    obj['id']='reference_'+row[0]
    obj['color']=[0.08,0.42,0.85]
    scene['objects'].append(obj)
    from schema import validate
    validate(scene)
    audit['scale_reference']={**reference,'geometry_id':obj['id'],
        'meters_per_scene_unit':length/obj['s'][axis], 'metric_scale_verified':False,
        'ruler_type':'numbered_meter_ticks_on_reference'}
    return scene,audit
