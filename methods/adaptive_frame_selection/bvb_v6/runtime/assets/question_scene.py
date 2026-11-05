"""Question-conditioned transport: target assets, walls, anonymous white boxes.

No answer labels are accepted. Keyword parsing runs on the host using the
existing object_sampling.questions.extract_question adapter.
"""
from schema import KINDS,validate

STRUCTURE={'wall','floor','ceiling','roof'}
ALIASES={'washing machine':'washing_machine','refrigerator':'fridge','heater':'radiator',
         'ceiling light':'lamp','computer mouse':'box','keyboard':'box','basket':'bin',
         'cutting board':'box','blanket':'towel','bowl':'pot','kettle':'pot','printer':'box'}


def asset_kind(label):
    return ALIASES.get(label,label if label in KINDS else 'box')


def expand(data,question):
    if not isinstance(data,dict) or set(data)!={'structure','targets','background','camera'}:
        raise ValueError('Return exactly structure, targets, background, camera')
    for key in ['structure','targets','background']:
        if not isinstance(data[key],list):raise ValueError(key+' must be an array')
    objects=[];targets=[];background=[];structure=[]
    for row in data['structure']:
        if not isinstance(row,list) or not 4<=len(row)<=8:raise ValueError('structure: compact object arrays required')
        if row[1] not in STRUCTURE:raise ValueError('Only wall/floor/ceiling/roof in structure')
        item={k:v for k,v in zip(['id','kind','p','s','r','color','style','openings'],row) if v is not None}
        if not isinstance(item['id'],str):raise ValueError('Structure id must be a string')
        item['id']='structure_'+item['id']
        if item['kind']!='wall':item['color']='white'
        objects.append(item);structure.append(item['id'])
    for row in data['targets']:
        if not isinstance(row,list) or not 4<=len(row)<=7:
            raise ValueError('target: [id,label,p,s,r?,color?,style?]')
        item={k:v for k,v in zip(['id','label','p','s','r','color','style'],row) if v is not None}
        label=item.pop('label')
        if label not in question['objects']:raise ValueError('Target label not in parsed question: '+str(label))
        if not isinstance(item['id'],str):raise ValueError('Target id must be a string')
        item['id']='target_'+item['id'];item['kind']=asset_kind(label)
        objects.append(item);targets.append(dict(id=item['id'],label=label,kind=item['kind']))
    for i,row in enumerate(data['background']):
        if not isinstance(row,list) or not 2<=len(row)<=3:raise ValueError('background: [center,size,rotation?]')
        item=dict(id=f'background_{i:03d}',kind='box',p=row[0],s=row[1],color='white')
        if len(row)==3 and row[2] is not None:item['r']=row[2]
        objects.append(item);background.append(item['id'])
    scene=validate(dict(objects=objects,camera=data['camera']))
    found={t['label'] for t in targets}
    audit=dict(question_id=question['id'],keywords=question['objects'],roles=question['roles'],
        targets=targets,structure=structure,background=background,
        missing_keywords=[x for x in question['objects'] if x not in found],
        background_forced_white_boxes=True,nonwall_structure_forced_white=True,
        ground_truth_used=False)
    return scene,audit


def prompt_for(question):
    import json
    public={k:question[k] for k in ['id','question_type','question','options','roles','objects'] if k in question}
    mapping={label:asset_kind(label) for label in question['objects']}
    return '''\nQUESTION-CONDITIONED RECONSTRUCTION (OVERRIDES previous output schema):
Reconstruct only question-relevant objects as semantic assets. The LOCAL parser
extracted the targets below from the public question, including reference objects
and answer-option objects. Do NOT answer the question. Do not infer invisible
objects merely because the question mentions them. No ground truth is supplied.
Return ONLY JSON with four keys: structure, targets, background, camera.

structure: compact arrays [id,kind,p,s,r,color,style,openings] as above, but ONLY
wall/floor/ceiling/roof. Preserve walls, their locations and door/window openings.
Non-wall structural slabs are rendered white. No other objects go in structure.

targets: arrays [id,label,p,s,r,color,style], first 4 required, later defaults may
be omitted/null. label MUST exactly match one parsed keyword. The compiler picks
the asset kind from the mapping below. Keep all visible instances of each target
category, with their individual positions, dimensions, orientation and color.

background: arrays [p,s,r], only p and s required; no name, kind, color or parts.
Each non-target physical object becomes ONE oriented white bounding box. Preserve
its location, size and occupancy, not its detailed structure or original color.
Do not emit separate furniture legs, handles, hinges, door rails, appliance knobs,
shower hoses or trim: include attached parts in their parent object's bounds.
Independent countable items remain separate background boxes. Do not duplicate
targets as background. Walls and their openings must not become solid white boxes.

camera: same [seconds,position,target] arrays as before. Coordinate conventions,
meters, bounding-box centers and the video interval remain unchanged. Unknown or
unseen target categories may have no entry: never invent evidence to fill them.
The local compiler enforces white boxes for background and white nonwall slabs.
PARSED QUESTION (public data only):
'''+json.dumps(public,ensure_ascii=False)+'\nTARGET ASSET MAPPING:\n'+json.dumps(mapping)+'\n'
