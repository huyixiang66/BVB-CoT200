"""Question targets, direct supports, walls and optional structural floor."""
import json
from question_scene import expand as expand_full,asset_kind
from schema import COLORS,validate

RELATIONS={'support'}


def expand(data,question,walls_only=False,walls_and_floor=False):
    if not isinstance(data,dict) or set(data)!={'structure','targets','context'}:
        raise ValueError('Return exactly structure, targets, context; no camera')
    if not isinstance(data['targets'],list) or not isinstance(data['context'],list):raise ValueError('Expected arrays')
    if walls_only or walls_and_floor:
        allowed={'wall','floor'} if walls_and_floor else {'wall'}
        if not isinstance(data['structure'],list) or any(not isinstance(row,list) or len(row)<2 or row[1] not in allowed for row in data['structure']):
            raise ValueError('Allowed structure kinds: '+', '.join(sorted(allowed))+'; no ceiling or roof')
    target_ids={row[0] for row in data['targets'] if isinstance(row,list) and row and isinstance(row[0],str)}
    background=[];relations=[];ids=set()
    for row in data['context']:
        if not isinstance(row,list) or not 5<=len(row)<=6:raise ValueError('context: [id,target_ids,relation,p,s,r?]')
        ident,refs,relation,p,s=row[:5]
        if not isinstance(ident,str) or not ident or ident in ids:raise ValueError('Context ids must be unique strings')
        ids.add(ident)
        if not isinstance(refs,list) or not refs or any(not isinstance(x,str) or x not in target_ids for x in refs):
            raise ValueError('Every context object must directly reference existing target IDs; no context chains')
        if relation not in RELATIONS:raise ValueError('Context must directly physically support a target; only support is allowed')
        block=[p,s]+(row[5:] if len(row)==6 else [])
        name=f'background_{len(background):03d}';background.append(block)
        relations.append(dict(id=name,source_id=ident,targets=['target_'+x for x in refs],relation=relation))
    # Reuse all target/wall/background validation; remove the compatibility
    # placeholder before compilation. No camera data is produced by the model.
    scene,audit=expand_full(dict(structure=data['structure'],targets=data['targets'],background=background,
        camera=[[0,[0,-1,1],[0,0,0]],[90,[0,-1,1],[0,0,0]]]),question)
    del scene['camera'];validate(scene)
    audit.update(context_relations=relations,background_scope='direct physical supports of question targets only',
        camera_policy='local static overview; no inferred trajectory')
    if walls_only:audit['structure_scope']='walls and their openings only'
    if walls_and_floor:audit['structure_scope']='walls, openings and floor; no ceiling or roof'
    return scene,audit


def prompt_for(question,walls_only=False,walls_and_floor=False):
    public={k:question[k] for k in ['id','question_type','question','options','roles','objects'] if k in question}
    prompt = '''You reconstruct an indoor scene from chronological video frames using
a local Blender asset compiler. Return ONLY JSON: structure, targets, context.
Do not generate Python, explanatory text, an answer to the question, or a camera.
Use only the observed first 90 seconds. No extra frames or external information.
Only public question fields are supplied; never invent invisible target objects.

Reconstruct ONLY visible question-relevant targets, objects that directly
physically support those targets, and major walls with their openings.
OMIT all other scene contents, even if nearby or useful as a visual landmark.

Coordinates: meters, world Z up, one shared coordinate system. p=[x,y,z] is the
center of the full bounding box; s=[width,depth,height] is its positive full size
in local axes. r=[rx,ry,rz] is Euler XYZ in degrees, default [0,0,0]. Asset fronts
face local -Y. A floor-supported object of height h has center z=floor_top+h/2.

structure: [id,kind,p,s,r?,color?,style?,openings?]. kind is ONLY wall, floor,
ceiling, roof. Keep a small set of major room boundary slabs and partitions,
their connectivity and door openings. Omit trim, tile strips, window frames,
beams, fixtures and decorative detail. A wall has local X=length,Y=thickness,
Z=height; openings=[[x,z,width,height],...] use metric coordinates relative to
its local CENTER and must fit inside its width/height. No solid infill in holes.
Nonwall structural slabs are forced white. Null skips an optional default.

targets: [id,label,p,s,r?,color?,style?]. label must exactly match a parsed target
keyword below. The local compiler selects the asset kind. Keep each visible
instance as a separate object, but do not emit legs, knobs, hinges or handles:
the asset handles those parts. style defaults standard; round for table/stool,
open for door/cabinet, office for chair, double for cabinet/fridge are supported.

context: [id,[target_id,...],"support",p,s,r?]. These become simple white boxes.
The ONLY allowed relation is support: a separate object directly physically
supports a visible TARGET, such as the stove under a pan or the table under a
basket. Every box must reference actual target IDs, not category labels or other
context IDs. Require visible physical support/contact evidence; proximity,
containment, attachment, route obstruction, and usefulness as a landmark alone
do NOT qualify. Do not relabel an anchor or obstacle as support to keep it.
Do not traverse support chains: if a stove directly supports the pan, do not
also add the cabinet supporting the stove merely for that indirect relation.
If a support is itself a question-relevant object, reconstruct it as a TARGET
and do not duplicate it as a white box. Do not add objects resting on a target
unless they are independently question-relevant. Integral legs, knobs, hinges,
handles and other asset parts are not separate context. Walls and allowed floor
slabs belong only in structure, never in context. No ceiling or roof.
Keep the observed extent of a direct support; do not extend it into surrounding
cabinetry or unrelated furniture. If direct support is uncertain, omit the box.

No camera field, pose extraction, trajectory, timestamp samples or viewpoint
reasoning is needed: fixed overview cameras are generated locally. Keep JSON
compact and ids short. The scope is sparse; no minimum output length is required.
Aim for roughly 1000–3000 visible tokens, but retain necessary geometric evidence.
Colors: '''+', '.join(COLORS)+'''; or [r,g,b] in [0,1]. Defaults: white, standard.
PUBLIC QUESTION AND PARSED KEYWORDS:
'''+json.dumps(public,ensure_ascii=False)+'\nTARGET ASSET MAPPING:\n'+json.dumps({x:asset_kind(x) for x in question['objects']})+'\n'
    if walls_only:
        prompt=prompt.replace('structure: [id,kind,p,s,r?,color?,style?,openings?]. kind is ONLY wall, floor,\nceiling, roof. Keep a small set of major room boundary slabs and partitions,',
            'structure: [id,kind,p,s,r?,color?,style?,openings?]. kind is ONLY wall.\nKeep only major walls and partitions; do not output any floor, ceiling, roof,\nground plane, or horizontal room slab, even as a context box. Preserve')
        prompt=prompt.replace('Nonwall structural slabs are forced white. Null skips an optional default.',
            'Use z=0 as the implicit floor reference; do not create floor geometry. Null skips an optional default.')
        prompt=prompt.replace('Aim for roughly 1000–3000 visible tokens, but retain necessary geometric evidence.',
            'Use as few visible tokens as necessary, retaining the required geometric evidence; there is no minimum length.')
    if walls_and_floor:
        prompt=prompt.replace('and major walls with their openings.',
            'major walls with their openings, and the observed floor.')
        prompt=prompt.replace('structure: [id,kind,p,s,r?,color?,style?,openings?]. kind is ONLY wall, floor,\nceiling, roof. Keep a small set of major room boundary slabs and partitions,',
            'structure: [id,kind,p,s,r?,color?,style?,openings?]. kind is ONLY wall or floor.\nPreserve major walls and partitions plus a simple floor slab for the observed\nlocal region. Do not generate any ceiling or roof. Preserve')
        prompt=prompt.replace('Nonwall structural slabs are forced white. Null skips an optional default.',
            'Floor slabs are white. Infer floor height and visible extent from the supplied\nimages and shared poses. Use a thin slab with its TOP at the inferred floor\nlevel; do not assume z=0, cover steps or fill unobserved rooms with one large\nrectangle. No floor texture or tile detail. Null skips an optional default.')
        prompt=prompt.replace('Aim for roughly 1000–3000 visible tokens, but retain necessary geometric evidence.',
            'Use as few visible tokens as necessary, retaining the required geometric evidence; there is no minimum length.')
    return prompt
