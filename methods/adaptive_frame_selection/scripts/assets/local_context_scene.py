"""One-hop context reconstruction; no generated camera or full-scene inventory."""
import json
from question_scene import expand as expand_full,asset_kind
from schema import COLORS,validate

RELATIONS={'support','container','attached','obstacle','anchor'}


def expand(data,question,walls_only=False):
    if not isinstance(data,dict) or set(data)!={'structure','targets','context'}:
        raise ValueError('Return exactly structure, targets, context; no camera')
    if not isinstance(data['targets'],list) or not isinstance(data['context'],list):raise ValueError('Expected arrays')
    if walls_only:
        if not isinstance(data['structure'],list) or any(not isinstance(row,list) or len(row)<2 or row[1]!='wall' for row in data['structure']):
            raise ValueError('Walls-only structure: every structure record must be wall; no floor, ceiling, roof, or ground plane')
    target_ids={row[0] for row in data['targets'] if isinstance(row,list) and row and isinstance(row[0],str)}
    background=[];relations=[];ids=set()
    for row in data['context']:
        if not isinstance(row,list) or not 5<=len(row)<=6:raise ValueError('context: [id,target_ids,relation,p,s,r?]')
        ident,refs,relation,p,s=row[:5]
        if not isinstance(ident,str) or not ident or ident in ids:raise ValueError('Context ids must be unique strings')
        ids.add(ident)
        if not isinstance(refs,list) or not refs or any(not isinstance(x,str) or x not in target_ids for x in refs):
            raise ValueError('Every context object must directly reference existing target IDs; no context chains')
        if relation not in RELATIONS:raise ValueError('Unknown context relation')
        block=[p,s]+(row[5:] if len(row)==6 else [])
        name=f'background_{len(background):03d}';background.append(block)
        relations.append(dict(id=name,source_id=ident,targets=['target_'+x for x in refs],relation=relation))
    # Reuse all target/wall/background validation; remove the compatibility
    # placeholder before compilation. No camera data is produced by the model.
    scene,audit=expand_full(dict(structure=data['structure'],targets=data['targets'],background=background,
        camera=[[0,[0,-1,1],[0,0,0]],[90,[0,-1,1],[0,0,0]]]),question)
    del scene['camera'];validate(scene)
    audit.update(context_relations=relations,background_scope='one-hop target context only',
        camera_policy='local static overview; no inferred trajectory')
    if walls_only:audit['structure_scope']='walls and their openings only'
    return scene,audit


def prompt_for(question,walls_only=False):
    public={k:question[k] for k in ['id','question_type','question','options','roles','objects'] if k in question}
    prompt = '''You reconstruct an indoor scene from chronological video frames using
a local Blender asset compiler. Return ONLY JSON: structure, targets, context.
Do not generate Python, explanatory text, an answer to the question, or a camera.
Use only the observed first 90 seconds. No extra frames or external information.
Only public question fields are supplied; never invent invisible target objects.

Focus on all visible instances of question-relevant categories, their immediate
physical context, and the coarse room layout. OMIT unrelated scene contents.

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

context: [id,[target_id,...],relation,p,s,r?]. These become white boxes, never
detailed assets. EVERY box must have a direct, visible relation to one or more
TARGET instances, referenced by their ids (not category labels):
support: physically supports a target, such as stove/counter under a pan;
container: houses a target, if needed to locate it;
attached: a separate meaningful object directly attached to a target;
obstacle: large object directly affects access to a target or the requested route;
anchor: immediately neighboring large object necessary to disambiguate location.
Keep only one-hop relations: do not preserve objects merely related to another
context box, merely in the same room, or simply visible. Omit distant furniture,
cosmetics, bottles, papers, clutter and unrelated small objects. Do not list
integral parts or surface decorations as context. A floor/wall already in
structure is not duplicated as a context box. Integrated fixtures may be one
coarse box; distinct route obstacles stay separate. Context geometry must match
the observed placement and extent. Do not fill walkable gaps with merged boxes.

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
    return prompt
