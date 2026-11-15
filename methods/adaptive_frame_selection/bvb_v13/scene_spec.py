"""Compact, data-only scene contract. No model-authored code is executed."""
import json,math
from ._schema_helpers import vector,rotation,integer

VERSION='compact-scene-v2'
SHAPES={'b':'box','c':'cylinder','s':'sphere','t':'torus'}

def validate(spec,packet):
    from .scene_parameters import expand
    spec=expand(spec)
    if not isinstance(spec,dict) or set(spec)-{'o','a','c','g'} or not {'o','a'}<=set(spec):raise ValueError('Expected scene keys o,a and optional c,g only')
    for key in ['o','a','c','g']:
        if not isinstance(spec.get(key,[]),list):raise ValueError('Scene arrays required')
    if sum(len(spec.get(k,[])) for k in ['o','a','c','g'])>1024:raise ValueError('Scene exceeds 1024 entities')
    room=packet.get('room_mode',False);categories=packet['question'].get('objects',[]);colors=packet.get('colors',{})
    frames=packet['frames'];counts={};objects=[];parts=0
    def times(indices):
        if not isinstance(indices,list) or not indices:raise ValueError('Architecture needs supplied frame indices')
        return [frames[integer(i,0,len(frames)-1)]['timestamp_s'] for i in indices]
    for i,item in enumerate(spec['o']):
        if not isinstance(item,list) or len(item)!=4:raise ValueError('Object = [category,position,rotation,parts]')
        cat,pos,rot,rawparts=item
        if room:
            if not isinstance(cat,str) or not cat.strip() or len(cat)>80:raise ValueError('Room furniture needs a short category')
        else:cat=categories[integer(cat,0,len(categories)-1)]
        if not isinstance(rawparts,list) or not rawparts:raise ValueError('Object needs visible primitive parts')
        parsed=[]
        for p in rawparts:
            if not isinstance(p,list) or len(p) not in (3,4) or p[0] not in SHAPES:raise ValueError('Part = [b|c|s|t,local_position,full_dimensions,optional_rotation]')
            parsed.append(dict(shape=SHAPES[p[0]],position=vector(p[1]),size=vector(p[2],positive=True),rotation=rotation(p[3] if len(p)==4 else 0)))
        parts+=len(parsed)
        if parts>4096:raise ValueError('Scene exceeds 4096 primitives')
        counts[cat]=counts.get(cat,0)+1
        limit=packet.get('instance_limits',{}).get(cat)
        if not room and limit is not None and counts[cat]>limit:raise ValueError('Instance limit exceeded: '+cat)
        color=colors.get(cat,[.35,.55,.72,1])
        objects.append(dict(id=f'object_{i:03d}',category=cat,position=vector(pos),rotation=rotation(rot),parts=parsed,color=color))
    if not room:
        allowed_absent=set(packet.get('counting_zero_evidence_categories',[])) if packet.get('counting_allow_all_rejected') else set()
        missing=set(categories)-set(counts)-allowed_absent
        if missing:raise ValueError('Missing supported target categories: '+', '.join(sorted(missing)))
    architecture=[]
    for i,item in enumerate(spec['a']):
        if not isinstance(item,list) or len(item)!=5 or item[0] not in {'f','w'}:raise ValueError('Architecture = [f|w,position,size,rotation,frame_indices]')
        architecture.append(dict(id=f'structure_{i:03d}',kind='floor' if item[0]=='f' else 'wall',position=vector(item[1]),size=vector(item[2],positive=True),rotation=rotation(item[3]),times=times(item[4])))
    if (packet.get('architecture_inventory_required') or room) and not architecture:raise ValueError('Evidenced architecture is required')
    if room and not any(a['kind']=='floor' for a in architecture):raise ValueError('Room area requires evidenced floor geometry')
    context=[]
    for i,item in enumerate(spec.get('c',[])):
        if not isinstance(item,list) or len(item)!=5 or item[0] not in {'s','t'}:raise ValueError('Context = [s|t,target_indices,position,size,rotation]')
        if not isinstance(item[1],list) or not item[1]:raise ValueError('Context needs direct target references')
        targets=[objects[integer(n,0,len(objects)-1)]['id'] for n in item[1]]
        context.append(dict(id=f'context_{i:03d}',relation='support' if item[0]=='s' else 'contact',targets=targets,position=vector(item[2]),size=vector(item[3],positive=True),rotation=rotation(item[4])))
    openings=[]
    for i,item in enumerate(spec.get('g',[])):
        if not isinstance(item,list) or len(item)!=2 or not isinstance(item[0],list):raise ValueError('Opening = [wall_indices,frame_indices]')
        walls=[architecture[integer(n,0,len(architecture)-1)] for n in item[0]]
        if len({w['id'] for w in walls})<2 or any(w['kind']!='wall' for w in walls):raise ValueError('Opening needs two distinct wall boundaries')
        openings.append(dict(id=f'opening_{i:03d}',walls=[w['id'] for w in walls],times=times(item[1])))
    return dict(version=VERSION,room_mode=room,objects=objects,architecture=architecture,context=context,openings=openings)

def compile_script(spec,packet,destination):
    from pathlib import Path
    parsed=validate(spec,packet)
    runtime=Path(__file__).with_name('scene_runtime.py').read_text()
    # JSON is quoted as data using repr; labels can never become executable code.
    return runtime+'\n\nif __name__ == "__main__":\n    build(json.loads('+repr(json.dumps(parsed,separators=(',',':'),ensure_ascii=False))+'), '+repr(str(Path(destination).resolve()))+')\n'
