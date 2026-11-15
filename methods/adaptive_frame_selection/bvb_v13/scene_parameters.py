"""Expand compact model parameters to the validated local scene contract."""

def expand(spec):
    if not isinstance(spec,dict):raise ValueError('Expected scene object')
    # Previously saved scenes remain readable; new inference uses v=2 explicitly.
    if 'v' not in spec:return spec
    if type(spec['v']) is not int or spec['v']!=2:raise ValueError('Expected scene parameter version 2')
    if set(spec)-{'v','o','a','c','g'} or not {'o','a'}<=set(spec):raise ValueError('Expected v,o,a and optional c,g')
    if any(not isinstance(spec.get(k,[]),list) for k in ['o','a','c','g']):raise ValueError('Scene arrays required')
    if sum(len(spec.get(k,[])) for k in ['o','a','c','g'])>1024:raise ValueError('Scene exceeds 1024 entities')
    result=dict(o=[],a=[])
    for row in spec['o']:
        if not isinstance(row,list) or len(row) not in (3,4):raise ValueError('Object = [category,position,geometry,optional_rotation]')
        cat,pos,geometry=row[:3];rot=row[3] if len(row)==4 else 0
        if isinstance(geometry,list) and len(geometry)==3:
            parts=[['b',[0,0,0],geometry]]
        elif isinstance(geometry,list) and len(geometry)==4 and geometry[0] in ('b','c','s','t'):
            parts=[[geometry[0],[0,0,0],geometry[1:]]]
        elif isinstance(geometry,dict) and set(geometry)=={'p'}:
            parts=geometry['p']
        else:raise ValueError('Geometry = [dx,dy,dz], [shape,dx,dy,dz] or {p:parts}')
        result['o'].append([cat,pos,rot,parts])
    for row in spec['a']:
        if not isinstance(row,list) or len(row) not in (4,5):raise ValueError('Architecture = [kind,position,size,frames,optional_rotation]')
        result['a'].append([row[0],row[1],row[2],row[4] if len(row)==5 else 0,row[3]])
    if 'c' in spec:
        result['c']=[]
        for row in spec['c']:
            if not isinstance(row,list) or len(row) not in (4,5):raise ValueError('Context = [relation,targets,position,size,optional_rotation]')
            result['c'].append(row+[0] if len(row)==4 else row)
    if 'g' in spec:result['g']=spec['g']
    return result
