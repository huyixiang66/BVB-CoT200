"""Check declared scene scope. This is not visual/physical contact verification."""
import json

def support_ids(value):
    if isinstance(value,str):
        try:value=json.loads(value)
        except (TypeError,ValueError):raise ValueError('supports must be a JSON array of target instance IDs')
    if not isinstance(value,list) or not value or not all(isinstance(x,str) and x for x in value):
        raise ValueError('Every context cube needs nonempty direct target instance references')
    return value

def validate(records,question,limits):
    categories=set(question['objects']);instances={};context=[];struct=[]
    for r in records:
        role=r.get('role')
        if role=='target':
            if r.get('category') not in categories:raise ValueError('Unrelated target category: '+r['name'])
            ident=r.get('instance_id')
            if not isinstance(ident,str) or not ident:raise ValueError('Target needs instance_id: '+r['name'])
            if ident in instances and instances[ident]!=r['category']:raise ValueError('Cross-category instance ID')
            instances[ident]=r['category']
        elif role=='context':context.append(r)
        elif role=='structure':
            if r.get('category') not in {'wall','floor'}:raise ValueError('Structure can only contain walls/opening boundaries and floors')
            struct.append(r)
        else:raise ValueError('Every mesh must be target, context, or structure: '+r['name'])
    for r in context+struct:
        color=r.get('color',[])
        if len(color)<3 or min(color[:3])<.9 or max(color[:3])-min(color[:3])>.001:
            raise ValueError('Non-target geometry must be white: '+r['name'])
        if r.get('vertices')!=8 or r.get('faces')!=6 or r.get('modifiers'):
            raise ValueError('Non-target geometry must be an unmodified cuboid: '+r['name'])
    for r in context:
        relation=r.get('relation')
        if relation not in {'support','contact'}:
            raise ValueError('Context must declare direct support or contact: '+r['name'])
        refs=support_ids(r.get('supports') if relation=='support' else r.get('contacts'))
        if not set(refs)<=set(instances):
            raise ValueError('Context must directly reference target instances; no support/contact chains: '+r['name'])
    for label in categories:
        count=sum(v==label for v in instances.values());limit=limits.get(label)
        if limit is not None and count>limit:raise ValueError('Final instance cap exceeded: '+label)
    return dict(schema_valid=True,instance_count=len(instances),context_blocks=len(context),
                structure_blocks=len(struct),missing_categories=sorted(categories-set(instances.values())),
                physical_support_verified=False,physical_contact_verified=False,geometric_accuracy_verified=False)

def validate_architecture(inventory,records,timestamps):
    """Validate declared evidence/mesh coverage, not visual structural completeness."""
    if isinstance(inventory,str):inventory=json.loads(inventory)
    if not isinstance(inventory,list) or not inventory:raise ValueError('Architecture inventory required')
    structures={r['name']:r for r in records if r.get('role')=='structure'}
    covered=set();seen=set()
    for item in inventory:
        ident=item.get('id');kind=item.get('kind');names=item.get('mesh_names');times=item.get('evidence_times_s')
        if not isinstance(ident,str) or not ident or ident in seen:raise ValueError('Architecture IDs must be unique')
        seen.add(ident)
        if kind not in {'wall','floor','opening'}:raise ValueError('Invalid architecture kind')
        if not isinstance(names,list) or not names or any(n not in structures for n in names):
            raise ValueError('Architecture must reference actual structure meshes')
        expected='floor' if kind=='floor' else 'wall'
        if any(structures[n]['category']!=expected for n in names):raise ValueError('Architecture mesh category mismatch')
        if kind=='opening' and len(set(names))<2:raise ValueError('Opening must reference at least two wall boundaries')
        if not isinstance(times,list) or not times or any(type(t) not in (int,float) or not any(abs(t-s)<1e-4 for s in timestamps) for t in times):
            raise ValueError('Architecture evidence must reference supplied timestamps')
        covered.update(names)
    if covered!=set(structures):raise ValueError('Structure meshes omitted from inventory')
    return dict(inventory_valid=True,items=len(inventory),declared_wall_items=sum(i['kind']=='wall' for i in inventory),
        declared_openings=sum(i['kind']=='opening' for i in inventory),declared_floor_items=sum(i['kind']=='floor' for i in inventory),
        visual_completeness_verified=False,opening_clearance_verified=False,inventory=inventory)
