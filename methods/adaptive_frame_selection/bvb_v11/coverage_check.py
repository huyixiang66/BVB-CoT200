"""Pure contract checks, importable inside isolated Blender and in tests."""
import json,math

def validate(data,records,inventory):
    # Evidence notes are optional diagnostics, not a gate on scene acceptance.
    warnings=[]
    if isinstance(inventory,str):
        try:inventory=json.loads(inventory)
        except (ValueError,TypeError):warnings.append('Evidence notes are not valid JSON');inventory=None
    if not isinstance(inventory,list):
        warnings.append('No optional evidence inventory');inventory=[]
    categories=data['question']['objects'];colors=data['colors']
    if len({tuple(colors[c]) for c in categories})!=len(categories):raise ValueError('Category colors are not unique')
    instances={c:set() for c in categories}
    for r in records:
        if r['role']!='target':continue
        c=r['category'];ident=r['instance_id']
        if c not in instances:raise ValueError('Unknown target category')
        if not all(math.isfinite(x) for x in r['dimensions']) or max(r['dimensions'])<=1e-6 or sum(x>1e-6 for x in r['dimensions'])<2:raise ValueError('Degenerate target geometry: '+r['name'])
        if r.get('hide_render'):raise ValueError('Target hidden from render: '+r['name'])
        expected=colors[c]
        if data.get('counting_mode'):
            if ident not in data.get('instance_colors',{}):raise ValueError('Missing counting instance color: '+str(ident))
            expected=data['instance_colors'][ident]
        if len(r['color'])!=4 or max(abs(x-y) for x,y in zip(r['color'],expected))>.005:raise ValueError('Wrong target color: '+r['name'])
        instances[c].add(ident)
    zero=set(data.get('counting_zero_evidence_categories',[]))
    if zero and (data['question'].get('question_type') not in {'object_counting_single','object_counting_multiple'} or not data.get('counting_allow_all_rejected')):
        raise ValueError('Zero-evidence exemption requires explicit counting policy')
    missing=[c for c,ids in instances.items() if not ids and c not in zero]
    if missing:raise ValueError('Approved question targets missing from reconstruction: '+', '.join(missing))
    return dict(valid=True,instances_by_category={c:sorted(ids) for c,ids in instances.items()},inventory=inventory,
                category_coverage_checked=True,zero_evidence_categories=sorted(zero),declared_evidence_coverage_checked=False,
                evidence_mapping_required=False,diagnostic_warnings=warnings,visual_identity_verified=False,geometric_accuracy_verified=False)
