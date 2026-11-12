"""Conservative measurement-scope repairs; never alter visible geometry."""
import re


def measurement_exclusions(ref, component):
    axes=list(component.get('measurement_exclude_axes', []))
    if any(type(a) is not int or a not in (0,1,2) for a in axes):
        raise ValueError('Invalid measurement exclusion axes')
    source=ref.get('source','').lower()
    label=ref.get('label','').lower()
    role=str(component.get('component_role','')).lower()
    name=str(getattr(component,'name',component.get('name',''))).lower()
    # Exact source scope is required. Do not drop accessories from whole-object sizes.
    rim=any(s in source for s in ('floor-to-rim','floor to rim','floor-to-counter','floor to counter'))
    basin=any(s in label for s in ('basin','sink','vanity'))
    tap=role in ('tap','faucet') or bool(re.search(r'\b(tap|faucet)\b',name))
    if rim and basin and tap:axes.append(2)
    leaf='door' in label and 'leaf' in source and 'thickness' in source
    leaf_body=role=='door_leaf' or bool(re.search(r'\bdoor leaf\b',name))
    if leaf and not leaf_body:axes.append(1)
    return sorted(set(axes))
