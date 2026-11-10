"""Check the actual Blender mesh declarations after generated code executes."""
import bpy,json,sys
from pathlib import Path
out=Path(sys.argv[sys.argv.index('--')+1]).resolve()
sys.path.insert(0,str(out))
from v10_contract import validate,validate_architecture
data=json.loads((out/'input.json').read_text())
records=[dict(name=o.name,role=o.get('role'),category=o.get('category'),instance_id=o.get('instance_id'),
              relation=o.get('relation'),supports=o.get('supports'),contacts=o.get('contacts'),color=list(o.color),vertices=len(o.data.vertices),
              faces=len(o.data.polygons),modifiers=[m.type for m in o.modifiers])
         for o in bpy.context.scene.objects if o.type=='MESH']
if not records:raise ValueError('No scene geometry')
result=validate(records,data['question'],data['instance_limits'])
if data.get('architecture_inventory_required'):
    result['architecture']=validate_architecture(bpy.context.scene.get('architecture_inventory'),records,[f['timestamp_s'] for f in data['frames']])
    (out/'architecture-inventory.json').write_text(json.dumps(result['architecture'],ensure_ascii=False,indent=2))
if Path(bpy.data.filepath).resolve()!=(out/'scene.blend'):raise ValueError('Unexpected scene save path')
(out/'scope-validation.json').write_text(json.dumps(dict(result,mesh_records=records),ensure_ascii=False,indent=2))
print('V10_SCOPE_VALID',result)
