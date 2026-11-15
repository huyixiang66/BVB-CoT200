"""Blender post-build coloring, preserves all instance identities and geometry."""
import bpy,json,sys
from pathlib import Path
from mathutils import Vector
out=Path(sys.argv[sys.argv.index('--')+1]).resolve();sys.path.insert(0,str(out))
from counting_mode import assign_colors
data=json.loads((out/'input.json').read_text())
assert data.get('counting_mode',False)
bpy.ops.wm.open_mainfile(filepath=str(out/'scene.blend'))
groups={}
for o in bpy.context.scene.objects:
    if o.type=='MESH' and o.get('role')=='target':groups.setdefault(o['instance_id'],[]).append(o)
rows=[]
for ident,objects in sorted(groups.items()):
    assert len({o['category'] for o in objects})==1
    points=[o.matrix_world@Vector(v) for o in objects for v in o.bound_box]
    center=[(min(p[i] for p in points)+max(p[i] for p in points))/2 for i in range(3)]
    rows.append(dict(instance_id=ident,category=objects[0]['category'],center=center))
palette=json.loads((out/'counting-palette.json').read_text())
result=assign_colors(rows,palette)
snapshot={o.name:(tuple(x for row in o.matrix_world for x in row),len(o.data.vertices),len(o.data.polygons)) for obs in groups.values() for o in obs}
for ident,objects in groups.items():
    for o in objects:o.color=result['instances'][ident]['rgba']
bpy.ops.wm.save_as_mainfile(filepath=str(out/'scene.blend'))
bpy.ops.wm.open_mainfile(filepath=str(out/'scene.blend'))
for name,(matrix,nv,nf) in snapshot.items():
    o=bpy.data.objects[name]
    assert tuple(x for row in o.matrix_world for x in row)==matrix and len(o.data.vertices)==nv and len(o.data.polygons)==nf
    assert max(abs(a-b) for a,b in zip(o.color,result['instances'][o['instance_id']]['rgba']))<1e-5
(out/'instance-color-map.json').write_text(json.dumps(result,indent=2))
(out/'color-map.json').write_text(json.dumps(dict(category_palette=result['category_palette'],
    same_category_same_color=False,instance_ids_internal_only=True,
    instance_map_file='instance-color-map.json',qa_legend_file='qa-color-legend.json'),indent=2))
# QA receives only the palette, never instance IDs, positions, or declared counts.
(out/'qa-color-legend.json').write_text(json.dumps(dict(category_palette=result['category_palette'],
    note='Same category may have several colors; non-neighbour instances may reuse colors. Count visible physical objects, not colors.'),indent=2))
data['instance_colors']={k:v['rgba'] for k,v in result['instances'].items()}
(out/'input.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
print('COUNTING_INSTANCE_COLORS',len(groups))
