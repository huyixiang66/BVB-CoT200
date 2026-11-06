"""Blender: verify actual question-conditioned geometry before preview rendering."""
import bpy
import hashlib
import json
from pathlib import Path
import sys
from mathutils import Vector

folder=Path(sys.argv[sys.argv.index('--')+1]).resolve()
spec=json.loads((folder/'scene-spec.json').read_text())
audit=json.loads((folder/'question-audit.json').read_text())
by_id={o['id']:o for o in spec['objects']}
assert set(by_id)==set(audit['background']+audit['structure']+[t['id'] for t in audit['targets']])
bpy.context.view_layer.update()
for name in audit['background']:
    record=by_id[name];col=bpy.data.collections[name]
    parts=[o for o in col.objects if o.type=='MESH']
    assert record['kind']=='box' and record['color']=='white' and len(parts)==1
    ob=parts[0];root=ob.parent
    assert root['asset_kind']=='box'
    color=ob.data.materials[0].node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value
    assert all(abs(color[i]-(.85,.85,.82)[i])<1e-5 for i in range(3))
    pts=[root.matrix_world.inverted()@ob.matrix_world@Vector(v) for v in ob.bound_box]
    for i in range(3):
        lo=min(v[i] for v in pts);hi=max(v[i] for v in pts)
        assert abs(hi-lo-record['s'][i])<1e-4 and abs(lo+hi)<1e-4
        assert abs(root.location[i]-record['p'][i])<1e-4
for t in audit['targets']:
    root=bpy.data.objects[t['id']+' [asset]']
    assert root['asset_kind']==t['kind']
    assert any(o.type=='MESH' for o in bpy.data.collections[t['id']].objects)
for name in audit['structure']:
    if audit.get('structure_scope')=='walls, openings and floor; no ceiling or roof':
        assert by_id[name]['kind'] in ('wall','floor')
        assert bpy.data.objects[name+' [asset]']['asset_kind']==by_id[name]['kind']
    if audit.get('structure_scope')=='walls and their openings only':
        assert by_id[name]['kind']=='wall'
        assert bpy.data.objects[name+' [asset]']['asset_kind']=='wall'
    if by_id[name]['kind']!='wall':
        for ob in bpy.data.collections[name].objects:
            if ob.type=='MESH':
                color=ob.data.materials[0].diffuse_color
                assert all(abs(color[i]-(.85,.85,.82)[i])<1e-5 for i in range(3))
if 'camera' in spec:assert bpy.context.scene.camera.animation_data
else:
    assert bpy.context.scene.camera and not bpy.context.scene.camera.animation_data
    assert bpy.context.scene.frame_start==bpy.context.scene.frame_end==1
result=dict(status='passed',background_single_white_boxes=len(audit['background']),
    background_bounds_and_positions_match=True,targets_match_labels=len(audit['targets']),
    nonwall_structure_white=True,animated_camera=bool(bpy.context.scene.camera.animation_data),
    generated_camera_trajectory='camera' in spec,
    source_blend_sha256=hashlib.sha256(Path(bpy.data.filepath).read_bytes()).hexdigest())
if audit.get('structure_scope')=='walls and their openings only':
    assert not any(ob.get('asset_kind') in ('floor','ceiling','roof') for ob in bpy.data.objects)
    result['walls_only_verified']=True
if audit.get('structure_scope')=='walls, openings and floor; no ceiling or roof':
    assert not any(ob.get('asset_kind') in ('ceiling','roof') for ob in bpy.data.objects)
    result['walls_and_floor_verified']=True
(folder/'geometry-validation.json').write_text(json.dumps(result,indent=2)+'\n')
print('QUESTION_GEOMETRY_VERIFIED',json.dumps(result))
