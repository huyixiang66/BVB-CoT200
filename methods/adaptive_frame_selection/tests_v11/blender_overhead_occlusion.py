"""Run with Blender --background --factory-startup --python this_file -- ROOT OUT."""
import sys
from pathlib import Path
import bpy

root, out = [Path(p).resolve() for p in sys.argv[sys.argv.index('--')+1:]]
sys.path.insert(0, str(root))
from bvb_v11.runtime_occlusion import overhead_blockers, render_with_overhead_removal

bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
out.mkdir(parents=True, exist_ok=True)

def cube(name, loc, dimensions, role='context', category=None):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
    o = bpy.context.object; o.name=name; o.dimensions=dimensions; o['role']=role
    if role=='target': o['instance_id']=name; o['category']=category or name
    return o

target=cube('target',(0,0,0),(1,1,1),'target')
support=cube('support',(0,0,-1),(3,3,.5))
cap1=cube('cap1',(0,0,1),(2,2,.2)); cap2=cube('cap2',(0,0,2),(2,2,.2))
adjacent=cube('adjacent',(4,0,2),(1,1,1))
data=bpy.data.cameras.new('camera');data.type='ORTHO';data.ortho_scale=8
cam=bpy.data.objects.new('camera',data);bpy.context.scene.collection.objects.link(cam);cam.location=(0,0,10)
s=bpy.context.scene;s.camera=cam;s.render.engine='BLENDER_WORKBENCH'
s.render.resolution_x=s.render.resolution_y=128;s.render.resolution_percentage=100;s.render.image_settings.file_format='PNG'
bpy.context.view_layer.update();meshes=[target,support,cap1,cap2,adjacent]
first=overhead_blockers(s,cam,meshes)
assert {r['name'] for r in first['removable']}=={'cap2'}, first
report=render_with_overhead_removal(s,cam,meshes,out/'test.png',out,'top')
assert {r['name'] for r in report['hidden_objects']}=={'cap1','cap2'}, report
assert all(not o.hide_render for o in meshes)
assert (out/'test-before-occlusion.png').exists()
# Another question target is protected even when physically above the first.
cap1.hide_render=cap2.hide_render=True
upper=cube('upper-target',(0,0,2),(2,2,.2),'target')
bpy.context.view_layer.update()
checked=overhead_blockers(s,cam,meshes+[upper])
assert not checked['removable'] and any(r['name']==upper.name for r in checked['protected']), checked
upper.hide_render=True
# Actual triangle test: torus bbox overlaps the tiny target but its hole does not.
target.dimensions=(.1,.1,.1)
bpy.ops.mesh.primitive_torus_add(major_radius=1,minor_radius=.1,location=(0,0,2))
ring=bpy.context.object;ring['role']='context'
bpy.context.view_layer.update()
checked=overhead_blockers(s,cam,meshes+[upper,ring])
assert not checked['removable'], checked
print('OCCLUSION_TESTS_PASSED: stacked caps, support below, adjacent object, protected target, hollow mesh, restored visibility')
