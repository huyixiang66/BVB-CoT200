"""Blender: inspect raw ablation outputs and render consistent material previews."""
import bpy
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from mathutils import Euler, Vector

folder=Path(sys.argv[sys.argv.index('--')+1]).resolve()
source=Path(bpy.data.filepath);source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
scene=bpy.context.scene
scene.frame_set(scene.frame_start)
meshes=[o for o in scene.objects if o.type=='MESH']
deps=bpy.context.evaluated_depsgraph_get()
def points_of(objects):
    return [o.evaluated_get(deps).matrix_world@Vector(p) for o in objects for p in o.evaluated_get(deps).bound_box]
def box(points):
    lo=[min(p[i] for p in points) for i in range(3)]
    hi=[max(p[i] for p in points) for i in range(3)]
    return {'min':lo,'max':hi,'size':[b-a for a,b in zip(lo,hi)]}
objects=[{'name':o.name,'collections':[c.name for c in o.users_collection],
          'bbox':box(points_of([o])),'vertices':len(o.data.vertices),'polygons':len(o.data.polygons)} for o in meshes]
groups=[{'name':c.name,'mesh_parts':len([o for o in c.all_objects if o in meshes])}
        for c in bpy.data.collections if any(o in meshes for o in c.all_objects)]
original_camera=scene.camera
camera=[]
if original_camera:
    for t in [0,30,60,88]:
        frame=min(scene.frame_end,scene.frame_start+round(t*scene.render.fps/scene.render.fps_base))
        scene.frame_set(frame);c=original_camera.evaluated_get(bpy.context.evaluated_depsgraph_get())
        camera.append({'time_s':t,'frame':frame,'matrix_world':[list(row) for row in c.matrix_world]})
scene.frame_set(scene.frame_start)
original_matrices={o.name:[list(r) for r in o.matrix_world] for o in meshes}
roof=[]
for o in meshes:
    label=' '.join([o.name]+[c.name for c in o.users_collection])
    if re.search(r'roof|ceiling',label,re.I) and not re.search(r'light|lamp',label,re.I):
        o.hide_render=True;roof.append(o.name)
visible=[o for o in meshes if not o.hide_render]
points=points_of(visible)
assert points
stats={'source_blend':str(source),'source_sha256':source_hash,'mesh_parts':len(meshes),
       'mesh_collections':len(groups),'collections':groups,'objects':objects,
       'bounds':box(points_of(meshes)),'camera_name':original_camera.name if original_camera else None,
       'cameras':sum(o.type=='CAMERA' for o in scene.objects),'lights':sum(o.type=='LIGHT' for o in scene.objects),
       'camera_samples':camera,'camera_motion_present':bool(camera and camera[0]['matrix_world']!=camera[-1]['matrix_world']),
       'frame_start':scene.frame_start,'frame_end':scene.frame_end,
       'animation_fps':scene.render.fps/scene.render.fps_base,'unit_scale':scene.unit_settings.scale_length,
       'image_textures':sum(1 for m in bpy.data.materials if m.use_nodes for n in m.node_tree.nodes if n.type=='TEX_IMAGE'),
       'roof_hidden_for_preview_only':roof,
       'preview_note':'Workbench material colors, studio lighting, identical world-axis views; automatic per-scene framing. Independent reconstructions are not geometrically registered. Original blend unchanged.'}
scene.render.engine='BLENDER_WORKBENCH'
# Match numeric Principled base colors even when generated scripts did not set
# viewport diffuse_color. This affects this in-memory preview only.
for material in bpy.data.materials:
    if material.use_nodes:
        principled=next((n for n in material.node_tree.nodes if n.type=='BSDF_PRINCIPLED'),None)
        if principled and not principled.inputs['Base Color'].is_linked:
            material.diffuse_color=principled.inputs['Base Color'].default_value
scene.render.resolution_x=1200;scene.render.resolution_y=1200;scene.render.resolution_percentage=100
scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB'
scene.render.film_transparent=False
sh=scene.display.shading;sh.light='STUDIO';sh.color_type='MATERIAL'
sh.show_shadows=True;sh.show_cavity=True;sh.cavity_type='BOTH'
sh.background_type='WORLD';scene.world.color=(.8,.8,.8)
sh.show_specular_highlight=True
scene.view_settings.view_transform='Standard'
data=bpy.data.cameras.new('Ablation preview camera');data.type='ORTHO';data.clip_end=1000
cam=bpy.data.objects.new('Ablation preview camera',data);scene.collection.objects.link(cam);scene.camera=cam
views={}
for name,angles in [('top',(0,0,0)),('oblique',(20,-20,0))]:
    rot=Euler([math.radians(v) for v in angles],'XYZ');basis=rot.to_matrix()
    local=[basis.transposed()@p for p in points];bounds=box(local)
    lo=Vector(bounds['min']);hi=Vector(bounds['max']);center=(lo+hi)/2
    data.ortho_scale=max(hi.x-lo.x,hi.y-lo.y)*1.15
    cam.rotation_euler=rot;cam.location=basis@Vector((center.x,center.y,hi.z+30))
    scene.render.filepath=str(folder/f'{name}.png');bpy.ops.render.render(write_still=True)
    views[name]={'rotation_degrees':angles,'ortho_scale':data.ortho_scale,'location':list(cam.location)}
assert all(original_matrices[o.name]==[list(r) for r in o.matrix_world] for o in meshes)
assert hashlib.sha256(source.read_bytes()).hexdigest()==source_hash
stats['preview_views']=views
(folder/'scene-inspection.json').write_text(json.dumps(stats,indent=2)+'\n')
print('ABLATION_PREVIEW_OK',len(meshes),len(groups))
