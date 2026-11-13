"""Blender: attach exact reference cameras and inspect a joint scene without moving geometry."""
import bpy
import sys,json,math,hashlib
from pathlib import Path
from mathutils import Matrix,Vector,Euler
from bpy_extras.object_utils import world_to_camera_view

out=Path(sys.argv[sys.argv.index('--')+1]).resolve()
sys.path.insert(0,str(out))
from v10_contract import validate,validate_architecture
inputs=json.loads((out/'input.json').read_text());rows=inputs['frames'];s=bpy.context.scene
meshes=[o for o in s.objects if o.type=='MESH']
assert meshes,'Empty geometry'
expected=set(inputs['question']['objects'])
targets={k:[o for o in meshes if o.get('role')=='target' and o.get('category')==k] for k in expected}
scope_records=[dict(name=o.name,role=o.get('role'),category=o.get('category'),instance_id=o.get('instance_id'),
    relation=o.get('relation'),supports=o.get('supports'),contacts=o.get('contacts'),color=list(o.color),vertices=len(o.data.vertices),
    faces=len(o.data.polygons),modifiers=[m.type for m in o.modifiers]) for o in meshes]
scope=validate(scope_records,inputs['question'],inputs['instance_limits'])
if inputs.get('architecture_inventory_required'):
    scope['architecture']=validate_architecture(s.get('architecture_inventory'),scope_records,[f['timestamp_s'] for f in rows])
assert all(o.get('role') in {'target','context','structure'} for o in meshes),'Untagged mesh'
snapshot={o.name:[list(r) for r in o.matrix_world] for o in meshes}
inspection=[dict(name=o.name,role=o.get('role'),category=o.get('category'),
                location=list(o.matrix_world.translation),world_bbox=[list(o.matrix_world@Vector(v)) for v in o.bound_box]) for o in meshes]
context_checks=[dict(name=o.name,vertices=len(o.data.vertices),color=list(o.color),
                    white=min(o.color[:3])>=.9 and max(o.color[:3])-min(o.color[:3])<1e-4) for o in meshes if o.get('role')=='context']
assert all(r['white'] and r['vertices']==8 for r in context_checks),'Non-target context must be white cuboids'
(out/'geometry-inspection.json').write_text(json.dumps(inspection,indent=2))
scene_hash=hashlib.sha256((out/'scene.blend').read_bytes()).hexdigest()
s.render.engine='BLENDER_WORKBENCH';s.render.image_settings.file_format='PNG';s.render.image_settings.color_mode='RGB'
s.render.film_transparent=False;s.render.use_border=False;s.render.resolution_percentage=100
s.view_settings.view_transform='Standard'
if not s.world:s.world=bpy.data.worlds.new('World')
s.world.color=(.8,.8,.8)
sh=s.display.shading;sh.light='STUDIO';sh.color_type='OBJECT';sh.show_shadows=True;sh.show_cavity=True;sh.cavity_type='BOTH';sh.background_type='WORLD'
refcol=bpy.data.collections.new('LingBot fixed reference cameras');s.collection.children.link(refcol)
camera_list=[];camera_tests=[];box_metrics=[]
(out/'pose-renders').mkdir(exist_ok=True);(out/'views').mkdir(exist_ok=True)
def points_for(obs):return [o.matrix_world@Vector(v) for o in obs for v in o.bound_box]
def projected_bounds(camera,objs):
    points=[world_to_camera_view(s,camera,p) for p in points_for(objs)]
    if not points or any(p.z<=camera.data.clip_start for p in points):return None
    x=[p.x*640 for p in points];y=[(1-p.y)*480 for p in points]
    return [max(0,min(x)),max(0,min(y)),min(640,max(x)),min(480,max(y))]
def iou(a,b):
    if a is None or a[2]<=a[0] or a[3]<=a[1]:return 0.
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union else 0.
for i,row in enumerate(rows):
    name=f'LingBot {row["timestamp_s"]:.3f}s'
    d=bpy.data.cameras.new(name);d.type='PERSP';d.sensor_fit='HORIZONTAL';d.sensor_width=36;d.clip_start=.01;d.clip_end=1000;d.display_size=.1
    cam=bpy.data.objects.new(name,d);refcol.objects.link(cam);cam.matrix_world=Matrix(row['c2w_blender'])
    k=row['intrinsics_640x480'];fx,fy,cx,cy=k[0][0],k[1][1],k[0][2],k[1][2]
    d.lens=fx*36/640;d.shift_x=(320-cx)/640;d.shift_y=(cy-240)*(fx/fy)/640
    cam['timestamp_s']=row['timestamp_s'];cam['source_frame_index']=row['source_frame_index'];cam['intrinsics_json']=json.dumps(k)
    # Blender clamps each pixel-aspect component to >= 1. Preserve y/x=fx/fy
    # even when fx < fy, rather than silently clamping the ratio to 1.
    s.camera=cam;s.render.resolution_x=640;s.render.resolution_y=480
    s.render.pixel_aspect_x=max(1.,fy/fx);s.render.pixel_aspect_y=max(1.,fx/fy)
    bpy.context.view_layer.update();errs=[]
    for u,v in [(0,0),(640,0),(640,480),(0,480),(320,240)]:
        cp=Vector(((u-cx)/fx,-(v-cy)/fy,-1));wp=cam.matrix_world@cp
        proj=world_to_camera_view(s,cam,wp);errs.append(max(abs(proj.x*640-u),abs((1-proj.y)*480-v)))
    matrix_error=max(abs(cam.matrix_world[a][b]-row['c2w_blender'][a][b]) for a in range(4) for b in range(4))
    assert max(errs)<.02 and matrix_error<1e-5,(errs,matrix_error)
    filename=f'frame-{row["source_frame_index"]:06d}.png'
    s.render.filepath=str(out/'pose-renders'/filename);bpy.ops.render.render(write_still=True)
    camera_tests.append(dict(timestamp_s=row['timestamp_s'],source_frame_index=row['source_frame_index'],render='pose-renders/'+filename,
      max_projection_error_px=max(errs),matrix_error=matrix_error))
    for obs in row['observations']:
        box=obs.get('box_xyxy');label=obs.get('object')
        if box and label in targets:
            prediction=projected_bounds(cam,targets[label])
            box_metrics.append(dict(timestamp_s=row['timestamp_s'],category=label,dino_score=obs.get('dino_score'),
              detection_box=box,projected_geometry_box=prediction,iou=iou(prediction,box)))
    camera_list.append((cam,row))
# Store a trajectory at the original timestamps. These keyframes are references,
# not an assertion that interpolation between sparse samples matches the video.
animdata=bpy.data.cameras.new('Reference trajectory');animdata.sensor_fit='HORIZONTAL';animdata.sensor_width=36
anim=bpy.data.objects.new('Reference trajectory - fixed sample poses',animdata);s.collection.objects.link(anim)
s.render.fps=30;s.frame_start=1;s.frame_end=max(1,int(round(rows[-1]['timestamp_s']*30))+1)
for cam,row in camera_list:
    frame=int(round(row['timestamp_s']*30))+1;anim.matrix_world=cam.matrix_world.copy()
    anim.keyframe_insert(data_path='location',frame=frame);anim.keyframe_insert(data_path='rotation_euler',frame=frame)
    for prop in ['lens','shift_x','shift_y']:
        setattr(animdata,prop,getattr(cam.data,prop));animdata.keyframe_insert(data_path=prop,frame=frame)
s.frame_set(1)
overview_data=bpy.data.cameras.new('Overview');overview_data.type='ORTHO';overview_data.clip_end=1000
overview=bpy.data.objects.new('Overview',overview_data);s.collection.objects.link(overview);s.camera=overview
s.render.pixel_aspect_x=s.render.pixel_aspect_y=1;s.render.resolution_x=s.render.resolution_y=900
points=points_for(meshes);views=[]
for name,angles in [('top',[0,0,0])]+[(f'oblique-{yaw:03d}',[20,-20,yaw]) for yaw in range(0,360,45)]:
    e=Euler([math.radians(v) for v in angles],'XYZ');basis=e.to_matrix();p=[basis.transposed()@v for v in points]
    lo=Vector([min(v[i] for v in p) for i in range(3)]);hi=Vector([max(v[i] for v in p) for i in range(3)]);center=(lo+hi)/2
    overview.rotation_euler=e;overview.location=basis@Vector((center.x,center.y,hi.z+30));overview_data.ortho_scale=max(hi.x-lo.x,hi.y-lo.y)*1.15
    path=out/'views'/(name+'.png');s.render.filepath=str(path);bpy.ops.render.render(write_still=True)
    views.append(dict(id=name,image=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
(out/'views/views.json').write_text(json.dumps(dict(views=views),indent=2))
for o in meshes:
    assert max(abs(o.matrix_world[a][b]-snapshot[o.name][a][b]) for a in range(4) for b in range(4))<1e-6
assert hashlib.sha256((out/'scene.blend').read_bytes()).hexdigest()==scene_hash
s['alignment_note']='Exact supplied LingBot reference cameras. Geometry jointly predicted by Astra; not optimized by ICP/reprojection.'
bpy.ops.wm.save_as_mainfile(filepath=str(out/'scene-with-reference-cameras.blend'))
(out/'validation.json').write_text(json.dumps(dict(camera_alignment_verified=True,geometry_unchanged=True,
 original_scene_sha256=scene_hash,geometric_accuracy_verified=False,cameras=camera_tests,mesh_count=len(meshes),
 scope_validation=scope,
 context_white_cuboids=context_checks,
 box_comparison_note='Projected union of target mesh bounding-box corners versus noisy DINO boxes; clipped to frame, ignores occlusion. Diagnostic only.',
 box_comparisons=box_metrics),indent=2))
print('POSE_VALIDATION_COMPLETE',len(rows),len(meshes))
