"""Blender reference-only artifact, independent of generated reconstruction geometry."""
import bpy,json,sys,math
from pathlib import Path
from mathutils import Matrix,Vector
out=Path(sys.argv[sys.argv.index('--')+1]).resolve();ref=json.loads((out/'spatial-reference.json').read_text());data=json.loads((out/'input.json').read_text())
bpy.ops.wm.read_factory_settings(use_empty=True);s=bpy.context.scene
colors=data['colors'];mats={}
for name,color in {**colors,'trajectory':[.4,.45,.5,1]}.items():
 m=bpy.data.materials.new(name);m.diffuse_color=color;mats[name]=m
points=[r['position'] for r in ref['trajectory']]
scale=max((Vector([max(p[i] for p in points) for i in range(3)])-Vector([min(p[i] for p in points) for i in range(3)])).length,.1)
def line(name,points,material,width):
 c=bpy.data.curves.new(name,'CURVE');c.dimensions='3D';c.bevel_depth=width;c.bevel_resolution=1
 sp=c.splines.new('POLY');sp.points.add(len(points)-1)
 for node,p in zip(sp.points,points):node.co=(*p,1)
 ob=bpy.data.objects.new(name,c);s.collection.objects.link(ob);c.materials.append(material);return ob
line('Complete LingBot trajectory - reference only',points,mats['trajectory'],scale*.001)
frames={r['source_frame_index']:r for r in data['frames']}
for f in ref['selected_frames']:
 matrix=Matrix(frames[f['source_frame_index']]['c2w_blender']);obj=f['objects'][0] if f['objects'] else 'scene context'
 depth=scale*.035;k=frames[f['source_frame_index']]['intrinsics_640x480'];fx,fy,cx,cy=k[0][0],k[1][1],k[0][2],k[1][2]
 corners=[matrix@Vector(((u-cx)/fx*depth,-(v-cy)/fy*depth,-depth)) for u,v in [(0,0),(640,0),(640,480),(0,480)]];origin=matrix.translation
 for i,c in enumerate(corners):line(f"{f['timestamp_s']}s edge {i}",[origin,c],mats[obj],scale*.0008)
 line(f"{f['timestamp_s']}s frustum",corners+[corners[0]],mats[obj],scale*.0008)
 c=bpy.data.cameras.new(f"Reference {f['timestamp_s']}s");c.display_size=scale*.015;cam=bpy.data.objects.new(c.name,c);s.collection.objects.link(cam);cam.matrix_world=matrix;cam['groups']=json.dumps(f['groups']);cam['timestamp_s']=f['timestamp_s']
for a in ref['anchors']:
 if a['center'] is None:continue
 center=Vector(a['center']);radius=scale*.012
 for plane in [(0,1),(0,2),(1,2)]:
  pts=[]
  for i in range(33):
   p=center.copy();p[plane[0]]+=radius*math.cos(i*math.tau/32);p[plane[1]]+=radius*math.sin(i*math.tau/32);pts.append(p)
  ob=line('SOFT '+a['object'],pts,mats[a['object']],scale*.0008);ob['confidence']=a['confidence'];ob['not_verified_object']=True
s['description']='Empty reference: camera trajectory, selected frame frusta, soft visible-surface anchors. No room mesh, no ground truth, no hard registration.'
for screen in bpy.data.screens:
 for area in screen.areas:
  if area.type=='VIEW_3D':
   area.spaces.active.region_3d.view_distance=scale*1.3;area.spaces.active.region_3d.view_location=Vector(points[len(points)//2]);area.spaces.active.shading.color_type='MATERIAL'
bpy.ops.wm.save_as_mainfile(filepath=str(out/'empty-trajectory-reference.blend'))
