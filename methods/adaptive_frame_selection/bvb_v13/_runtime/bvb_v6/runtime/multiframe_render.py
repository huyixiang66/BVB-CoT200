"""Apply fitted corrections to a copy; render two original overview cameras."""
import json,math,sys
from pathlib import Path
import bpy
from mathutils import Euler,Matrix,Vector

source=Path(sys.argv[sys.argv.index('--')+1]);out=Path(sys.argv[sys.argv.index('--')+2])
alignment=json.loads((out/'alignment.json').read_text(encoding='utf8'))
merge=json.loads((source/'merge.json').read_text(encoding='utf8'))
parents={p['group_id']:bpy.data.objects[p['group_id']+' | shared-world frame'] for p in merge['parts']}
before={name:obj.matrix_world.copy() for name,obj in parents.items()}
geometry=[o for o in bpy.context.scene.objects if o.type=='MESH' and o.get('source_fragment')]
def points():
    return [o.matrix_world@Vector(c) for o in geometry for c in o.bound_box]
before_points=points()
for row in alignment['groups']:
    if row['status']!='fitted':continue
    parent=bpy.data.objects[row['group_id']+' | shared-world frame']
    pivot=Vector(row['pivot']);delta=Vector(row['translation'])
    transform=Matrix.Translation(pivot+delta)@Matrix.Rotation(math.radians(row['yaw_degrees']),4,'Z')@Matrix.Translation(-pivot)
    meshes=[o for o in bpy.context.scene.objects if o.type=='MESH' and o.get('source_fragment')==row['group_id']]
    prior={o.name:o.matrix_world.copy() for o in meshes}
    parent.matrix_world=transform@parent.matrix_world;bpy.context.view_layer.update()
    for o in meshes:
        expected=transform@prior[o.name]
        assert max(abs(expected[i][j]-o.matrix_world[i][j]) for i in range(4) for j in range(4))<1e-4
    part=next(p for p in merge['parts'] if p['group_id']==row['group_id'])
    part['world_from_fragment_before_alignment']=part['world_from_fragment']
    part['world_from_fragment']=[list(v) for v in transform]
    part['multiframe_correction']={'pivot':row['pivot'],'translation':row['translation'],'yaw_degrees':row['yaw_degrees']}
after={name:obj.matrix_world.copy() for name,obj in parents.items()}
both_points=before_points+points()
scene=bpy.context.scene;scene.render.engine='BLENDER_WORKBENCH';scene.render.resolution_x=1400;scene.render.resolution_y=1400;scene.render.resolution_percentage=100
scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB'
camera=bpy.data.objects['Merged overview'];scene.camera=camera
for view in ('top','oblique'):
    d=merge['views'][view];rotation=Euler([math.radians(v) for v in d['rotation_degrees']],'XYZ');basis=rotation.to_matrix()
    projected=[basis.transposed()@p for p in both_points]
    lo=Vector([min(p[i] for p in projected) for i in range(3)]);hi=Vector([max(p[i] for p in projected) for i in range(3)])
    center=(lo+hi)/2;camera.rotation_euler=rotation;camera.data.ortho_scale=max(hi.x-lo.x,hi.y-lo.y)*1.12
    camera.location=basis@Vector((center.x,center.y,hi.z+30))
    for label,transforms in (('before',before),('aligned',after)):
        for gid,matrix in transforms.items():parents[gid].matrix_world=matrix
        bpy.context.view_layer.update()
        scene.render.filepath=str((out/(label+'-'+view+'.png')).resolve());bpy.ops.render.render(write_still=True)
bpy.ops.wm.save_as_mainfile(filepath=str((out/'aligned.blend').resolve()))
(out/'aligned-merge.json').write_text(json.dumps(merge,ensure_ascii=False,indent=2),encoding='utf8')
print('ALIGNED',sum(r['status']=='fitted' for r in alignment['groups']))
