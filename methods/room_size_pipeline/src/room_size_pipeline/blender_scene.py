"""Postprocess upstream V10_2 scene: fixed architecture, unchanged furniture, top view."""
import json
import math
import sys
from pathlib import Path
import bpy
from mathutils import Vector, Matrix, Euler
from bpy_extras.object_utils import world_to_camera_view

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from room_size_pipeline.contracts import validate_scene, edge, validate_reference_size, validate_bounds
from room_size_pipeline.pipeline import read, write


def build(out):
    out=Path(out);spec=read(out/'estimate/data.json')
    scene=bpy.context.scene
    model={'layout':json.loads(scene['region_layout']),'openings':json.loads(scene['openings'])}
    rects=validate_scene(spec,model);dest=out/'render';dest.mkdir(exist_ok=True)
    # Keep upstream-generated furniture intact; only architecture is replaced, as in frozen runs.
    for obj in list(bpy.context.scene.objects):
        if obj.type=='MESH' and obj.get('category') in ['floor','wall','ceiling']:
            bpy.data.objects.remove(obj,do_unlink=True)
    scene=bpy.context.scene;scene.unit_settings.system='METRIC';scene.unit_settings.scale_length=1
    materials={};groups={}
    def material(color):
        key=tuple(color)
        if key not in materials:
            mat=bpy.data.materials.new('Material');mat.diffuse_color=(*color,1);materials[key]=mat
        return materials[key]
    def cube(name,center,size,color):
        bpy.ops.mesh.primitive_cube_add(size=1,location=center);o=bpy.context.object;o.name=name;o.dimensions=size
        bpy.ops.object.transform_apply(location=False,rotation=False,scale=True);o.data.materials.append(material(color));return o
    for key,(x0,x1,y0,y1) in rects.items():
        floor=cube('Floor_'+key,((x0+x1)/2,(y0+y1)/2,-.05),(x1-x0,y1-y0,.1),(.65,.59,.48))
        floor['region_id']=key;floor['kind']='floor'
        for side in ['north','south','east','west']:
            axis,line,lo,hi=edge(rects[key],side)
            gaps=sorted((lo+g['start'],lo+g['end']) for g in model['openings'] if g['region']==key and g['side']==side)
            parts=[];start=lo
            for a,b in gaps:
                if a>start:parts.append((start,a))
                start=max(start,b)
            if start<hi:parts.append((start,hi))
            offset=.04 if side in ['east','north'] else -.04
            for n,(a,b) in enumerate(parts):
                if b-a<1e-5:continue
                pos=((a+b)/2,line+offset,1.2) if axis=='h' else (line+offset,(a+b)/2,1.2)
                size=(b-a,.08,2.4) if axis=='h' else (.08,b-a,2.4)
                cube(f'Wall_{key}_{side}_{n}',pos,size,(.86,.86,.82))['kind']='wall'
    bpy.context.view_layer.update()
    for obj in scene.objects:
        if obj.type!='MESH' or obj.get('kind') in ['floor','wall']:continue
        region=obj.get('region_id')
        if region not in rects:raise ValueError('Missing region_id: '+obj.name)
        points=[obj.matrix_world@Vector(v) for v in obj.bound_box];x0,x1,y0,y1=rects[region]
        validate_bounds(rects[region],points)
        if obj.get('instance_id'):groups.setdefault(obj['instance_id'],[]).append(obj)
    actual=[];byid={}
    for ref in spec['references']:
        obs=groups.get(ref['id'],[])
        if not obs:raise ValueError('Reference absent: '+ref['id'])
        spaces={x['id']:x['space_id'] for x in spec['regions']}
        if any(spaces[o.get('region_id')]!=spaces[ref['region_id']] for o in obs):raise ValueError('Reference physical space mismatch')
        angles=[float(o['reference_yaw']) for o in obs]
        if max(angles)-min(angles)>1e-5:raise ValueError('Inconsistent reference assembly axes')
        rotation=Matrix.Rotation(-angles[0],4,'Z')
        if 'reference_euler' in obs[0]:rotation=Euler(tuple(obs[0]['reference_euler']),'XYZ').to_matrix().to_4x4().inverted()
        sizes=[]
        for axis in range(3):
            measured=[o for o in obs if axis not in list(o.get('measurement_exclude_axes',[]))]
            if not measured:raise ValueError('No measurement components for reference axis')
            points=[(Matrix([list(x) for x in o['measurement_matrix']]) if 'measurement_matrix' in o else rotation)@o.matrix_world@Vector(v) for o in measured for v in o.bound_box]
            sizes.append(max(p[axis] for p in points)-min(p[axis] for p in points))
        validate_reference_size(ref,sizes)
        world=[o.matrix_world@Vector(v) for o in obs for v in o.bound_box]
        center=[(min(p[i] for p in world)+max(p[i] for p in world))/2 for i in range(3)]
        actual.append({'id':ref['id'],'actual_local_size_m':sizes})
        byid[ref['id']]={'id':ref['id'],'label':ref['label'],'position':center,'size_m':sizes}
    areas=[]
    for key,p in rects.items():
        obj=bpy.data.objects['Floor_'+key]
        if abs(obj.dimensions.x-(p[1]-p[0]))>1e-5 or abs(obj.dimensions.y-(p[3]-p[2]))>1e-5:raise ValueError('Actual floor mismatch')
        areas.append(obj.dimensions.x*obj.dimensions.y)
    furniture=[o for o in scene.objects if o.type=='MESH' and o.get('kind') not in ['floor','wall']]
    write(dest/'geometry.json',{'actual_dimensions_pass':True,'objects':actual,'diagnostic_area_m2':sum(areas),'camera_alignment_preserved':False,
          'preserved_furniture':[{'name':o.name,'vertices':len(o.data.vertices),'instance_id':o.get('instance_id')} for o in furniture]})
    meshpoints=[o.matrix_world@Vector(v) for o in scene.objects if o.type=='MESH' for v in o.bound_box]
    lo=Vector([min(p[i] for p in meshpoints) for i in range(3)]);hi=Vector([max(p[i] for p in meshpoints) for i in range(3)])
    center=(lo+hi)/2;data=bpy.data.cameras.new('Top');cam=bpy.data.objects.new('Top',data);scene.collection.objects.link(cam)
    z=hi.z+max(hi.x-lo.x,hi.y-lo.y)*2+5;cam.location=(center.x,center.y,z);cam.rotation_euler=(0,0,0)
    data.type='ORTHO';data.ortho_scale=max(hi.x-lo.x,(hi.y-lo.y)*1.2)*1.06;data.clip_end=z+10;scene.camera=cam
    scene.render.engine='BLENDER_WORKBENCH';scene.display.shading.light='STUDIO';scene.display.shading.color_type='MATERIAL'
    scene.display.shading.show_shadows=True;scene.display.shading.show_cavity=True
    scene.render.resolution_x=1200;scene.render.resolution_y=1000;scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG';scene.render.filepath=str(dest/'top.png')
    bpy.context.view_layer.update();deps=bpy.context.evaluated_depsgraph_get()
    candidates=[]
    for ref in spec['references']:
        if not {0,1}.issubset(ref['qa_axes']):continue
        ref=dict(ref,qa_axes=[0,1])
        obs=groups[ref['id']]
        if 'reference_euler' in obs[0]:
            rotation=Euler(tuple(obs[0]['reference_euler']),'XYZ').to_matrix()
            if any(abs(rotation.col[a].z)>.1 for a in ref['qa_axes']):continue
        pts=[o.matrix_world@Vector(v) for o in obs for v in o.bound_box]
        x0,x1=min(p.x for p in pts),max(p.x for p in pts);y0,y1=min(p.y for p in pts),max(p.y for p in pts)
        # Compare scene ray visibility against isolated reference hits (not bbox fill ratio).
        hit_ref=visible=0
        for ix in range(15):
            for iy in range(15):
                origin=Vector((x0+(ix+.5)*(x1-x0)/15,y0+(iy+.5)*(y1-y0)/15,z));direction=Vector((0,0,-1))
                intersects=False
                for obj in obs:
                    inv=obj.matrix_world.inverted();localdir=(inv.to_3x3()@direction).normalized()
                    if obj.ray_cast(inv@origin,localdir)[0]:intersects=True;break
                if not intersects:continue
                hit_ref+=1;hit=scene.ray_cast(deps,origin,direction)
                if hit[0] and hit[4].get('instance_id')==ref['id']:visible+=1
        ratio=visible/max(1,hit_ref)
        projected=[world_to_camera_view(scene,cam,p) for p in pts]
        area=(max(p.x for p in projected)-min(p.x for p in projected))*(max(p.y for p in projected)-min(p.y for p in projected))
        complete=all(.01<=p.x<=.99 and .01<=p.y<=.99 for p in projected)
        print("REFERENCE_VISIBILITY",ref["id"],ratio,area,complete,flush=True)
        if ratio>=.8 and area>=.005 and complete:
            candidates.append((area*ratio,ref,ratio))
    if not candidates:
        write(dest/'reference.json',{'accepted':False,'reason':'No sufficiently visible, complete reference'})
    else:
        _,ref,visibility=max(candidates,key=lambda x:(x[0],x[1]['id']))
        item=byid[ref['id']];pos=world_to_camera_view(scene,cam,Vector(item['position']))
        horizontal='左侧' if pos.x<.33 else '右侧' if pos.x>.67 else '中部'
        vertical='下方' if pos.y<.33 else '上方' if pos.y>.67 else '中间'
        label=ref['label']
        if any(x in label for x in ['面积','平方米','㎡','m²']):raise ValueError('Invalid reference label')
        same=[o for o in byid.values() if o['label']==label]
        same.sort(key=lambda o:(o['position'][0],-o['position'][1],o['id']))
        ordinal=f'（同名物体按从左到右、同列从上到下数第{[o["id"] for o in same].index(ref["id"])+1}个）' if len(same)>1 else ''
        axes=ref['qa_axes'];dimensions=[]
        for axis in axes:
            name='长边' if item['size_m'][axis]>=item['size_m'][1-axis] else '短边'
            dimensions.append(f'{name}约{item["size_m"][axis]:.3g}米')
        text=f'图中{horizontal}{vertical}的{label}{ordinal}，'+ '、'.join(dimensions)
        write(dest/'reference.json',{'accepted':True,'id':ref['id'],'text':text,'visible_fraction':visibility,
                                     'projected_center':[pos.x,pos.y],'source':'adopted original-frame estimate, not measured'})
    bpy.ops.wm.save_as_mainfile(filepath=str(dest/'scene.blend'))
    bpy.ops.render.render(write_still=True)


if __name__=='__main__':build(Path(sys.argv[sys.argv.index('--')+1]).resolve())
