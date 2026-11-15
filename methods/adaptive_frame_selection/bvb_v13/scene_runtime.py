"""Fixed Blender compiler runtime; contains no model-authored Python."""
import json,math

def build(spec,destination):
    import bpy
    from mathutils import Matrix,Euler,Vector
    bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
    scene=bpy.context.scene
    scene.render.engine='BLENDER_WORKBENCH'
    scene.world.color=(.8,.8,.8)
    scene.render.resolution_x=900;scene.render.resolution_y=900;scene.render.resolution_percentage=100
    materials={}
    def transform(pos,rot):return Matrix.Translation(Vector(pos)) @ Euler([math.radians(v) for v in rot],'XYZ').to_matrix().to_4x4()
    def primitive(name,shape,pos,size,rot,color,parent=None):
        if shape=='box':bpy.ops.mesh.primitive_cube_add(size=1)
        elif shape=='cylinder':bpy.ops.mesh.primitive_cylinder_add(vertices=24,radius=.5,depth=1)
        elif shape=='sphere':bpy.ops.mesh.primitive_uv_sphere_add(segments=24,ring_count=12,radius=.5)
        elif shape=='torus':bpy.ops.mesh.primitive_torus_add(major_segments=32,minor_segments=12,major_radius=.35,minor_radius=.15)
        else:raise ValueError('Unknown compiled shape')
        obj=bpy.context.object;obj.name=name;obj.dimensions=size
        bpy.context.view_layer.update();bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
        obj.matrix_world=(parent if parent is not None else Matrix.Identity(4)) @ transform(pos,rot)
        obj.color=color;key=tuple(color)
        if key not in materials:
            mat=bpy.data.materials.new('label_'+str(len(materials)));mat.diffuse_color=color;mat.use_nodes=True
            mat.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value=color
            mat.node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value=.7;materials[key]=mat
        obj.data.materials.append(materials[key]);return obj
    for entity in spec['objects']:
        parent=transform(entity['position'],entity['rotation'])
        for n,part in enumerate(entity['parts']):
            obj=primitive(entity['id']+f'_part_{n:03d}',part['shape'],part['position'],part['size'],part['rotation'],entity['color'],parent)
            obj['role']='target';obj['category']=entity['category'];obj['instance_id']=entity['id'];obj['bvb_role']='furniture'
    inventory=[]
    for entity in spec['architecture']:
        obj=primitive(entity['id'],'box',entity['position'],entity['size'],entity['rotation'],[1,1,1,1])
        obj['role']='structure';obj['category']=entity['kind'];obj['bvb_role']=entity['kind']
        inventory.append(dict(id=entity['id'],kind=entity['kind'],label=entity['kind'],evidence_times_s=entity['times'],mesh_names=[obj.name]))
    for entity in spec['context']:
        obj=primitive(entity['id'],'box',entity['position'],entity['size'],entity['rotation'],[1,1,1,1])
        obj['role']='context';obj['category']='support';obj['relation']=entity['relation']
        obj['supports' if entity['relation']=='support' else 'contacts']=json.dumps(entity['targets'])
    for opening in spec['openings']:
        inventory.append(dict(id=opening['id'],kind='opening',label='opening',evidence_times_s=opening['times'],mesh_names=opening['walls']))
    scene['architecture_inventory']=json.dumps(inventory)
    scene['target_evidence_inventory']='[]';scene['compiler']='bvb_v13/'+spec['version']
    bpy.ops.object.camera_add(location=(8,-8,8));camera=bpy.context.object
    camera.rotation_euler=(Vector((0,0,0))-camera.location).to_track_quat('-Z','Y').to_euler();scene.camera=camera
    bpy.ops.object.light_add(type='AREA',location=(0,0,8));bpy.context.object.data.energy=1000;bpy.context.object.data.shape='DISK';bpy.context.object.data.size=8
    bpy.context.view_layer.update();bpy.ops.wm.save_as_mainfile(filepath=destination)
