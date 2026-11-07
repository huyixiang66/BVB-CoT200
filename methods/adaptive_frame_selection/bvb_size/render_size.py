"""Blender entry: blender -b scene.blend --python render_size.py -- groups.json output_dir."""
import json
import math
import sys
from pathlib import Path
import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bvb_size.policy import choose_reference, load_priors, needs_detail, reference_prompt


def points(objects):
    return [obj.matrix_world @ Vector(p) for obj in objects for p in obj.bound_box]


def bounds(objects):
    ps = points(objects)
    return Vector([min(p[i] for p in ps) for i in range(3)]), Vector([max(p[i] for p in ps) for i in range(3)])


def objects_for(item):
    objects = [bpy.data.objects.get(n) for n in item['meshes']]
    if not objects or any(o is None or o.type != 'MESH' for o in objects):
        raise ValueError('Missing or non-mesh object in ' + str(item['id']))
    return objects


def prepare(item):
    objects = objects_for(item)
    lo, hi = bounds(objects)
    return dict(item, extent=list(hi-lo)), objects


def projected(objects, right, up):
    return [(p.dot(right), p.dot(up)) for p in points(objects)]


def frame(objects, right, up):
    p = projected(objects, right, up)
    lo, hi = bounds(objects)
    center = (lo+hi)*0.5
    cx = (min(v[0] for v in p)+max(v[0] for v in p))*0.5
    cy = (min(v[1] for v in p)+max(v[1] for v in p))*0.5
    center += right*(cx-center.dot(right))+up*(cy-center.dot(up))
    width = max(v[0] for v in p)-min(v[0] for v in p)
    height = max(v[1] for v in p)-min(v[1] for v in p)
    return center, max(width, 1.5*height)*1.15, width, height


def main():
    args = sys.argv[sys.argv.index('--')+1:]
    spec = json.loads(Path(args[0]).read_text(encoding='utf-8'))
    out = Path(args[1]); out.mkdir(parents=True, exist_ok=True)
    target, target_objects = prepare(spec['target'])
    if not target.get('verified'):
        raise ValueError('Target identity and mesh group must be verified')
    candidates, lookup = [], {}
    for item in spec.get('candidates', []):
        if set(item['meshes']) & set(spec['target']['meshes']):
            continue
        candidate, objects = prepare(item)
        candidates.append(candidate); lookup[candidate['id']] = objects
    reference, reason = choose_reference(target, candidates, load_priors())
    ref_objects = lookup[reference['id']] if reference else []
    scene = bpy.context.scene
    for obj in scene.objects:
        obj.hide_render = obj not in target_objects + ref_objects
    for obj in target_objects + ref_objects:
        for mat in obj.data.materials:
            if mat and mat.use_nodes and mat.node_tree:
                shader = next((n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
                if shader and not shader.inputs['Base Color'].is_linked:
                    mat.diffuse_color = shader.inputs['Base Color'].default_value
    size = Vector(target['extent'])
    elevation = math.radians(25 if size.z >= max(size.x, size.y) else 35)
    view = Vector((math.cos(math.pi/4)*math.cos(elevation), -math.sin(math.pi/4)*math.cos(elevation), math.sin(elevation)))
    rot = (-view).to_track_quat('-Z','Y'); right = rot @ Vector((1,0,0)); up = rot @ Vector((0,1,0))
    if ref_objects:
        a, b = projected(target_objects,right,up), projected(ref_objects,right,up)
        lo,hi = bounds(target_objects); rlo,rhi = bounds(ref_objects)
        offset = max(p[0] for p in a)+max(.15,max(size)*.1)-min(p[0] for p in b)
        depth = Vector((view.x,view.y,0)).normalized(); delta = ((lo+hi-rlo-rhi)*.5).dot(depth)
        shift = right*offset+depth*delta; shift.z += lo.z-rlo.z
        for obj in ref_objects: obj.location += shift
        bpy.context.view_layer.update()
    included = target_objects+ref_objects
    center, scale, _, _ = frame(included,right,up)
    camdata=bpy.data.cameras.new('size_camera');camera=bpy.data.objects.new('size_camera',camdata);scene.collection.objects.link(camera)
    camera.location=center+view*10;camera.rotation_euler=rot.to_euler();camdata.type='ORTHO';camdata.ortho_scale=scale;scene.camera=camera
    scene.render.engine='BLENDER_WORKBENCH';scene.display.shading.light='STUDIO';scene.display.shading.color_type='MATERIAL'
    scene.render.resolution_x=1200;scene.render.resolution_y=800;scene.render.resolution_percentage=100
    scene.render.filepath=str(out/'main.png');bpy.ops.render.render(write_still=True)
    _,_,width,height=frame(target_objects,right,up); pixels=max(width,height)*1200/scale
    axis=max(range(3),key=lambda i:size[i]);v=Vector([int(i==axis) for i in range(3)])
    retention=math.sqrt(max(0,1-v.dot(view)**2));images=['main.png'];notes=[]
    if needs_detail(pixels,retention):
        if pixels < 200:
            # Recognition detail has different scale and must not be used as a ruler.
            for obj in ref_objects:obj.hide_render=True
            bpy.context.view_layer.update();center,camdata.ortho_scale,_,_=frame(target_objects,right,up)
            camera.location=center+view*10;scene.render.filepath=str(out/'detail.png');bpy.ops.render.render(write_still=True)
            images.append('detail.png');notes.append('detail has independent scale')
            if retention < .8: notes.append('comparison-axis compression remains unresolved')
        else:
            view2=Vector((view.x*.45,view.y*.45,.9)).normalized();rot2=(-view2).to_track_quat('-Z','Y')
            right2=rot2@Vector((1,0,0));up2=rot2@Vector((0,1,0));center,camdata.ortho_scale,_,_=frame(included,right2,up2)
            camera.location=center+view2*10;camera.rotation_euler=rot2.to_euler();scene.render.filepath=str(out/'complement.png');bpy.ops.render.render(write_still=True)
            images.append('complement.png');notes.append('complement has independent scale')
    result={'target':target['category'],'reference':reference['category'] if reference else None,'reference_id':reference['id'] if reference else None,'reason':reason,'images':images,'target_major_pixels':pixels,'axis_retention':retention,'prompt':reference_prompt(spec['question'],target['category'],reference['category'] if reference else None),'notes':notes}
    (out/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
