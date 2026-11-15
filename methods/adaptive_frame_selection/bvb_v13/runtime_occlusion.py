"""Blender-only view-local overhead removal; preserve targets and saved geometry."""
import json
import shutil
import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def geometry(objects, depsgraph):
    vertices, faces, owners = [], [], []
    for obj in objects:
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            mesh.calc_loop_triangles()
            offset = len(vertices)
            vertices.extend(evaluated.matrix_world @ v.co for v in mesh.vertices)
            for tri in mesh.loop_triangles:
                faces.append(tuple(offset+i for i in tri.vertices)); owners.append(obj)
        finally:
            evaluated.to_mesh_clear()
    return vertices, faces, owners


def tree(objects, depsgraph):
    vertices, faces, owners = geometry(objects, depsgraph)
    return (BVHTree.FromPolygons(vertices, faces, all_triangles=True) if faces else None), vertices, faces, owners


def overhead_blockers(scene, camera, meshes, grid=16):
    """Exact ray intersections on evaluated triangles, sampled over each target.

    Detect only a frontmost, physically higher non-target object. A support below
    an object and an overlapping bounding box alone are insufficient. Other target
    instances and walls are protected and reported, never removed to fake coverage.
    """
    if camera.data.type != 'ORTHO': raise ValueError('Overhead inspection requires an orthographic overview')
    depsgraph = bpy.context.evaluated_depsgraph_get()
    visible = [o for o in meshes if not o.hide_render]
    all_tree, _, _, owners = tree(visible, depsgraph)
    instances = {}
    for o in visible:
        if o.get('role') == 'target': instances.setdefault((o.get('category'), o.get('instance_id')), []).append(o)
    inverse = camera.matrix_world.inverted()
    direction = -(camera.matrix_world.to_quaternion() @ Vector((0, 0, 1)))
    direction.normalize()
    removable, protected, samples = {}, {}, 0
    for (category, ident), parts in instances.items():
        target_tree, vertices, faces, _ = tree(parts, depsgraph)
        if target_tree is None: continue
        coords = [inverse @ p for p in vertices]
        lo = [min(p[i] for p in coords) for i in range(2)]
        hi = [max(p[i] for p in coords) for i in range(2)]
        xy = [(lo[0]+(i+.5)*(hi[0]-lo[0])/grid, lo[1]+(j+.5)*(hi[1]-lo[1])/grid)
              for i in range(grid) for j in range(grid)]
        # Triangle centers supplement thin silhouettes missed by the regular grid.
        for face in faces[::max(1, len(faces)//256)]:
            p = inverse @ (sum((vertices[i] for i in face), Vector())/3)
            xy.append((p.x, p.y))
        own = {o.name for o in parts}
        for x, y in xy:
            origin = camera.matrix_world @ Vector((x, y, 0))
            hit, _, _, distance = target_tree.ray_cast(origin, direction)
            if hit is None: continue
            samples += 1
            front, _, index, front_distance = all_tree.ray_cast(origin, direction, distance+1e-5)
            if front is None or front_distance >= distance-1e-5: continue
            obj = owners[index]
            if obj.name in own or front.z <= hit.z+1e-5: continue
            forbidden = obj.get('role') == 'target' or obj.get('category') == 'wall'
            records = protected if forbidden else removable
            record = records.setdefault(obj.name, dict(name=obj.name, role=obj.get('role'),
                category=obj.get('category'), target_instances=[], sampled_blocked_rays=0))
            label = dict(category=category, instance_id=ident)
            if label not in record['target_instances']: record['target_instances'].append(label)
            record['sampled_blocked_rays'] += 1
    return dict(removable=list(removable.values()), protected=list(protected.values()), target_surface_samples=samples)


def render_with_overhead_removal(scene, camera, meshes, path, out, view_id):
    """Render first, remove evidenced overhead blockers, rerender until stable."""
    states = {o.name: o.hide_render for o in meshes}
    report = dict(view_id=view_id, enabled=True, policy='sampled_triangle_rays_physically_higher',
                  hidden_objects=[], passes=[], protected_blockers=[], geometry_modified=False,
                  targets_removed=False, before_image=None, after_image=path.name)
    try:
        scene.render.filepath = str(path); bpy.ops.render.render(write_still=True)
        for _ in range(len(meshes)+1):
            bpy.context.view_layer.update()
            checked = overhead_blockers(scene, camera, meshes)
            report['passes'].append(checked)
            report['protected_blockers'] = checked['protected']
            if not checked['removable']: break
            if report['before_image'] is None:
                before = path.with_name(path.stem+'-before-occlusion.png')
                shutil.copy2(path, before); report['before_image'] = before.name
            for item in checked['removable']:
                bpy.data.objects[item['name']].hide_render = True
                report['hidden_objects'].append(item)
            bpy.context.view_layer.update()
            scene.render.filepath = str(path); bpy.ops.render.render(write_still=True)
        else:
            raise RuntimeError('Overhead removal did not converge')
        report['remaining_removable_blockers'] = 0
        return report
    finally:
        for name, hidden in states.items(): bpy.data.objects[name].hide_render = hidden
        bpy.context.view_layer.update()
