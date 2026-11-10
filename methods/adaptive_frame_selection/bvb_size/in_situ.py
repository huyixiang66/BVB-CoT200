"""Blender-only, original-scene camera selection. No reconstruction or QA.

blender -b --python in_situ.py -- --blend scene.blend
  --target target_l --reference target_t --out output_directory
Mesh properties role=target and category identify complete object groups; noun resolution is upstream.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view


LIMITS = dict(margin=.05, minimum_area=.025, minimum_short_side=.08,
              minimum_unblocked_rays=.75, maximum_distance_ratio=1.15,
              maximum_depth_ratio=1.15, maximum_direction_angle_deg=20.)


def blockers(views, distance_ratio=1., depth_ratio=1., angle=0.):
    reasons = []
    for i, view in enumerate(views):
        if not view['in_frame']:
            reasons.append(f'object_{i}:cropped_or_behind_camera')
        if view['area'] < LIMITS['minimum_area'] or view['short_side'] < LIMITS['minimum_short_side']:
            reasons.append(f'object_{i}:too_small')
        if view['unblocked_ray_fraction'] < LIMITS['minimum_unblocked_rays']:
            reasons.append(f'object_{i}:external_occlusion')
        if view['own_visible_fraction'] < .20:
            reasons.append(f'object_{i}:insufficient_identity_surface')
    if distance_ratio > LIMITS['maximum_distance_ratio']:
        reasons.append('distance_difference')
    if depth_ratio > LIMITS['maximum_depth_ratio']:
        reasons.append('perspective_scale_difference')
    if angle > LIMITS['maximum_direction_angle_deg']:
        reasons.append('viewing_direction_difference')
    return reasons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--blend', type=Path, required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--check-pair', action='store_true', help='Diagnostic only: exercise pair branch even if one view passes')
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    args.out.mkdir(parents=True, exist_ok=False)
    digest = hashlib.sha256(args.blend.read_bytes()).hexdigest()
    bpy.ops.wm.open_mainfile(filepath=str(args.blend))
    scene = bpy.context.scene
    original = {o.name: ([list(r) for r in o.matrix_world], o.hide_render)
                for o in scene.objects if o.type == 'MESH'}
    groups = [[o for o in scene.objects if o.type == 'MESH' and o.get('role') == 'target' and o.get('category') == n]
              for n in (args.target, args.reference)]
    assert all(groups) and not set(groups[0]).intersection(groups[1])
    geometry = []
    for objects in groups:
        points = [o.matrix_world @ Vector(p) for o in objects for p in o.bound_box]
        low = Vector([min(p[i] for p in points) for i in range(3)])
        high = Vector([max(p[i] for p in points) for i in range(3)])
        samples = []
        for obj in objects:
            obj.data.calc_loop_triangles()
            triangles = list(obj.data.loop_triangles)
            for tri in triangles[::max(1, len(triangles)//24)]:
                samples.append(obj.matrix_world @ (sum((obj.data.vertices[j].co for j in tri.vertices), Vector())/3))
        geometry.append(dict(points=points, center=(low+high)/2, samples=samples,
                             names={o.name for o in objects}, low=low, high=high, radius=(high-low).length/2))
    # Permit normal mutual cover only when the two object bounds touch/overlap.
    tolerance = .03 * max(g['radius'] for g in geometry)
    contact = all(max(geometry[0]['low'][i], geometry[1]['low'][i]) <= min(geometry[0]['high'][i], geometry[1]['high'][i]) + tolerance for i in range(3))
    pair_names = geometry[0]['names'] | geometry[1]['names']
    midpoint = (geometry[0]['center']+geometry[1]['center'])/2
    radius = max(g['radius'] for g in geometry)
    source_camera_position = scene.camera.location.copy() if scene.camera else None
    cam = bpy.data.objects.new('Adaptive comparison camera', bpy.data.cameras.new('Adaptive comparison camera'))
    scene.collection.objects.link(cam)
    scene.camera = cam
    cam.data.type = 'PERSP'
    cam.data.lens = 50
    cam.data.clip_start = .001
    cam.data.clip_end = max(1000, radius*100)
    scene.render.resolution_x, scene.render.resolution_y = 900, 700
    scene.render.resolution_percentage = 100

    def camera(center, direction, distance):
        cam.location = center + direction*distance
        cam.rotation_euler = (-direction).to_track_quat('-Z', 'Y').to_euler()
        bpy.context.view_layer.update()

    def inspect(g):
        uv = [world_to_camera_view(scene, cam, p) for p in g['points']]
        width = max(p.x for p in uv)-min(p.x for p in uv)
        height = max(p.y for p in uv)-min(p.y for p in uv)
        margin = LIMITS['margin']
        penalty = sum(max(margin-p.x, 0, p.x-(1-margin)) + max(margin-p.y, 0, p.y-(1-margin))
                      + (10 if p.z <= 0 else 0) for p in uv)
        hits = own = mutual = 0
        dep = bpy.context.evaluated_depsgraph_get()
        for point in g['samples']:
            delta = point-cam.location
            hit, _, _, _, obj, _ = scene.ray_cast(dep, cam.location, delta.normalized(), distance=delta.length+1e-4)
            # Same-object self-occlusion is not an environmental obstruction.
            name = obj.original.name if hit else None
            own += bool(name in g['names'])
            mutual += bool(contact and name in pair_names and name not in g['names'])
            hits += bool(name in g['names'] or (contact and name in pair_names))
        return dict(in_frame=penalty == 0, framing_penalty=penalty,
                    area=width*height, short_side=min(width, height),
                    unblocked_ray_fraction=hits/max(len(g['samples']), 1),
                    own_visible_fraction=own/max(len(g['samples']), 1),
                    normal_mutual_cover_fraction=mutual/max(len(g['samples']), 1))

    def evaluate(x, pair):
        azimuth, elevation, distance = x
        direction = Vector((math.cos(elevation)*math.sin(azimuth),
                            -math.cos(elevation)*math.cos(azimuth), math.sin(elevation)))
        views, poses = [], []
        for i, g in enumerate(geometry):
            aim = g['center'] if pair else midpoint
            camera(aim, direction, distance)
            views.append(inspect(g))
            poses.append(dict(position=list(cam.location), lookat=list(aim), direction=list(direction), lens_mm=50))
        positions = [Vector(p['position']) for p in poses]
        distances = [(positions[i]-g['center']).length for i, g in enumerate(geometry)]
        depths = [(positions[i]-g['center']).dot(direction) for i, g in enumerate(geometry)]
        ratio = max(distances)/max(min(distances), 1e-8)
        depth_ratio = max(depths)/max(min(depths), 1e-8)
        angle = math.degrees((positions[0]-geometry[0]['center']).angle(positions[1]-geometry[1]['center']))
        reasons = blockers(views, ratio, depth_ratio, angle)
        loss = sum(100*v['framing_penalty'] + 8*max(0, .75-v['unblocked_ray_fraction']) + 5*max(0,.20-v['own_visible_fraction'])
                   + 100*max(0, LIMITS['minimum_area']-v['area'])
                   + 30*max(0, LIMITS['minimum_short_side']-v['short_side']) for v in views)
        loss += 10*max(0, ratio-1.15)+10*max(0, depth_ratio-1.15)+max(0, angle-20)*.1
        loss += .03*distance/radius + .5*(elevation-.55)**2
        return loss, dict(views=views, poses=poses, distance_ratio=ratio, depth_ratio=depth_ratio,
                          direction_angle_deg=angle, blockers=reasons, loss=loss, parameters=x.copy())

    def solve(pair, source_start=False):
        initial_radius = radius if pair else max((p-midpoint).length for g in geometry for p in g['points'])
        x, steps = [0., .55, initial_radius*4.], [.5, .2, radius]
        if source_start and source_camera_position is not None:
            delta = source_camera_position-midpoint
            x[0] = math.atan2(delta.x, -delta.y)
            x[1] = min(1.15, max(.25, math.atan2(delta.z, math.hypot(delta.x,delta.y))))
        best, record = evaluate(x, pair)
        feasible = record if not record['blockers'] else None
        evaluations = 1
        for _ in range(32):
            improved = False
            for k in range(3):
                for sign in (-1, 1):
                    y = x.copy()
                    y[k] += sign*steps[k]
                    y[1] = min(1.15, max(.25, y[1]))
                    y[2] = min(initial_radius*25, max(radius*1.5, y[2]))
                    score, rec = evaluate(y, pair)
                    evaluations += 1
                    if not rec['blockers'] and (feasible is None or score < feasible['loss']):
                        feasible = rec
                    if score < best:
                        x, best, record, improved = y, score, rec, True
            if not improved:
                steps = [s*.5 for s in steps]
            if max(steps[0], steps[1], steps[2]/radius) < .002:
                break
        chosen = feasible or record
        chosen['evaluations'] = evaluations
        return chosen

    common = solve(False)
    if common['blockers'] and source_camera_position is not None:
        fallback = solve(False, source_start=True)
        if not fallback['blockers'] or fallback['loss'] < common['loss']:
            common = fallback
    pair = solve(True) if common['blockers'] or args.check_pair else None
    mode = 'single' if not common['blockers'] else ('pair' if not pair['blockers'] else 'unresolved')
    selected = common if mode == 'single' else pair
    # Normalize illumination for native-scale scenes; do not alter geometry.
    for light in scene.objects:
        if light.type == 'LIGHT':light.data.energy = 0
    scene.view_settings.view_transform = 'AgX'
    scene.view_settings.exposure = 0
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 16
    scene.render.threads_mode = 'FIXED'
    scene.render.threads = 2
    scene.world.use_nodes = True
    scene.world.node_tree.nodes.get('Background').inputs['Color'].default_value = (.75,.75,.75,1)
    scene.world.node_tree.nodes.get('Background').inputs['Strength'].default_value = .8
    scene.display.shading.light = 'STUDIO'
    scene.display.shading.color_type = 'MATERIAL'
    scene.display.shading.show_cavity = True
    for mat in bpy.data.materials:
        if mat.use_nodes:
            node = next((n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
            if node and not node.inputs['Base Color'].is_linked:
                mat.diffuse_color = node.inputs['Base Color'].default_value
    images = []
    if selected is not None:
        for i, pose in enumerate(selected['poses'][:1] if mode == 'single' else selected['poses']):
            cam.location = Vector(pose['position'])
            cam.rotation_euler = (Vector(pose['lookat'])-cam.location).to_track_quat('-Z','Y').to_euler()
            filename = f'{mode}-{i+1}.png'  # unresolved images are diagnostic, not accepted views
            scene.render.filepath = str(args.out/filename)
            bpy.ops.render.render(write_still=True)
            images.append(filename)
    assert original == {o.name: ([list(r) for r in o.matrix_world], o.hide_render)
                        for o in scene.objects if o.type == 'MESH'}
    assert hashlib.sha256(args.blend.read_bytes()).hexdigest() == digest
    report = dict(source=str(args.blend), source_sha256=digest, mode=mode, images=images,
                  limits=LIMITS, contact_bounds_overlap=contact, common=common, pair=pair, pair_diagnostic_only=bool(args.check_pair and mode=='single'),
                  geometry_and_visibility_unchanged=True, qa_run=False,
                  method='continuous coordinate descent; source camera initialization fallback on failure; no fixed camera grid or global optimality guarantee',
                  limitations=['ray first-hit sampling is not pixel visibility or semantic recognition',
                               'same world orientation does not guarantee matching semantic front faces',
                               'reference reliability and question-to-instance mapping are upstream inputs'])
    (args.out/'result.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(mode=mode, common_blockers=common['blockers'], pair_blockers=pair['blockers'] if pair else None)))


if __name__ == '__main__':
    main()
