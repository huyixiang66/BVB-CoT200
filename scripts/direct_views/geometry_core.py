"""Fit and refine cameras in the frozen Astra scene. No API or QA answers."""
import bisect
from collections import Counter
import json
import math
from pathlib import Path
import sys
import time

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fit_camera import bbox_corners, basis, center, fit_camera, project as pinhole_project, union_box
from fit_camera import viewing_ray


def cli_arg(name, default=None):
    argv = sys.argv
    if "--" not in argv:
        return default
    args = argv[argv.index("--") + 1:]
    for index, value in enumerate(args):
        if value == name and index + 1 < len(args):
            return args[index + 1]
    return default


SCENE = cli_arg('--scene', '0d2ee665be')
SPEC_PATH = Path(cli_arg('--spec', str(HERE / f'spec_{SCENE}.json')))
GROUPS_PATH = Path(cli_arg('--groups', str(HERE / f'object_groups_{SCENE}.json')))
MARGIN = .07
MIN_AREA = .0008
MIN_VISIBLE = .20
SAMPLES = 64
MAX_VIEWS = 6


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def bbox(obj):
    pts = [obj.matrix_world @ Vector(p) for p in obj.bound_box]
    return {'min': [min(p[i] for p in pts) for i in range(3)],
            'max': [max(p[i] for p in pts) for i in range(3)]}


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def unit_delta(a, b):
    delta = [y-x for x, y in zip(a, b)]
    length = math.sqrt(dot(delta, delta))
    return [v/length for v in delta] if length > 1e-8 else [0, 0, 0]


def project(point, camera):
    depth = dot([point[i]-camera['location'][i] for i in range(3)], camera['forward'])
    if camera.get('ortho_scale') is not None:
        delta = [point[i]-camera['location'][i] for i in range(3)]
        return (.5+dot(delta,camera['right'])/camera['ortho_scale'],
                .5+dot(delta,camera['up'])/camera['ortho_scale'], depth)
    return pinhole_project(point, camera) if depth > 0 else (0., 0., depth)


started = time.monotonic()
spec = read(SPEC_PATH)
groups = read(GROUPS_PATH)
questions = spec['questions']
current_question = None
scene = bpy.context.scene
ceiling_meshes = set(groups['room'].get('ceiling', []))
if groups['room'].get('remove_walls', False):
    ceiling_meshes.update(groups['room'].get('walls', []))
for obj in bpy.data.objects:
    if obj.type == 'MESH' and obj.name in ceiling_meshes:
        obj.hide_viewport = True
        obj.hide_render = True
bpy.context.view_layer.update()
depsgraph = bpy.context.evaluated_depsgraph_get()
instances = {}
for semantic, items in groups['groups'].items():
    instances[semantic] = [{'key': f'{semantic}#{i}', 'names': names,
        'box': union_box([bbox(bpy.data.objects[n]) for n in names])}
        for i, names in enumerate(items)]
floor_names = groups['room'].get('floor', [])
if not floor_names:
    raise ValueError('room.floor must name at least one Blender mesh')
floor = union_box([bbox(bpy.data.objects[n]) for n in floor_names])
instances['__room__'] = [{'key': 'room#0', 'names': groups['room']['floor'], 'box': floor}]
room_span = max(floor['max'][i]-floor['min'][i] for i in (0, 1))
ceiling_names = groups['room'].get('ceiling', [])
ceiling = (union_box([bbox(bpy.data.objects[n]) for n in ceiling_names])
           if ceiling_names else {'min': [floor['min'][0], floor['min'][1], floor['max'][2]+2.5],
                                  'max': [floor['max'][0], floor['max'][1], floor['max'][2]+3.0]})

# A single world-space BVH avoids repeated dependency-graph traversal per ray.
vertices, faces, face_names = [], [], []
triangles = {}
solid_boxes = []
for obj in bpy.data.objects:
    if obj.type != 'MESH' or obj.hide_render or obj.hide_viewport:
        continue
    solid_boxes.append(bbox(obj))
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    mesh.calc_loop_triangles()
    offset = len(vertices)
    pts = [obj.matrix_world @ v.co for v in mesh.vertices]
    vertices.extend(pts)
    own = []
    for triangle in mesh.loop_triangles:
        ids = tuple(triangle.vertices)
        a, b, c = (pts[i] for i in ids)
        area = (b-a).cross(c-a).length*.5
        if area > 1e-12:
            own.append((a.copy(), b.copy(), c.copy(), area))
        faces.append(tuple(offset+i for i in ids))
        face_names.append(obj.name)
    triangles[obj.name] = own
    evaluated.to_mesh_clear()
bvh = BVHTree.FromPolygons(vertices, faces, all_triangles=True)
print('BVH', len(vertices), len(faces), flush=True)


def halton(n, base):
    value, fraction = 0., 1.
    while n:
        fraction /= base
        value += fraction*(n % base)
        n //= base
    return value


for items in instances.values():
    for item in items:
        tris = [t for n in item['names'] for t in triangles.get(n, [])]
        cumulative = []
        for tri in tris:
            cumulative.append((cumulative[-1] if cumulative else 0)+tri[3])
        if not cumulative:
            raise ValueError('No surface samples: '+item['key'])
        item['points'] = []
        for i in range(SAMPLES):
            tri = tris[min(len(tris)-1, bisect.bisect_left(cumulative,
                       cumulative[-1]*(i+.5)/SAMPLES))]
            a, b, c, _ = tri
            u, v = math.sqrt(halton(i+1, 2)), halton(i+1, 3)
            item['points'].append(a*(1-u)+b*(u*(1-v))+c*(u*v))


def camera_free(location):
    # Conservative rejection of locations inside a mesh AABB; no scene edits.
    for box in solid_boxes:
        if all(box['min'][i]-.025 <= location[i] <= box['max'][i]+.025 for i in range(3)):
            return False
    return True


def evaluate(camera, items):
    metrics = {}
    for item in items:
        projected = [project(p, camera) for p in bbox_corners(item['box'])]
        positive = all(p[2] > .03 for p in projected)
        xs, ys = [p[0] for p in projected], [p[1] for p in projected]
        bounds = [min(xs), min(ys), max(xs), max(ys)]
        margin = min(bounds[0], bounds[1], 1-bounds[2], 1-bounds[3]) if positive else -1.
        area = (max(0, min(1, bounds[2])-max(0, bounds[0]))*max(0, min(1, bounds[3])-max(0, bounds[1]))) if positive else 0.
        visible, blockers = 0, Counter()
        for point in item['points']:
            x, y, depth = project(point, camera)
            if depth <= .03 or not (0 <= x <= 1 and 0 <= y <= 1):
                continue
            ray_origin, ray_direction, distance = viewing_ray(point,camera)
            if distance < 1e-8:
                continue
            hit, _, index, _ = bvh.ray_cast(Vector(ray_origin), Vector(ray_direction), distance+.025)
            name = face_names[index] if index is not None else None
            # Self-occluded back/interior surfaces are not external blockers:
            # the sampled ray still sees this same physical target instance.
            if name in item['names']:
                visible += 1
            elif name:
                blockers[name] += 1
        fraction = visible/SAMPLES
        metrics[item['key']] = {'visible_fraction': fraction, 'area_fraction': area,
            'edge_margin': margin, 'projected_box': bounds,
            'readable': positive and margin >= MARGIN-1e-5 and area >= MIN_AREA and fraction >= MIN_VISIBLE,
            'top_blockers': dict(blockers.most_common(3))}
    return metrics
