"""Image-plane visibility sampling against opaque world-space triangle BVHs.

The target-only silhouette is the denominator; the scene's first hit decides
visible samples. This measures geometry visibility, not VLM recognizability,
and ignores material transparency. Grid estimates can miss very thin parts.
"""
import itertools
import math


def _dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def image_ray(camera, x, y):
    """Return origin and unit direction for normalized image coordinates."""
    origin = tuple(camera['location'])
    right, up, forward = camera['right'], camera['up'], camera['forward']
    scale = camera.get('ortho_scale')
    if scale is not None:
        origin = tuple(origin[i]+(x-.5)*scale*right[i]+(y-.5)*scale*up[i]
                       for i in range(3))
        return origin, tuple(forward)
    tangent = camera.get('tan_half_fov')
    if tangent is None:
        tangent = camera.get('sensor_mm', 36.)/(2*camera['lens_mm'])
    direction = tuple(forward[i]+(2*x-1)*tangent*right[i]+(2*y-1)*tangent*up[i]
                      for i in range(3))
    length = math.sqrt(_dot(direction, direction))
    return origin, tuple(value/length for value in direction)


def make_target_bvh(item, triangles_by_name):
    """Build one physical instance's BVH from existing world-space triangles."""
    from mathutils.bvhtree import BVHTree
    vertices, faces = [], []
    for name in item['names']:
        for a, b, c, area in triangles_by_name.get(name, []):
            if area <= 1e-12:
                continue
            start = len(vertices)
            vertices.extend((a, b, c))
            faces.append((start, start+1, start+2))
    if not faces:
        raise ValueError('No target triangles: '+str(item.get('key', item['names'])))
    return BVHTree.FromPolygons(vertices, faces, all_triangles=True)


def _projected_bounds(camera, box):
    projected, depths = [], []
    scale = camera.get('ortho_scale')
    tangent = camera.get('tan_half_fov')
    if scale is None and tangent is None:
        tangent = camera.get('sensor_mm', 36.)/(2*camera['lens_mm'])
    for point in itertools.product(*zip(box['min'], box['max'])):
        delta = tuple(point[i]-camera['location'][i] for i in range(3))
        depth = _dot(delta, camera['forward'])
        depths.append(depth)
        if scale is not None:
            denominator = scale
        elif depth > .03:
            denominator = 2*tangent*depth
        else:
            continue
        projected.append((.5+_dot(delta, camera['right'])/denominator,
                          .5+_dot(delta, camera['up'])/denominator))
    if max(depths) <= .03:
        return None
    if scale is None and min(depths) <= .03:
        # A near-plane-crossing box can project beyond every image boundary.
        return (0., 0., 1., 1.)
    low = [max(0., min(p[i] for p in projected)) for i in (0, 1)]
    high = [min(1., max(p[i] for p in projected)) for i in (0, 1)]
    return (*low, *high) if all(high[i] > low[i] for i in (0, 1)) else None


def measure(camera, item_box, target_bvh, scene_bvh, face_names, item_names, grid=14):
    """Sample the clipped target bbox in image space (square 512px camera).

    Visibility fraction is relative to target-only silhouette hits; visible
    image fraction uses normalized image area. Width/height are sampled spans,
    not a guarantee that the complete target outline is visible.
    """
    from mathutils import Vector
    if grid < 2:
        raise ValueError('Use at least a 2x2 sampling grid')
    result = {'silhouette_visibility_fraction': 0., 'visible_image_fraction': 0.,
              'visible_width_px': 0., 'visible_height_px': 0.,
              'silhouette_samples': 0, 'visible_samples': 0, 'grid': grid,
              'projected_box': None}
    bounds = _projected_bounds(camera, item_box)
    if bounds is None:
        return result
    x0, y0, x1, y1 = bounds
    result['projected_box'] = list(bounds)
    target_names, visible_cells = set(item_names), []
    for row in range(grid):
        for col in range(grid):
            x = x0+(col+.5)*(x1-x0)/grid
            y = y0+(row+.5)*(y1-y0)/grid
            origin, direction = image_ray(camera, x, y)
            ray = (Vector(origin), Vector(direction), camera.get('clip_end', 100.))
            _, _, target_index, _ = target_bvh.ray_cast(*ray)
            if target_index is None:
                continue
            result['silhouette_samples'] += 1
            _, _, scene_index, _ = scene_bvh.ray_cast(*ray)
            if scene_index is not None and face_names[scene_index] in target_names:
                visible_cells.append((col, row))
    count = len(visible_cells)
    result['visible_samples'] = count
    if result['silhouette_samples']:
        result['silhouette_visibility_fraction'] = count/result['silhouette_samples']
    result['visible_image_fraction'] = (x1-x0)*(y1-y0)*count/(grid*grid)
    if count:
        result['visible_width_px'] = (max(c[0] for c in visible_cells)-min(c[0] for c in visible_cells)+1)*(x1-x0)*512/grid
        result['visible_height_px'] = (max(c[1] for c in visible_cells)-min(c[1] for c in visible_cells)+1)*(y1-y0)*512/grid
    return result
