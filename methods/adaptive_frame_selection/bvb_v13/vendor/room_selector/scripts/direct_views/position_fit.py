"""Fit a bounded-FOV camera at a fixed, physically chosen position.

This only solves framing. Scene collision, occlusion and target readability
remain the caller's responsibility. Coordinates follow Blender's -Z/+Y camera.
"""
import math

from fit_camera import basis, bbox_corners, center, dot, project, sub, union_box, unit


def from_position(boxes, location, target=None, min_lens=24, max_lens=35,
                  margin=.07):
    """Return the longest fitting focal length at this position, or None.

    All corners must be in front by more than .05 scene units. Lenses shorter
    than min_lens are rejected instead of moving the camera or clipping targets.
    The square sensor is fixed at 36mm, matching the existing rendering setup.
    """
    if not 0 <= margin < .5 or not 0 < min_lens <= max_lens:
        raise ValueError('Use a margin in [0,.5) and positive ordered lens limits')
    box = union_box(boxes)
    target = tuple(center(box) if target is None else target)
    location = tuple(location)
    delta = sub(location, target)
    distance = math.sqrt(dot(delta, delta))
    if distance < 1e-12:
        return None
    outward = unit(delta)
    right, up, forward = basis(outward)
    points = [point for box in boxes for point in bbox_corners(box)]
    needed_tan = 0.
    for point in points:
        relative = sub(point, location)
        depth = dot(relative, forward)
        if depth <= .05:
            return None
        needed_tan = max(needed_tan, abs(dot(relative, right))/depth,
                         abs(dot(relative, up))/depth)
    needed_tan /= 1-2*margin
    lens = min(max_lens, 36/(2*needed_tan)) if needed_tan else max_lens
    if lens < min_lens*(1-1e-10):
        return None
    lens = max(min_lens, lens)
    camera = {
        'location': list(location), 'target': list(target),
        'right': list(right), 'up': list(up), 'forward': list(forward),
        'tan_half_fov': 36/(2*lens), 'lens_mm': lens, 'sensor_mm': 36.,
        'ortho_scale': None, 'distance': distance, 'margin': margin,
        'elevation_deg': math.degrees(math.atan2(outward[2],
                                              math.hypot(outward[0], outward[1]))),
        'azimuth_deg': math.degrees(math.atan2(outward[1], outward[0])) % 360,
    }
    projections = [project(point, camera) for point in points]
    camera['projected_bounds'] = {
        'min': [min(p[i] for p in projections) for i in (0, 1)],
        'max': [max(p[i] for p in projections) for i in (0, 1)],
    }
    return camera


def perimeter_positions(floor_box, heights=(1.4, 1.9, 2.3), inset=.18):
    """Eight inset corner/edge positions per height, without collision checks."""
    if inset < 0:
        raise ValueError('Inset must be nonnegative')
    x0, y0 = (floor_box['min'][i]+inset for i in (0, 1))
    x1, y1 = (floor_box['max'][i]-inset for i in (0, 1))
    if x0 >= x1 or y0 >= y1:
        return []
    xm, ym = (x0+x1)/2, (y0+y1)/2
    xy = ((x0, y0), (xm, y0), (x1, y0), (x1, ym),
          (x1, y1), (xm, y1), (x0, y1), (x0, ym))
    return [[x, y, floor_box['max'][2]+height]
            for height in heights for x, y in xy]
