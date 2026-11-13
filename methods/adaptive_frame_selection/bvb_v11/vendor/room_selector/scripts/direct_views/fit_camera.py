"""Analytic square-camera framing for world-space axis-aligned boxes.

Coordinates are right-handed. The camera looks along local -Z with local +Y up,
as Blender does. Image coordinates increase rightward and upward from (0, 0).
Only framing is solved here; visibility, collision and readability are separate.
"""

import itertools
import math


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def unit(vector):
    length = math.sqrt(dot(vector, vector))
    if length < 1e-12:
        raise ValueError("A camera direction must be nonzero")
    return tuple(value / length for value in vector)


def bbox_corners(box):
    return [tuple(values) for values in itertools.product(
        *zip(box["min"], box["max"]))]


def union_box(boxes):
    if not boxes:
        raise ValueError("At least one target box is required")
    return {
        "min": [min(box["min"][i] for box in boxes) for i in range(3)],
        "max": [max(box["max"][i] for box in boxes) for i in range(3)],
    }


def center(box):
    return tuple((low + high) / 2 for low, high in zip(box["min"], box["max"]))


def basis(outward):
    """Return (right, up, forward) for a direction from target toward camera."""
    outward = unit(outward)
    forward = tuple(-value for value in outward)
    reference_up = (0.0, 0.0, 1.0)
    if abs(dot(forward, reference_up)) > 1.0 - 1e-10:
        reference_up = (0.0, 1.0, 0.0)
    right = unit(cross(forward, reference_up))
    up = unit(cross(right, forward))
    return right, up, forward


def project(point, camera):
    """Project a world point to normalized (x, y, positive-forward depth)."""
    delta = sub(point, camera["location"])
    depth = dot(delta, camera["forward"])
    if depth <= 0:
        raise ValueError("Cannot project a point on or behind the camera")
    scale = 2.0 * camera["tan_half_fov"] * depth
    return (0.5 + dot(delta, camera["right"]) / scale,
            0.5 + dot(delta, camera["up"]) / scale, depth)


def viewing_ray(point, camera):
    """Ray through a world point; orthographic rays have shifted origins."""
    origin = tuple(camera['location'])
    if camera.get('ortho_scale') is not None:
        delta = sub(point, origin)
        x, y = dot(delta, camera['right']), dot(delta, camera['up'])
        origin = tuple(origin[i]+x*camera['right'][i]+y*camera['up'][i] for i in range(3))
    direction = sub(point, origin)
    distance = math.sqrt(dot(direction,direction))
    return origin, unit(direction), distance


def fit_camera(boxes, azimuth_deg, elevation_deg, margin=0.07,
               lens_mm=35.0, sensor_mm=36.0):
    """Find the minimum framing distance along one specified outward direction.

    Azimuth is measured from world +X toward +Y; elevation is above the XY
    plane. For each corner, the inequalities are linear in camera distance:
    d >= z + abs(x)/(tan(fov/2)*(1-2*margin)), and equivalently for y.
    All object boxes are constrained, including their invisible/back corners.
    """
    if not 0 <= margin < 0.5 or lens_mm <= 0 or sensor_mm <= 0:
        raise ValueError("Use a margin in [0, .5) and positive camera intrinsics")
    target = center(union_box(boxes))
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    outward = (math.cos(elevation) * math.cos(azimuth),
               math.cos(elevation) * math.sin(azimuth), math.sin(elevation))
    right, up, forward = basis(outward)
    tan_half_fov = sensor_mm / (2.0 * lens_mm)
    usable_half_fov = tan_half_fov * (1.0 - 2.0 * margin)
    points = [point for box in boxes for point in bbox_corners(box)]
    distance = 0.05
    for point in points:
        delta = sub(point, target)
        z = dot(delta, outward)
        distance = max(distance, z + 0.05,
                       z + abs(dot(delta, right)) / usable_half_fov,
                       z + abs(dot(delta, up)) / usable_half_fov)
    distance += 1e-8 * max(1.0, distance)
    camera = {
        "location": [target[i] + distance * outward[i] for i in range(3)],
        "target": list(target), "right": list(right), "up": list(up),
        "forward": list(forward), "distance": distance,
        "tan_half_fov": tan_half_fov, "lens_mm": lens_mm,
        "sensor_mm": sensor_mm, "margin": margin,
        "azimuth_deg": float(azimuth_deg), "elevation_deg": float(elevation_deg),
    }
    projections = [project(point, camera) for point in points]
    camera["projected_bounds"] = {
        "min": [min(point[i] for point in projections) for i in range(2)],
        "max": [max(point[i] for point in projections) for i in range(2)],
    }
    return camera
