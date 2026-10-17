"""Blender candidate projections, without rendering or reading image pixels."""

import argparse
import json
from pathlib import Path
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
import refine_blender as b
import refine_probe as p


def bind_features(moving, features):
    """Unproject baseline pixels onto the first hit; never bind to another object."""
    scene = bpy.context.scene
    graph = bpy.context.evaluated_depsgraph_get()
    camera = scene.camera.evaluated_get(graph)
    if camera.data.type != "PERSP":
        raise ValueError("Partial feature binding currently requires a perspective camera")
    inverse = camera.calc_matrix_camera(
        graph, x=scene.render.resolution_x, y=scene.render.resolution_y,
        scale_x=scene.render.pixel_aspect_x, scale_y=scene.render.pixel_aspect_y).inverted()
    names = {obj.name for obj in moving}
    anchors = []
    for feature in features:
        xy = feature.get("baseline")
        if (not isinstance(xy, list) or len(xy) != 2 or
                any(type(v) not in (int, float) or not 0 <= v <= 1 for v in xy) or
                not isinstance(feature.get("id"), str) or feature.get("same_physical_point") is not True):
            continue
        local = inverse @ Vector((2 * xy[0] - 1, 1 - 2 * xy[1], -1, 1))
        local /= local.w
        origin = camera.matrix_world.translation
        direction = (camera.matrix_world @ local.to_3d() - origin).normalized()
        hit, location, _, _, obj, _ = scene.ray_cast(graph, origin, direction, distance=camera.data.clip_end)
        if hit and obj.name in names:
            anchors.append({"id": feature["id"], "object_name": obj.name,
                            "local_point": list(obj.matrix_world.inverted() @ location)})
    return anchors


def project_features(moving, anchors):
    scene = bpy.context.scene
    graph = bpy.context.evaluated_depsgraph_get()
    camera = scene.camera.evaluated_get(graph)
    by_name = {obj.name: obj.evaluated_get(graph) for obj in moving}
    records = []
    for anchor in anchors:
        if anchor["object_name"] not in by_name:
            raise ValueError("Observation anchor is outside the frozen transform group")
        point = by_name[anchor["object_name"]].matrix_world @ Vector(anchor["local_point"])
        ndc = world_to_camera_view(scene, camera, point)
        delta = point - camera.matrix_world.translation
        visible = False
        if 0 <= ndc.x <= 1 and 0 <= ndc.y <= 1 and camera.data.clip_start < ndc.z < camera.data.clip_end:
            hit, location, _, _, obj, _ = scene.ray_cast(
                graph, camera.matrix_world.translation, delta.normalized(), distance=delta.length + 0.002)
            visible = bool(hit and obj.name == anchor["object_name"] and (location - point).length < 0.002)
        records.append({**anchor, "xy": [ndc.x, 1 - ndc.y], "visible": visible})
    return records


def project(case, frames, candidates, observations=None, anchors=None):
    scene = bpy.context.scene
    scene.frame_set(frames[0])
    moving, reference, _ = b.state(case)
    matrices = {obj.name: obj.matrix_world.copy() for obj in moving}
    cameras = b.camera_trajectory(frames)
    anchors = dict(anchors or {})
    if observations is not None:
        for frame in frames:
            scene.frame_set(frame)
            anchors[str(frame)] = bind_features(moving, observations.get(str(frame), []))
    results = {}
    for candidate in candidates:
        scene.frame_set(frames[0])
        b.restore(moving, matrices)
        bpy.context.view_layer.update()
        b.apply_candidate(candidate, moving, reference)
        bpy.context.view_layer.update()
        records = []
        for frame in frames:
            scene.frame_set(frame)
            records.append({"scene_frame": frame, **p.visibility(moving),
                            "features": project_features(moving, anchors.get(str(frame), []))})
        if b.camera_trajectory(frames) != cameras:
            raise ValueError("Source camera changed")
        results[candidate["id"]] = records
    b.restore(moving, matrices)
    bpy.context.view_layer.update()
    return {"candidates": results, "source_camera_unchanged": True, "no_image_access": True,
            "transform_members": sorted(matrices)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    case, spec = b.read(args.case), b.read(args.spec)
    result = project(case, spec["scene_frames"], spec["candidates"],
                     spec.get("observations"), spec.get("anchors"))
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
