"""Batch position/scale box projections and optional saved preview; no pixel reads."""

import argparse
import json
from pathlib import Path
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import refine_blender as b
import refine_probe as p


def unclipped_box(objects):
    """Unclipped AABB projection; invalidate near-plane crossing instead of truncating it."""
    scene = bpy.context.scene
    graph = bpy.context.evaluated_depsgraph_get()
    camera = scene.camera.evaluated_get(graph)
    points = []
    for obj in objects:
        evaluated = obj.evaluated_get(graph)
        for corner in evaluated.bound_box:
            ndc = world_to_camera_view(scene, camera, evaluated.matrix_world @ Vector(corner))
            if not camera.data.clip_start < ndc.z < camera.data.clip_end:
                return None
            points.append((ndc.x, 1-ndc.y))
    return [min(x for x, y in points), min(y for x, y in points),
            max(x for x, y in points), max(y for x, y in points)]


def visible_surface_box(objects):
    """Project sampled mesh surfaces that survive scene occlusion, without pixels."""
    scene = bpy.context.scene
    graph = bpy.context.evaluated_depsgraph_get()
    camera = scene.camera.evaluated_get(graph)
    origin = camera.matrix_world.translation
    visible, inside = [], 0
    for obj in objects:
        evaluated = obj.evaluated_get(graph)
        mesh = evaluated.to_mesh()
        try:
            mesh.calc_loop_triangles()
            points = [v.co.copy() for v in list(mesh.vertices)[::max(1, len(mesh.vertices) // 256)]]
            triangles = list(mesh.loop_triangles)
            for triangle in triangles[::max(1, len(triangles) // 256)]:
                a, c, d = [mesh.vertices[i].co for i in triangle.vertices]
                points.extend(((a+c+d)/3, (a+c)/2, (a+d)/2, (c+d)/2))
            for local in points:
                point = evaluated.matrix_world @ local
                ndc = world_to_camera_view(scene, camera, point)
                if not (0 <= ndc.x <= 1 and 0 <= ndc.y <= 1 and
                        camera.data.clip_start < ndc.z < camera.data.clip_end):
                    continue
                inside += 1
                direction = point-origin
                hit, location, _, _, blocker, _ = scene.ray_cast(
                    graph, origin, direction.normalized(), distance=direction.length + .002)
                if hit and blocker.name == obj.name and (location-point).length <= .002:
                    visible.append((ndc.x, 1-ndc.y))
        finally:
            evaluated.to_mesh_clear()
    projected = ([min(x for x, y in visible), min(y for x, y in visible),
                  max(x for x, y in visible), max(y for x, y in visible)] if visible else None)
    return {"projected_bbox_xyxy": projected, "first_hit_group_samples": len(visible),
            "frustum_samples": inside, "projection_method": "occlusion_tested_surface_samples"}


def transform(moving, matrices, parameters, observed=None):
    b.restore(moving, matrices)
    bpy.context.view_layer.update()
    box = b.bbox(observed if observed is not None else moving)
    pivot = Vector([*box["center_xyz"][:2], box["min_xyz"][2]])
    x, y, scale = parameters
    matrix = Matrix.Translation(Vector((x, y, 0))) @ Matrix.Translation(pivot) @ Matrix.Scale(
        scale, 4) @ Matrix.Translation(-pivot)
    for obj in moving:
        obj.matrix_world = matrix @ matrices[obj.name]
    bpy.context.view_layer.update()


def execute(case, spec, directory):
    scene = bpy.context.scene
    frames = spec["scene_frames"]
    scene.frame_set(frames[0])
    moving, reference, before = b.state(case)
    expected_names = case["objects"]["moving"].get("names")
    if expected_names and set(expected_names) != {o.name for o in moving}:
        raise ValueError("Missing target parts")
    observed = moving[:]
    carried_names = case.get("box_carried_names", [])
    if carried_names:
        carried = b.matching_objects({"label": "carried accessories", "names": carried_names})
        if {o.name for o in carried} != set(carried_names):
            raise ValueError("Missing carried parts")
        if {o.name for o in carried} & {o.name for o in moving + reference}:
            raise ValueError("Carried parts overlap target or reference")
        moving += carried
    matrices = {obj.name: obj.matrix_world.copy() for obj in moving}
    cameras = b.camera_trajectory(frames)
    result = {"before": before, "candidates": {}, "candidate_states": {}, "source_camera_unchanged": True,
              "observed_members": sorted(o.name for o in observed),
              "transformed_members": sorted(o.name for o in moving)}
    project = visible_surface_box if case.get("box_visible_surface") else p.visibility
    for candidate in spec["candidates"]:
        scene.frame_set(frames[0])
        transform(moving, matrices, candidate["parameters"], observed)
        result["candidate_states"][candidate["id"]] = b.state(case)[2]
        records = []
        for frame in frames:
            scene.frame_set(frame)
            records.append({"scene_frame": frame, "candidate": candidate, **project(observed)})
            if case.get("box_censored_edges"):
                records[-1]["unclipped_bbox_top_left"] = unclipped_box(observed)
        result["candidates"][candidate["id"]] = records
    if spec.get("save_preview"):
        if len(spec["candidates"]) != 1:
            raise ValueError("Save exactly one selected proposal")
        scene.frame_set(frames[0])
        transform(moving, matrices, spec["candidates"][0]["parameters"], observed)
        result["after"] = b.state(case)[2]
        result["transforms"] = {
            obj.name: {"before": [list(row) for row in matrices[obj.name]],
                       "after": [list(row) for row in obj.matrix_world]} for obj in moving}
        output = directory / "proposal.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(output))
        result["proposal_blend"] = str(output)
        size = (scene.render.resolution_x, scene.render.resolution_y)
        b.configure_render()
        factor = 640 / max(size)
        scene.render.resolution_x, scene.render.resolution_y = [round(v * factor) for v in size]
        result["frames"] = b.render_frames(directory / "frames", frames)
    if b.camera_trajectory(frames) != cameras:
        raise ValueError("Camera changed")
    b.restore(moving, matrices)
    bpy.context.view_layer.update()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    result = execute(b.read(args.case), b.read(args.spec), args.output.parent)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
