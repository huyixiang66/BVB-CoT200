"""Blender geometry-only visibility and transform probes. Never reads pixels."""

import argparse
import json
from pathlib import Path
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
import refine_blender as b


def visibility(objects):
    scene = bpy.context.scene
    graph = bpy.context.evaluated_depsgraph_get()
    camera = scene.camera.evaluated_get(graph)
    origin = camera.matrix_world.translation
    names = {obj.name for obj in objects}
    projected = []
    hits = inside = 0
    for obj in objects:
        evaluated = obj.evaluated_get(graph)
        corners = [Vector(point) for point in evaluated.bound_box]
        center = sum(corners, Vector()) / 8
        points = [center, *corners]
        for axis in range(3):
            for extreme in (min, max):
                point = center.copy()
                point[axis] = extreme(corner[axis] for corner in corners)
                points.append(point)
        for local in points:
            point = evaluated.matrix_world @ local
            ndc = world_to_camera_view(scene, camera, point)
            if camera.data.clip_start < ndc.z < camera.data.clip_end:
                projected.append((ndc.x, ndc.y))
            if not (0 <= ndc.x <= 1 and 0 <= ndc.y <= 1
                    and camera.data.clip_start < ndc.z < camera.data.clip_end):
                continue
            inside += 1
            direction = point - origin
            hit, _, _, _, blocker, _ = scene.ray_cast(
                graph, origin, direction.normalized(), distance=direction.length + 1e-4)
            if hit and blocker.name in names:
                hits += 1
    box = None
    if projected:
        box = [max(0, min(p[0] for p in projected)), max(0, min(p[1] for p in projected)),
               min(1, max(p[0] for p in projected)), min(1, max(p[1] for p in projected))]
    area = max(0, box[2] - box[0]) * max(0, box[3] - box[1]) if box else 0
    return {"frustum_samples": inside, "first_hit_group_samples": hits,
            "projected_bbox_xyxy": box, "projected_area_fraction": area}


def components(objects, tolerance=0.10):
    boxes = {obj.name: b.bbox([obj]) for obj in objects}
    pending = set(boxes)
    result = []
    while pending:
        group = {min(pending)}
        pending -= group
        frontier = list(group)
        while frontier:
            name = frontier.pop()
            neighbours = {other for other in pending
                          if b.aabb_distance(boxes[name], boxes[other]) <= tolerance}
            pending -= neighbours
            group |= neighbours
            frontier.extend(sorted(neighbours))
        box = b.bbox([obj for obj in objects if obj.name in group])
        result.append({"members": sorted(group), "world_bbox": box})
    return result


def probe(case, frames, candidates):
    scene = bpy.context.scene
    # Projection uses the source aspect, matching the comparison renderer.
    scene.frame_set(frames[0])
    moving, reference, initial = b.state(case)
    objects = moving + reference
    matrices = {obj.name: obj.matrix_world.copy() for obj in objects}
    records = []
    for frame in frames:
        scene.frame_set(frame)
        records.append({"scene_frame": frame, "camera": b.camera_state(),
                        "moving": visibility(moving), "reference": visibility(reference)})
    scene.frame_set(frames[0])
    candidate_records = []
    for candidate in candidates:
        b.restore(objects, matrices)
        bpy.context.view_layer.update()
        b.apply_candidate(candidate, moving, reference)
        bpy.context.view_layer.update()
        _, _, intended = b.state(case)
        states = []
        for frame in sorted({frames[0], frames[-1]}):
            scene.frame_set(frame)
            _, _, actual = b.state(case)
            states.append({"scene_frame": frame, "state": actual})
        candidate_records.append({"candidate": candidate, "intended_state": intended,
                                  "after_frame_set": states})
    b.restore(objects, matrices)
    bpy.context.view_layer.update()
    return {
        "case_id": case["case_id"], "initial_state": initial, "frames": records,
        "components": {role: components(group)
                       for role, group in (("moving", moving), ("reference", reference))},
        "candidate_transforms": candidate_records,
        "dynamic_members": [obj.name for obj in objects
                            if obj.animation_data or obj.constraints or obj.parent],
        "limitations": "Sparse bbox rays, not pixel masks. Zero hits do not prove invisibility. "
                        "AABB proximity groups are not verified instances.",
        "no_image_access": True, "no_render": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    case, spec = b.read(args.case), b.read(args.spec)
    if spec.get("action") == "apply":
        scene = bpy.context.scene
        scene.frame_set(spec["scene_frames"][0])
        moving, reference, before = b.state(case)
        cameras = b.camera_trajectory(spec["scene_frames"])
        b.apply_candidate(spec["candidate"], moving, reference)
        bpy.context.view_layer.update()
        _, _, after = b.state(case)
        if b.camera_trajectory(spec["scene_frames"]) != cameras:
            raise ValueError("Source camera changed")
        blend = args.output.parent / "scene_after.blend"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(blend))
        result = {"candidate": spec["candidate"], "before_state": before, "after_state": after,
                  "scene_after_blend": str(blend), "source_camera_unchanged": True}
    elif spec.get("action") == "render":
        scene = bpy.context.scene
        scene.frame_set(spec["scene_frames"][0])
        moving, reference, _ = b.state(case)
        cameras = b.camera_trajectory(spec["scene_frames"])
        source_size = (scene.render.resolution_x, scene.render.resolution_y)
        b.configure_render()
        factor = 640 / max(source_size)
        scene.render.resolution_x = round(source_size[0] * factor)
        scene.render.resolution_y = round(source_size[1] * factor)
        b.apply_candidate(spec["candidate"], moving, reference)
        bpy.context.view_layer.update()
        _, _, changed = b.state(case)
        records = b.render_frames(args.output.parent / "frames", spec["scene_frames"])
        if [frame["camera"] for frame in records] != cameras:
            raise ValueError("Source camera changed")
        result = {"frames": records, "state": changed, "source_camera_unchanged": True,
                  "materials": "original_unmodified", "aspect": list(source_size)}
    else:
        result = probe(case, spec["scene_frames"], spec.get("candidates", []))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"case_id": case["case_id"], "frames": len(result.get("frames", [])),
                      "components": {k: len(v) for k, v in result.get("components", {}).items()},
                      "no_image_access": True}))


if __name__ == "__main__":
    main()
