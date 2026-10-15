"""Blender-side object grouping, candidate rendering, and non-destructive finalization."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Vector


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def matching_objects(spec):
    names = set(spec.get("names", []))
    prefixes = tuple(spec.get("prefixes", []))
    objects = [obj for obj in bpy.data.objects if obj.type == "MESH" and
               (obj.name in names or (prefixes and obj.name.startswith(prefixes)))]
    objects.sort(key=lambda obj: obj.name)
    if not objects:
        raise ValueError("No mesh objects matched " + spec["label"])
    return objects


def bbox(objects):
    points = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    low = [min(point[index] for point in points) for index in range(3)]
    high = [max(point[index] for point in points) for index in range(3)]
    return {"min_xyz": low, "max_xyz": high,
            "center_xyz": [(low[index] + high[index]) / 2 for index in range(3)],
            "size_xyz": [high[index] - low[index] for index in range(3)]}


def aabb_distance(left, right):
    gaps = [max(0.0, left["min_xyz"][index] - right["max_xyz"][index],
                right["min_xyz"][index] - left["max_xyz"][index]) for index in range(3)]
    return math.sqrt(sum(value * value for value in gaps))


def groups(case):
    moving = matching_objects(case["objects"]["moving"])
    reference = matching_objects(case["objects"]["reference"])
    overlap = {obj.name for obj in moving} & {obj.name for obj in reference}
    if overlap:
        raise ValueError("Object groups overlap: " + ", ".join(sorted(overlap)))
    return moving, reference


def state(case):
    moving, reference = groups(case)
    moving_box, reference_box = bbox(moving), bbox(reference)
    return moving, reference, {
        "moving": {"label": case["objects"]["moving"]["label"],
                   "members": [obj.name for obj in moving], "world_bbox": moving_box},
        "reference": {"label": case["objects"]["reference"]["label"],
                      "members": [obj.name for obj in reference], "world_bbox": reference_box},
        "surface_distance_m": aabb_distance(moving_box, reference_box),
    }


def configure_render():
    scene = bpy.context.scene
    if scene.camera is None:
        raise ValueError("Scene has no active camera")
    for engine in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        try:
            scene.render.engine = engine
            break
        except Exception:
            pass
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.use_file_extension = True


def render_frames(directory, frames):
    directory.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    records = []
    for frame in frames:
        scene.frame_set(int(frame))
        path = directory / f"frame_{int(frame):06d}.png"
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        records.append({"scene_frame": int(frame), "file": str(path.resolve()),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return records


def apply_candidate(candidate, moving, reference):
    operation = candidate["operation"]
    if operation == "no_change":
        return
    moving_box, reference_box = bbox(moving), bbox(reference)
    if operation == "translate":
        direction = Vector(reference_box["center_xyz"]) - Vector(moving_box["center_xyz"])
        if candidate.get("horizontal_only", True):
            direction.z = 0
        if direction.length == 0:
            raise ValueError("Moving and reference centers coincide")
        direction.normalize()
        signed_distance = float(candidate["distance_m"])
        if candidate.get("direction", "toward") == "away":
            signed_distance *= -1
        delta = direction * signed_distance
        for obj in moving:
            obj.matrix_world.translation += delta
        return
    if operation == "scale":
        factor = float(candidate["factor"])
        pivot = Vector(moving_box["center_xyz"])
        transform = Matrix.Translation(pivot) @ Matrix.Scale(factor, 4) @ Matrix.Translation(-pivot)
        for obj in moving:
            obj.matrix_world = transform @ obj.matrix_world
        return
    raise ValueError("Unsupported operation")


def restore(objects, matrices):
    for obj in objects:
        obj.matrix_world = matrices[obj.name].copy()


def inspect(case, output):
    _, _, scene_state = state(case)
    scene = bpy.context.scene
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"active_camera": scene.camera.name if scene.camera else None,
                                  "timeline": [int(scene.frame_start), int(scene.frame_end)],
                                  "instances": scene_state}, indent=2), encoding="utf-8")


def render_candidates(case, spec, output):
    configure_render()
    moving, reference, initial = state(case)
    objects = moving + reference
    matrices = {obj.name: obj.matrix_world.copy() for obj in objects}
    manifest = {"initial_state": initial, "candidates": []}
    for candidate in spec["candidates"]:
        restore(objects, matrices)
        apply_candidate(candidate, moving, reference)
        _, _, changed = state(case)
        frames = render_frames(output / candidate["id"], spec["scene_frames"])
        manifest["candidates"].append({"candidate": candidate, "state": changed, "frames": frames})
    restore(objects, matrices)
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def finalize(case, spec, output):
    configure_render()
    moving, reference, before = state(case)
    before_frames = render_frames(output / "before" / "qa", spec["qa_scene_frames"])
    before_matrices = {obj.name: [list(row) for row in obj.matrix_world] for obj in moving}
    apply_candidate(spec["candidate"], moving, reference)
    _, _, after = state(case)
    after_frames = render_frames(output / "after" / "qa", spec["qa_scene_frames"])
    after_matrices = {obj.name: [list(row) for row in obj.matrix_world] for obj in moving}
    output.mkdir(parents=True, exist_ok=True)
    blend_path = output / "scene_after.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    (output / "manifest.json").write_text(json.dumps({
        "candidate": spec["candidate"], "before_state": before, "after_state": after,
        "moving_matrices_before": before_matrices, "moving_matrices_after": after_matrices,
        "before_qa": before_frames, "after_qa": after_frames,
        "scene_after_blend": str(blend_path.resolve()),
    }, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("inspect", "render_candidates", "finalize"))
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spec", type=Path)
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    args = parser.parse_args(argv)
    case = read(args.case)
    if args.action == "inspect":
        inspect(case, args.output)
    elif args.action == "render_candidates":
        render_candidates(case, read(args.spec), args.output)
    else:
        finalize(case, read(args.spec), args.output)


if __name__ == "__main__":
    main()
