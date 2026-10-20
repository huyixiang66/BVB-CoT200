"""Apply one shared 3D correction and render every paired camera frame."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Euler, Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import refine_blender as b
from visual_feedback import action_components, blender_intrinsic_settings, validate_action


def apply_camera(camera, part, reference_rotation):
    if camera.parent or camera.constraints:
        raise ValueError("Multi-view camera correction requires an unparented, unconstrained camera")
    delta = reference_rotation @ Vector(part["translation_local"])
    local = Euler(tuple(math.radians(value) for value in part["rotation_local_degrees"]), "XYZ").to_quaternion()
    world_rotation = reference_rotation @ local @ reference_rotation.inverted()
    if camera.animation_data and (camera.animation_data.nla_tracks or camera.animation_data.drivers):
        raise ValueError("Multi-view camera correction does not support NLA tracks or drivers")
    if world_rotation.angle < 1e-8 and delta.length < 1e-10:
        return {"world_translation": list(delta), "keyframes_updated": 0}

    scene = bpy.context.scene
    if not camera.animation_data or not camera.animation_data.action:
        location, rotation, scale = camera.matrix_world.decompose()
        moved_rotation = world_rotation @ rotation
        moved_rotation.normalize()
        camera.location = location + delta
        camera.rotation_mode = "QUATERNION"
        camera.rotation_quaternion = moved_rotation
        camera.scale = scale
        return {"world_translation": list(delta), "keyframes_updated": 0}

    samples = []
    for frame in range(int(scene.frame_start), int(scene.frame_end) + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        location, rotation, scale = camera.matrix_world.decompose()
        moved_rotation = world_rotation @ rotation
        moved_rotation.normalize()
        if samples and samples[-1][2].dot(moved_rotation) < 0:
            moved_rotation.negate()
        samples.append((frame, location + delta, moved_rotation, scale))
    camera.animation_data_clear()
    camera.rotation_mode = "QUATERNION"
    for frame, location, rotation, scale in samples:
        camera.location = location
        camera.rotation_quaternion = rotation
        camera.scale = scale
        camera.keyframe_insert(data_path="location", frame=frame)
        camera.keyframe_insert(data_path="rotation_quaternion", frame=frame)
    action = camera.animation_data.action
    slot = camera.animation_data.action_slot
    for layer in action.layers:
        for strip in layer.strips:
            for curve in strip.channelbag(slot).fcurves:
                for point in curve.keyframe_points:
                    point.interpolation = "LINEAR"
    return {"world_translation": list(delta), "keyframes_updated": len(samples)}


def apply_target(case, part, reference_rotation):
    moving = b.matching_objects(case["objects"]["moving"])
    if any(obj.animation_data or obj.constraints for obj in moving):
        raise ValueError("Multi-view target correction does not support animated or constrained target meshes")
    old_matrices = {obj.name: obj.matrix_world.copy() for obj in moving}
    points = [obj.matrix_world @ Vector(corner) for obj in moving for corner in obj.bound_box]
    low = Vector(tuple(min(point[index] for point in points) for index in range(3)))
    high = Vector(tuple(max(point[index] for point in points) for index in range(3)))
    pivot = (low + high) * 0.5
    delta = reference_rotation @ Vector(part["translation_local"])
    transform = (Matrix.Translation(delta) @ Matrix.Translation(pivot) @
                 Matrix.Scale(float(part["scale_ratio"]), 4) @ Matrix.Translation(-pivot))

    def parent_depth(obj):
        depth, parent = 0, obj.parent
        while parent:
            depth += 1
            parent = parent.parent
        return depth

    for obj in sorted(moving, key=parent_depth):
        obj.matrix_world = transform @ old_matrices[obj.name]
    return {"world_translation": list(delta), "scale_ratio": float(part["scale_ratio"]),
            "pivot_world": list(pivot), "objects_moved": [obj.name for obj in moving]}


def apply_action(case, pairs, action):
    action = validate_action(action)
    if action["action"] != "adjust":
        raise ValueError("Multi-view renderer needs an adjustment")
    reference = Matrix(pairs[0]["reference_camera_matrix"])
    reference_rotation = reference.to_3x3().to_quaternion()
    parts = {part["subject"]: part for part in action_components(action)}
    scene = bpy.context.scene
    results = {"camera": None, "target": None}
    if "camera" in parts:
        results["camera"] = apply_camera(scene.camera, parts["camera"], reference_rotation)
    if "target" in parts:
        results["target"] = apply_target(case, parts["target"], reference_rotation)
    return results


def render_pairs(pairs, directory, target_center):
    scene = bpy.context.scene
    camera = scene.camera
    if camera is None:
        raise ValueError("Scene has no active camera")
    old_render = {
        "engine": scene.render.engine,
        "resolution_x": scene.render.resolution_x,
        "resolution_y": scene.render.resolution_y,
        "resolution_percentage": scene.render.resolution_percentage,
        "pixel_aspect_x": scene.render.pixel_aspect_x,
        "pixel_aspect_y": scene.render.pixel_aspect_y,
        "filepath": scene.render.filepath,
        "image_format": scene.render.image_settings.file_format,
        "color_mode": scene.render.image_settings.color_mode,
        "cycles_device": scene.cycles.device,
        "cycles_samples": scene.cycles.samples,
        "cycles_denoising": scene.cycles.use_denoising,
    }
    updated = []
    try:
        for pair in pairs:
            scene.frame_set(int(pair["scene_frame"]))
            bpy.context.view_layer.update()
            evaluated = scene.camera.evaluated_get(bpy.context.evaluated_depsgraph_get())
            matrix = evaluated.matrix_world.copy()
            record = {**pair, "camera_matrix": [list(row) for row in matrix]}
            camera = scene.camera
            camera_data = camera.data
            temporary_data = camera_data.copy()
            temporary_data.animation_data_clear()
            camera.data = temporary_data
            try:
                settings = blender_intrinsic_settings(pair["width"], pair["height"], pair["fx_px"], pair["fy_px"],
                                                     pair.get("cx_px"), pair.get("cy_px"))
                for key in ("lens", "sensor_width", "shift_x", "shift_y"):
                    setattr(camera.data, key, settings[key])
                scene.render.pixel_aspect_x = settings["pixel_aspect_x"]
                scene.render.pixel_aspect_y = settings["pixel_aspect_y"]
                scene.render.resolution_x = pair["width"]
                scene.render.resolution_y = pair["height"]
                scene.render.resolution_percentage = 100
                scene.render.engine = "CYCLES"
                scene.cycles.device = "CPU"
                scene.cycles.samples = 8
                scene.cycles.use_denoising = True
                scene.render.image_settings.file_format = "PNG"
                scene.render.image_settings.color_mode = "RGB"
                output = directory / f"render_pair_{pair['pair_id']}.png"
                if output.exists():
                    raise FileExistsError("Do not overwrite an existing multi-view render")
                scene.render.filepath = str(output.resolve())
                bpy.context.view_layer.update()
                bpy.ops.render.render(write_still=True)
                record["target_center_world"] = target_center
                record["render"] = {"file": str(output.resolve()),
                                    "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
                updated.append(record)
            finally:
                camera.data = camera_data
                bpy.data.cameras.remove(temporary_data)
    finally:
        for key, value in old_render.items():
            if key == "image_format":
                scene.render.image_settings.file_format = value
            elif key == "color_mode":
                scene.render.image_settings.color_mode = value
            elif key.startswith("cycles_"):
                cycle_key = {"cycles_device": "device", "cycles_samples": "samples",
                             "cycles_denoising": "use_denoising"}[key]
                setattr(scene.cycles, cycle_key, value)
            else:
                setattr(scene.render, key, value)
        scene.frame_set(int(pairs[-1]["scene_frame"]))
        bpy.context.view_layer.update()
    return updated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    directory = args.spec.parent
    if any((directory / f"render_pair_{pair['pair_id']}.png").exists() for pair in spec["pairs"]):
        raise FileExistsError("Do not overwrite an existing multi-view render")
    application = apply_action(spec["case"], spec["pairs"], spec["action"])
    target = b.bbox(b.matching_objects(spec["case"]["objects"]["moving"]))
    saved = directory / "scene.blend"
    if saved.exists():
        raise FileExistsError("Do not overwrite a previous scene edit")
    bpy.ops.wm.save_as_mainfile(filepath=str(saved.resolve()))
    pairs = render_pairs(spec["pairs"], directory, target["center_xyz"])
    (directory / "state.json").write_text(
        json.dumps({"pairs": pairs, "blend": str(saved.resolve()), "application": application}, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({"pairs": len(pairs), "blend": str(saved.resolve()), "application": application}), flush=True)


if __name__ == "__main__":
    main()
