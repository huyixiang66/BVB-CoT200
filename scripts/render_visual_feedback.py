"""Execute one model correction in Blender and save a rendered scene copy."""

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--")+1:])
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    action = validate_action(spec["action"])
    if action["action"] != "adjust":
        raise ValueError("Renderer needs an adjustment")
    directory = args.spec.parent
    output, saved = directory / "render.png", directory / "scene.blend"
    if output.exists() or saved.exists():
        raise FileExistsError("Do not overwrite a previous correction")
    pair, scene = spec["pair"], bpy.context.scene
    scene.frame_set(int(pair["scene_frame"]))
    original_camera = scene.camera
    data = original_camera.data.copy()
    if data.animation_data:
        data.animation_data_clear()
    camera = bpy.data.objects.new("VisualFeedbackCamera", data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    matrix = Matrix(pair["camera_matrix"])
    axes = matrix.to_3x3().copy()
    for part in action_components(action):
        delta = axes @ Vector(part["translation_local"])
        if part["subject"] == "camera":
            matrix.translation += delta
            matrix = matrix @ Euler([math.radians(x) for x in part["rotation_local_degrees"]], "XYZ").to_matrix().to_4x4()
        else:
            moving = b.matching_objects(spec["case"]["objects"]["moving"])
            if not moving:
                raise ValueError("No matching target objects")
            bounds = b.bbox(moving)
            pivot = Vector([*bounds["center_xyz"][:2], bounds["min_xyz"][2]])
            transform = Matrix.Translation(delta) @ Matrix.Translation(pivot) @ Matrix.Scale(
                part["scale_ratio"], 4) @ Matrix.Translation(-pivot)
            for obj in moving:
                if obj.animation_data:
                    obj.animation_data_clear()
                obj.matrix_world = transform @ obj.matrix_world
    camera.matrix_world = matrix
    camera.data.type = "PERSP"
    camera.data.sensor_fit = "HORIZONTAL"
    settings = blender_intrinsic_settings(pair["width"], pair["height"], pair["fx_px"],
                                          pair["fy_px"], pair.get("cx_px"), pair.get("cy_px"))
    for key in ("lens", "sensor_width", "shift_x", "shift_y"):
        setattr(camera.data, key, settings[key])
    scene.render.pixel_aspect_x = settings["pixel_aspect_x"]
    scene.render.pixel_aspect_y = settings["pixel_aspect_y"]
    scene.render.resolution_x, scene.render.resolution_y = pair["width"], pair["height"]
    scene.render.resolution_percentage = 100
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 8
    scene.cycles.use_denoising = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.filepath = str(output.resolve())
    bpy.context.view_layer.update()
    bpy.ops.render.render(write_still=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(saved.resolve()))
    moving = b.matching_objects(spec["case"]["objects"]["moving"])
    updated = {**pair, "camera_matrix": [list(row) for row in matrix],
               "target_center_world": b.bbox(moving)["center_xyz"],
               "render": {"file": str(output.resolve()), "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}}
    (directory / "state.json").write_text(json.dumps({"pair": updated, "blend": str(saved.resolve())},
                                                     indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
