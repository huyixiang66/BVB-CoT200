"""Render bounded object-transform candidates from fixed camera poses in Blender."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def rotation_error_degrees(expected, actual):
    relative = expected.to_3x3().transposed() @ actual.to_3x3()
    cosine = max(-1.0, min(1.0, (sum(relative[i][i] for i in range(3)) - 1) / 2))
    return math.degrees(math.acos(cosine))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=16)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    sys.path.insert(0, str(args.repo_root))
    sys.path.insert(0, str(args.repo_root / "scripts"))
    import refine_blender as b
    from refine_box_blender import transform

    case = json.loads(args.case.read_text(encoding="utf-8"))
    alignment = json.loads(args.alignment.read_text(encoding="utf-8"))
    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
    views = alignment.get("paired_views", [])
    if not alignment.get("accepted_for_render") or len(views) < 3:
        raise ValueError("Need an accepted alignment with at least three views")
    if args.samples < 1 or not candidates:
        raise ValueError("Need candidates and positive render samples")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Output directory must be new or empty")

    scene = bpy.context.scene
    scene.frame_set(int(views[0]["scene_frame"]))
    moving, _, _ = b.state(case)
    expected_names = case["objects"]["moving"].get("names")
    if expected_names and set(expected_names) != {obj.name for obj in moving}:
        raise ValueError("Target parts differ from the frozen case")
    matrices = {obj.name: obj.matrix_world.copy() for obj in moving}

    source_camera = scene.camera
    if source_camera is None:
        raise ValueError("Blend has no active camera")
    camera_data = source_camera.data.copy()
    if camera_data.animation_data:
        camera_data.animation_data_clear()
    camera = bpy.data.objects.new("RenderRankCamera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.data.type = "PERSP"
    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.sensor_width = 36.0

    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = 1.0

    args.output.mkdir(parents=True, exist_ok=True)
    frames_dir = args.output / "renders"
    frames_dir.mkdir()
    records = []
    for candidate in candidates:
        for pair in views:
            scene.frame_set(int(pair["scene_frame"]))
            b.restore(moving, matrices)
            transform(moving, matrices, candidate["parameters"], moving)
            width, height = int(pair["width"]), int(pair["height"])
            if (width, height) != (640, 480):
                raise ValueError("This pilot expects 640x480 paired views")
            camera.data.lens = float(pair["fx_px"]) * camera.data.sensor_width / width
            camera.data.shift_x = camera.data.shift_y = 0.0
            expected = Matrix(pair["camera_matrix"])
            camera.matrix_world = expected
            scene.render.resolution_x, scene.render.resolution_y = width, height
            bpy.context.view_layer.update()
            actual = camera.matrix_world.copy()
            position_error = (actual.translation - expected.translation).length
            angle_error = rotation_error_degrees(expected, actual)
            if position_error > 1e-4 or angle_error > 0.01:
                raise RuntimeError("Temporary render camera did not preserve the requested pose")

            output = frames_dir / f"{candidate['id']}_pair_{int(pair['pair_id']):02d}.png"
            scene.render.filepath = str(output)
            bpy.ops.render.render(write_still=True)
            records.append({
                "candidate_id": candidate["id"],
                "parameters": candidate["parameters"],
                "pair_id": pair["pair_id"],
                "source_frame_index": pair["source_frame_index"],
                "scene_frame": pair["scene_frame"],
                "file": str(output.resolve()),
                "sha256": digest(output),
                "camera_position_error_blender_units": position_error,
                "camera_rotation_error_deg": angle_error,
            })
            print(f"rendered {candidate['id']} pair {pair['pair_id']}", flush=True)

    report = {
        "protocol": "fixed_camera_render_rank_candidates_v1",
        "source_blend": bpy.data.filepath,
        "source_blend_modified": False,
        "render_device": "CPU",
        "samples": args.samples,
        "target": case["objects"]["moving"]["label"],
        "records": records,
    }
    (args.output / "render_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"render_count": len(records), "render_device": "CPU",
                      "report": str((args.output / 'render_report.json').resolve())}, indent=2),
          flush=True)


if __name__ == "__main__":
    main()
