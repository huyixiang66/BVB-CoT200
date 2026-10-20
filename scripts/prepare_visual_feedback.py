"""Prepare one image pair or a set of time-matched BVB views for feedback."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.render_visual_feedback import main as render
import refine_blender as b


def camera_record(scene, pair_id, scene_frame, source, rendered, width, height, moving_objects):
    scene.frame_set(scene_frame)
    bpy.context.view_layer.update()
    camera = scene.camera
    if camera is None or camera.data.type != "PERSP":
        raise ValueError(f"Scene frame {scene_frame} needs an active perspective camera")
    graph = bpy.context.evaluated_depsgraph_get()
    evaluated_camera = camera.evaluated_get(graph)
    projection = camera.calc_matrix_camera(graph, x=width, y=height,
                                           scale_x=scene.render.pixel_aspect_x,
                                           scale_y=scene.render.pixel_aspect_y)
    record = {
        "pair_id": pair_id,
        "scene_frame": scene_frame,
        "source": {"file": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
        "render": {"file": str(rendered), "sha256": hashlib.sha256(rendered.read_bytes()).hexdigest()},
        "camera_matrix": [list(row) for row in evaluated_camera.matrix_world],
        "width": width,
        "height": height,
        "fx_px": projection[0][0] * width / 2,
        "fy_px": projection[1][1] * height / 2,
        "cx_px": (1 - projection[0][2]) * width / 2,
        "cy_px": (1 + projection[1][2]) * height / 2,
        "lens_mm": camera.data.lens,
        "sensor_width_mm": camera.data.sensor_width,
        "target_center_world": b.bbox(moving_objects)["center_xyz"],
    }
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--pairs-manifest", type=Path,
                        help="JSON list of {pair_id, scene_frame, source_frame, render_frame} records")
    parser.add_argument("--source-frame", type=Path)
    parser.add_argument("--initial-render-frame", type=Path,
                        help="Use an existing BVB-rendered video frame and skip the initial Blender render")
    parser.add_argument("--scene-frame", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    case = json.loads(args.case.read_text(encoding="utf-8"))
    scene = bpy.context.scene
    if Path(case["inputs"]["blend"]).resolve() != Path(bpy.data.filepath).resolve():
        raise ValueError("Loaded Blender file does not match case.inputs.blend")
    if args.output.exists():
        raise FileExistsError("Use a new preparation directory")

    if args.pairs_manifest:
        specs = json.loads(args.pairs_manifest.read_text(encoding="utf-8"))
        if not isinstance(specs, list) or len(specs) < 2:
            raise ValueError("A multi-view manifest must contain at least two matched pairs")
        pair_ids = [entry.get("pair_id") for entry in specs]
        if any(type(pair_id) is not int for pair_id in pair_ids) or len(set(pair_ids)) != len(pair_ids):
            raise ValueError("Pair IDs must be unique integers")
        records = []
        moving_objects = b.matching_objects(case["objects"]["moving"])
        for entry in specs:
            source = Path(entry["source_frame"]).expanduser().resolve()
            rendered = Path(entry["render_frame"]).expanduser().resolve()
            if not source.is_file() or not rendered.is_file():
                raise FileNotFoundError(f"Missing source or BVB render for pair {entry['pair_id']}")
            width, height = int(entry.get("width", args.width)), int(entry.get("height", args.height))
            if min(width, height) <= 0:
                raise ValueError("Image dimensions must be positive")
            record = camera_record(scene, int(entry["pair_id"]), int(entry["scene_frame"]),
                                  source, rendered, width, height, moving_objects)
            record["source_frame_index_0based"] = entry.get("source_frame_index_0based")
            record["bvb_frame_index"] = entry.get("bvb_frame_index")
            records.append(record)
        reference = records[-1]["camera_matrix"]
        reference_pair_id = records[-1]["pair_id"]
        for record in records:
            record["reference_camera_matrix"] = reference
            record["reference_pair_id"] = reference_pair_id
        args.output.mkdir(parents=True)
        (args.output / "case.json").write_text(json.dumps(case, indent=2) + "\n", encoding="utf-8")
        (args.output / "pairs.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        print(f"Prepared {len(records)} paired BVB views; reference camera is pair {reference_pair_id}", flush=True)
        return

    if args.source_frame is None or args.scene_frame is None:
        parser.error("Provide --pairs-manifest or both --source-frame and --scene-frame")
    source = args.source_frame.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if min(args.width, args.height) <= 0:
        raise ValueError("Image dimensions must be positive")
    scene.frame_set(args.scene_frame)
    bpy.context.view_layer.update()
    camera = scene.camera
    if camera is None or camera.data.type != "PERSP":
        raise ValueError("Scene needs an active perspective camera")
    graph = bpy.context.evaluated_depsgraph_get()
    projection = camera.calc_matrix_camera(graph, x=args.width, y=args.height,
                                          scale_x=scene.render.pixel_aspect_x,
                                          scale_y=scene.render.pixel_aspect_y)
    pair = {"pair_id": 0, "scene_frame": args.scene_frame,
            "source": {"file": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
            "camera_matrix": [list(row) for row in camera.evaluated_get(graph).matrix_world],
            "width": args.width, "height": args.height,
            "fx_px": projection[0][0]*args.width/2, "fy_px": projection[1][1]*args.height/2,
            "cx_px": (1-projection[0][2])*args.width/2,
            "cy_px": (1+projection[1][2])*args.height/2,
            "target_center_world": b.bbox(b.matching_objects(case["objects"]["moving"]))["center_xyz"]}
    if args.initial_render_frame:
        initial_render = args.initial_render_frame.resolve()
        if not initial_render.is_file():
            raise FileNotFoundError(initial_render)
        pair["render"] = {"file": str(initial_render),
                          "sha256": hashlib.sha256(initial_render.read_bytes()).hexdigest()}
        args.output.mkdir(parents=True)
        (args.output / "case.json").write_text(json.dumps(case, indent=2)+"\n", encoding="utf-8")
        (args.output / "pairs.json").write_text(json.dumps([pair], indent=2)+"\n", encoding="utf-8")
        print(f"Prepared existing render frame {initial_render}; skipped initial Blender render", flush=True)
        return
    args.output.mkdir(parents=True)
    spec = args.output / "spec.json"
    spec.write_text(json.dumps({"case": case, "pair": pair, "action": {
        "action": "adjust", "subject": "camera", "translation_local": [0, 0, 0],
        "rotation_local_degrees": [0, 0, 0], "scale_ratio": 1}}, indent=2), encoding="utf-8")
    sys.argv = ["blender", "--", "--spec", str(spec.resolve())]
    render()
    state = json.loads((args.output / "state.json").read_text(encoding="utf-8"))
    case["inputs"] = {"blend": state["blend"]}
    (args.output / "case.json").write_text(json.dumps(case, indent=2)+"\n", encoding="utf-8")
    (args.output / "pairs.json").write_text(json.dumps([state["pair"]], indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
