"""Blender: prepare a source frame and the scene's active camera for visual feedback."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.render_visual_feedback import main as render
import refine_blender as b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--source-frame", type=Path, required=True)
    parser.add_argument("--scene-frame", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args(sys.argv[sys.argv.index("--")+1:])
    case = json.loads(args.case.read_text(encoding="utf-8"))
    source = args.source_frame.resolve()
    scene = bpy.context.scene
    if Path(case["inputs"]["blend"]).resolve() != Path(bpy.data.filepath).resolve():
        raise ValueError("Loaded Blender file does not match case.inputs.blend")
    if args.output.exists():
        raise FileExistsError("Use a new preparation directory")
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
