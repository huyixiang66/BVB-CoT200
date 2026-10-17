"""Audit frozen coordinate claims against scene geometry; no pixels or GT."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import refine_blender as b
import refine_alignment as a


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ray_record(point):
    scene = bpy.context.scene
    graph = bpy.context.evaluated_depsgraph_get()
    camera = scene.camera.evaluated_get(graph)
    inverse = camera.calc_matrix_camera(
        graph, x=scene.render.resolution_x, y=scene.render.resolution_y,
        scale_x=scene.render.pixel_aspect_x, scale_y=scene.render.pixel_aspect_y).inverted()
    local = inverse @ Vector((2 * point[0] - 1, 1 - 2 * point[1], -1, 1))
    local /= local.w
    origin = camera.matrix_world.translation
    direction = (camera.matrix_world @ local.to_3d() - origin).normalized()
    hit, location, _, _, obj, _ = scene.ray_cast(
        graph, origin, direction, distance=camera.data.clip_end)
    if not hit:
        return {"hit": False}
    return {"hit": True, "object_name": obj.name,
            "collections": sorted(c.name for c in obj.users_collection),
            "world_point": list(location),
            "local_point": list(obj.matrix_world.inverted() @ location),
            "camera_depth_m": -(camera.matrix_world.inverted() @ location).z}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    if args.output.exists():
        raise ValueError("Refusing to overwrite frozen audit")
    case = b.read(args.parent / "resolved_case.json")
    pairs = b.read(args.parent / "proposed_pairs.json")
    source = Path(bpy.data.filepath)
    source_hash = digest(source)
    if source_hash != b.read(args.parent / "config.json")["input_sha256"]["blend"]:
        raise ValueError("Source changed")
    cameras = b.camera_trajectory([p["scene_frame"] for p in pairs])
    moving = set(case["objects"]["moving"]["names"])
    results = []
    for pair in pairs:
        path = args.measurements / f"measurement_{pair['pair_id']}.json"
        row = b.read(path)["annotation"]
        bpy.context.scene.frame_set(pair["scene_frame"])
        records = {}
        for role in ("features", "landmarks"):
            records[role] = []
            for point in row[role]:
                record = ray_record(point["baseline"])
                records[role].append({**point, **record,
                                      "hits_moving_group": record.get("object_name") in moving})
        # Diagnostic fit only: never enables a rejected pair or selects edits.
        fits = []
        if len(row["landmarks"]) >= 3:
            source_xy = [p["baseline"] for p in row["landmarks"]]
            original_xy = [p["original"] for p in row["landmarks"]]
            for model in ("translation", "similarity"):
                params = a._fit(source_xy, original_xy, model)
                if params is None:
                    continue
                alignment = {"parameters": params, "support_baseline": source_xy}
                fits.append({
                    "model": model, "parameters": params,
                    "target_support": [
                        {"id": p["id"], "baseline_supported": a.supports_point(p["baseline"], alignment),
                         "desired_supported": a.supports_point(a.inverse_point(p["original"], alignment), alignment),
                         "diagnostic_desired_xy": a.inverse_point(p["original"], alignment)}
                        for p in row["features"]],
                })
        results.append({**pair, **records, "background_check": a.background_alignment(row),
                        "diagnostic_unvalidated_fits": fits,
                        "measurement_sha256": digest(path)})
    if cameras != b.camera_trajectory([p["scene_frame"] for p in pairs]) or digest(source) != source_hash:
        raise ValueError("Source or camera changed")
    payload = {
        "protocol": "frozen_correspondence_geometry_audit_v1",
        "source_sha256": source_hash, "qa_calls": 0, "api_calls": 0,
        "gt_used": False, "no_image_access": True, "edits": 0,
        "source_camera_unchanged": True, "pairs": results,
        "code_sha256": {name: digest(Path(__file__).resolve().parents[1] / name) for name in (
            "scripts/audit_refinement_correspondences.py", "refine_alignment.py")},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps({"pairs": len(results), "output": str(args.output), "no_image_access": True}))


if __name__ == "__main__":
    main()
