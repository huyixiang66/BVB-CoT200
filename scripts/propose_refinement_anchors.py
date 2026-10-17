"""Propose visible mesh surface anchors using geometry only, never image pixels."""

import argparse
import itertools
import json
import math
from pathlib import Path
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import refine_blender as b
import refine_grounding as g
import refine_structure as s


def visible_extrema(obj, include_top=False):
    scene = bpy.context.scene
    graph = bpy.context.evaluated_depsgraph_get()
    camera = scene.camera.evaluated_get(graph)
    evaluated = obj.evaluated_get(graph)
    vertices = evaluated.data.vertices
    if not vertices or obj.hide_render:
        return []
    indices = {min(vertices, key=lambda v: (v.co - Vector(corner)).length_squared).index
               for corner in evaluated.bound_box}
    if include_top:
        top = max(vertices, key=lambda v: ((evaluated.matrix_world @ v.co).z, -v.index)).index
        indices.add(top)
    records = []
    for index in sorted(indices):
        local = vertices[index].co
        world = evaluated.matrix_world @ local
        ndc = world_to_camera_view(scene, camera, world)
        if not (0.005 < ndc.x < 0.995 and 0.005 < ndc.y < 0.995
                and camera.data.clip_start < ndc.z < camera.data.clip_end):
            continue
        delta = world - camera.matrix_world.translation
        hit, location, _, _, hit_obj, _ = scene.ray_cast(
            graph, camera.matrix_world.translation, delta.normalized(), distance=delta.length + 0.002)
        if hit and hit_obj.name == obj.name and (location - world).length < 0.002:
            records.append({"id": f"{obj.name}:vertex:{index}", "object_name": obj.name,
                            "local_point": list(local), "xy": [ndc.x, 1 - ndc.y],
                            "depth_m": ndc.z, "visible": True})
            if include_top:
                records[-1]["point_description"] = s.point_description(
                    world, b.bbox([obj]), [c.name for c in obj.users_collection],
                    "topmost_mesh_vertex" if index == top else "surface_vertex_near_bbox_corner")
    return records


def structural_points(obj):
    if not s.structural_name(obj.name):
        return []
    points = visible_extrema(obj, include_top=True)
    if len(points) < 2 or max(math.dist(a["xy"], b["xy"]) for a in points for b in points) < s.POLICY["min_projected_span"]:
        return []
    return points


def spread(points, limit, max_per_mesh=None):
    remaining = sorted(points, key=lambda p: p["id"])
    selected = []
    while remaining and len(selected) < limit:
        point = (max(remaining, key=lambda p: min(math.dist(p["xy"], q["xy"]) for q in selected))
                 if selected else remaining[0])
        remaining.remove(point)
        if selected and min(math.dist(point["xy"], p["xy"]) for p in selected) < g.POLICY["min_point_spacing"]:
            break
        selected.append(point)
        if max_per_mesh and sum(p["object_name"] == point["object_name"] for p in selected) >= max_per_mesh:
            remaining = [p for p in remaining if p["object_name"] != point["object_name"]]
    return selected


def background_exclusions(case, moving, shared_collections=False):
    excluded = {obj.name for obj in moving}
    for role in case.get("object_resolution", {}).get("roles", []):
        if role["role"] == "moving":
            excluded.update(role["objects"])
    if shared_collections:
        collections = {collection for obj in moving for collection in obj.users_collection}
        excluded.update(obj.name for collection in collections for obj in collection.all_objects)
    return excluded


def propose(case, shared_collections=False, structural=False):
    moving, _, _ = b.state(case)
    features = spread(list(itertools.chain.from_iterable(visible_extrema(obj) for obj in moving)),
                      g.POLICY["max_target_points"])
    excluded = background_exclusions(case, moving, shared_collections)
    background = []
    if features:
        for obj in sorted(bpy.context.scene.objects, key=lambda obj: obj.name):
            if obj.type != "MESH" or obj.name in excluded:
                continue
            for point in (structural_points(obj) if structural else visible_extrema(obj)):
                if min(math.dist(point["xy"], f["xy"]) for f in features) <= g.POLICY["max_background_distance"]:
                    background.append(point)
    return {"features": features, "landmarks": spread(
                background, g.POLICY["max_background_points"],
                s.POLICY["max_points_per_mesh"] if structural else None),
            "transform_members": sorted(obj.name for obj in moving),
            "background_excluded_members": sorted(excluded)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-shared-collections", action="store_true")
    parser.add_argument("--structural-background", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    if args.output.exists():
        raise ValueError("Refusing overwrite")
    case = b.read(args.parent / "resolved_case.json")
    pairs = b.read(args.parent / "proposed_pairs.json")
    cameras = b.camera_trajectory([p["scene_frame"] for p in pairs])
    records = []
    for pair in pairs:
        bpy.context.scene.frame_set(pair["scene_frame"])
        records.append({**pair, **propose(case, args.exclude_shared_collections, args.structural_background)})
    if cameras != b.camera_trajectory([p["scene_frame"] for p in pairs]):
        raise ValueError("Camera changed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump({"protocol": g.PROTOCOL, "policy": g.POLICY, "pairs": records,
                   "background_exclusion": ("shared_target_collections_v1"
                                            if args.exclude_shared_collections else "resolved_names_v1"),
                   "anchor_policy": s.POLICY if args.structural_background else None,
                   "no_image_access": True, "source_camera_unchanged": True}, handle, indent=2)
    print(json.dumps([{"pair_id": p["pair_id"], "features": len(p["features"]),
                       "landmarks": len(p["landmarks"])} for p in records]))


if __name__ == "__main__":
    main()
