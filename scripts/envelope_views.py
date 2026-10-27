"""Geometry-only question-conditioned view selection for a Blender scene.

The script never reads pixels. It samples a camera sphere around the union
envelope of each question's target objects, tests frustum visibility and
first-hit occlusion, and greedily finds the smallest readable view set.

Inputs are a question spec JSON and an object-groups JSON. The scene itself is
opened by Blender before this script runs.
"""

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def cli_arg(name, default=None):
    argv = sys.argv
    if "--" not in argv:
        return default
    args = argv[argv.index("--") + 1:]
    for index, value in enumerate(args):
        if value == name and index + 1 < len(args):
            return args[index + 1]
    return default


SCENE = cli_arg("--scene", "")
SPEC_PATH = Path(cli_arg(
    "--spec", str(ROOT / "data" / "envelope_views" / "0d2ee665be" / "spec.json")))
GROUPS_PATH = Path(cli_arg(
    "--groups", str(ROOT / "data" / "envelope_views" / "0d2ee665be" / "object_groups.json")))
QA_FILTER = set((cli_arg("--qa", "") or "").split(",")) - {""}
OUT_TAG = "_" + "_".join(sorted(QA_FILTER)) if QA_FILTER else ""
OUT_PATH = Path(cli_arg(
    "--out", str(ROOT / "runs" / "envelope_views" / f"geometry{OUT_TAG}.json")))
SUMMARY_PATH = Path(cli_arg("--summary", str(OUT_PATH.with_suffix(".md"))))

FULL_AZIMUTHS = range(0, 360, 45)
FULL_ELEVATIONS = (20, 40, 60)
TARGET_AZIMUTHS = range(0, 360, 60)
TARGET_ELEVATIONS = (20, 45, 65)
DISTANCE_FACTORS = (1.1,)
LENS_MM = 35.0
SENSOR_MM = 36.0


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def bbox_of(objects):
    points = []
    for obj in objects:
        points.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
    low = [min(point[i] for point in points) for i in range(3)]
    high = [max(point[i] for point in points) for i in range(3)]
    return {"min": low, "max": high}


def union_bbox(boxes):
    return {
        "min": [min(box["min"][i] for box in boxes) for i in range(3)],
        "max": [max(box["max"][i] for box in boxes) for i in range(3)],
    }


def bbox_center(box):
    return Vector([(box["min"][i] + box["max"][i]) / 2 for i in range(3)])


def bbox_dims(box):
    return [box["max"][i] - box["min"][i] for i in range(3)]


def bbox_radius(box):
    return 0.5 * math.sqrt(sum(value * value for value in bbox_dims(box)))


def bbox_points(box):
    low, high = box["min"], box["max"]
    center = [(low[i] + high[i]) / 2 for i in range(3)]
    points = [Vector(center)]
    points.extend(Vector(corner) for corner in (
        (low[0], low[1], low[2]), (low[0], low[1], high[2]),
        (low[0], high[1], low[2]), (low[0], high[1], high[2]),
        (high[0], low[1], low[2]), (high[0], low[1], high[2]),
        (high[0], high[1], low[2]), (high[0], high[1], high[2]),
    ))
    for axis in range(3):
        for value in (low[axis], high[axis]):
            point = Vector(center)
            point[axis] = value
            points.append(point)
    return points


def object_points(objects):
    points = []
    for obj in objects:
        box = bbox_of([obj])
        center = bbox_center(box)
        points.append(center)
        for axis in range(3):
            for value in (box["min"][axis], box["max"][axis]):
                point = center.copy()
                point[axis] = value
                points.append(point)
    points.append(bbox_center(bbox_of(objects)))
    return points


def look_at_euler(location, target):
    direction = Vector(target) - Vector(location)
    if direction.length < 1e-8:
        direction = Vector((0, 0, -1))
    return direction.to_track_quat("-Z", "Y").to_euler()


def clamp(value, low, high):
    return max(low, min(high, value))


def clamp_to_room(point, bounds):
    return Vector((
        clamp(point.x, bounds["min"][0], bounds["max"][0]),
        clamp(point.y, bounds["min"][1], bounds["max"][1]),
        clamp(point.z, bounds["min"][2], bounds["max"][2]),
    ))


def dedupe_points(points, precision=3):
    seen = set()
    result = []
    for point in points:
        key = tuple(round(value, precision) for value in point)
        if key not in seen:
            seen.add(key)
            result.append(point)
    return result


def projected_box_fraction(points):
    if not points:
        return 0.0
    x0 = max(0.0, min(point[0] for point in points))
    x1 = min(1.0, max(point[0] for point in points))
    y0 = max(0.0, min(point[1] for point in points))
    y1 = min(1.0, max(point[1] for point in points))
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def edge_margin(points):
    if not points:
        return 0.0
    return min(min(x, 1.0 - x, y, 1.0 - y) for x, y in points)


spec = load_json(SPEC_PATH)
group_data = load_json(GROUPS_PATH)
scene_name = SCENE or spec.get("scene") or Path(bpy.data.filepath).stem
scene = bpy.context.scene
scene.render.resolution_x = scene.render.resolution_y = 512
scene.render.resolution_percentage = 100

objects_by_name = {obj.name: obj for obj in bpy.data.objects if obj.type == "MESH"}
missing = []
semantic_objects = {}
for semantic, instances in group_data["groups"].items():
    semantic_objects[semantic] = []
    for instance in instances:
        resolved = []
        for name in instance:
            obj = objects_by_name.get(name)
            if obj is None:
                missing.append(name)
            else:
                resolved.append(obj)
        semantic_objects[semantic].append(resolved)
if missing:
    raise RuntimeError("Unresolved object names: %s" % sorted(set(missing)))

room_objects = {}
for role, names in group_data["room"].items():
    room_objects[role] = []
    for name in names:
        obj = objects_by_name.get(name)
        if obj is None:
            missing.append(name)
        else:
            room_objects[role].append(obj)
if missing:
    raise RuntimeError("Unresolved room object names: %s" % sorted(set(missing)))


floor_box = bbox_of(room_objects["floor"])
ceiling_box = bbox_of(room_objects["ceiling"])
room_bounds = {
    "min": [floor_box["min"][0] + 0.18, floor_box["min"][1] + 0.18, floor_box["max"][2] + 0.35],
    "max": [floor_box["max"][0] - 0.18, floor_box["max"][1] - 0.18, ceiling_box["min"][2] - 0.12],
}
room_center = bbox_center(floor_box)
room_dims = bbox_dims(floor_box)
top_z = ceiling_box["max"][2] + 0.7

# The ceiling is never a useful occluder for a room-scale spatial question.
ceiling_hidden = []
for obj in room_objects["ceiling"]:
    if not obj.hide_viewport:
        obj.hide_viewport = True
        ceiling_hidden.append(obj.name)
bpy.context.view_layer.update()
depsgraph = bpy.context.evaluated_depsgraph_get()

object_to_semantic = {}
for semantic, instances in semantic_objects.items():
    for instance in instances:
        for obj in instance:
            object_to_semantic[obj.name] = semantic

camera_data = bpy.data.cameras.new("envelope_probe_camera")
camera_data.lens = LENS_MM
camera_data.sensor_width = SENSOR_MM
camera_data.clip_start = 0.03
camera_data.clip_end = 100.0
camera = bpy.data.objects.new("envelope_probe_camera", camera_data)
scene.collection.objects.link(camera)
scene.camera = camera


def set_camera(location, target, ortho_scale=None):
    if ortho_scale is None:
        camera.data.type = "PERSP"
        rotation = look_at_euler(location, target)
    else:
        camera.data.type = "ORTHO"
        camera.data.ortho_scale = ortho_scale
        rotation = (0.0, 0.0, 0.0)
    camera.location = Vector(location)
    camera.rotation_euler = rotation
    bpy.context.view_layer.update()


def camera_is_free(location):
    origin = Vector(location)
    for axis in range(3):
        for sign in (-1.0, 1.0):
            direction = Vector((0, 0, 0))
            direction[axis] = sign
            hit, _, _, distance, blocker, _ = scene.ray_cast(
                depsgraph, origin, direction, distance=0.16)
            if hit and blocker is not None and blocker.type == "MESH" and distance < 0.12:
                return False
    return True


def measure_instance(points, semantic):
    origin = camera.matrix_world.translation.copy()
    in_frustum = 0
    visible = 0
    depths = []
    projected = []
    blockers = Counter()
    for point in points:
        ndc = world_to_camera_view(scene, camera, point)
        if ndc.z <= 0 or not (0 <= ndc.x <= 1 and 0 <= ndc.y <= 1):
            continue
        in_frustum += 1
        projected.append((float(ndc.x), float(ndc.y)))
        depths.append(float(ndc.z))
        direction = point - origin
        distance = direction.length
        if distance < 1e-8:
            continue
        hit, _, _, _, blocker, _ = scene.ray_cast(
            depsgraph, origin, direction.normalized(), distance=distance + 0.01)
        if hit and blocker is not None and object_to_semantic.get(blocker.name) == semantic:
            visible += 1
        elif hit and blocker is not None:
            blockers[blocker.name] += 1
    total = max(1, len(points))
    return {
        "points": len(points),
        "in_frustum_points": in_frustum,
        "visible_points": visible,
        "in_frustum_fraction": in_frustum / total,
        "visible_fraction": visible / total,
        "area_fraction": projected_box_fraction(projected),
        "edge_margin": edge_margin(projected),
        "mean_depth": sum(depths) / len(depths) if depths else None,
        "top_blockers": [{"name": name, "count": count}
                         for name, count in blockers.most_common(5)],
    }


def measure_camera(semantics):
    instances = {}
    for semantic in semantics:
        instances[semantic] = []
        for index, objects in enumerate(semantic_objects[semantic]):
            metric = measure_instance(object_points(objects), semantic)
            metric["instance"] = index
            instances[semantic].append(metric)
    semantic_metrics = {}
    for semantic, metrics in instances.items():
        semantic_metrics[semantic] = {
            "min_visible_fraction": min(item["visible_fraction"] for item in metrics),
            "mean_visible_fraction": sum(item["visible_fraction"] for item in metrics) / len(metrics),
            "min_in_frustum_fraction": min(item["in_frustum_fraction"] for item in metrics),
            "area_fraction_sum": sum(item["area_fraction"] for item in metrics),
            "min_area_fraction": min(item["area_fraction"] for item in metrics),
            "min_edge_margin": min(item["edge_margin"] for item in metrics),
        }
    return {"instances": instances, "semantics": semantic_metrics}


def candidate_key(candidate):
    return candidate["id"]


def make_candidate(candidate_id, method, location, target, ortho_scale=None, metadata=None):
    return {
        "id": candidate_id,
        "method": method,
        "location": [float(value) for value in location],
        "target": [float(value) for value in target],
        "ortho_scale": ortho_scale,
        "lens_mm": LENS_MM if ortho_scale is None else None,
        "metadata": metadata or {},
    }


def sphere_candidates(prefix, envelope, azimuths, elevations, include_top=True):
    center = bbox_center(envelope)
    dims = bbox_dims(envelope)
    radius = max(0.22, bbox_radius(envelope))
    candidates = []
    if include_top:
        top_scale = max(0.45, max(dims[0], dims[1]) * 1.35)
        candidates.append(make_candidate(
            f"{prefix}_overhead", "target_overhead",
            (center.x, center.y, top_z), center, top_scale,
            {"envelope_radius_m": radius}))
    seen = {(tuple(round(value, 3) for value in item["location"]),
             tuple(round(value, 3) for value in item["target"]))
            for item in candidates}
    for elevation in elevations:
        elevation_rad = math.radians(elevation)
        for azimuth in azimuths:
            azimuth_rad = math.radians(azimuth)
            for factor in DISTANCE_FACTORS:
                distance = max(0.8, 2.2 * radius) * factor
                ideal = center + Vector((
                    distance * math.cos(elevation_rad) * math.cos(azimuth_rad),
                    distance * math.cos(elevation_rad) * math.sin(azimuth_rad),
                    distance * math.sin(elevation_rad),
                ))
                location = clamp_to_room(ideal, room_bounds)
                if (location - center).length < 0.3:
                    continue
                key = (tuple(round(value, 3) for value in location),
                       tuple(round(value, 3) for value in center))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(make_candidate(
                    f"{prefix}_sphere_e{elevation}_a{azimuth}_f{factor:g}",
                    "envelope_sphere",
                    location, center,
                    metadata={"elevation_deg": elevation, "azimuth_deg": azimuth,
                              "distance_factor": factor,
                              "ideal_distance_m": distance,
                              "envelope_radius_m": radius,
                              "clamped_to_room": list(location) != [float(v) for v in ideal]}))
    return candidates, radius


def question_candidates(question, semantic_boxes):
    if semantic_boxes:
        envelope = union_bbox([box for boxes in semantic_boxes.values() for box in boxes])
    else:
        envelope = floor_box
    candidates, radius = sphere_candidates(
        "env_full", envelope, FULL_AZIMUTHS, FULL_ELEVATIONS)

    if question["question_type"] in {"object_counting", "room_size_estimation"}:
        candidates.append(make_candidate(
            "overhead_room", "room_overhead",
            (room_center.x, room_center.y, top_z), room_center,
            max(room_dims[0], room_dims[1]) * 1.08,
            {"room_span_m": max(room_dims[0], room_dims[1])}))

    if question["question_type"] == "room_size_estimation":
        return candidates, envelope, radius

    seen = {(tuple(round(value, 3) for value in item["location"]),
             tuple(round(value, 3) for value in item["target"]))
            for item in candidates}
    for semantic, boxes in semantic_boxes.items():
        target_candidates, _ = sphere_candidates(
            "env_" + semantic.replace(" ", "_"), union_bbox(boxes),
            TARGET_AZIMUTHS, TARGET_ELEVATIONS)
        for candidate in target_candidates:
            key = (tuple(round(value, 3) for value in candidate["location"]),
                   tuple(round(value, 3) for value in candidate["target"]))
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)
        if question["question_type"] == "object_counting" and len(boxes) > 1:
            for index, box in enumerate(boxes):
                instance_candidates, _ = sphere_candidates(
                    f"env_{semantic.replace(' ', '_')}_{index}", box,
                    TARGET_AZIMUTHS, TARGET_ELEVATIONS)
                for candidate in instance_candidates:
                    key = (tuple(round(value, 3) for value in candidate["location"]),
                           tuple(round(value, 3) for value in candidate["target"]))
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(candidate)
    return candidates, envelope, radius


def requirement_for(question_type, semantic, instance_count):
    if question_type == "room_size_estimation":
        return {"visible": 0.0, "area": 0.0}
    if question_type == "object_size_estimation":
        return {"visible": 0.20, "area": 0.012}
    if question_type == "object_counting":
        return {"visible": 0.20, "area": 0.0004}
    if question_type in {"object_abs_distance", "object_rel_distance"}:
        return {"visible": 0.20, "area": 0.0008}
    return {"visible": 0.20, "area": 0.0010}


def satisfaction(candidate, question, semantics_requirements):
    satisfied = set()
    for semantic in semantics_requirements:
        requirement = semantics_requirements[semantic]
        metrics = candidate["metrics"]["instances"].get(semantic, [])
        if question["question_type"] == "object_counting" and len(metrics) > 1:
            for metric in metrics:
                key = f"{semantic}#{metric['instance']}"
                if (metric["visible_fraction"] >= requirement["visible"] and
                        metric["area_fraction"] >= requirement["area"]):
                    satisfied.add(key)
        elif metrics:
            best = max(metrics, key=lambda item: (item["visible_fraction"], item["area_fraction"]))
            if (best["visible_fraction"] >= requirement["visible"] and
                    best["area_fraction"] >= requirement["area"]):
                satisfied.add(semantic)
    return satisfied


def candidate_quality(candidate, question, semantics_requirements, all_keys):
    metrics = candidate["metrics"]["instances"]
    min_visible = 1.0
    mean_visible = 0.0
    count = 0
    min_area = 1.0
    min_margin = 1.0
    for semantic in semantics_requirements:
        for metric in metrics.get(semantic, []):
            min_visible = min(min_visible, metric["visible_fraction"])
            mean_visible += metric["visible_fraction"]
            min_area = min(min_area, metric["area_fraction"])
            min_margin = min(min_margin, metric["edge_margin"])
            count += 1
    mean_visible = mean_visible / max(1, count)
    newly = len(satisfaction(candidate, question, semantics_requirements))
    return (newly, min_visible + 0.25 * mean_visible + 0.5 * min(0.25, min_area), min_margin)


def greedy_select(candidates, question, semantics_requirements, require_area):
    requirements = {}
    for semantic in semantics_requirements:
        base = semantics_requirements[semantic]
        requirements[semantic] = {
            "visible": base["visible"],
            "area": base["area"] if require_area else 0.0,
        }
    all_metrics = {}
    for semantic in requirements:
        all_metrics[semantic] = candidates[0]["metrics"]["instances"].get(semantic, [])
    all_keys = set()
    for semantic, metrics in all_metrics.items():
        if question["question_type"] == "object_counting" and len(metrics) > 1:
            for metric in metrics:
                all_keys.add(f"{semantic}#{metric['instance']}")
        else:
            all_keys.add(semantic)
    chosen = []
    covered = set()
    remaining = list(candidates)
    while remaining and covered != all_keys:
        ranked = sorted(
            remaining,
            key=lambda item: (
                len(satisfaction(item, question, requirements) - covered),
                candidate_quality(item, question, requirements, all_keys),
            ),
            reverse=True)
        best = ranked[0]
        newly = satisfaction(best, question, requirements) - covered
        if not newly:
            break
        chosen.append(best)
        covered |= newly
        remaining.remove(best)
    return chosen, covered, all_keys


questions = []
summary_by_type = defaultdict(list)
for question in spec["questions"]:
    qa_id = str(question["qa_id"])
    if QA_FILTER and qa_id not in QA_FILTER:
        continue
    print("Q", qa_id, question["question_type"], flush=True)
    semantics = list(dict.fromkeys(question["targets"]))
    unresolved = [semantic for semantic in semantics if semantic not in semantic_objects]
    if unresolved:
        questions.append({"qa_id": qa_id, "question_type": question["question_type"],
                          "unresolved": unresolved})
        continue
    if question["question_type"] == "room_size_estimation":
        semantic_boxes = {}
    else:
        semantic_boxes = {
            semantic: [bbox_of(objects) for objects in semantic_objects[semantic]]
            for semantic in semantics
        }
    candidates, envelope, radius = question_candidates(question, semantic_boxes)
    requirements = {
        semantic: requirement_for(question["question_type"], semantic,
                                  len(semantic_objects[semantic]))
        for semantic in semantics
    }
    for candidate in candidates:
        set_camera(candidate["location"], candidate["target"], candidate["ortho_scale"])
        if candidate["method"] == "envelope_sphere" and not camera_is_free(candidate["location"]):
            candidate["metrics"] = {"instances": {}, "semantics": {}}
            candidate["free_space"] = False
            continue
        candidate["free_space"] = True
        candidate["metrics"] = measure_camera(semantics)
    usable = [candidate for candidate in candidates if candidate.get("free_space", True)]
    if not usable:
        usable = candidates
    if question["question_type"] == "room_size_estimation":
        chosen_visibility = [candidate for candidate in candidates if candidate["id"] == "overhead_room"]
        chosen_readable = chosen_visibility
        covered_visibility = {"room"}
        covered_readable = {"room"}
        all_keys = {"room"}
    else:
        chosen_visibility, covered_visibility, all_keys = greedy_select(
            usable, question, requirements, require_area=False)
        chosen_readable, covered_readable, _ = greedy_select(
            usable, question, requirements, require_area=True)
    record = {
        "qa_id": qa_id,
        "question_type": question["question_type"],
        "question": question["question"],
        "targets": semantics,
        "object_instances": {semantic: len(semantic_objects[semantic]) for semantic in semantics},
        "envelope": envelope if semantic_boxes else None,
        "envelope_radius_m": radius if semantic_boxes else None,
        "unresolved": [],
        "requirements": requirements,
        "k_visibility": len(chosen_visibility),
        "k_readable": len(chosen_readable),
        "covered_visibility": sorted(covered_visibility),
        "covered_readable": sorted(covered_readable),
        "missing_visibility": sorted(all_keys - covered_visibility),
        "missing_readable": sorted(all_keys - covered_readable),
        "candidate_count": len(usable),
        "best_by_semantic": {
            semantic: max(
                ({"candidate_id": candidate["id"],
                  "visible_fraction": metric["visible_fraction"],
                  "area_fraction": metric["area_fraction"],
                  "in_frustum_fraction": metric["in_frustum_fraction"]}
                 for candidate in usable
                 for metric in candidate["metrics"]["instances"].get(semantic, [])),
                key=lambda item: (item["visible_fraction"], item["area_fraction"]),
                default=None)
            for semantic in semantics
        },
        "chosen_visibility": [{key: value for key, value in candidate.items()
                               if key not in {"metrics"}} for candidate in chosen_visibility],
        "chosen_readable": [{key: value for key, value in candidate.items()
                             if key not in {"metrics"}} for candidate in chosen_readable],
        "chosen_visibility_metrics": [candidate["metrics"] for candidate in chosen_visibility],
        "chosen_readable_metrics": [candidate["metrics"] for candidate in chosen_readable],
        "top_candidates": [
            {
                "id": candidate["id"],
                "method": candidate["method"],
                "location": candidate["location"],
                "ortho_scale": candidate["ortho_scale"],
                "free_space": candidate["free_space"],
                "metrics": candidate["metrics"]["semantics"],
            }
            for candidate in sorted(
                usable,
                key=lambda item: candidate_quality(item, question, requirements, all_keys),
                reverse=True)[:8]
        ],
    }
    questions.append(record)
    summary_by_type[question["question_type"]].append(record)

summary = {}
for question_type, records in sorted(summary_by_type.items()):
    summary[question_type] = {
        "n": len(records),
        "mean_k_visibility": sum(record["k_visibility"] for record in records) / len(records),
        "mean_k_readable": sum(record["k_readable"] for record in records) / len(records),
        "k_visibility_distribution": dict(Counter(record["k_visibility"] for record in records)),
        "k_readable_distribution": dict(Counter(record["k_readable"] for record in records)),
        "ids": [record["qa_id"] for record in records],
    }

payload = {
    "scene": scene_name,
    "blend": bpy.data.filepath,
    "selection_method": "question_conditioned_envelope_sphere_greedy_set_cover",
    "no_render": True,
    "no_image_access": True,
    "camera_assumptions": {
        "lens_mm": LENS_MM,
        "sensor_mm": SENSOR_MM,
        "resolution": [512, 512],
        "full_envelope_azimuth_step_deg": 45,
        "full_envelope_elevations_deg": list(FULL_ELEVATIONS),
        "target_envelope_azimuth_step_deg": 60,
        "target_envelope_elevations_deg": list(TARGET_ELEVATIONS),
        "distance_factors": list(DISTANCE_FACTORS),
    },
    "room": {
        "floor_bbox": floor_box,
        "ceiling_bbox": ceiling_box,
        "camera_bounds": room_bounds,
        "top_z": top_z,
        "floor_dims_m": room_dims,
        "ceiling_objects": [obj.name for obj in room_objects["ceiling"]],
        "ceiling_hidden_for_rays": ceiling_hidden,
    },
    "object_groups": {
        semantic: {"instances": len(instances), "objects": [obj.name for obj in sum(instances, [])]}
        for semantic, instances in semantic_objects.items()
    },
    "question_count": len(questions),
    "summary_by_type": summary,
    "questions": questions,
}
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

lines = [
    "# Geometry-only envelope view selection — %s" % scene_name,
    "",
    "No rendering and no image access. Ceilings were hidden for ray casts.",
    "",
    "| question type | n | mean k visibility | mean k readable | readable distribution |",
    "|---|---:|---:|---:|---|",
]
for question_type, item in summary.items():
    lines.append("| %s | %d | %.2f | %.2f | %s |" % (
        question_type, item["n"], item["mean_k_visibility"], item["mean_k_readable"],
        item["k_readable_distribution"]))
lines.extend(["", "| qa_id | type | targets | k_vis | k_read | chosen readable views |",
              "|---|---|---|---:|---:|---|"])
for record in sorted(questions, key=lambda item: int(item["qa_id"])):
    chosen = ", ".join(view["id"] for view in record.get("chosen_readable", []))
    lines.append("| %s | %s | %s | %s | %s | %s |" % (
        record["qa_id"], record["question_type"], ", ".join(record.get("targets", [])),
        record.get("k_visibility", "-"), record.get("k_readable", "-"), chosen))
SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("ENVELOPE_VIEWS_OK", len(questions))
print(SUMMARY_PATH)
