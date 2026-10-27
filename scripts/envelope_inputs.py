"""Build question specs and object groups for envelope view selection.

Runs inside Blender. Reuses the release recipes, the conservative name resolver,
and spatial bbox connectivity instead of a hand-written per-scene input file.
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from mathutils import Vector


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fixed_protocol import resolve_roles
from name_resolver import COMPOUND_BLOCK, SYN, head_term
from scene_protocol import legacy_roles


STATIC_TYPES = {
    "object_counting",
    "object_abs_distance",
    "object_rel_distance",
    "object_size_estimation",
    "room_size_estimation",
}

FLOOR_KEYWORDS = ("floor", "foundation", "ground")
CEILING_KEYWORDS = ("ceiling", "roof")
ROOM_PIECE_EXCLUDES = (
    "background", "lampshade", "lamp", "light", "fixture", "fan", "vent",
    "basket", "shelf", "cabinet", "rug", "mat", "curtain", "blind",
    "sofa", "couch", "bed", "chair", "table", "desk", "dresser", "wardrobe",
    "drawer", "nightstand", "counter", "island", "stool", "bench", "ottoman",
    "storage",
)


def supported_type(question_type):
    return question_type in STATIC_TYPES or question_type.startswith("object_rel_direction")


def bbox(obj):
    points = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    return (
        min(point.x for point in points), max(point.x for point in points),
        min(point.y for point in points), max(point.y for point in points),
        min(point.z for point in points), max(point.z for point in points),
    )


def bbox_gap(first, second):
    return max(
        0.0,
        first[0] - second[1], second[0] - first[1],
        first[2] - second[3], second[2] - first[3],
        first[4] - second[5], second[4] - first[5],
    )


def bbox_diagonal(box):
    return math.sqrt(
        (box[1] - box[0]) ** 2 + (box[3] - box[2]) ** 2 + (box[5] - box[4]) ** 2)


def bbox_area(box):
    return max(0.0, box[1] - box[0]) * max(0.0, box[3] - box[2])


def group_by_gap(objects, threshold):
    """Join touching/nearby AABBs into connected components."""
    if not objects:
        return []
    boxes = {obj.name: bbox(obj) for obj in objects}
    parent = {obj.name: obj.name for obj in objects}

    def find(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(first, second):
        first, second = find(first), find(second)
        if first != second:
            parent[second] = first

    for index, first in enumerate(objects):
        for second in objects[index + 1:]:
            if bbox_gap(boxes[first.name], boxes[second.name]) <= threshold:
                union(first.name, second.name)

    groups = defaultdict(list)
    for obj in objects:
        groups[find(obj.name)].append(obj)
    ordered = sorted(
        groups.values(),
        key=lambda group: (
            min(bbox(obj)[0] for obj in group),
            min(bbox(obj)[2] for obj in group),
        ),
    )
    return ordered


def group_objects(objects):
    """Join touching/nearby AABBs into instances."""
    if not objects:
        return []
    sizes = sorted(bbox_diagonal(bbox(obj)) for obj in objects)
    largest = sizes[-1]
    threshold = (0.50 if largest >= 0.35
                 else min(0.15, max(0.05, 0.50 * largest)))
    return group_by_gap(objects, threshold)


def highlight_names_by_color(recipe):
    by_color = defaultdict(list)
    spec = recipe.get("spec") or {}
    for group in spec.get("highlights") or []:
        color = str(group.get("color", "")).lower()
        by_color[color].extend(group.get("match") or [])
    return by_color


def resolved_entries(qa, recipe, mesh_names):
    """Return role entries with objects, preferring exact recipe highlights."""
    roles = recipe.get("roles")
    if roles:
        return resolve_roles(roles, mesh_names, SYN, head_term, COMPOUND_BLOCK), "recipe_roles"

    legacy = legacy_roles(qa)
    highlights = highlight_names_by_color(recipe)
    resolved = []
    for role in legacy:
        exact = [name for name in highlights.get(role["color"].lower(), [])
                 if name in mesh_names]
        if exact:
            resolved.append(dict(role, objects=sorted(exact), status="highlight_exact",
                                 conflicts=[]))
        else:
            resolved.append(resolve_roles(
                [role], mesh_names, SYN, head_term, COMPOUND_BLOCK)[0])
    return resolved, "recipe_highlights"


def build_question_record(qa, recipe, mesh_objects):
    question_type = qa["question_type"]
    if question_type == "room_size_estimation":
        return {
            "qa_id": str(qa["qa_id"]),
            "question_type": question_type,
            "question": qa["question"],
            "targets": [],
            "source": "room_size",
            "unresolved": [],
        }, {}, [], {}

    entries, source = resolved_entries(qa, recipe, sorted(mesh_objects))
    targets = []
    grouped = {}
    unresolved = []
    term_status = {}
    seen = set()
    for entry in entries:
        if entry.get("anchor"):
            continue
        term = entry["term"]
        if term in seen:
            continue
        seen.add(term)
        targets.append(term)
        term_status[term] = entry.get("status", "")
        objects = [mesh_objects[name] for name in entry.get("objects", [])
                   if name in mesh_objects]
        if not objects:
            unresolved.append(term)
            continue
        grouped[term] = group_objects(objects)

    return {
        "qa_id": str(qa["qa_id"]),
        "question_type": question_type,
        "question": qa["question"],
        "targets": targets,
        "source": source,
        "unresolved": unresolved,
    }, grouped, unresolved, term_status


def room_pieces(meshes, keywords):
    found = []
    for obj in meshes:
        name = obj.name.lower()
        if not any(keyword in name for keyword in keywords):
            continue
        if any(word in name for word in ROOM_PIECE_EXCLUDES):
            continue
        box = bbox(obj)
        footprint = max(0.0, box[1] - box[0]) * max(0.0, box[3] - box[2])
        thickness = box[5] - box[4]
        if footprint < 0.05:
            continue
        if thickness > 0.35 and footprint < 1.0:
            continue
        found.append(obj)
    return found


def room_regions(meshes, keywords, min_area=3.0, max_area=400.0):
    """Merge floor/ceiling pieces, then keep room-sized connected regions."""
    pieces = room_pieces(meshes, keywords)
    regions = []
    for group in group_by_gap(pieces, 0.25):
        boxes = [bbox(obj) for obj in group]
        box = (
            min(item[0] for item in boxes), max(item[1] for item in boxes),
            min(item[2] for item in boxes), max(item[3] for item in boxes),
            min(item[4] for item in boxes), max(item[5] for item in boxes),
        )
        area = bbox_area(box)
        if min_area <= area <= max_area:
            regions.append({
                "objects": sorted(obj.name for obj in group),
                "bbox": [round(value, 6) for value in box],
                "area_m2": round(area, 4),
            })
    regions.sort(key=lambda item: (item["area_m2"], item["objects"]))
    return regions


def detect_room(meshes):
    floors = room_regions(meshes, FLOOR_KEYWORDS)
    ceilings = room_regions(meshes, CEILING_KEYWORDS)
    room = {
        "floor": sorted({name for region in floors for name in region["objects"]}),
        "ceiling": sorted({name for region in ceilings for name in region["objects"]}),
    }
    return room, {"floor": floors, "ceiling": ceilings}


def xy_overlap_area(first, second):
    width = max(0.0, min(first[1], second[1]) - max(first[0], second[0]))
    depth = max(0.0, min(first[3], second[3]) - max(first[2], second[2]))
    return width * depth


def choose_room(question_type, target_objects, regions, slack=0.8):
    """Pick the smallest floor region covering the question's target objects."""
    floors = regions["floor"]
    ceilings = regions["ceiling"]
    if not floors:
        return {"floor": [], "ceiling": [], "selection": "missing_floor"}

    if question_type == "room_size_estimation":
        return {
            "floor": sorted({name for region in floors for name in region["objects"]}),
            "ceiling": sorted({name for region in ceilings for name in region["objects"]}),
            "selection": "all_regions",
        }

    target_boxes = [bbox(obj) for obj in target_objects]
    scored = []
    for region in floors:
        floor = region["bbox"]
        contained = 0
        for box in target_boxes:
            center_x = (box[0] + box[1]) / 2
            center_y = (box[2] + box[3]) / 2
            inside_xy = (
                floor[0] - slack <= center_x <= floor[1] + slack and
                floor[2] - slack <= center_y <= floor[3] + slack)
            height = box[4] - floor[5]
            if inside_xy and -0.5 <= height <= 3.0:
                contained += 1
        scored.append((contained, region["area_m2"], region))

    if not scored:
        return {"floor": [], "ceiling": [], "selection": "missing_floor"}

    best_count = max(item[0] for item in scored)
    if target_boxes and best_count == len(target_boxes):
        selected = [min(
            (item for item in scored if item[0] == best_count),
            key=lambda item: item[1])[2]]
        selection = "smallest_containing_region"
    else:
        selected = [item[2] for item in scored if item[0] == best_count and item[0] > 0]
        selection = "partial_region" if selected else "nearest_region"
        if not selected:
            selected = [min(scored, key=lambda item: item[1])[2]]

    floor_boxes = [region["bbox"] for region in selected]
    selected_ceilings = []
    for ceiling in ceilings:
        if any(xy_overlap_area(floor, ceiling["bbox"]) > 0.0 for floor in floor_boxes):
            selected_ceilings.append(ceiling)

    return {
        "floor": sorted({name for region in selected for name in region["objects"]}),
        "ceiling": sorted({name for region in selected_ceilings
                           for name in region["objects"]}),
        "selection": selection,
    }


def build_spec_and_groups(meshes, questions_path, recipes_path, scene_name,
                          qa_filter=None, scene_model="astra"):
    qa_filter = qa_filter or set()
    questions = []
    with Path(questions_path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                questions.append(json.loads(line))
    recipes = json.loads(Path(recipes_path).read_text(encoding="utf-8"))
    mesh_objects = {obj.name: obj for obj in meshes}

    records = []
    term_names = defaultdict(set)
    exact_term_names = defaultdict(set)
    skipped = []
    unresolved = []
    for qa in questions:
        if qa.get("scene_name") != scene_name:
            continue
        qa_id = str(qa["qa_id"])
        if qa_filter and qa_id not in qa_filter:
            continue
        question_type = qa["question_type"]
        if not supported_type(question_type):
            skipped.append({"qa_id": qa_id, "question_type": question_type,
                            "reason": "temporal_question_type"})
            continue
        recipe = recipes.get(qa_id)
        if not recipe:
            skipped.append({"qa_id": qa_id, "question_type": question_type,
                            "reason": "missing_recipe"})
            continue
        try:
            record, term_groups, missing, term_status = build_question_record(
                qa, recipe, mesh_objects)
        except (ValueError, KeyError) as error:
            skipped.append({"qa_id": qa_id, "question_type": question_type,
                            "reason": str(error)})
            continue
        records.append(record)
        if missing:
            unresolved.append({"qa_id": qa_id, "terms": missing})
        for term, instances in term_groups.items():
            names = {obj.name for instance in instances for obj in instance}
            term_names[term].update(names)
            if term_status.get(term) == "highlight_exact":
                exact_term_names[term].update(names)

    groups = {}
    group_instances = {}
    for term, names in term_names.items():
        canonical = exact_term_names.get(term) or names
        instances = group_objects([mesh_objects[name] for name in sorted(canonical)])
        groups[term] = [[obj.name for obj in instance] for instance in instances]
        group_instances[term] = len(instances)

    room, room_regions_data = detect_room(meshes)
    room_assignments = []
    for record in records:
        question = next(item for item in questions if str(item["qa_id"]) == record["qa_id"])
        target_objects = []
        seen_objects = set()
        record_terms = record["targets"]
        for term, instances in groups.items():
            if term not in record_terms:
                continue
            for instance in instances:
                for name in instance:
                    if name not in seen_objects:
                        seen_objects.add(name)
                        target_objects.append(mesh_objects[name])
        room_choice = choose_room(
            question["question_type"], target_objects, room_regions_data)
        record["room"] = room_choice
        room_assignments.append({"qa_id": record["qa_id"], **room_choice})

    spec = {"scene": scene_name, "question_count": len(records), "questions": records}
    group_data = {"scene": scene_name, "groups": groups, "room": room}
    resolution = {
        "mode": "auto",
        "scene_model": scene_model,
        "questions_used": len(records),
        "skipped": skipped,
        "unresolved": unresolved,
        "room": room,
        "room_regions": room_regions_data,
        "room_assignments": room_assignments,
        "group_instances": group_instances,
    }
    return spec, group_data, resolution
