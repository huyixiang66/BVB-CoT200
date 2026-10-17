"""Blender-side object grouping, candidate rendering, and non-destructive finalization."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import bpy
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from fixed_protocol import COLORS, resolve_roles
from name_resolver import COMPOUND_BLOCK, SYN, head_term


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


def resolve_objects(case, output):
    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH" and not obj.hide_render]
    roles = [
        {"role": "moving", "term": case["objects"]["moving"]["label"],
         "color": "RED", "anchor": False},
        {"role": "reference", "term": case["objects"]["reference"]["label"],
         "color": "BLUE", "anchor": False},
    ]
    resolved = resolve_roles(roles, [obj.name for obj in meshes], SYN, head_term, COMPOUND_BLOCK)
    by_role = {entry["role"]: entry for entry in resolved}
    unresolved = [role for role in ("moving", "reference") if not by_role[role]["objects"]]
    payload = {
        "case_id": case["case_id"],
        "scene": case["scene"],
        "blend_sha256": hashlib.sha256(Path(bpy.data.filepath).read_bytes()).hexdigest(),
        "objects": {
            role: {"label": case["objects"][role]["label"], "names": by_role[role]["objects"]}
            for role in ("moving", "reference")
        },
        "resolution": {
            "method": "fixed_protocol_name_candidates",
            "instance_verified": False,
            "complete": not unresolved,
            "unresolved_roles": unresolved,
            "mesh_count": len(meshes),
            "roles": resolved,
            "guardrail": "Name matches are candidates, not verified physical instance IDs.",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def state(case):
    unresolved = [role for role in ("moving", "reference")
                  if not (case["objects"][role].get("names") or case["objects"][role].get("prefixes"))]
    if unresolved:
        return [], [], {
            "status": "unresolved",
            "unresolved_roles": unresolved,
            "moving": {"label": case["objects"]["moving"]["label"], "members": [], "world_bbox": None},
            "reference": {"label": case["objects"]["reference"]["label"], "members": [], "world_bbox": None},
            "surface_distance_m": None,
        }
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


def color_socket(node):
    name = {
        "BSDF_PRINCIPLED": "Base Color",
        "BSDF_DIFFUSE": "Color",
        "BSDF_GLOSSY": "Color",
        "EMISSION": "Color",
    }.get(node.type)
    return node.inputs.get(name) if name else None


def tinted_material(original, color, cache):
    key = (original.name, color)
    if key in cache:
        return cache[key]
    material = original.copy()
    material.name = "REFINE_" + color + "_" + original.name
    rgba = COLORS[color]
    if material.use_nodes:
        for node in list(material.node_tree.nodes):
            socket = color_socket(node)
            if socket is None:
                continue
            if socket.is_linked:
                source = socket.links[0].from_socket
                mix = material.node_tree.nodes.new("ShaderNodeMixRGB")
                mix.blend_type = "MIX"
                mix.inputs[0].default_value = 0.75
                mix.inputs[2].default_value = rgba
                material.node_tree.links.new(source, mix.inputs[1])
                material.node_tree.links.new(mix.outputs[0], socket)
            else:
                previous = socket.default_value[:]
                socket.default_value = tuple(
                    0.25 * previous[index] + 0.75 * rgba[index] for index in range(3)
                ) + (previous[3],)
    previous = material.diffuse_color[:]
    material.diffuse_color = tuple(
        0.25 * previous[index] + 0.75 * rgba[index] for index in range(3)
    ) + (previous[3],)
    cache[key] = material
    return material


def grayscale_material(original, cache):
    if original.name in cache:
        return cache[original.name]
    material = original.copy()
    material.name = "REFINE_GRAY_" + original.name
    if material.use_nodes:
        for node in list(material.node_tree.nodes):
            socket = color_socket(node)
            if socket is None:
                continue
            if socket.is_linked:
                source = socket.links[0].from_socket
                converter = material.node_tree.nodes.new("ShaderNodeRGBToBW")
                material.node_tree.links.new(source, converter.inputs[0])
                material.node_tree.links.new(converter.outputs[0], socket)
            else:
                red, green, blue, alpha = socket.default_value[:]
                gray = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                socket.default_value = (gray, gray, gray, alpha)
    red, green, blue, alpha = material.diffuse_color[:]
    gray = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    material.diffuse_color = (gray, gray, gray, alpha)
    cache[original.name] = material
    return material


def apply_target_colours(moving, reference):
    role_by_name = {obj.name: "RED" for obj in moving}
    role_by_name.update({obj.name: "BLUE" for obj in reference})
    tint_cache = {}
    grayscale_cache = {}
    coloured = []
    grayscale = []
    for obj in bpy.data.objects:
        if obj.type != "MESH" or obj.hide_render or not obj.data.materials:
            continue
        color = role_by_name.get(obj.name)
        replacements = [
            tinted_material(material, color, tint_cache) if color else grayscale_material(material, grayscale_cache)
            for material in obj.data.materials
        ]
        obj.data = obj.data.copy()
        for index, material in enumerate(replacements):
            if material is not None:
                obj.data.materials[index] = material
        (coloured if color else grayscale).append(obj.name)
    return {
        "target_colours_only": True,
        "non_target_meshes_grayscale": True,
        "role_colours": {"moving": "RED", "reference": "BLUE"},
        "coloured_objects": sorted(coloured),
        "grayscale_objects": sorted(grayscale),
    }


def camera_state():
    scene = bpy.context.scene
    camera = scene.camera.evaluated_get(bpy.context.evaluated_depsgraph_get())
    data = camera.data
    return {
        "name": scene.camera.name,
        "matrix_world": [list(row) for row in camera.matrix_world],
        **{key: getattr(data, key) for key in (
            "type", "lens", "sensor_width", "sensor_height", "sensor_fit",
            "shift_x", "shift_y", "ortho_scale", "clip_start", "clip_end",
        )},
    }


def camera_trajectory(frames):
    scene = bpy.context.scene
    previous = scene.frame_current
    records = []
    for frame in frames:
        scene.frame_set(int(frame))
        records.append(camera_state())
    scene.frame_set(previous)
    return records


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
                        "camera": camera_state(),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return records


def copy_frames(directory, records):
    directory.mkdir(parents=True, exist_ok=True)
    copied = []
    for record in records:
        source = Path(record["file"])
        target = directory / source.name
        shutil.copy2(source, target)
        copied.append({**record, "file": str(target.resolve()),
                       "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    return copied


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
        if candidate.get("pivot") == "bottom":
            pivot.z = moving_box["min_xyz"][2]
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
    source_cameras = camera_trajectory(spec["scene_frames"])
    moving, reference, initial = state(case)
    visual_ablation = apply_target_colours(moving, reference)
    objects = moving + reference
    matrices = {obj.name: obj.matrix_world.copy() for obj in objects}
    manifest = {"initial_state": initial, "visual_ablation": visual_ablation, "candidates": []}
    if not objects and any(candidate["operation"] != "no_change" for candidate in spec["candidates"]):
        raise ValueError("Unresolved object groups only permit no_change")
    for candidate in spec["candidates"]:
        restore(objects, matrices)
        apply_candidate(candidate, moving, reference)
        _, _, changed = state(case)
        frames = render_frames(output / candidate["id"], spec["scene_frames"])
        if [frame["camera"] for frame in frames] != source_cameras:
            raise ValueError("Candidate changed the source camera trajectory")
        manifest["candidates"].append({"candidate": candidate, "state": changed, "frames": frames})
    restore(objects, matrices)
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def finalize(case, spec, output):
    configure_render()
    source_cameras = camera_trajectory(spec["qa_scene_frames"])
    moving, reference, before = state(case)
    visual_ablation = apply_target_colours(moving, reference)
    before_frames = render_frames(output / "before" / "qa", spec["qa_scene_frames"])
    before_matrices = {obj.name: [list(row) for row in obj.matrix_world] for obj in moving}
    apply_candidate(spec["candidate"], moving, reference)
    _, _, after = state(case)
    no_change = spec["candidate"]["operation"] == "no_change"
    after_frames = (copy_frames(output / "after" / "qa", before_frames) if no_change else
                    render_frames(output / "after" / "qa", spec["qa_scene_frames"]))
    if any([frame["camera"] for frame in frames] != source_cameras
           for frames in (before_frames, after_frames)):
        raise ValueError("Final QA changed the source camera trajectory")
    after_matrices = {obj.name: [list(row) for row in obj.matrix_world] for obj in moving}
    output.mkdir(parents=True, exist_ok=True)
    blend_path = output / "scene_after.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    (output / "manifest.json").write_text(json.dumps({
        "candidate": spec["candidate"], "before_state": before, "after_state": after,
        "moving_matrices_before": before_matrices, "moving_matrices_after": after_matrices,
        "before_qa": before_frames, "after_qa": after_frames,
        "source_camera_trajectory": source_cameras,
        "source_camera_unchanged": True,
        "qa_frames_byte_identical": no_change,
        "visual_ablation": visual_ablation,
        "scene_after_blend": str(blend_path.resolve()),
    }, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("resolve", "inspect", "render_candidates", "finalize"))
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spec", type=Path)
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    args = parser.parse_args(argv)
    case = read(args.case)
    if args.action == "resolve":
        resolve_objects(case, args.output)
    elif args.action == "inspect":
        inspect(case, args.output)
    elif args.action == "render_candidates":
        render_candidates(case, read(args.spec), args.output)
    else:
        finalize(case, read(args.spec), args.output)


if __name__ == "__main__":
    main()
