"""Run inside Blender: original-camera frames, reversible highlights, one overhead."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixed_protocol import COLORS, VERSION, bvb_sample_frames, resolve_roles
from name_resolver import COMPOUND_BLOCK, SYN, head_term


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def camera_record(scene):
    camera = scene.camera.evaluated_get(bpy.context.evaluated_depsgraph_get())
    data = camera.data
    return {
        "name": scene.camera.name, "frame": scene.frame_current,
        "matrix_world": [float(v) for row in camera.matrix_world for v in row],
        "type": data.type, "lens": data.lens, "sensor_fit": data.sensor_fit,
        "sensor_width": data.sensor_width, "sensor_height": data.sensor_height,
        "shift_x": data.shift_x, "shift_y": data.shift_y,
        "clip_start": data.clip_start, "clip_end": data.clip_end,
        "ortho_scale": data.ortho_scale,
        "resolution": [scene.render.resolution_x, scene.render.resolution_y],
        "pixel_aspect": [scene.render.pixel_aspect_x, scene.render.pixel_aspect_y],
    }


def geometry_record(meshes):
    return {o.name: {"data": o.data.name, "verts": len(o.data.vertices),
                     "polygons": len(o.data.polygons), "hide_render": o.hide_render,
                     "matrix_world": [float(v) for row in o.matrix_world for v in row]}
            for o in meshes}


def near_equal(left, right):
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(near_equal(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(near_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, float):
        return math.isclose(left, right, abs_tol=1e-6, rel_tol=1e-6)
    return left == right


def original_material_signature(meshes):
    return {o.name: [slot.material.name if slot.material else None for slot in o.material_slots]
            for o in meshes}


def tinted_material(original, color, cache):
    key = (original.name, color)
    if key in cache:
        return cache[key]
    material = original.copy()
    material.name = "FC_" + color + "_" + original.name
    touched = False
    rgba = COLORS[color]
    if material.use_nodes:
        tree = material.node_tree
        for node in list(tree.nodes):
            socket_name = {
                "BSDF_PRINCIPLED": "Base Color", "BSDF_DIFFUSE": "Color",
                "BSDF_GLOSSY": "Color", "EMISSION": "Color",
            }.get(node.type)
            if not socket_name:
                continue
            socket = node.inputs.get(socket_name)
            if socket is None:
                continue
            # Tint the colour input, preserving alpha, transmission, normals and lighting.
            if socket.is_linked:
                previous = socket.links[0].from_socket
                mix = tree.nodes.new("ShaderNodeMixRGB")
                mix.blend_type = "MIX"
                mix.inputs[0].default_value = 0.75
                mix.inputs[2].default_value = rgba
                tree.links.new(previous, mix.inputs[1])
                tree.links.new(mix.outputs[0], socket)
            else:
                previous = socket.default_value[:]
                socket.default_value = tuple(0.25 * previous[i] + 0.75 * rgba[i] for i in range(3)) + (previous[3],)
            touched = True
    else:
        previous = material.diffuse_color[:]
        material.diffuse_color = tuple(0.25 * previous[i] + 0.75 * rgba[i] for i in range(3)) + (previous[3],)
        touched = True
    if not touched:
        bpy.data.materials.remove(material)
        cache[key] = None
    else:
        cache[key] = material
    return cache[key]


def apply_highlights(resolved, meshes, cache):
    by_name = {o.name: o for o in meshes}
    saved = []
    changed = set()
    for role in resolved:
        supported = []
        unsupported = []
        for name in role["objects"]:
            obj = by_name[name]
            materials = list(obj.data.materials)
            replacements = [tinted_material(mat, role["color"], cache) if mat else None
                            for mat in materials]
            if not replacements or not any(mat is not None for mat in replacements):
                unsupported.append(name)
                continue
            if name in changed:
                raise ValueError("Cross-role material overwrite: " + name)
            saved.append((obj, obj.data))
            obj.data = obj.data.copy()
            for index, material in enumerate(replacements):
                if material is not None:
                    obj.data.materials[index] = material
            supported.append(name)
            changed.add(name)
        role["objects"] = supported
        role["unsupported_materials"] = unsupported
        role["status"] = "name_candidates" if supported else "unresolved"
    return saved, changed


def restore_highlights(saved):
    for obj, original in saved:
        copied = obj.data
        obj.data = original
        bpy.data.meshes.remove(copied)


def frame_output(scene, path, view, camera=None):
    scene.render.filepath = str(path)
    result = bpy.ops.render.render(write_still=True)
    if "FINISHED" not in result or not path.is_file():
        raise RuntimeError("Render failed: " + str(path))
    return {"file": str(path), "view": view, "sha256": digest(path),
            "bytes": path.stat().st_size, "camera": camera or camera_record(scene)}


def overhead_settings(scene, meshes):
    # One scene-wide framing, never a QA-dependent floor selection or target crop.
    visible = [o for o in meshes if not o.hide_render]
    corners = [o.matrix_world @ Vector(point) for o in visible for point in o.bound_box]
    if not corners:
        raise ValueError("Scene has no visible mesh bounds")
    lo = [min(p[i] for p in corners) for i in range(3)]
    hi = [max(p[i] for p in corners) for i in range(3)]
    aspect = (scene.render.resolution_x * scene.render.pixel_aspect_x /
              (scene.render.resolution_y * scene.render.pixel_aspect_y))
    span = max(hi[0] - lo[0], (hi[1] - lo[1]) * aspect, 1.0) * 1.08
    return {"bounds": [lo, hi], "scale": span,
            "location": [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, hi[2] + max(span, 2.0)]}


def render_scene(spec_path):
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    scene = bpy.context.scene
    if scene.camera is None:
        raise ValueError("BVB scene has no active camera")
    for engine in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        try:
            scene.render.engine = engine
            break
        except TypeError:
            pass
    else:
        raise ValueError("No EEVEE renderer")
    scene.render.resolution_x = scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.use_file_extension = True
    video_frames = spec.get("video_frames")
    if video_frames is None:
        video_frames = min(64, scene.frame_end - scene.frame_start + 1)
    frame_ids = bvb_sample_frames(scene.frame_start, scene.frame_end, video_frames)
    meshes = [o for o in scene.objects if o.type == "MESH"]
    baseline_materials = original_material_signature(meshes)
    initial_camera = scene.camera
    scene.frame_set(frame_ids[0])
    baseline_geometry = geometry_record(meshes)
    c_only = spec.get("arms") == ["C"]
    plan_settings = None if c_only else overhead_settings(scene, meshes)
    output = Path(spec["output"])
    shared = output / "shared" / spec["scene"]
    shared.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    shared_manifest_path = shared / "manifest.json"
    if shared_manifest_path.exists():
        shared_manifest = json.loads(shared_manifest_path.read_text(encoding="utf-8"))
        if shared_manifest["render_hash"] != spec["render_hash"]:
            raise ValueError("Existing shared render belongs to a different protocol")
        if any(not Path(fr["file"]).exists() or digest(Path(fr["file"])) != fr["sha256"]
               for fr in shared_manifest["frames"] if "file" in fr):
            raise ValueError("Existing shared render is incomplete or changed")
    else:
        frames = []
        for i, frame in enumerate(frame_ids):
            scene.frame_set(frame)
            frames.append({"camera": camera_record(scene)} if c_only else
                          frame_output(scene, shared / f"frame_{i + 1:02d}.png",
                                       "original_camera", camera_record(scene)))
        shared_manifest = {"scene": spec["scene"], "version": VERSION,
                           "render_hash": spec["render_hash"], "blend_sha256": spec["blend_sha256"],
                           "source_video_frames": video_frames,
                           "source_timeline": [scene.frame_start, scene.frame_end],
                           "timeline_samples": frame_ids, "frames": frames,
                           "engine": scene.render.engine}
        write_json(shared_manifest_path, shared_manifest)
    cache = {}
    for qa in spec["questions"]:
        qaid = str(qa["qa_id"])
        destination = output / "renders" / spec["scene"] / qaid
        manifest_path = destination / "manifest.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing["render_hash"] != spec["render_hash"]:
                raise ValueError("Existing question render belongs to a different protocol")
            print("FIXED_CAMERA_SKIP q" + qaid, flush=True)
            continue
        destination.mkdir(parents=True, exist_ok=True)
        scene.camera = initial_camera
        scene.frame_set(frame_ids[0])
        if not near_equal(geometry_record(meshes), baseline_geometry):
            raise ValueError("Geometry or visibility leaked between questions")
        if original_material_signature(meshes) != baseline_materials:
            raise ValueError("Material state leaked between questions")
        resolved = resolve_roles(qa["roles"], [o.name for o in meshes if not o.hide_render],
                                 SYN, head_term, COMPOUND_BLOCK)
        saved, changed = apply_highlights(resolved, meshes, cache)
        frames = []
        roof_hidden = []
        plan_camera = None
        try:
            for i, frame in enumerate(frame_ids):
                original_record = shared_manifest["frames"][i]["camera"]
                scene.frame_set(frame)
                scene.camera = bpy.data.objects[original_record["name"]]
                record = camera_record(scene)
                if not near_equal(record, original_record):
                    raise ValueError(f"Original camera changed at frame {frame}")
                frames.append(frame_output(scene, destination / f"frame_{i + 1:02d}.png",
                                           "original_camera", record))
            if not c_only:
                scene.frame_set(frame_ids[0])
                for obj in meshes:
                    lower = obj.name.lower()
                    if not obj.hide_render and re_roof(lower) and obj.name not in changed:
                        roof_hidden.append(obj)
                        obj.hide_render = True
                data = bpy.data.cameras.new("FC_Plan")
                plan_camera = bpy.data.objects.new("FC_Plan", data)
                scene.collection.objects.link(plan_camera)
                plan_camera.location = plan_settings["location"]
                plan_camera.rotation_euler = (0, 0, 0)
                data.type = "ORTHO"
                data.ortho_scale = plan_settings["scale"]
                data.clip_start = 0.01
                data.clip_end = max(1000.0, plan_settings["scale"] * 10)
                scene.camera = plan_camera
                frames.append(frame_output(scene, destination / "frame_17_plan.png", "overhead"))
        finally:
            for obj in roof_hidden:
                obj.hide_render = False
            scene.camera = initial_camera
            if plan_camera is not None:
                data = plan_camera.data
                bpy.data.objects.remove(plan_camera, do_unlink=True)
                bpy.data.cameras.remove(data)
            restore_highlights(saved)
            scene.frame_set(frame_ids[0])
        if original_material_signature(meshes) != baseline_materials:
            raise ValueError("Failed to restore original materials")
        if not near_equal(geometry_record(meshes), baseline_geometry):
            raise ValueError("Failed to restore geometry or visibility")
        if changed != {name for role in resolved for name in role["objects"]}:
            raise ValueError("Material changes outside current question's role mapping")
        write_json(manifest_path, {
            "version": VERSION, "qa_id": qaid, "scene": spec["scene"],
            "render_hash": spec["render_hash"], "blend_sha256": spec["blend_sha256"],
            "source_timeline": [scene.frame_start, scene.frame_end],
            "timeline_samples": frame_ids, "frames": frames, "roles": resolved,
            "changed_objects": sorted(changed), "plan": plan_settings,
            "roof_hidden_in_plan_only": [o.name for o in roof_hidden],
            "qc": {"original_camera_unchanged": True, "restored_materials": True,
                   "restored_geometry_visibility": True, "question_isolation": True,
                   "original_frames": 16, "extra_overhead_frames": 0 if c_only else 1,
                   "numeric_answer_annotations": False},
        })
        print(f"FIXED_CAMERA_OK q{qaid} highlighted_objects={len(changed)} "
              f"elapsed={time.monotonic() - started:.1f}", flush=True)
    print("FIXED_CAMERA_SCENE_OK " + spec["scene"], flush=True)


def re_roof(name):
    import re
    return (bool(re.search(r"\b(ceiling|roof)\b", name)) and
            not re.search(r"\b(light|lamp|fixture|fan|diffuser)\b", name))


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:]
    render_scene(Path(args[args.index("--spec") + 1]))
