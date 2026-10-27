"""Render geometry-selected views without reading or inspecting pixels."""

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import bpy
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


GEOMETRY_PATH = Path(cli_arg(
    "--geometry", str(ROOT / "runs" / "envelope_views" / "geometry.json")))
OUT_ROOT = Path(cli_arg(
    "--out", str(ROOT / "runs" / "envelope_views" / "renders")))
MANIFEST_PATH = Path(cli_arg(
    "--manifest", str(ROOT / "runs" / "envelope_views" / "render_manifest.json")))
QA_FILTER = set((cli_arg("--qa", "") or "").split(",")) - {""}
RESOLUTION = int(cli_arg("--res", "512"))


def look_at_euler(location, target):
    direction = Vector(target) - Vector(location)
    if direction.length < 1e-8:
        direction = Vector((0, 0, -1))
    return direction.to_track_quat("-Z", "Y").to_euler()


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


geometry = json.loads(GEOMETRY_PATH.read_text(encoding="utf-8"))
scene_name = geometry.get("scene") or Path(bpy.data.filepath).stem
scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x = RESOLUTION
scene.render.resolution_y = RESOLUTION
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGB"
scene.render.film_transparent = False
try:
    scene.eevee.taa_render_samples = 16
except Exception:
    pass

ceiling_names = set(geometry.get("room", {}).get("ceiling_objects", []))
for obj in bpy.data.objects:
    if obj.type == "MESH" and (obj.name in ceiling_names or "ceiling" in obj.name.lower()):
        obj.hide_render = True

if not any(obj.type == "LIGHT" for obj in bpy.data.objects):
    light_data = bpy.data.lights.new("render_sun", type="SUN")
    light_data.energy = 3.0
    light = bpy.data.objects.new("render_sun", light_data)
    light.rotation_euler = (math.radians(50), 0, math.radians(30))
    scene.collection.objects.link(light)

camera_data = bpy.data.cameras.new("selected_view_camera")
camera_data.sensor_width = 36.0
camera_data.clip_start = 0.03
camera_data.clip_end = 100.0
camera = bpy.data.objects.new("selected_view_camera", camera_data)
scene.collection.objects.link(camera)
scene.camera = camera

manifest = {
    "scene": scene_name,
    "blend": bpy.data.filepath,
    "resolution": [RESOLUTION, RESOLUTION],
    "engine": scene.render.engine,
    "no_image_access": True,
    "questions": [],
}
started = time.time()
MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

for question in geometry["questions"]:
    qa_id = str(question["qa_id"])
    if QA_FILTER and qa_id not in QA_FILTER:
        continue
    views = question.get("chosen_readable") or []
    if not views:
        print("SKIP no views", qa_id, flush=True)
        continue
    out_dir = OUT_ROOT / qa_id
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "qa_id": qa_id,
        "question_type": question["question_type"],
        "targets": question["targets"],
        "views": [],
    }
    for index, view in enumerate(views, 1):
        camera.location = Vector(view["location"])
        if view["ortho_scale"] is not None:
            camera.data.type = "ORTHO"
            camera.data.ortho_scale = float(view["ortho_scale"])
            camera.rotation_euler = (0.0, 0.0, 0.0)
        else:
            camera.data.type = "PERSP"
            camera.data.lens = float(view["lens_mm"])
            camera.rotation_euler = look_at_euler(view["location"], view["target"])
        bpy.context.view_layer.update()
        filepath = out_dir / f"{index:02d}_{view['id']}.png"
        scene.render.filepath = str(filepath.resolve())
        bpy.ops.render.render(write_still=True)
        record["views"].append({
            "index": index,
            "view_id": view["id"],
            "method": view["method"],
            "camera_type": camera.data.type,
            "location": view["location"],
            "target": view["target"],
            "ortho_scale": view["ortho_scale"],
            "lens_mm": view["lens_mm"],
            "file": str(filepath.resolve()),
            "sha256": sha256(filepath),
            "bytes": filepath.stat().st_size,
        })
        print("RENDER", qa_id, index, view["id"], filepath.stat().st_size, flush=True)
    manifest["questions"].append(record)

manifest["elapsed_seconds"] = round(time.time() - started, 2)
manifest["view_count"] = sum(len(item["views"]) for item in manifest["questions"])
MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print("RENDER_OK", manifest["view_count"], MANIFEST_PATH, flush=True)
