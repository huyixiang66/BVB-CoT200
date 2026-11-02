"""Render geometry-selected views without reading or inspecting pixels."""

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import bpy
from mathutils import Matrix, Vector
from bpy_extras.object_utils import world_to_camera_view


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fit_camera import bbox_corners, project


def cli_arg(name, default=None):
    argv = sys.argv
    if "--" not in argv:
        return default
    args = argv[argv.index("--") + 1:]
    for index, value in enumerate(args):
        if value == name and index + 1 < len(args):
            return args[index + 1]
    return default


QA_FILTER = set((cli_arg("--qa", "") or "").split(",")) - {""}
RESOLUTION = int(cli_arg("--res", "512"))
SCENE = cli_arg("--scene", "0d2ee665be")
GEOMETRY_PATH = Path(cli_arg("--geometry", str(HERE / f"geometry_{SCENE}.json")))
OUT_ROOT = Path(cli_arg("--out-root", str(HERE / "renders" / SCENE)))
MANIFEST_PATH = Path(cli_arg("--manifest", str(HERE / "render_manifest.json")))
GROUPS_PATH = Path(cli_arg("--groups", str(HERE / f"object_groups_{SCENE}.json")))
RENDER_DIAGNOSTICS = '--render-diagnostics' in sys.argv


def look_at_euler(location, target):
    direction = Vector(target) - Vector(location)
    if direction.length < 1e-8:
        direction = Vector((0, 0, -1))
    return direction.to_track_quat("-Z", "Y").to_euler()


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.sha256(stream.read()).hexdigest()


geometry = json.loads(GEOMETRY_PATH.read_text(encoding="utf-8"))
scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x = RESOLUTION
scene.render.resolution_y = RESOLUTION
scene.render.resolution_percentage = 100
scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1.0
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGB"
scene.render.film_transparent = False
try:
    scene.eevee.taa_render_samples = 16
except Exception:
    pass

groups = json.loads(GROUPS_PATH.read_text(encoding="utf-8"))
ceiling_meshes = set(groups["room"]["ceiling"])
if groups['room'].get('remove_walls', False):
    ceiling_meshes.update(groups['room'].get('walls', []))
for obj in bpy.data.objects:
    if obj.type == "MESH" and obj.name in ceiling_meshes:
        obj.hide_render = True

if not any(obj.type == "LIGHT" for obj in bpy.data.objects):
    light_data = bpy.data.lights.new("render_sun", type="SUN")
    light_data.energy = 3.0
    light = bpy.data.objects.new("render_sun", light_data)
    light.rotation_euler = (math.radians(50), 0, math.radians(30))
    scene.collection.objects.link(light)

camera_data = bpy.data.cameras.new("selected_view_camera")
camera_data.sensor_width = 36.0
camera_data.sensor_fit = 'HORIZONTAL'
camera_data.clip_start = 0.03
camera_data.clip_end = 100.0
camera = bpy.data.objects.new("selected_view_camera", camera_data)
scene.collection.objects.link(camera)
scene.camera = camera

manifest = {
    "scene": SCENE,
    "blend": bpy.data.filepath,
    "resolution": [RESOLUTION, RESOLUTION],
    "engine": scene.render.engine,
    "visual_inspection_allowed": True,
    "questions": [],
}
started = time.time()

for question in geometry["questions"]:
    qa_id = str(question["qa_id"])
    if QA_FILTER and qa_id not in QA_FILTER:
        continue
    views = [(view, 'views') for view in question.get('chosen_readable', [])]
    if RENDER_DIAGNOSTICS:
        views += [(view, 'diagnostic_views') for view in question.get('diagnostic_views', [])]
    record = {
        "qa_id": qa_id,
        "question_type": question["question_type"],
        "targets": question["targets"],
        "selection_status": question['selection_status'],
        "qa_ready": question['qa_ready'],
        "views": [],
        "diagnostic_views": [],
    }
    for index, (view, kind) in enumerate(views, 1):
        if kind=='views':
            assert view['acceptance']['passed'] and view['geometry_safe'], 'Unaccepted view cannot enter QA inputs'
        out_dir = (OUT_ROOT / qa_id if kind=='views' else OUT_ROOT / 'diagnostics' / qa_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        camera.location = Vector(view["location"])
        if view["ortho_scale"] is not None:
            camera.data.type = "ORTHO"
            camera.data.ortho_scale = float(view["ortho_scale"])
            camera.rotation_euler = (0.0, 0.0, 0.0)
        else:
            camera.data.type = "PERSP"
            camera.data.lens = float(view["lens_mm"])
            if 'right' in view:
                camera.rotation_euler = Matrix((view['right'], view['up'],
                    [-v for v in view['forward']])).transposed().to_euler()
            else:
                camera.rotation_euler = look_at_euler(view["location"], view["target"])
        if 'right' in view:
            camera.rotation_euler = Matrix((view['right'], view['up'],
                [-v for v in view['forward']])).transposed().to_euler()
        bpy.context.view_layer.update()
        max_error = 0.0
        for semantic, boxes in question.get('object_boxes', {}).items():
            for instance, box in enumerate(boxes):
                check_keys = view.get('metadata', {}).get('projection_check_keys', view.get('metadata', {}).get('frame_keys', []))
                if f'{semantic}#{instance}' not in check_keys:
                    continue
                for corner in bbox_corners(box):
                    if view['ortho_scale'] is None:
                        depth = sum((corner[i]-view['location'][i])*view['forward'][i] for i in range(3))
                        if depth<=.03:
                            continue
                        expected = project(corner, view)
                        # Near-horizon off-screen corners amplify float error;
                        # validate visible image coordinates for standing views.
                        if view.get('metadata', {}).get('role')=='standing_reference' and not (0<=expected[0]<=1 and 0<=expected[1]<=1):
                            continue
                    else:
                        delta = [corner[i]-view['location'][i] for i in range(3)]
                        expected = [.5+sum(delta[i]*view[axis][i] for i in range(3))/view['ortho_scale'] for axis in ('right','up')]
                    actual = world_to_camera_view(scene, camera, Vector(corner))
                    max_error = max(max_error, abs(expected[0]-actual.x), abs(expected[1]-actual.y))
        assert max_error < 1e-5, ('projection disagreement', max_error)
        filepath = out_dir / f"{index:02d}_{view['id']}.png"
        scene.render.filepath = str(filepath.resolve())
        bpy.ops.render.render(write_still=True)
        record[kind].append({
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
            "projection_max_error": max_error,
            "acceptance": view['acceptance'],
            "geometry_accepted": kind=='views',
            "qa_eligible": kind=='views' and question['qa_ready'],
        })
        print("RENDER", qa_id, index, view["id"], filepath.stat().st_size, flush=True)
    manifest["questions"].append(record)

manifest["elapsed_seconds"] = round(time.time() - started, 2)
manifest["view_count"] = sum(len(item["views"]) for item in manifest["questions"])
manifest['diagnostic_view_count'] = sum(len(item['diagnostic_views']) for item in manifest['questions'])
MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print("RENDER_OK", manifest["view_count"], MANIFEST_PATH, flush=True)
