"""Blender regression checks using synthetic geometry, without rendering."""

from pathlib import Path
import sys
import json

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import refine_blender as b
import refine_probe as probe

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.resolution_x = scene.render.resolution_y = 512

def cube(name, location, scale):
    bpy.ops.mesh.primitive_cube_add(size=2, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.context.view_layer.update()
    return obj

moving = cube("Stove hob", (0, 0, 0.5), (0.5, 0.5, 0.5))
oven = cube("Built in oven glass", (4, 0, 0.5), (0.5, 0.5, 0.5))
reference = cube("Fridge", (7, 0, 1), (0.5, 0.5, 1))
bpy.ops.object.camera_add(location=(0, -8, 2))
scene.camera = bpy.context.object
scene.camera.rotation_euler = (Vector((0, 0, 0.5)) - scene.camera.location).to_track_quat("-Z", "Y").to_euler()
bpy.context.view_layer.update()
assert len(probe.components([moving, oven])) == 2
assert probe.visibility([moving])["first_hit_group_samples"] > 0
occluder = cube("Opaque blocker", (0, -3, 1), (3, 0.1, 3))
assert probe.visibility([moving])["first_hit_group_samples"] == 0
bpy.data.objects.remove(occluder, do_unlink=True)
bpy.context.view_layer.update()
initial = moving.matrix_world.copy()
floor_z = b.bbox([moving])["min_xyz"][2]
b.apply_candidate({"operation": "scale", "factor": 1.1, "pivot": "bottom"}, [moving], [reference])
bpy.context.view_layer.update()
assert abs(b.bbox([moving])["min_xyz"][2] - floor_z) < 1e-5
b.restore([moving], {moving.name: initial})
bpy.context.view_layer.update()
before = b.bbox([moving])["center_xyz"]
b.apply_candidate({"operation": "translate", "distance_m": 0.3, "direction": "toward"}, [moving], [reference])
bpy.context.view_layer.update()
scene.frame_set(10)
assert abs(b.bbox([moving])["center_xyz"][0] - before[0] - 0.3) < 1e-5
print(json.dumps({"status": "PASS", "tests": [
    "disconnected_components", "visible_target", "occluded_target",
    "scale_support_preserved", "translation_survives_frame_set"],
    "rendered": False, "image_access": False}))
