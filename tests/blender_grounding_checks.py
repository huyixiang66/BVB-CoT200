"""Verify geometry proposals against actual first hits without rendering."""

from pathlib import Path
import sys

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.propose_refinement_anchors import propose, visible_extrema, background_exclusions, structural_points
import refine_project as p

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.resolution_x, scene.render.resolution_y = 640, 480
bpy.ops.mesh.primitive_cube_add(size=1)
target = bpy.context.object
target.name = "Target"
target_collection = bpy.data.collections.new("Target assembly")
scene.collection.children.link(target_collection)
for collection in list(target.users_collection):
    collection.objects.unlink(target)
target_collection.objects.link(target)
bpy.ops.mesh.primitive_cube_add(size=0.3, location=(0, 0, 0.8))
accessory = bpy.context.object
accessory.name = "Accessory"
for collection in list(accessory.users_collection):
    collection.objects.unlink(accessory)
target_collection.objects.link(accessory)
bpy.ops.mesh.primitive_cube_add(size=0.1, location=(0.4, 0, 0.8))
unresolved_part = bpy.context.object
unresolved_part.name = "Unresolved knob"
for collection in list(unresolved_part.users_collection):
    collection.objects.unlink(unresolved_part)
target_collection.objects.link(unresolved_part)
bpy.ops.mesh.primitive_cube_add(size=0.6, location=(1.3, 0, 0))
reference = bpy.context.object
reference.name = "Reference"
bpy.ops.object.camera_add(location=(2, -5, 2))
scene.camera = bpy.context.object
scene.camera.rotation_euler = (-scene.camera.location).to_track_quat("-Z", "Y").to_euler()
bpy.context.view_layer.update()
case = {"objects": {"moving": {"label": "target", "names": ["Target"]},
                    "reference": {"label": "reference", "names": ["Reference"]}},
        "object_resolution": {"roles": [{"role": "moving", "objects": ["Target", "Accessory"]}]}}
matrices = {obj.name: obj.matrix_world.copy() for obj in scene.objects}
row = propose(case)
assert len(row["features"]) >= 2
assert row["transform_members"] == ["Target"]
assert row["background_excluded_members"] == ["Accessory", "Target"]
assert all(f["object_name"] not in {"Target", "Accessory"} for f in row["landmarks"])
assert all(f["visible"] for f in p.project_features([target], row["features"]))
assert all(obj.matrix_world == matrices[obj.name] for obj in scene.objects)
assert (scene.render.resolution_x, scene.render.resolution_y) == (640, 480)
excluded = background_exclusions(case, [target], shared_collections=True)
assert {"Accessory", "Target", "Unresolved knob"} <= excluded
assert "Reference" not in excluded
filtered = propose(case, shared_collections=True)
assert filtered["transform_members"] == ["Target"]
assert filtered["features"] == row["features"]
assert all(p["object_name"] not in excluded for p in filtered["landmarks"])
reference.name = "frame rail"
structure = structural_points(reference)
assert structure and all("point_description" in f for f in structure)
assert all(f["visible"] for f in p.project_features([reference], structure))
reference.name = "soft duvet"
assert structural_points(reference) == []
reference.name = "Reference"
target.hide_render = True
assert visible_extrema(target) == []
target.hide_render = False
bpy.ops.mesh.primitive_cube_add(size=2, location=scene.camera.location / 2)
blocker = bpy.context.object
blocker.scale = (2, 2, 2)
bpy.context.view_layer.update()
assert visible_extrema(target) == []
print("PASS: source anchors reproject visibly; accessory exclusion, frozen group, occlusion, source restore; no pixels.")
