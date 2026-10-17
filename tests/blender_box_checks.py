"""Verify target-box search and group transforms with synthetic geometry only."""

from pathlib import Path
import sys

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.refine_box_blender import execute, transform
import refine_blender as b
import refine_box as box
from refine_measurement import top_left_box

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.resolution_x, scene.render.resolution_y = 640, 480
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, .5))
target = bpy.context.object
target.name = "Target"
bpy.ops.mesh.primitive_cube_add(size=.2, location=(0, 0, 1.1))
part = bpy.context.object
part.name = "Part"
bpy.ops.mesh.primitive_cube_add(size=1, location=(4, 0, .5))
reference = bpy.context.object
reference.name = "Reference"
bpy.ops.object.camera_add(location=(0, -8, 2))
scene.camera = bpy.context.object
scene.camera.rotation_euler = (Vector((0, 0, .5)) - scene.camera.location).to_track_quat("-Z", "Y").to_euler()
bpy.context.view_layer.update()
case = {"objects": {"moving": {"label": "target", "names": ["Target", "Part"]},
                    "reference": {"label": "reference", "names": ["Reference"]}}}
matrices = {obj.name: obj.matrix_world.copy() for obj in (target, part, reference)}
candidates = [{"id": "identity", "parameters": [0, 0, 1]}, {"id": "move", "parameters": [.3, 0, 1]},
              {"id": "scale", "parameters": [0, 0, 1.2]}]
result = execute(case, {"scene_frames": [1, 10], "candidates": candidates}, Path("."))
boxes = {k: top_left_box(v[0]["projected_bbox_xyxy"]) for k, v in result["candidates"].items()}
assert boxes["move"][0] > boxes["identity"][0]
assert box.vector(boxes["scale"])[2] > box.vector(boxes["identity"])[2]
best, _ = box.select([v[0] for v in result["candidates"].values()],
                     {"baseline_box": boxes["identity"], "original_box": boxes["move"]}, boxes["identity"])
assert best["candidate"]["id"] == "move"
assert all(obj.matrix_world == matrices[obj.name] for obj in (target, part, reference))
body_case = {"objects": {"moving": {"label": "target", "names": ["Target"]},
                         "reference": case["objects"]["reference"]},
             "box_carried_names": ["Part"]}
body_result = execute(body_case, {"scene_frames": [1], "candidates": candidates}, Path("."))
assert body_result["transformed_members"] == ["Part", "Target"]
assert body_result["observed_members"] == ["Target"]
assert body_result["candidates"]["identity"][0]["projected_bbox_xyxy"] != result["candidates"]["identity"][0]["projected_bbox_xyxy"]
assert all(obj.matrix_world == matrices[obj.name] for obj in (target, part, reference))
for invalid in (
        {**body_case, "box_carried_names": ["Missing"]},
        {**body_case, "box_carried_names": ["Reference"]}):
    try:
        execute(invalid, {"scene_frames": [1], "candidates": candidates}, Path("."))
    except ValueError:
        pass
    else:
        raise AssertionError("Missing or reference accessory must fail before mutation")
transform([target, part], matrices, [.3, .2, 1.2], observed=[target])
assert abs(part.matrix_world.translation.x - .3) < 1e-6
assert abs(part.matrix_world.translation.z - 1.32) < 1e-6
assert abs(target.matrix_world.translation.z - .6) < 1e-6
b.restore([target, part], matrices)
transform([target, part], matrices, [.3, .2, 1.2])
assert abs(b.bbox([target, part])["min_xyz"][2]) < 1e-6
assert reference.matrix_world == matrices["Reference"]
assert result["source_camera_unchanged"]
assert (scene.render.resolution_x, scene.render.resolution_y) == (640, 480)
print("PASS: box search improves synthetic displacement; parts move together; floor pivot, reference/camera and restore unchanged; no images.")

# An occluder covers the left half. The objective must not retain hidden extent.
b.restore([target, part], matrices)
bpy.ops.mesh.primitive_cube_add(size=1, location=(-.4, -2, .5))
occluder = bpy.context.object
occluder.scale = (.6, .2, 2)
bpy.context.view_layer.update()
from scripts.refine_box_blender import visible_surface_box
visible = visible_surface_box([target])
full = __import__("refine_probe").visibility([target])
assert visible["projected_bbox_xyxy"][0] > full["projected_bbox_xyxy"][0] + .01
assert visible["first_hit_group_samples"] > 0
occluder.location.x = 0
occluder.scale.x = 3
bpy.context.view_layer.update()
assert visible_surface_box([target])["projected_bbox_xyxy"] is None
print("PASS: visible surface box excludes occluded extent and vanishes behind full occlusion.")

from scripts.refine_box_blender import unclipped_box
bpy.ops.object.camera_add(location=(0, -8, .5))
scene.camera = bpy.context.object
scene.camera.rotation_euler = (Vector((0, 0, .5))-scene.camera.location).to_track_quat("-Z", "Y").to_euler()
scene.camera.data.type = "ORTHO"
scene.camera.data.ortho_scale = 2
b.restore([target], matrices)
target.location.x = -.8
bpy.context.view_layer.update()
raw = unclipped_box([target])
assert raw[0] < 0 < raw[2] < 1
clipped = top_left_box(__import__("refine_probe").visibility([target])["projected_bbox_xyxy"])
assert clipped[0] == 0
assert abs(raw[2]-clipped[2]) < 1e-6
target.location.y = -8
bpy.context.view_layer.update()
assert unclipped_box([target]) is None
print("PASS: raw projection retains offscreen extent; near-plane crossing is invalid.")
