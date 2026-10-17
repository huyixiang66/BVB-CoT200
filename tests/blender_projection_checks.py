"""Check candidate projection direction and frame identity using only geometry."""

from pathlib import Path
import sys

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import refine_project as p
import refine_measurement as m
from bpy_extras.object_utils import world_to_camera_view

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.resolution_x, scene.render.resolution_y = 768, 576
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.5))
moving = bpy.context.object
moving.name = "Moving"
bpy.ops.mesh.primitive_cube_add(size=1, location=(4, 0, 0.5))
reference = bpy.context.object
reference.name = "Reference"
bpy.ops.object.camera_add(location=(0, -8, 2))
scene.camera = bpy.context.object
scene.camera.rotation_euler = (Vector((0, 0, 0.5)) - scene.camera.location).to_track_quat("-Z", "Y").to_euler()
bpy.context.view_layer.update()
case = {"case_id": "synthetic", "objects": {
    "moving": {"label": "moving", "names": ["Moving"]},
    "reference": {"label": "reference", "names": ["Reference"]}}}
candidates = [
    {"id": "no_change", "operation": "no_change"},
    {"id": "toward", "operation": "translate", "direction": "toward", "distance_m": 0.3},
    {"id": "away", "operation": "translate", "direction": "away", "distance_m": 0.3},
]
result = p.project(case, [1, 10], candidates)
boxes = {key: rows[0]["projected_bbox_xyxy"] for key, rows in result["candidates"].items()}
assert boxes["toward"][0] > boxes["no_change"][0] > boxes["away"][0]
for rows in result["candidates"].values():
    assert [row["scene_frame"] for row in rows] == [1, 10]
    assert rows[0]["projected_bbox_xyxy"] == rows[1]["projected_bbox_xyxy"]
    assert all(row["first_hit_group_samples"] > 0 for row in rows)
evidence = {"desired_box_top_left": m.top_left_box(boxes["toward"])}
assert m.loss(boxes["toward"], evidence) == 0
assert m.loss(boxes["away"], evidence) > m.loss(boxes["no_change"], evidence)
assert (scene.render.resolution_x, scene.render.resolution_y) == (768, 576)
print("PASS: projection direction, candidate restore, frame identity, top-left conversion, source aspect; no rendering.")

# Observe only the front face of one part, but transform both frozen parts.
bpy.ops.mesh.primitive_cube_add(size=0.4, location=(0, 0, -0.3))
support = bpy.context.object
support.name = "Support"
case["objects"]["moving"]["names"].append("Support")
bpy.context.view_layer.update()
initial = {obj.name: obj.matrix_world.copy() for obj in (moving, support, reference)}
features = []
for index, point in enumerate((Vector((-0.2, -0.5, 0.6)), Vector((0.2, -0.5, 0.6)))):
    ndc = world_to_camera_view(scene, scene.camera, point)
    features.append({"id": str(index), "baseline": [ndc.x, 1 - ndc.y],
                     "same_physical_point": True})
partial = p.project(case, [1], candidates, observations={"1": features})
base = partial["candidates"]["no_change"][0]["features"]
assert len(base) == 2 and all(f["visible"] and f["object_name"] == "Moving" for f in base)
for f, expected in zip(base, features):
    assert max(abs(a - b) for a, b in zip(f["xy"], expected["baseline"])) < 1e-5
assert partial["transform_members"] == ["Moving", "Support"]
assert all(obj.matrix_world == initial[obj.name] for obj in (moving, support, reference))
p.b.apply_candidate(candidates[1], [moving, support], [reference])
bpy.context.view_layer.update()
moving_delta = moving.matrix_world.translation - initial["Moving"].translation
support_delta = support.matrix_world.translation - initial["Support"].translation
assert (moving_delta - support_delta).length < 1e-6 and moving_delta.length > 0.29
assert reference.matrix_world == initial["Reference"]
p.b.restore([moving, support], initial)
bpy.context.view_layer.update()
shifted = partial["candidates"]["toward"][0]["features"]
assert all(after["xy"][0] > before["xy"][0] for before, after in zip(base, shifted))
bound = [{"id": f["id"], "object_name": f["object_name"], "local_point": f["local_point"]} for f in base]
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, -2, 1))
blocker = bpy.context.object
blocker.scale = (3, 0.1, 3)
bpy.context.view_layer.update()
assert p.bind_features([moving, support], features) == []
assert all(not f["visible"] for f in p.project_features([moving, support], bound))
try:
    p.project_features([support], bound)
    raise AssertionError("Out-of-group anchor accepted")
except ValueError:
    pass
print("PASS: partial anchors, distinct transform group, source restore, foreground occlusion, out-of-group rejection.")
