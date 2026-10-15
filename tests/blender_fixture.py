"""Create a tiny procedural fixture; run only inside Blender."""

import sys
from pathlib import Path

import bpy
from mathutils import Vector

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 24
scene.render.resolution_percentage = 100
material = bpy.data.materials.new("FixtureMaterial")
material.use_nodes = True
material.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.5, 0.5, 0.5, 1)
for name, location, scale in (
    ("floor", (0, 0, -0.1), (5, 5, 0.1)),
    ("table", (2, 1, 0.5), (1, 0.6, 0.5)),
    ("chair", (-1, 1, 0.5), (0.4, 0.4, 0.5)),
    ("door", (-2, -3, 1), (0.1, 0.5, 1)),
):
    bpy.ops.mesh.primitive_cube_add(size=2, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    obj.data.materials.append(material)
bpy.ops.object.light_add(type="SUN", location=(0, 0, 5))
bpy.context.object.data.energy = 2
bpy.ops.object.camera_add(location=(7, -9, 6))
camera = bpy.context.object
camera.name = "FixtureCamera"
scene.camera = camera
for frame, location in ((1, (7, -9, 6)), (24, (-7, -9, 6))):
    camera.location = location
    camera.rotation_euler = (Vector((0, 0, 0.7)) - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.keyframe_insert(data_path="location", frame=frame)
    camera.keyframe_insert(data_path="rotation_euler", frame=frame)
scene.frame_set(1)
output = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
output.parent.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=str(output))
