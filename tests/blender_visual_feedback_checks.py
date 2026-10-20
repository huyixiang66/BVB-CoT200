"""Verify model actions execute exactly in Blender; inspect numbers, never pixels."""

import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile

import bpy
from mathutils import Euler, Vector
from bpy_extras.object_utils import world_to_camera_view

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import refine_blender as b
from scripts.render_visual_feedback import main
from scripts.prepare_visual_feedback import main as prepare


with tempfile.TemporaryDirectory(prefix="visual-feedback-check-") as temp:
    root = Path(temp)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_cube_add()
    bpy.context.object.name = "target"
    data = bpy.data.cameras.new("camera")
    camera = bpy.data.objects.new("camera", data)
    bpy.context.scene.collection.objects.link(camera)
    camera.location = (0, -4, 2)
    camera.rotation_euler = (-camera.location).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = camera
    bpy.context.view_layer.update()
    matrix = camera.matrix_world.copy()
    original = root / "original.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(original))
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    case = {"objects": {"moving": {"label": "target", "names": ["target"]}}}
    pair = {"scene_frame": 1, "camera_matrix": [list(row) for row in matrix],
            "width": 32, "height": 32, "fx_px": 30, "fy_px": 29, "cx_px": 13, "cy_px": 19}
    for subject, translation, scale in (("camera", [0, 0, .3], 1), ("target", [.1, 0, 0], 1.1),
                                        ("both", [.1, .08, 0], .9)):
        bpy.ops.wm.open_mainfile(filepath=str(original))
        before = b.bbox([bpy.data.objects["target"]])
        directory = root / subject
        directory.mkdir()
        spec = directory / "spec.json"
        action = {"action": "adjust", "subject": subject, "translation_local": translation,
                  "rotation_local_degrees": [0, 0, 6] if subject == "camera" else [0, 0, 0],
                  "scale_ratio": scale}
        if subject == "both":
            action = {"action": "adjust", "subject": "both",
                      "camera": {"translation_local": [0, 0, .3], "rotation_local_degrees": [0, 0, 6]},
                      "target": {"translation_local": translation, "scale_ratio": scale}}
        spec.write_text(json.dumps({"case": case, "pair": pair, "action": action}), encoding="utf-8")
        sys.argv = ["blender", "--", "--spec", str(spec)]
        main()
        after = b.bbox([bpy.data.objects["target"]])
        delta = matrix.to_3x3() @ Vector(translation)
        actual = bpy.context.scene.camera.matrix_world.translation
        if subject in ("camera", "both"):
            camera_delta = matrix.to_3x3() @ Vector([0, 0, .3])
            assert (actual-matrix.translation-camera_delta).length < 1e-5
            expected = matrix.to_3x3() @ Euler((0, 0, math.radians(6)), "XYZ").to_matrix()
            assert max(abs(expected[i][j]-bpy.context.scene.camera.matrix_world[i][j])
                       for i in range(3) for j in range(3)) < 1e-5
        else:
            assert (actual-matrix.translation).length < 1e-5
        if subject == "camera":
            assert max(abs(a-c) for a, c in zip(before["size_xyz"], after["size_xyz"])) < 1e-5
            assert max(abs(a-c) for a, c in zip(before["center_xyz"], after["center_xyz"])) < 1e-5
        else:
            assert max(abs(a*scale-c) for a, c in zip(before["size_xyz"], after["size_xyz"])) < 1e-5
            assert abs(after["min_xyz"][2]-before["min_xyz"][2]-delta.z) < 1e-5
            assert max(abs(after["center_xyz"][i]-before["center_xyz"][i]-delta[i])
                       for i in (0, 1)) < 1e-5
        assert (directory / "render.png").exists() and (directory / "scene.blend").exists()
        active = bpy.context.scene.camera
        point = active.matrix_world @ Vector([1, 2, -5])
        uv = world_to_camera_view(bpy.context.scene, active, point)
        assert abs(uv.x*32-(13+30/5)) < 1e-4
        assert abs((1-uv.y)*32-(19-29*2/5)) < 1e-4
        assert hashlib.sha256(original.read_bytes()).hexdigest() == digest
    bpy.ops.wm.open_mainfile(filepath=str(original))
    bpy.context.scene.camera.data.sensor_fit = "VERTICAL"
    bpy.context.scene.camera.data.shift_x = .1
    bpy.context.scene.camera.data.shift_y = -.08
    bpy.context.scene.render.pixel_aspect_x = 1.3
    bpy.context.scene.render.pixel_aspect_y = 1.0
    expected_projection = bpy.context.scene.camera.calc_matrix_camera(
        bpy.context.evaluated_depsgraph_get(), x=32, y=24,
        scale_x=bpy.context.scene.render.pixel_aspect_x, scale_y=bpy.context.scene.render.pixel_aspect_y)
    source = root / "source.jpg"
    source.write_bytes(b"fixture frame; no pixel inspection")
    case["inputs"] = {"blend": str(original)}
    case_path = root / "input-case.json"
    case_path.write_text(json.dumps(case), encoding="utf-8")
    output = root / "prepared"
    sys.argv = ["blender", "--", "--case", str(case_path), "--source-frame", str(source),
                "--scene-frame", "1", "--output", str(output), "--width", "32", "--height", "24"]
    prepare()
    prepared = json.loads((output / "pairs.json").read_text())[0]
    prepared_case = json.loads((output / "case.json").read_text())
    assert prepared["source"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert prepared_case["inputs"]["blend"] == str((output / "scene.blend").resolve())
    assert max(abs(a-c) for ra, rb in zip(prepared["camera_matrix"], matrix)
               for a, c in zip(ra, rb)) < 1e-5
    actual_projection = bpy.context.scene.camera.calc_matrix_camera(
        bpy.context.evaluated_depsgraph_get(), x=32, y=24,
        scale_x=bpy.context.scene.render.pixel_aspect_x, scale_y=bpy.context.scene.render.pixel_aspect_y)
    assert max(abs(expected_projection[i][j]-actual_projection[i][j])
               for i in range(4) for j in range(4)) < 1e-5
    assert hashlib.sha256(original.read_bytes()).hexdigest() == digest
print("PASS: camera, target, joint actions with pre-edit axes, source preservation; no pixel inspection")
