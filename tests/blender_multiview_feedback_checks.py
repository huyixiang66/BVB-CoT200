"""Exercise multi-view preparation, shared edits, and renders without viewing image pixels."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile

import bpy
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import refine_blender as b
from scripts.prepare_visual_feedback import main as prepare
from scripts.render_visual_feedback_multiview import main as render


with tempfile.TemporaryDirectory(prefix="multiview-feedback-check-") as temp:
    root = Path(temp)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, 3
    bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))
    target = bpy.context.object
    target.name = "target"
    camera_data = bpy.data.cameras.new("camera")
    camera = bpy.data.objects.new("camera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = (0, -4, 2)
    camera.rotation_euler = (1.1, 0, 0)
    camera.keyframe_insert(data_path="location", frame=1)
    camera.keyframe_insert(data_path="rotation_euler", frame=1)
    camera.location.x = 1
    camera.keyframe_insert(data_path="location", frame=3)
    camera.keyframe_insert(data_path="rotation_euler", frame=3)
    scene.camera = camera

    original_blend = root / "original.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(original_blend))
    original_hash = hashlib.sha256(original_blend.read_bytes()).hexdigest()
    case = {"inputs": {"blend": str(original_blend)},
            "objects": {"moving": {"label": "target", "names": ["target"]}}}
    case_path = root / "case.json"
    case_path.write_text(json.dumps(case), encoding="utf-8")
    manifest = []
    for pair_id, frame in ((2, 1), (5, 3)):
        source, bvb = root / f"source_{pair_id}.jpg", root / f"bvb_{pair_id}.png"
        source.write_bytes(f"source fixture {pair_id}".encode())
        bvb.write_bytes(f"render fixture {pair_id}".encode())
        manifest.append({"pair_id": pair_id, "scene_frame": frame,
                         "source_frame": str(source), "render_frame": str(bvb),
                         "width": 32, "height": 32})
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    prepared_dir = root / "prepared"
    sys.argv = ["blender", "--", "--case", str(case_path), "--pairs-manifest", str(manifest_path),
                "--output", str(prepared_dir)]
    prepare()
    prepared_pairs = json.loads((prepared_dir / "pairs.json").read_text(encoding="utf-8"))
    prepared_case = json.loads((prepared_dir / "case.json").read_text(encoding="utf-8"))
    assert len(prepared_pairs) == 2
    assert prepared_pairs[-1]["reference_pair_id"] == 5
    assert prepared_pairs[0]["source"]["sha256"] == hashlib.sha256(
        Path(prepared_pairs[0]["source"]["file"]).read_bytes()).hexdigest()

    edit_dir = root / "edit_01"
    edit_dir.mkdir()
    spec = {"case": prepared_case, "pairs": prepared_pairs,
            "action": {"action": "adjust", "subject": "both",
                       "camera": {"translation_local": [0, 0, .2],
                                  "rotation_local_degrees": [0, 2, 0]},
                       "target": {"translation_local": [.1, 0, 0], "scale_ratio": 1.1}}}
    spec_path = edit_dir / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    sys.argv = ["blender", "--", "--spec", str(spec_path)]
    render()

    state = json.loads((edit_dir / "state.json").read_text(encoding="utf-8"))
    assert len(state["pairs"]) == 2
    reference = Matrix(prepared_pairs[0]["reference_camera_matrix"])
    camera_delta = reference.to_3x3().to_quaternion() @ Vector((0, 0, .2))
    target_delta = reference.to_3x3().to_quaternion() @ Vector((.1, 0, 0))
    for before, after in zip(prepared_pairs, state["pairs"]):
        before_position = Matrix(before["camera_matrix"]).translation
        after_position = Matrix(after["camera_matrix"]).translation
        assert (after_position - before_position - camera_delta).length < 1e-4
        assert Path(after["render"]["file"]).is_file()
        assert after["render"]["sha256"] == hashlib.sha256(Path(after["render"]["file"]).read_bytes()).hexdigest()
        expected_target = Vector(before["target_center_world"]) + target_delta
        assert (Vector(after["target_center_world"]) - expected_target).length < 1e-4
    assert hashlib.sha256(original_blend.read_bytes()).hexdigest() == original_hash
print("PASS: multi-view input prep, shared trajectory/object edit, CPU renders, source preservation; no pixel inspection")
