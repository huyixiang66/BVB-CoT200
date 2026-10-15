"""Actual Blender render smoke using synthetic geometry; never opens images."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pipeline as p
from fixed_protocol import bvb_sample_frames
from scene_protocol import luna_recipe, legacy_views, render_prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blender", required=True)
    parser.add_argument("--scene-model", choices=("astra", "luna"), default="astra")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    blender, version = p.check_blender(args.blender)
    out = (args.out or Path("runs") / ("synthetic-smoke-" + args.scene_model)).resolve()
    out.mkdir(parents=True, exist_ok=True)
    blend = out / "fixture.blend"

    def execute(script, tail, source=None):
        command = [blender, "--background", "--disable-autoexec"]
        if source:
            command.append(str(source))
        command += ["--threads", "2", "--python-exit-code", "1", "--python", str(script), "--", *tail]
        log = out / (script.stem + "_" + str(len(list(out.glob("*.log")))) + ".log")
        with log.open("w", encoding="utf-8") as handle:
            subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, timeout=600, check=True,
                           env=p.blender_environment())

    luna = args.scene_model == "luna"
    start, end = (11, 42) if luna else (1, 24)
    execute(p.ROOT / "tests/blender_fixture.py", [str(blend), str(start), str(end)])
    original_hash = p.digest(blend)
    fixed = {
        "scene": "fixture", "blend_sha256": original_hash, "video_frames": None if luna else 24,
        "output": str(out / "fixed"), "arms": ["C"], "render_hash": "synthetic-C",
        "questions": [{
            "qa_id": "1", "scene_name": "fixture", "question_type": "object_size_estimation",
            "question": "What is the longest dimension of the table in centimetres?", "options": None,
            "roles": [
                {"role": "target object", "term": "table", "color": "RED", "anchor": False},
                {"role": "scale reference", "term": "door", "color": "BLUE", "anchor": True},
            ],
        }],
    }
    p.save_json(out / "fixed.json", fixed)
    execute(p.ROOT / "render_fixed.py", ["--spec", str(out / "fixed.json")], blend)
    manifest = p.read_json(out / "fixed/renders/fixture/1/manifest.json")
    baseline = p.read_json(out / "fixed/shared/fixture/manifest.json")
    assert len(manifest["frames"]) == 16
    assert manifest["timeline_samples"] == bvb_sample_frames(start, end, end - start + 1)
    assert [f["camera"] for f in manifest["frames"]] == [f["camera"] for f in baseline["frames"]]
    assert manifest["qc"]["extra_overhead_frames"] == 0
    assert manifest["qc"]["restored_materials"] and manifest["qc"]["restored_geometry_visibility"]
    checks = {"fixed": 16}
    legacy_base = {
        "scene": "fixture", "res": 96, "frame": 1, "hide": [], "grid": {"enable": True},
        "highlights": [{"color": "red", "match": ["table"]}],
    }
    cases = {
        "counting": ({"plan": 1, "oblique": 4, "focus": 8, "plan_end": 1}, 14),
        "direction": ({"startcam": {"at": ["chair"], "look": ["table"]},
                       "plan": 1, "focus": 8, "plan_end": 1}, 11),
        "appearance": ({"trajectory": {"n": 24}}, 24),
    }
    luna_questions = {
        "counting": ("object_counting", "How many chair(s) are in this room?"),
        "direction": ("object_rel_direction", "If I am standing by the chair and facing the table, is the door to my left?"),
        "appearance": ("obj_appearance_order", "What is the first-time appearance order in the video: table, chair, door, sofa?"),
        "missing_direction": ("object_rel_direction", "If I am standing by the bed and facing the tv, is the toilet to my left?"),
        "missing_counting": ("object_counting", "How many refrigerator(s) are in this room?"),
    }
    if luna:
        cases.update(missing_direction=({}, 2), missing_counting=({}, 6))
    for name, (views, expected) in cases.items():
        spec = dict(legacy_base, qa_id=name, views=views, out_dir=str(out / name))
        if name == "appearance":
            spec["grid"] = {"enable": False}
        if luna:
            kind, question = luna_questions[name]
            qa = {"qa_id": name, "scene_name": "fixture", "question_type": kind,
                  "question": question, "options": None}
            spec = dict(luna_recipe(qa, {"spec": {}})["spec"], res=96, out_dir=str(out / name))
        path = out / (name + ".json")
        p.save_json(path, spec)
        execute(p.ROOT / "render_legacy.py", ["--spec", str(path)], blend)
        record = p.read_json(out / name / "manifest.json")
        assert len(record["frames"]) == expected, (name, len(record["frames"]))
        if luna:
            actual_views = [f["view"] for f in record["frames"]]
            assert actual_views == legacy_views(kind, record["roles"], record["source_timeline"])
            prompt = render_prompt(qa, record["roles"], actual_views)
            assert f"The {expected} frames" in prompt
            assert record["spec"]["exact_names"]
            if name.startswith("missing"):
                assert "No reliable colour guide" in prompt
        for frame in record["frames"]:
            image_path = out / name / frame["file"]
            assert image_path.stat().st_size > 100
            p.digest(image_path)
        checks[name] = expected
    assert p.digest(blend) == original_hash
    result = {"status": "pass", "blender": version, "scene_model": args.scene_model, "frames": checks,
              "source_blend_unchanged": True, "visual_inspection": False, "api_calls": 0}
    p.save_json(out / "smoke_result.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
