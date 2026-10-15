"""Actual Blender render smoke using synthetic geometry; never opens images."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pipeline as p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blender", required=True)
    parser.add_argument("--out", type=Path, default=Path("runs/synthetic-smoke"))
    args = parser.parse_args()
    blender, version = p.check_blender(args.blender)
    out = args.out.resolve()
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

    execute(p.ROOT / "tests/blender_fixture.py", [str(blend)])
    original_hash = p.digest(blend)
    fixed = {
        "scene": "fixture", "blend_sha256": original_hash, "video_frames": 24,
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
    for name, (views, expected) in cases.items():
        spec = dict(legacy_base, qa_id=name, views=views, out_dir=str(out / name))
        if name == "appearance":
            spec["grid"] = {"enable": False}
        path = out / (name + ".json")
        p.save_json(path, spec)
        execute(p.ROOT / "render_legacy.py", ["--spec", str(path)], blend)
        record = p.read_json(out / name / "manifest.json")
        assert len(record["frames"]) == expected, (name, len(record["frames"]))
        for frame in record["frames"]:
            image_path = out / name / frame["file"]
            assert image_path.stat().st_size > 100
            p.digest(image_path)
        checks[name] = expected
    assert p.digest(blend) == original_hash
    result = {"status": "pass", "blender": version, "frames": checks,
              "source_blend_unchanged": True, "visual_inspection": False, "api_calls": 0}
    p.save_json(out / "smoke_result.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
