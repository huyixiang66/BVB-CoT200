"""Blender smoke for the refinement renderer; never opens rendered images."""

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
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    blender, version = p.check_blender(args.blender)
    out = (args.out or Path("runs/synthetic-refine-smoke")).resolve()
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

    execute(p.ROOT / "tests/blender_fixture.py", [str(blend), "1", "8"])
    original_hash = p.digest(blend)
    case = {
        "case_id": "fixture", "scene": "fixture", "question_type": "object_abs_distance",
        "question": "What is the distance between the table and chair?",
        "inputs": {"original_video": "unused", "bvb_video": "unused", "blend": str(blend)},
        "objects": {
            "moving": {"label": "table", "names": ["table"]},
            "reference": {"label": "chair", "names": ["chair"]},
        },
        "candidates": [
            {"id": "no_change", "operation": "no_change"},
            {"id": "toward_050", "operation": "translate", "distance_m": 0.5},
        ],
    }
    p.save_json(out / "case.json", case)
    renderer = p.ROOT / "refine_blender.py"
    execute(renderer, ["inspect", "--case", str(out / "case.json"),
                       "--output", str(out / "scene_before.json")], blend)
    candidate_spec = {"candidates": case["candidates"], "scene_frames": [1, 8]}
    p.save_json(out / "candidate_spec.json", candidate_spec)
    execute(renderer, ["render_candidates", "--case", str(out / "case.json"),
                       "--spec", str(out / "candidate_spec.json"),
                       "--output", str(out / "candidates")], blend)
    final_spec = {"candidate": case["candidates"][1], "qa_scene_frames": [1, 8], "mode": "blind"}
    p.save_json(out / "final_spec.json", final_spec)
    execute(renderer, ["finalize", "--case", str(out / "case.json"),
                       "--spec", str(out / "final_spec.json"),
                       "--output", str(out / "final")], blend)
    manifest = p.read_json(out / "final/manifest.json")
    assert manifest["after_state"]["surface_distance_m"] < manifest["before_state"]["surface_distance_m"]
    assert len(manifest["before_qa"]) == 2 and len(manifest["after_qa"]) == 2
    assert (out / "final/scene_after.blend").is_file()
    assert p.digest(blend) == original_hash
    result = {"status": "pass", "blender": version,
              "before_distance_m": manifest["before_state"]["surface_distance_m"],
              "after_distance_m": manifest["after_state"]["surface_distance_m"],
              "source_blend_unchanged": True, "visual_inspection": False, "api_calls": 0}
    p.save_json(out / "smoke_result.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
