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
    auto_case = dict(case)
    auto_case["objects"] = {
        "moving": {"label": "table", "auto_match": True},
        "reference": {"label": "chair", "auto_match": True},
    }
    p.save_json(out / "auto_case.json", auto_case)
    execute(renderer, ["resolve", "--case", str(out / "auto_case.json"),
                       "--output", str(out / "object_resolution.json")], blend)
    resolution = p.read_json(out / "object_resolution.json")
    assert resolution["objects"]["moving"]["names"] == ["table"]
    assert resolution["objects"]["reference"]["names"] == ["chair"]
    assert resolution["resolution"]["instance_verified"] is False
    unresolved_case = dict(case)
    unresolved_case["objects"] = {
        "moving": {"label": "missing moving", "auto_match": True},
        "reference": {"label": "missing reference", "auto_match": True},
    }
    p.save_json(out / "unresolved_case.json", unresolved_case)
    execute(renderer, ["resolve", "--case", str(out / "unresolved_case.json"),
                       "--output", str(out / "unresolved_resolution.json")], blend)
    unresolved = p.read_json(out / "unresolved_resolution.json")
    assert unresolved["resolution"]["complete"] is False
    assert unresolved["resolution"]["unresolved_roles"] == ["moving", "reference"]
    no_change = {"id": "no_change", "operation": "no_change"}
    unresolved_runtime = dict(unresolved_case)
    unresolved_runtime["candidates"] = [no_change]
    p.save_json(out / "unresolved_runtime.json", unresolved_runtime)
    p.save_json(out / "unresolved_candidate_spec.json", {
        "candidates": [no_change], "scene_frames": [1],
    })
    execute(renderer, ["render_candidates", "--case", str(out / "unresolved_runtime.json"),
                       "--spec", str(out / "unresolved_candidate_spec.json"),
                       "--output", str(out / "unresolved_candidates")], blend)
    unresolved_candidates = p.read_json(out / "unresolved_candidates/manifest.json")
    assert unresolved_candidates["initial_state"]["status"] == "unresolved"
    p.save_json(out / "unresolved_final_spec.json", {
        "candidate": no_change, "qa_scene_frames": [1], "mode": "blind",
    })
    execute(renderer, ["finalize", "--case", str(out / "unresolved_runtime.json"),
                       "--spec", str(out / "unresolved_final_spec.json"),
                       "--output", str(out / "unresolved_final")], blend)
    unresolved_final = p.read_json(out / "unresolved_final/manifest.json")
    assert unresolved_final["before_state"]["status"] == "unresolved"
    assert unresolved_final["after_state"]["status"] == "unresolved"
    assert unresolved_final["qa_frames_byte_identical"] is True
    assert unresolved_final["before_qa"][0]["sha256"] == unresolved_final["after_qa"][0]["sha256"]
    assert unresolved_final["before_qa"][0]["camera"] == unresolved_final["after_qa"][0]["camera"]
    assert unresolved_final["source_camera_unchanged"] is True
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
    assert manifest["qa_frames_byte_identical"] is False
    assert manifest["source_camera_unchanged"] is True
    assert manifest["source_camera_trajectory"][0] != manifest["source_camera_trajectory"][1]
    for name in ("before_qa", "after_qa"):
        assert [frame["camera"] for frame in manifest[name]] == manifest["source_camera_trajectory"]
    assert manifest["visual_ablation"]["target_colours_only"] is True
    assert manifest["visual_ablation"]["non_target_meshes_grayscale"] is True
    assert manifest["visual_ablation"]["role_colours"] == {"moving": "RED", "reference": "BLUE"}
    assert (out / "final/scene_after.blend").is_file()
    assert p.digest(blend) == original_hash
    result = {"status": "pass", "blender": version,
              "before_distance_m": manifest["before_state"]["surface_distance_m"],
              "after_distance_m": manifest["after_state"]["surface_distance_m"],
              "source_blend_unchanged": True, "automatic_object_resolution": True,
              "source_camera_unchanged": True,
              "unresolved_name_gate": "no_change_only",
              "target_colours_only": True,
              "visual_inspection": False, "api_calls": 0}
    p.save_json(out / "smoke_result.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
