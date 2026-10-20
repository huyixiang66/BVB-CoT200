"""Two images -> model correction -> Blender edit -> repeat, without geometric registration."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import refinement as r
from visual_feedback import validate_action


def parse_action(result):
    if result.get("finish_reason") != "stop":
        raise ValueError("Incomplete model response")
    return validate_action(r.parse_json_response(result))


def comparison_prompt(label, step):
    return (
        f"Compare SOURCE VIDEO with CURRENT BLENDER RENDER. Target: {label}. Round: {step}. "
        "Directly decide the smallest useful correction to make the render match the source. "
        "You may correct the CAMERA position/orientation or the TARGET position/size. "
        "Use the whole images to decide which needs correction; do not return boxes, landmarks, "
        "confidence scores or a registration procedure. You may change camera, target, or BOTH in one round. "
        "All translation values are Blender units, not automatically centimeters. Both subjects use "
        "the CURRENT CAMERA's local axes BEFORE this round's changes: +X right in the image, "
        "+Y up, +Z backward/toward the viewer. For a joint action both use these same pre-edit axes. "
        "For a camera, +Z moves it away from the scene; e.g. [0,0,0.2] retreats 0.2 Blender units. "
        "Translation vector length is at most 0.5 for camera, 0.25 for target. Camera rotation uses "
        "local XYZ Euler increments in degrees, each within [-8,8]; scale_ratio must be 1. "
        "For target, rotation must be [0,0,0], and uniform scale_ratio is within [0.8,1.25]. "
        "Do not shrink/move an object merely to compensate for an incorrect camera. "
        "If already satisfactory or you cannot determine a useful correction (including incompatible "
        "target identity/shape), return action=stop and explain. Otherwise return JSON only: "
        '{"action":"adjust","subject":"camera","translation_local":[0,0,0],'
        '"rotation_local_degrees":[0,0,0],"scale_ratio":1,"reason":"brief"}. '
        'For joint changes return {"action":"adjust","subject":"both",'
        '"camera":{"translation_local":[0,0,0],"rotation_local_degrees":[0,0,0]},'
        '"target":{"translation_local":[0,0,0],"scale_ratio":1},"reason":"brief"}. '
        'Stop schema: {"action":"stop","reason":"brief"}.'
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--pair-id", type=int, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--blender", required=True)
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL"), required=False)
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()
    if not args.model or not 1 <= args.rounds <= 5:
        parser.error("Provide a model and 1-5 edit rounds")
    if args.run.exists():
        raise FileExistsError("Use a new run directory")
    case = r.read_json(args.case)
    pair = next(x for x in r.read_json(args.pairs) if x["pair_id"] == args.pair_id)
    for frame in (pair["source"], pair["render"]):
        if r.digest(frame["file"]) != frame["sha256"]:
            raise ValueError("Input image changed")
    blend = Path(case["inputs"]["blend"]).resolve()
    original_digest = r.digest(blend)
    args.run.mkdir(parents=True)
    r.freeze(args.run / "inputs.json", {"case": case, "pair": pair, "blend_sha256": original_digest,
                                      "model": args.model, "maximum_edit_rounds": args.rounds})
    history = []
    for step in range(args.rounds):
        frames = [{**pair["source"], "source": "SOURCE VIDEO"},
                  {**pair["render"], "source": "CURRENT BLENDER RENDER"}]
        response = r.call_model(args.run, f"compare_{step:02d}", args.model, args.base_url,
                                comparison_prompt(case["objects"]["moving"]["label"], step),
                                frames, 4096, "high", json_mode=True)
        action = parse_action(response)
        history.append({**action, "applied": False})
        print(f"Round {step}: {action}", flush=True)
        if action["action"] == "stop":
            break
        directory = args.run / f"edit_{step+1:02d}"
        directory.mkdir()
        spec = directory / "spec.json"
        r.freeze(spec, {"case": case, "pair": pair, "action": action})
        command = [r.blender_command(args.blender), "--background", "--disable-autoexec", str(blend), "--threads", "1",
                   "--python-exit-code", "1", "--python",
                   str(Path(__file__).with_name("render_visual_feedback.py")), "--", "--spec", str(spec.resolve())]
        with (directory / "blender.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                           env=r.blender_environment())
        state = r.read_json(directory / "state.json")
        pair, blend = state["pair"], Path(state["blend"])
        history[-1]["applied"] = True
    if r.digest(case["inputs"]["blend"]) != original_digest:
        raise RuntimeError("Source blend changed")
    r.freeze(args.run / "result.json", {"protocol": "two_image_model_blender_feedback_v2_joint",
             "history": history, "pair": pair, "result_blend": str(blend),
             "source_blend_modified": False, "model_stopped": action["action"] == "stop",
             "termination_reason": "model_stopped" if action["action"] == "stop" else "edit_limit_reached",
             "applied_edit_rounds": sum(x["applied"] for x in history),
             "effectiveness_independently_verified": False})


if __name__ == "__main__":
    main()
