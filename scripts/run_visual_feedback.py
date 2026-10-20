"""Send matched image pairs to a model and apply its shared Blender correction."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import refinement as r
from visual_feedback import validate_action


def parse_action(result, moving_label=None, multiview=False):
    if result.get("finish_reason") != "stop":
        raise ValueError("Incomplete model response")
    return normalize_action(r.parse_json_response(result), moving_label, multiview)


def normalize_action(payload, moving_label=None, multiview=False):
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object")
    nested = payload.get("action")
    action = nested if isinstance(nested, dict) else payload
    if (action.get("action") == "adjust" and moving_label and
            action.get("subject") == moving_label):
        action = {**action, "subject": "target"}
    if multiview and action.get("action") == "stop":
        action = {"action": "no_change", "reason": action.get("reason", "")}
    if action.get("action") == "no_change" and not multiview:
        raise ValueError("no_change is only supported in the multi-view protocol")
    return validate_action(action)


def comparison_prompt(label, step, reference_label=None):
    reference = (f" The spatial reference object is {reference_label}; keep its geometry fixed and, "
                 "when needed, correct the target's position/scale relative to it. "
                 "Compare the target-reference relationship as well as the overall view."
                 if reference_label else "")
    return (
        f"Compare SOURCE VIDEO with CURRENT BLENDER RENDER. Target: {label}. Round: {step}. "
        f"{reference} "
        "Directly decide the smallest useful correction to make the render match the source. "
        "You may correct the CAMERA position/orientation or the TARGET position/size. "
        "Use the whole images to decide which needs correction; do not return boxes, landmarks, "
        "confidence scores or a registration procedure. You may change camera, target, or BOTH in one round. "
        "All translation values are Blender units, not automatically centimeters. Both subjects use "
        "the CURRENT CAMERA's local axes BEFORE this round's changes: +X right in the image, "
        "+Y up, +Z backward/toward the viewer. For a joint action both use these same pre-edit axes. "
        "For a camera, +Z moves it away from the scene; e.g. [0,0,0.2] retreats 0.2 Blender units. "
        "Check the Euclidean length of each translation vector before returning: it must be at most "
        "0.5 for camera and 0.25 for target. If the desired change is larger, scale it down to the limit. "
        "Camera rotation uses local XYZ Euler increments in degrees, each within [-8,8]; scale_ratio must be 1. "
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


def camera_poses(pairs):
    reference = pairs[0].get("reference_camera_matrix") or pairs[-1]["camera_matrix"]
    for pair in pairs:
        other = pair.get("reference_camera_matrix")
        if other and any(abs(float(other[row][col]) - float(reference[row][col])) > 1e-6
                         for row in range(4) for col in range(4)):
            raise ValueError("Pairs do not share one fixed reference camera matrix")
    reference_rotation = [[float(reference[row][col]) for col in range(3)] for row in range(3)]
    reference_position = [float(reference[row][3]) for row in range(3)]
    views = []
    for pair in pairs:
        matrix = pair["camera_matrix"]
        position = [sum(reference_rotation[row][axis] *
                        (float(matrix[row][3]) - reference_position[row]) for row in range(3))
                    for axis in range(3)]
        axes = {}
        for axis, name in enumerate(("right", "up", "back")):
            axes[name] = [sum(reference_rotation[row][basis] * float(matrix[row][axis])
                              for row in range(3)) for basis in range(3)]
        views.append({"pair_id": pair["pair_id"],
                      "bvb_frame_index": pair.get("bvb_frame_index"),
                      "source_frame_index_0based": pair.get("source_frame_index_0based"),
                      "position_ref": position, "axes_ref": axes,
                      "lens_mm": pair.get("lens_mm"),
                      "sensor_width_mm": pair.get("sensor_width_mm"),
                      "resolution_px": [pair["width"], pair["height"]]})
    return {"reference_camera_matrix": reference,
            "reference_pair_id": pairs[0].get("reference_pair_id", pairs[-1]["pair_id"]),
            "views": views}


def pose_text(view):
    short = lambda values: [round(float(value), 3) for value in values]
    axes = view["axes_ref"]
    return (f"pair {view['pair_id']}: position_ref={short(view['position_ref'])}; "
            f"right_ref={short(axes['right'])}; up_ref={short(axes['up'])}; "
            f"back_ref={short(axes['back'])}; lens_mm={view['lens_mm']}; "
            f"resolution={view['resolution_px']}")


def multiview_comparison_prompt(label, step, pairs, reference_label=None):
    poses = camera_poses(pairs)
    view_lines = "\n".join(pose_text(view) for view in poses["views"])
    reference = (f" The spatial reference object is {reference_label}; keep its geometry fixed and "
                 "compare the target-reference relationship across views."
                 if reference_label else "")
    return (
        f"Use all {len(pairs)} matched views to recommend ONE small shared 3D correction. "
        f"Target: {label}. Round: {step}.{reference}\n"
        "Each pair contains the same-time SOURCE VIDEO frame and CURRENT BLENDER RENDER. "
        "The camera-pose metadata below is measured from the BVB camera path, in one fixed reference-camera "
        "basis. A single 3D object or camera correction can appear in opposite screen directions from "
        "different camera poses. Do not reject a shared correction just because left/right image errors "
        "reverse between views; reason about the predicted effect using the listed axes and positions. "
        "Use every pair, and rely on a view for a target only when that same target is identifiable in both images. "
        "You may adjust the full camera trajectory, the target object, or both in one round. "
        "Return no_change only if the scene is already close or every bounded shared correction would make the "
        "combined views worse. "
        "Camera translation and target translation are expressed in the fixed reference-camera axes: "
        "+X right, +Y up, +Z backward. Values use Blender scene units. A camera correction shifts the full "
        "trajectory; a target correction moves the target once in world space. Camera rotation is an XYZ "
        "increment about the fixed reference axes. Target rotation is fixed at [0,0,0]. "
        "Translation length limits: camera <= 0.5 and target <= 0.25; scale the requested vector down if needed. "
        "Camera rotation is within [-8,8] degrees per axis; target uniform scale_ratio is within [0.8,1.25]. "
        "For each pair, briefly identify the object visible in both images and its source-versus-render mismatch; "
        "include a low/medium/high confidence label, but use it only to describe the diagnosis. "
        "Do not use ground-truth measurements. Do not return boxes, landmarks, or a registration procedure.\n"
        "Camera poses:\n" + view_lines + "\n"
        "Return JSON only, with one diagnosis per pair and exactly one action: adjust camera, target, both, or no_change. "
        "no_change means no edit this round; it does not end the requested run. Never return stop. "
        'Use this shape: {"per_view":[{"pair_id":2,"visible_object":"door",'
        '"mismatch":"brief","confidence":"low|medium|high"}],'
        '"action":{"action":"adjust","subject":"camera","translation_local":[0,0,0],'
        '"rotation_local_degrees":[0,0,0],"scale_ratio":1,"reason":"brief"}. '
        'Replace action with target: {"action":"adjust","subject":"target","translation_local":[0,0,0],'
        '"rotation_local_degrees":[0,0,0],"scale_ratio":1,"reason":"brief"}; '
        'or both: {"action":"adjust","subject":"both","camera":{"translation_local":[0,0,0],'
        '"rotation_local_degrees":[0,0,0]},"target":{"translation_local":[0,0,0],'
        '"scale_ratio":1},"reason":"brief"}; '
        'or no_change: {"action":"no_change","reason":"brief"}.'
    )


def validate_pair(pair, multiview=False):
    if multiview:
        required = ("scene_frame", "camera_matrix", "reference_camera_matrix", "width", "height",
                    "fx_px", "fy_px")
        missing = [key for key in required if key not in pair]
        if missing:
            raise ValueError("Multi-view pair is missing: " + ", ".join(missing))
    for frame in (pair["source"], pair["render"]):
        if r.digest(frame["file"]) != frame["sha256"]:
            raise ValueError("Input image changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--pair-id", type=int, help="Run the original single-pair protocol")
    selection.add_argument("--pair-ids", help="Comma-separated matched views for shared 3D feedback")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--blender", required=True)
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL"), required=False)
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()
    multiview = args.pair_ids is not None
    if not args.model or not 1 <= args.rounds <= 5:
        parser.error("Provide a model and 1-5 edit rounds")
    if args.run.exists():
        raise FileExistsError("Use a new run directory")
    case = r.read_json(args.case)
    available = r.read_json(args.pairs)
    if multiview:
        ids = [int(value.strip()) for value in args.pair_ids.split(",") if value.strip()]
        if len(ids) < 2 or len(set(ids)) != len(ids):
            parser.error("--pair-ids requires at least two unique pair IDs")
        by_id = {pair["pair_id"]: pair for pair in available}
        if any(pair_id not in by_id for pair_id in ids):
            raise ValueError("One or more selected pair IDs are missing")
        pairs = [by_id[pair_id] for pair_id in ids]
    else:
        pairs = [next(pair for pair in available if pair["pair_id"] == args.pair_id)]
    for pair in pairs:
        validate_pair(pair, multiview)
    blend = Path(case["inputs"]["blend"]).resolve()
    original_digest = r.digest(blend)
    args.run.mkdir(parents=True)
    frozen = {"case": case, "pairs": pairs, "blend_sha256": original_digest,
              "model": args.model, "maximum_edit_rounds": args.rounds,
              "protocol": "pose_aware_multiview" if multiview else "single_pair"}
    r.freeze(args.run / "inputs.json", frozen)
    history = []
    renderer = ("render_visual_feedback_multiview.py" if multiview else "render_visual_feedback.py")
    for step in range(args.rounds):
        reference_label = case.get("objects", {}).get("reference", {}).get("label")
        if multiview:
            pose_info = camera_poses(pairs)
            frames = []
            for pair in pairs:
                pose = next(view for view in pose_info["views"] if view["pair_id"] == pair["pair_id"])
                marker = f"TRAIN VIEW {pair['pair_id']} / {pose_text(pose)}"
                frames.extend([{**pair["source"], "source": marker + " / SOURCE VIDEO"},
                               {**pair["render"], "source": marker + " / CURRENT BLENDER RENDER"}])
            prompt = multiview_comparison_prompt(case["objects"]["moving"]["label"], step, pairs, reference_label)
        else:
            pair = pairs[0]
            frames = [{**pair["source"], "source": "SOURCE VIDEO"},
                      {**pair["render"], "source": "CURRENT BLENDER RENDER"}]
            prompt = comparison_prompt(case["objects"]["moving"]["label"], step, reference_label)
        response = r.call_model(args.run, f"compare_{step:02d}", args.model, args.base_url,
                                prompt, frames, 4096, "high", json_mode=True)
        if response.get("finish_reason") != "stop":
            raise ValueError("Incomplete model response")
        payload = r.parse_json_response(response)
        action = normalize_action(payload, case.get("objects", {}).get("moving", {}).get("label"), multiview)
        decision = {**action, "applied": False}
        if multiview and isinstance(payload.get("per_view"), list):
            decision["per_view"] = payload["per_view"]
        history.append(decision)
        print(f"Round {step}: {action}", flush=True)
        if action["action"] == "stop":
            break
        if action["action"] == "no_change":
            continue
        directory = args.run / f"edit_{step+1:02d}"
        directory.mkdir()
        spec = directory / "spec.json"
        payload = {"case": case, "action": action}
        payload.update({"pairs": pairs} if multiview else {"pair": pairs[0]})
        r.freeze(spec, payload)
        command = [r.blender_command(args.blender), "--background", "--disable-autoexec", str(blend), "--threads", "1",
                   "--python-exit-code", "1", "--python", str(Path(__file__).with_name(renderer)),
                   "--", "--spec", str(spec.resolve())]
        with (directory / "blender.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                           env=r.blender_environment(), timeout=1800)
        state = r.read_json(directory / "state.json")
        if multiview:
            pairs = state["pairs"]
        else:
            pairs = [state["pair"]]
        blend = Path(state["blend"])
        history[-1]["applied"] = True
    if r.digest(case["inputs"]["blend"]) != original_digest:
        raise RuntimeError("Source blend changed")
    final_action = history[-1]["action"]
    stopped = final_action == "stop"
    r.freeze(args.run / "result.json", {
        "protocol": "pose_aware_multiview_feedback_v1" if multiview else "two_image_model_blender_feedback_v2_joint",
        "history": history, "pairs": pairs if multiview else None,
        "pair": None if multiview else pairs[0], "result_blend": str(blend),
        "source_blend_modified": False, "model_stopped": stopped,
        "termination_reason": "model_stopped" if stopped else "edit_limit_reached",
        "applied_edit_rounds": sum(item["applied"] for item in history),
        "no_change_rounds": sum(item["action"] == "no_change" for item in history),
        "effectiveness_independently_verified": False})


if __name__ == "__main__":
    main()
