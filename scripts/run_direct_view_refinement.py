"""Propose and check one bounded position/scale update from matched image pairs."""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import refinement as r
import refine_measured as measured
from vggt_camera_consistency import check_vggt_camera_consistency
from scripts.run_box_refinement import geometry


def prompt(label):
    return (
        f"Compare these two views of the same scene and the named target: {label}. "
        "The next image is labeled SOURCE REFERENCE; the image after it is CURRENT RECONSTRUCTION. "
        "They are intended to be the same camera view. Do not draw, describe, or return boxes. "
        "If the same physical target is visible in both, directly estimate the smallest correction needed "
        "to make CURRENT RECONSTRUCTION match SOURCE REFERENCE. Return screen_delta as fractions of the "
        "full image width and height: positive x is right, positive y is down, and [-0.03,0.0] means "
        "move the rendered target left by about 3% of the image width. Each component must be within "
        "[-0.05,0.05]. Return scale_ratio in [0.9,1.1], where 1.0 means no size change. The program "
        "converts the screen correction to a bounded Blender-space translation using the paired camera "
        "and target depth. Use null screen_delta or scale_ratio if uncertain. Do not guess when identity, "
        "visibility, camera match, crop, or scale is unclear; do not infer hidden extents. Return only JSON: "
        "{\"same_target\":true,\"visible_both\":true,\"confidence\":0.9,"
        "\"screen_delta\":[-0.03,0.0],\"scale_ratio\":0.95,\"reason\":\"brief\"}."
    )


def observe(run, name, label, model, url, source, render):
    source = {**source, "source": "SOURCE REFERENCE"}
    render = {**render, "source": "CURRENT RECONSTRUCTION"}
    return r.call_model(run, name, model, url, prompt(label), [source, render], 8192, "high",
                        json_mode=True)


def parse_observation(result):
    if result.get("finish_reason") != "stop":
        raise ValueError("Direct comparison response is incomplete")
    value = r.parse_json_response(result)
    if not isinstance(value, dict):
        raise ValueError("Direct comparison must be a JSON object")
    if not isinstance(value.get("same_target"), bool) or not isinstance(value.get("visible_both"), bool):
        raise ValueError("Direct comparison lacks target identity/visibility")
    confidence = value.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError("Invalid direct-comparison confidence")
    screen_delta = value.get("screen_delta")
    scale_ratio = value.get("scale_ratio")
    valid_delta = (isinstance(screen_delta, list) and len(screen_delta) == 2 and
                   all(isinstance(item, (int, float)) and not isinstance(item, bool)
                       and math.isfinite(item) for item in screen_delta))
    valid_scale = (isinstance(scale_ratio, (int, float)) and not isinstance(scale_ratio, bool)
                   and math.isfinite(scale_ratio))
    if screen_delta is not None and not valid_delta:
        raise ValueError("screen_delta must be a two-number list or null")
    if scale_ratio is not None and not valid_scale:
        raise ValueError("scale_ratio must be a number or null")
    if valid_delta and any(abs(item) > 0.05 for item in screen_delta):
        raise ValueError("screen_delta exceeds the 5% per-axis bound")
    if valid_scale and not 0.9 <= scale_ratio <= 1.1:
        raise ValueError("scale_ratio must be within [0.9, 1.1]")
    value["screen_delta"] = screen_delta
    value["apparent_scale_ratio"] = scale_ratio
    if not valid_delta or not valid_scale:
        value["confidence"] = min(value["confidence"], 0.5)
    if value["same_target"] and value["visible_both"] and valid_delta and valid_scale:
        value["direct_actions"] = {"screen_delta": screen_delta, "scale_ratio": scale_ratio}
    return value


def residual(observation):
    delta = observation.get("screen_delta")
    if (not observation.get("same_target") or not observation.get("visible_both") or
            delta is None or observation.get("apparent_scale_ratio") is None):
        return None
    position = math.hypot(*delta)
    ratio = observation.get("apparent_scale_ratio")
    scale = abs(math.log(ratio)) if ratio else 0.0
    return position + 0.5 * scale


def transform_delta(pair, observation, gain=0.5):
    matrix = pair["camera_matrix"]
    center = pair["target_center_world"]
    camera = [matrix[i][3] for i in range(3)]
    right = [matrix[i][0] for i in range(3)]
    up = [matrix[i][1] for i in range(3)]
    back = [matrix[i][2] for i in range(3)]
    depth = -sum((center[i] - camera[i]) * back[i] for i in range(3))
    if depth <= 0:
        raise ValueError("Target is behind the matched camera")
    width, height = pair["width"], pair["height"]
    kx = pair["fx_px"] / (width * depth)
    ky = pair["fy_px"] / (height * depth)
    a, b = kx * right[0], kx * right[1]
    c, d = -ky * up[0], -ky * up[1]
    determinant = a * d - b * c
    if abs(determinant) < 1e-7:
        raise ValueError("Camera cannot map screen correction into world XY")
    sx, sy = observation["screen_delta"]
    dx = gain * (sx * d - b * sy) / determinant
    dy = gain * (a * sy - sx * c) / determinant
    distance = math.hypot(dx, dy)
    if distance > 0.15:
        dx, dy = dx * 0.15 / distance, dy * 0.15 / distance
    ratio = observation.get("apparent_scale_ratio")
    scale = 1 + gain * (ratio - 1) if ratio else 1.0
    scale = min(1.05, max(0.95, scale))
    return [round(dx, 6), round(dy, 6), round(scale, 6)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--train-pair", type=int, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--blender", required=True)
    parser.add_argument("--sourcecam-render-config", type=Path)
    parser.add_argument("--vggt-render-config", type=Path,
                        help="Alignment and renderer used to re-render proposals at validated VGGT cameras")
    parser.add_argument("--vggt-summary", type=Path,
                        help="Optional numeric VGGT trajectory summary used as a camera-pair gate")
    parser.add_argument("--base-url", default=os.environ.get(
        "OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL"),
                        help="GPT vision model id; defaults to OPENAI_MODEL")
    args = parser.parse_args()
    if not args.model:
        parser.error("pass --model or set OPENAI_MODEL to a GPT vision model id")
    args.case, args.pairs, args.run = [p.resolve() for p in (args.case, args.pairs, args.run)]
    if args.vggt_summary:
        args.vggt_summary = args.vggt_summary.resolve()
    if args.vggt_render_config:
        args.vggt_render_config = args.vggt_render_config.resolve()
    if args.vggt_render_config and args.sourcecam_render_config:
        raise ValueError("Choose one external camera renderer")
    if args.vggt_summary and args.vggt_render_config:
        raise ValueError("Use the validated alignment report instead of rechecking the same VGGT poses")
    case = r.read_json(args.case)
    rows = r.read_json(args.pairs)
    if not rows or len({row["pair_id"] for row in rows}) != len(rows):
        raise ValueError("Need unique paired views")
    by_id = {row["pair_id"]: row for row in rows}
    if args.train_pair not in by_id:
        raise ValueError("Training pair is missing")
    for row in rows:
        for key in ("source", "render"):
            frame = row[key]
            if r.digest(frame["file"]) != frame["sha256"]:
                raise ValueError("Frozen image changed: " + frame["file"])
    vggt_config = r.read_json(args.vggt_render_config) if args.vggt_render_config else None
    vggt_alignment = None
    vggt_render_report_path = None
    if vggt_config:
        vggt_alignment = r.read_json(Path(vggt_config["alignment"]).resolve())
        vggt_render_report_path = Path(vggt_config["render_report"]).resolve()
        vggt_render_report = r.read_json(vggt_render_report_path)
        if not vggt_alignment.get("accepted_for_render"):
            raise ValueError("VGGT camera alignment did not pass its render gate")
        if (vggt_render_report.get("protocol") != "vggt_camera_blender_render_pairs_v1" or
                vggt_render_report.get("source_blend_modified") or
                len(vggt_render_report.get("records", [])) != len(rows)):
            raise ValueError("Baseline render report is missing or invalid")
        expected = {pair["pair_id"]: pair for pair in vggt_alignment.get("paired_views", [])}
        baseline = {record["pair_id"]: record for record in vggt_render_report["records"]}
        if set(expected) != set(by_id):
            raise ValueError("Pair ids do not match the validated VGGT alignment")
        for pair_id, row in by_id.items():
            anchor = expected[pair_id]
            rendered = baseline.get(pair_id)
            if rendered is None:
                raise ValueError(f"Render report is missing pair {pair_id}")
            for key in ("source_frame_index", "scene_frame", "camera_matrix", "target_center_world",
                        "width", "height", "fx_px", "fy_px"):
                if row.get(key) != anchor.get(key):
                    raise ValueError(f"Pair {pair_id} differs from the validated VGGT alignment: {key}")
            if row["source"] != anchor["source"]:
                raise ValueError(f"Pair {pair_id} source differs from the validated VGGT alignment")
            if row["render"] != {"file": rendered["file"], "sha256": rendered["sha256"]}:
                raise ValueError(f"Pair {pair_id} render differs from the render report")
            if (rendered["camera_matrix"] != anchor["camera_matrix"] or
                    rendered["camera_position_error_blender_units"] > 1e-4 or
                    rendered["camera_rotation_error_deg"] > 0.01):
                raise ValueError(f"Pair {pair_id} render camera does not match the validated pose")
    r.freeze(args.run / "config.json", {
        "protocol": "direct_view_position_scale_v1", "case_id": case["case_id"],
        "model": args.model, "pair_ids": [row["pair_id"] for row in rows],
        "train_pair": args.train_pair,
        "holdout_pairs": [row["pair_id"] for row in rows if row["pair_id"] != args.train_pair],
        "qa_calls": 0, "gt_used_for_selection": False, "source_blend_modified": False,
        "code_sha256": {"refinement.py": r.digest(r.ROOT / "refinement.py"),
                        "scripts/run_direct_view_refinement.py": r.digest(Path(__file__)),
                        "vggt_camera_consistency.py": r.digest(
                            r.ROOT / "vggt_camera_consistency.py")},
        "vggt_summary_sha256": (r.digest(args.vggt_summary)
                                if args.vggt_summary else None),
        "vggt_render_config_sha256": (r.digest(args.vggt_render_config)
                                      if args.vggt_render_config else None),
        "vggt_alignment_sha256": (r.digest(Path(vggt_config["alignment"]).resolve())
                                  if vggt_config else None),
        "vggt_baseline_render_report_sha256": (r.digest(vggt_render_report_path)
                                               if vggt_render_report_path else None),
    })
    if any(row.get("camera_match", "validated") != "validated" for row in rows):
        r.freeze(args.run / "outcome.json", {
            "accepted": False, "stop": "camera_pair_not_validated",
            "source_blend_modified": False, "qa_calls": 0,
        })
        return
    if args.vggt_summary:
        consistency = check_vggt_camera_consistency(r.read_json(args.vggt_summary), rows)
        r.freeze(args.run / "vggt_camera_consistency.json", consistency)
        if not consistency["consistent"]:
            r.freeze(args.run / "outcome.json", {
                "accepted": False, "stop": "vggt_camera_consistency_failed",
                "vggt_stop_reasons": consistency["stop_reasons"],
                "source_blend_modified": False, "qa_calls": 0,
            })
            return
    before = {}
    for row in rows:
        pair_id = row["pair_id"]
        before[pair_id] = parse_observation(observe(
            args.run, f"before_{pair_id}", case["objects"]["moving"]["label"], args.model,
            args.base_url, row["source"], row["render"]))
        r.freeze(args.run / f"before_{pair_id}.json", before[pair_id])
    train = by_id[args.train_pair]
    train_observation = before[args.train_pair]
    if (train_observation["confidence"] < 0.6 or
            not train_observation["same_target"] or not train_observation["visible_both"]):
        r.freeze(args.run / "outcome.json", {"accepted": False, "stop": "unmatched_or_low_confidence"})
        return
    parameters = transform_delta(train, train_observation)
    r.freeze(args.run / "proposal.json", {"parameters": parameters,
                                           "source_observation": train_observation,
                                           "screen_residual_before": residual(train_observation)})
    pair_specs = [{"pair_id": row["pair_id"], "scene_frame": row["scene_frame"]} for row in rows]
    rendered = geometry(case, args.run / "rendered_proposal", args.blender, pair_specs,
                        [{"id": "direct_view_update", "parameters": parameters}], save=True)
    if args.vggt_render_config:
        render_dir = args.run / "vggt_proposal_renders"
        command = [args.blender, "--background", "--disable-autoexec",
                   str(Path(rendered["proposal_blend"]).resolve()), "--threads", "2",
                   "--python-exit-code", "1", "--python",
                   str(Path(vggt_config["renderer"]).resolve()), "--",
                   "--alignment", str(Path(vggt_config["alignment"]).resolve()),
                   "--output", str(render_dir.resolve()),
                   "--width", str(vggt_config.get("width", 640)),
                   "--height", str(vggt_config.get("height", 480)),
                   "--cycles-device", vggt_config.get("cycles_device", "CPU"),
                   "--samples", str(vggt_config.get("samples", 8))]
        subprocess.run(command, check=True)
        report = r.read_json(render_dir / "render_report.json")
        expected = {pair["pair_id"]: pair for pair in vggt_alignment["paired_views"]}
        if (report.get("protocol") != "vggt_camera_blender_render_pairs_v1" or
                report.get("source_blend_modified") or
                {frame["pair_id"] for frame in report.get("records", [])} != set(expected)):
            raise ValueError("VGGT proposal render report is missing or invalid")
        for frame in report["records"]:
            anchor = expected[frame["pair_id"]]
            if (frame["camera_matrix"] != anchor["camera_matrix"] or
                    frame["camera_position_error_blender_units"] > 1e-4 or
                    frame["camera_rotation_error_deg"] > 0.01):
                raise ValueError(f"Proposal render camera mismatch for pair {frame['pair_id']}")
        after_frames = {frame["pair_id"]: {"file": frame["file"], "sha256": frame["sha256"]}
                        for frame in report["records"]}
    elif args.sourcecam_render_config:
        config = r.read_json(args.sourcecam_render_config.resolve())
        render_dir = args.run / "sourcecam_proposal_renders"
        candidate_file = render_dir / "identity.json"
        r.freeze(candidate_file, [{"id": "identity", "parameters": [0, 0, 1]}])
        command = [args.blender, "--background", "--disable-autoexec",
                   str(Path(rendered["proposal_blend"]).resolve()), "--threads", "2",
                   "--python-exit-code", "1", "--python", str(Path(config["renderer"]).resolve()),
                   "--", "--case", str(args.case), "--pairs", str(Path(config["pairs"]).resolve()),
                   "--candidates", str(candidate_file), "--registration",
                   str(Path(config["registration"]).resolve()), "--pose-root",
                   str(Path(config["pose_root"]).resolve()), "--intrinsic",
                   str(Path(config["intrinsic"]).resolve()), "--output", str(render_dir.resolve()),
                   "--width", str(config.get("width", 640)), "--height", str(config.get("height", 480))]
        subprocess.run(command, check=True)
        report = r.read_json(render_dir / "render_report.json")
        after_frames = {frame["scene_frame"]: {"file": frame["file"], "sha256": frame["sha256"]}
                        for frame in report["records"]}
    else:
        after_frames = {frame["scene_frame"]: frame for frame in rendered["frames"]}
    after = {}
    for row in rows:
        pair_id = row["pair_id"]
        replacement = (after_frames[pair_id] if args.vggt_render_config
                       else after_frames[row["scene_frame"]])
        after[pair_id] = parse_observation(observe(
            args.run, f"after_{pair_id}", case["objects"]["moving"]["label"], args.model,
            args.base_url, row["source"], replacement))
        r.freeze(args.run / f"after_{pair_id}.json", after[pair_id])
    gains = {pair_id: (residual(before[pair_id]) - residual(after[pair_id])
                       if residual(before[pair_id]) is not None and residual(after[pair_id]) is not None
                       else None) for pair_id in before}
    holdouts = [row["pair_id"] for row in rows if row["pair_id"] != args.train_pair]
    accepted = bool(holdouts and all(
        after[pid]["confidence"] >= 0.7 and after[pid]["same_target"] and
        after[pid]["visible_both"] and gains[pid] >= 0.005 for pid in holdouts
    ) and gains[args.train_pair] >= 0.005)
    outcome = {"accepted": accepted, "parameters": parameters, "residual_gain": gains,
               "train_pair": args.train_pair, "holdout_pairs": holdouts,
               "source_blend_modified": False, "proposal_blend": rendered.get("proposal_blend"),
               "gt_used_for_selection": False, "qa_calls": 0,
               "stop": None if accepted else ("no_independent_holdout" if not holdouts else "holdout_gate_failed")}
    r.freeze(args.run / "outcome.json", outcome)
    print(json.dumps(outcome, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
