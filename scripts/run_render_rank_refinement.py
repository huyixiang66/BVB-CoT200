"""Search bounded Blender position/scale candidates by multi-view model ranking."""

import argparse
import json
import os
from pathlib import Path
import random
import shlex
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import refinement as r


def candidate_grid():
    return [
        {"id": "identity", "parameters": [0.0, 0.0, 1.0]},
        {"id": "x_minus_005", "parameters": [-0.05, 0.0, 1.0]},
        {"id": "x_plus_005", "parameters": [0.05, 0.0, 1.0]},
        {"id": "y_minus_005", "parameters": [0.0, -0.05, 1.0]},
        {"id": "y_plus_005", "parameters": [0.0, 0.05, 1.0]},
        {"id": "shrink_5pct", "parameters": [0.0, 0.0, 0.95]},
        {"id": "enlarge_5pct", "parameters": [0.0, 0.0, 1.05]},
    ]


def blinded_presentation(candidates, pair_id):
    order = list(candidates)
    random.Random(20261019 + int(pair_id)).shuffle(order)
    labels = {item["id"]: f"option_{index + 1}" for index, item in enumerate(order)}
    return order, labels


def choose_training_winner(rankings, candidate_ids):
    if len(rankings) < 2:
        return {"candidate_id": None, "reason": "need_two_training_views"}
    scores = {candidate_id: 0 for candidate_id in candidate_ids}
    for ranking in rankings:
        order = ranking["ranked_candidate_ids"]
        if set(order) != set(candidate_ids) or len(order) != len(candidate_ids):
            return {"candidate_id": None, "reason": "invalid_ranking"}
        for rank, candidate_id in enumerate(order):
            scores[candidate_id] += len(candidate_ids) - rank
    winner = min(scores, key=lambda candidate_id: (-scores[candidate_id], candidate_id))
    rank_positions = [item["ranked_candidate_ids"].index(winner) for item in rankings]
    if winner == "identity":
        return {"candidate_id": None, "reason": "identity_ranked_best", "scores": scores}
    if max(rank_positions) > 2:
        return {"candidate_id": None, "reason": "training_views_disagree", "scores": scores}
    if any(item["confidence"] < 0.6 or not item["same_target"] for item in rankings):
        return {"candidate_id": None, "reason": "low_confidence_or_target_mismatch", "scores": scores}
    return {"candidate_id": winner, "reason": None, "scores": scores,
            "rank_positions": rank_positions}


def presentation_order_bias(rankings):
    return bool(rankings) and all(item.get("ranked_candidate_ids") ==
                                  item.get("presented_candidate_order")
                                  for item in rankings)


def parse_ranking(result, candidate_ids):
    if result.get("finish_reason") != "stop":
        raise ValueError("Candidate ranking response is incomplete")
    value = r.parse_json_response(result)
    order = value.get("ranked_candidate_ids")
    confidence = value.get("confidence")
    if not isinstance(value.get("same_target"), bool):
        raise ValueError("Candidate ranking lacks same_target")
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError("Candidate ranking confidence is invalid")
    if not isinstance(order, list) or len(order) != len(candidate_ids) or set(order) != set(candidate_ids):
        raise ValueError("Candidate ranking must order every candidate exactly once")
    return {"same_target": value["same_target"], "confidence": float(confidence),
            "ranked_candidate_ids": order, "reason": value.get("reason", "")}


def api_rank(run, model, base_url, prompt, frames, name):
    result = r.call_model(run, name, model, base_url, prompt, frames, 1200, "high",
                          json_mode=True)
    return result


def quote(value):
    return shlex.quote(str(value))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--blender", default=os.environ.get("BLENDER_BIN"),
                        help="Local Blender executable; defaults to BLENDER_BIN")
    parser.add_argument("--remote-host",
                        help="Optional SSH host for rendering instead of local Blender")
    parser.add_argument("--remote-blender", default=os.environ.get("REMOTE_BLENDER_BIN"),
                        help="Blender executable on --remote-host; defaults to REMOTE_BLENDER_BIN")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL"),
                        help="GPT vision model id; defaults to OPENAI_MODEL")
    parser.add_argument("--base-url", default=os.environ.get(
        "OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--reuse-renders", action="store_true",
                        help="Resume ranking from a completed local render run without rerendering")
    args = parser.parse_args()
    if not args.model:
        parser.error("pass --model or set OPENAI_MODEL to a GPT vision model id")
    if not args.reuse_renders:
        if args.remote_host and not args.remote_blender:
            parser.error("pass --remote-blender or set REMOTE_BLENDER_BIN with --remote-host")
        if not args.remote_host and not args.blender:
            parser.error("pass --blender or set BLENDER_BIN for local rendering")
    args.case, args.alignment, args.run = [p.resolve() for p in (args.case, args.alignment, args.run)]
    if args.run.exists() and not args.reuse_renders:
        raise FileExistsError("Use a new experiment directory")
    if args.reuse_renders and not args.run.is_dir():
        raise FileNotFoundError("Cannot resume: experiment directory does not exist")
    if not args.reuse_renders:
        args.run.mkdir(parents=True)
    case = r.read_json(args.case)
    alignment = r.read_json(args.alignment)
    views = alignment.get("paired_views", [])
    if not alignment.get("accepted_for_render") or len(views) != 3:
        raise ValueError("This pilot needs three paired views passing the camera render gate")
    if any(r.digest(view["source"]["file"]) != view["source"]["sha256"] for view in views):
        raise ValueError("A frozen source frame changed")
    if args.samples < 1:
        raise ValueError("Samples must be positive")

    run_id = f"{case['case_id']}-{time.strftime('%Y%m%d-%H%M%S')}"
    remote_root = f"/tmp/render-rank-refinement/{run_id}" if args.remote_host else None
    remote_code = f"{remote_root}/code" if remote_root else None
    remote_output = f"{remote_root}/output" if remote_root else None
    candidates = r.read_json(args.run / "candidates.json") if args.reuse_renders else candidate_grid()
    if not args.reuse_renders:
        r.freeze(args.run / "candidates.json", candidates)
        r.freeze(args.run / "case.json", case)
        r.freeze(args.run / "alignment.json", alignment)
        r.freeze(args.run / "config.json", {
            "protocol": "render_rank_direct_view_refinement_pilot_v1",
            "case_id": case["case_id"], "target": case["objects"]["moving"]["label"],
            "training_pairs": [view["pair_id"] for view in views[:2]],
            "holdout_pair": views[2]["pair_id"], "model": args.model,
            "candidate_count": len(candidates), "samples": args.samples,
            "camera_alignment": str(args.alignment),
            "camera_nuisance_note": "camera poses are fixed by the supplied accepted alignment",
            "remote_scratch": remote_root,
            "source_blend_modified": False, "qa_calls": 0, "gt_used_for_selection": False,
        })
    run_config = r.read_json(args.run / "config.json")
    remote_root = run_config.get("remote_scratch") or remote_root

    dependencies = ["refinement.py", "refine_blender.py", "fixed_protocol.py", "name_resolver.py",
                    "refine_probe.py", "refine_box.py", "refine_measurement.py"]
    code_root = Path(__file__).resolve().parents[1]
    script_files = [code_root / name for name in dependencies]
    script_files += [code_root / "scripts" / name for name in
                     ("refine_box_blender.py", "render_rank_candidates.py")]
    remote_inputs = ([(args.case, f"{remote_root}/case.json"),
                     (args.alignment, f"{remote_root}/alignment.json"),
                     (args.run / "candidates.json", f"{remote_root}/candidates.json")]
                     if remote_root else [])
    if not args.reuse_renders:
        if args.remote_host:
            ssh_setup = (f"mkdir -p {quote(remote_code + '/scripts')} {quote(remote_output)} "
                         f"&& test -r {quote(case['inputs']['blend'])} "
                         f"&& test -x {quote(args.remote_blender)}")
            subprocess.run(["ssh", args.remote_host, ssh_setup], check=True, timeout=60)
            for path in script_files:
                remote_dest = f"{remote_code}/scripts/" if path.parent.name == "scripts" else f"{remote_code}/"
                subprocess.run(["scp", str(path), f"{args.remote_host}:{remote_dest}"], check=True, timeout=120)
            for local_path, remote_path in remote_inputs:
                subprocess.run(["scp", str(local_path), f"{args.remote_host}:{remote_path}"], check=True, timeout=120)

            remote_script = f"{remote_code}/scripts/render_rank_candidates.py"
            command = " ".join(quote(item) for item in [
                args.remote_blender, "--background", "--disable-autoexec", case["inputs"]["blend"],
                "--threads", "1", "--python-exit-code", "1", "--python", remote_script,
                "--", "--repo-root", remote_code, "--case", f"{remote_root}/case.json",
                "--alignment", f"{remote_root}/alignment.json", "--candidates",
                f"{remote_root}/candidates.json", "--output", remote_output,
                "--samples", str(args.samples),
            ])
            remote_command = f"cd {quote(remote_root)} && nice -n 19 {command}"
            log_path = args.run / "remote_render.log"
            print(f"Rendering {len(candidates) * len(views)} candidates on remote CPU (one thread).", flush=True)
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.Popen(["ssh", args.remote_host, remote_command],
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, bufsize=1)
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line.rstrip(), flush=True)
                if process.wait() != 0:
                    raise RuntimeError(f"Remote rendering failed; see {log_path}")
            subprocess.run(["scp", f"{args.remote_host}:{remote_output}/render_report.json",
                            str(args.run / "render_report.json")], check=True, timeout=120)
            subprocess.run(["scp", "-r", f"{args.remote_host}:{remote_output}/renders",
                            str(args.run / "renders")], check=True, timeout=300)
        else:
            local_output = args.run / "render_output"
            render_script = Path(__file__).with_name("render_rank_candidates.py")
            command = [args.blender, "--background", "--disable-autoexec",
                       str(Path(case["inputs"]["blend"]).resolve()), "--threads", "1",
                       "--python-exit-code", "1", "--python", str(render_script), "--",
                       "--repo-root", str(code_root), "--case", str(args.case),
                       "--alignment", str(args.alignment), "--candidates",
                       str((args.run / "candidates.json").resolve()), "--output",
                       str(local_output.resolve()), "--samples", str(args.samples)]
            subprocess.run(command, check=True)
            shutil.copy2(local_output / "render_report.json", args.run / "render_report.json")
            shutil.copytree(local_output / "renders", args.run / "renders")
    render_report = r.read_json(args.run / "render_report.json")
    if render_report.get("records"):
        remote_root = str(Path(render_report["records"][0]["file"]).parents[2])
    if (render_report.get("protocol") != "fixed_camera_render_rank_candidates_v1" or
            render_report.get("source_blend_modified") or
            len(render_report.get("records", [])) != len(candidates) * len(views)):
        raise ValueError("Rendered candidate report is incomplete or invalid")

    records = {(record["candidate_id"], record["pair_id"]): record
               for record in render_report["records"]}
    candidate_ids = [candidate["id"] for candidate in candidates]
    rankings = []
    for view in views[:2]:
        pair_id = view["pair_id"]
        frames = [{**view["source"], "source": "SOURCE REFERENCE",
                   "frame_index": view["source_frame_index"]}]
        presentation, option_by_candidate = blinded_presentation(candidates, pair_id)
        for candidate in presentation:
            record = records[(candidate["id"], pair_id)]
            local_file = args.run / "renders" / Path(record["file"]).name
            if r.digest(local_file) != record["sha256"]:
                raise ValueError("Copied candidate render hash mismatch")
            frames.append({"file": str(local_file), "sha256": record["sha256"],
                           "source": "OPTION " + option_by_candidate[candidate["id"]],
                           "candidate_id": option_by_candidate[candidate["id"]],
                           "frame_index": view["source_frame_index"]})
        prompt = (
            f"Compare the same physical {case['objects']['moving']['label']} in the SOURCE REFERENCE "
            "and the options. All options use the exact same camera and scene; only the target's "
            "3D position or scale changes. Option labels are randomized and carry no meaning. "
            "Ignore rendering style and global framing. Judge target position and apparent size relative "
            "to nearby fixed room geometry. Do not draw boxes. Rank every option from best to worst; "
            "do not copy the presentation order. If identity is unclear, set same_target=false. "
            "Return only JSON with every option exactly once: {\"same_target\":true,"
            "\"confidence\":0.8,\"ranked_candidate_ids\":[\"option_3\",\"option_1\","
            "\"option_6\",\"option_2\",\"option_7\",\"option_4\","
            "\"option_5\"],\"reason\":\"brief\"}.")
        print(f"Ranking candidates on training pair {pair_id}.", flush=True)
        retry_suffix = "_blind_retry1" if args.reuse_renders else "_blind"
        raw = api_rank(args.run, args.model, args.base_url, prompt, frames,
                       f"train_pair_{pair_id}{retry_suffix}")
        option_ids = list(option_by_candidate.values())
        blinded = parse_ranking(raw, option_ids)
        option_to_candidate = {option: candidate_id
                               for candidate_id, option in option_by_candidate.items()}
        ranking = {**blinded, "ranked_candidate_ids": [
            option_to_candidate[option] for option in blinded["ranked_candidate_ids"]]}
        ranking["blind_presentation"] = option_by_candidate
        ranking["presented_candidate_order"] = sorted(
            option_by_candidate, key=lambda candidate_id: int(
                option_by_candidate[candidate_id].split("_")[1]))
        ranking["rank_matches_presentation_order"] = (
            ranking["ranked_candidate_ids"] == ranking["presented_candidate_order"])
        ranking["pair_id"] = pair_id
        rankings.append(ranking)
        ranking_suffix = "_blind" if args.reuse_renders else ""
        r.freeze(args.run / f"ranking_pair_{pair_id}{ranking_suffix}.json", ranking)

    order_bias_detected = presentation_order_bias(rankings)
    selection = ({"candidate_id": None, "reason": "presentation_order_bias"}
                 if order_bias_detected else choose_training_winner(rankings, candidate_ids))
    holdout = views[2]
    holdout_result = None
    if selection["candidate_id"]:
        winner = selection["candidate_id"]
        frames = [{**holdout["source"], "source": "SOURCE REFERENCE",
                   "frame_index": holdout["source_frame_index"]}]
        presentation = [("identity", "option_a"), (winner, "option_b")]
        random.Random(20261019 + int(holdout["pair_id"])).shuffle(presentation)
        option_by_candidate = dict(presentation)
        for candidate_id, option in presentation:
            record = records[(candidate_id, holdout["pair_id"])]
            local_file = args.run / "renders" / Path(record["file"]).name
            frames.append({"file": str(local_file), "sha256": record["sha256"],
                           "source": "OPTION " + option, "candidate_id": option,
                           "frame_index": holdout["source_frame_index"]})
        prompt = (
            f"Compare the options against the SOURCE REFERENCE for the same physical "
            f"{case['objects']['moving']['label']}. Camera and background geometry are fixed; only "
            "the target changes. Which option's target position/size matches the source better relative "
            "to nearby fixed room geometry? Labels do not identify the transform. Do not draw boxes. "
            "Return only JSON: {\"same_target\":true,\"confidence\":0.8,"
            "\"closer_candidate\":\"option_a\",\"reason\":\"brief\"}. Use UNCERTAIN if ambiguous.")
        print(f"Checking selected candidate on held-out pair {holdout['pair_id']}.", flush=True)
        retry_suffix = "_retry1" if args.reuse_renders else ""
        raw = api_rank(args.run, args.model, args.base_url, prompt, frames,
                       f"holdout_pair{retry_suffix}")
        value = r.parse_json_response(raw)
        if (raw.get("finish_reason") != "stop" or not isinstance(value, dict) or
                value.get("closer_candidate") not in {"option_a", "option_b", "UNCERTAIN"} or
                not isinstance(value.get("confidence"), (int, float))):
            raise ValueError("Invalid held-out candidate comparison")
        selected_option = value.get("closer_candidate")
        holdout_result = {"same_target": value.get("same_target"),
                          "confidence": value.get("confidence"),
                          "closer_candidate": ("IDENTITY" if
                              option_by_candidate["identity"] == selected_option else
                              "PROPOSED UPDATE" if option_by_candidate[winner] == selected_option else
                              "UNCERTAIN"),
                          "reason": value.get("reason"),
                          "blind_presentation": option_by_candidate}
        holdout_suffix = "_blind" if args.reuse_renders else ""
        r.freeze(args.run / f"holdout{holdout_suffix}.json", holdout_result)

    accepted = bool(selection["candidate_id"] and holdout_result and
                    holdout_result["same_target"] is True and
                    holdout_result["confidence"] >= 0.7 and
                    holdout_result["closer_candidate"] == "PROPOSED UPDATE")
    winner_spec = next((item for item in candidates if item["id"] == selection["candidate_id"]), None)
    report = {
        "protocol": "render_rank_direct_view_refinement_pilot_v1",
        "case_id": case["case_id"], "target": case["objects"]["moving"]["label"],
        "candidate_count": len(candidates), "training_rankings": rankings,
        "training_selection": selection, "holdout": holdout_result,
        "presentation_order_bias_detected": order_bias_detected,
        "selected_candidate": winner_spec,
        "pilot_holdout_passed": accepted,
        "production_ready": False,
        "limitations": [
            "Only two candidate-selection views and one held-out view were available.",
            "Camera poses were held fixed at a local-right +0.45 Blender-unit diagnostic offset previously ranked at confidence 0.55.",
            "A passing single holdout is a pilot signal, not proof of generalization.",
        ],
        "source_blend_modified": False, "qa_calls": 0, "gt_used_for_selection": False,
        "remote_scratch": remote_root,
    }
    report_path = args.run / ("validation_report_blind.json" if args.reuse_renders
                              else "validation_report.json")
    r.freeze(report_path, report)
    print(json.dumps({"pilot_holdout_passed": accepted, "selected_candidate": winner_spec,
                      "training_selection_reason": selection.get("reason"),
                      "holdout": holdout_result,
                      "report": str(report_path)},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
