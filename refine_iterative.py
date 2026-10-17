"""Opt-in blind, evidence-gated refinement. No QA or label access in this module."""

import argparse
import copy
import json
import math
import os
from pathlib import Path
import re
import subprocess

import refinement as r

PROTOCOL = "evidence_gated_rounds_v4"
POLICY = {
    "max_rounds": 2, "probe_samples": 128, "max_alignment_pairs": 8,
    "time_separation": 0.04, "holdout_time_separation": 0.08, "component_gap_m": 0.10,
    "translation_steps_m": [0.30, 0.15],
    "scale_factors": [[0.90, 1.10], [0.95, 1.05]],
    "qa_calls": 0, "label_access": False,
}
NO_CHANGE = {"id": "no_change", "operation": "no_change"}


def geometry_stage(case, run, blender, name, spec):
    directory = Path(run) / name
    spec_path = directory / "spec.json"
    spec = {**spec, "source_sha256": r.digest(case["inputs"]["blend"]),
            "case_sha256": r.data_hash(r.blind_case(case))}
    r.freeze(spec_path, spec)
    output = directory / "result.json"
    if output.exists():
        result = r.read_json(output)
        for frame in result.get("frames", []):
            if "file" in frame and r.digest(frame["file"]) != frame["sha256"]:
                raise ValueError("Cached render hash changed")
        if result.get("scene_after_blend") and r.digest(result["scene_after_blend"]) != result["blend_sha256"]:
            raise ValueError("Cached accepted scene changed")
        return result
    case_path = directory / "case.json"
    r.freeze(case_path, r.blind_case(case))
    command = [
        r.blender_command(blender), "--background", "--disable-autoexec", case["inputs"]["blend"],
        "--threads", "2", "--python-exit-code", "1", "--python", str(r.ROOT / "refine_probe.py"),
        "--", "--case", str(case_path), "--spec", str(spec_path), "--output", str(output),
    ]
    with (directory / "blender.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=600,
                       check=True, env=r.blender_environment())
    if r.digest(case["inputs"]["blend"]) != spec["source_sha256"]:
        raise ValueError("Source blend changed")
    result = r.read_json(output)
    if result.get("scene_after_blend"):
        result["blend_sha256"] = r.digest(result["scene_after_blend"])
        r.save_json(output, result)
    return result


def pick_component(label, components):
    if len(components) == 1:
        return components[0]
    words = re.sub(r"[^a-z0-9]+", " ", label.lower()).strip()
    matches = [
        group for group in components
        if any(re.search(r"\b" + re.escape(words) + r"\b",
                         re.sub(r"[^a-z0-9]+", " ", name.lower())) for name in group["members"])
    ]
    return matches[0] if len(matches) == 1 else None


def propose_pairs(probe, original_count, timeline):
    start, end = timeline
    eligible = [
        frame for frame in probe["frames"]
        if frame["moving"]["first_hit_group_samples"] >= 2
        and frame["moving"]["projected_area_fraction"] >= 0.002
    ]
    eligible.sort(key=lambda frame: (-frame["moving"]["first_hit_group_samples"], frame["scene_frame"]))
    selected = []
    for frame in eligible:
        progress = (frame["scene_frame"] - start) / max(1, end - start)
        if any(abs(progress - item["normalized_time"]) < POLICY["time_separation"] for item in selected):
            continue
        selected.append({
            "pair_id": len(selected), "scene_frame": frame["scene_frame"],
            "original_frame_index_0based": round(progress * (original_count - 1)),
            "normalized_time": progress,
            "mapping_method": "uniform_full_timeline_approximation",
        })
        if len(selected) == POLICY["max_alignment_pairs"]:
            break
    return selected


def split_evidence(result, pairs):
    by_id = {item["pair_id"]: item for item in pairs}
    observations = result.get("pairs", [])
    ids = [item.get("pair_id") for item in observations]
    if len(ids) != len(set(ids)) or set(ids) != set(by_id):
        raise ValueError("Alignment must cover every frozen pair exactly once")
    usable = [by_id[item["pair_id"]] for item in observations if all(
        item.get(key) is True for key in (
            "moving_visible_original", "moving_visible_reconstruction",
            "same_physical_target", "background_consistent",
        )
    )]
    # Ordering depends on geometry visibility, not model scores or GT.
    usable.sort(key=lambda item: item["pair_id"])
    train = usable[:2]
    holdouts = []
    for pair in usable[2:]:
        if all(abs(pair["normalized_time"] - other["normalized_time"]) >=
               POLICY["holdout_time_separation"] for other in train + holdouts):
            holdouts.append(pair)
        if len(holdouts) == POLICY["max_rounds"]:
            break
    return {"train": train, "holdouts": holdouts}


def round_candidates(index):
    distance = POLICY["translation_steps_m"][index]
    candidates = [NO_CHANGE]
    for direction in ("toward", "away"):
        candidates.append({"id": f"{direction}_{round(distance * 100):03d}", "operation": "translate",
                           "direction": direction, "distance_m": distance, "horizontal_only": True})
    for factor in POLICY["scale_factors"][index]:
        candidates.append({"id": f"scale_{round(factor * 100):03d}", "operation": "scale",
                           "factor": factor, "pivot": "bottom"})
    return candidates


def accept_validation(result, pairs):
    rows = result.get("pairs", [])
    expected = {pair["pair_id"] for pair in pairs}
    if len(rows) != len(expected) or {row.get("pair_id") for row in rows} != expected:
        return False
    return all(row.get("winner") == "proposal" and all(
        row.get(key) is True for key in (
            "moving_visible_both", "same_physical_target", "background_consistent",
            "position_or_size_better", "no_new_artifacts",
        )
    ) for row in rows)


def judge(run, name, model, base_url, prompt, frames):
    result = r.call_model(run, name, model, base_url, prompt, frames, 3500, "high")
    if result.get("finish_reason") != "stop":
        raise ValueError("Incomplete model decision")
    return r.parse_json_response(result)


def render_pair_frames(case, run, blender, name, pairs, candidate):
    result = geometry_stage(case, run, blender, name, {
        "action": "render", "scene_frames": sorted({p["scene_frame"] for p in pairs}),
        "candidate": candidate,
    })
    lookup = {frame["scene_frame"]: frame for frame in result["frames"]}
    return [{**lookup[pair["scene_frame"]], "pair_id": pair["pair_id"]} for pair in pairs]


def baseline_pair_frames(records, pairs):
    by_id = {frame["pair_id"]: frame for frame in records}
    if len(by_id) != len(records):
        raise ValueError("Duplicate baseline pair IDs")
    selected = [by_id[pair["pair_id"]] for pair in pairs]
    if any(frame["scene_frame"] != pair["scene_frame"] for frame, pair in zip(selected, pairs)):
        raise ValueError("Baseline scene frame mismatch")
    return selected


def run_case(source, run, blender, model, base_url, preflight=False):
    run = Path(run).resolve()
    case = r.load_case(source)
    r.freeze(run / "config.json", {
        "protocol": PROTOCOL, "policy": POLICY, "model": model, "base_url": base_url,
        "case_sha256": r.digest(source),
        "input_sha256": {key: r.digest(path) for key, path in case["inputs"].items()},
        "code_sha256": {name: r.digest(r.ROOT / name) for name in (
            "refine_iterative.py", "refine_probe.py", "refine_blender.py", "refinement.py",
            "fixed_protocol.py", "name_resolver.py",
        )},
    })
    r.doctor(case, blender)
    case = r.resolve_case_objects(case, run, blender)
    r.freeze(run / "blind_case.json", r.blind_case(case))
    scene = r.inspect(case, run, blender)
    timeline = scene["timeline"]
    frames = sorted({timeline[0] + index for index in r.uniform_indices(
        timeline[1] - timeline[0] + 1, POLICY["probe_samples"])})
    probe = geometry_stage(case, run, blender, "probe_raw", {"scene_frames": frames})
    raw_gap = probe["initial_state"]["surface_distance_m"]
    resolution = {}
    outcome = {"protocol": PROTOCOL, "case_id": case["case_id"], "rounds": [],
               "raw_group_gap_m": raw_gap, "qa_calls": 0, "gt_used": False}

    def finish(reason):
        outcome["stop_reason"] = reason
        outcome["accepted_edits"] = sum(row.get("accepted", False) for row in outcome["rounds"])
        outcome["final_blend"] = case["inputs"]["blend"]
        original = r.read_json(run / "config.json")["input_sha256"]
        if {key: r.digest(path) for key, path in r.load_case(source)["inputs"].items()} != original:
            raise ValueError("Original source files changed")
        r.freeze(run / ("preflight.json" if preflight else "outcome.json"), outcome)
        print(json.dumps(outcome), flush=True)
        return outcome

    if probe["initial_state"].get("status") == "unresolved":
        outcome["unresolved_roles"] = probe["initial_state"]["unresolved_roles"]
        return finish("unresolved_object_names")
    if probe["dynamic_members"]:
        return finish("unsupported_dynamic_or_parented_target")
    for role in ("moving", "reference"):
        component = pick_component(case["objects"][role]["label"], probe["components"][role])
        if component is None:
            return finish("ambiguous_instance_" + role)
        old = case["objects"][role].get("names", [])
        case["objects"][role] = {"label": case["objects"][role]["label"], "names": component["members"]}
        resolution[role] = {"selected": component["members"], "excluded": sorted(set(old) - set(component["members"])),
                            "method": "unique_literal_label_component_or_single_component",
                            "instance_verified": False}
    r.freeze(run / "component_resolution.json", resolution)
    r.freeze(run / "resolved_case.json", r.blind_case(case))
    probe = geometry_stage(case, run, blender, "probe_resolved", {"scene_frames": frames})
    outcome["before_gap_m"] = outcome["after_gap_m"] = probe["initial_state"]["surface_distance_m"]
    metadata = r.video_metadata(case["inputs"]["original_video"])
    pairs = propose_pairs(probe, metadata["frame_count"], timeline)
    r.freeze(run / "proposed_pairs.json", pairs)
    outcome["proposed_pairs"] = len(pairs)
    if len(pairs) < 3:
        return finish("insufficient_visible_temporally_separated_views")
    if preflight:
        return finish("ready_for_alignment")
    original_frames = r.extract_frames(case["inputs"]["original_video"], run / "original",
                                      [p["original_frame_index_0based"] for p in pairs], 640)
    original_by_pair = {p["pair_id"]: {**frame, "pair_id": p["pair_id"], "source": "original"}
                        for p, frame in zip(pairs, original_frames)}
    baseline = render_pair_frames(case, run, blender, "alignment_baseline", pairs, NO_CHANGE)
    prompt = (
        f"Audit same-normalized-time ORIGINAL/reconstruction pairs for target "
        f"{case['objects']['moving']['label']} relative to {case['objects']['reference']['label']}. "
        "The reconstruction uses original materials. Camera trajectories are approximate. "
        "Assess each pair independently. Background consistency means stable nearby landmarks support comparing "
        "the SAME physical target, not just a similar room. Do not infer a target from metadata. "
        "Never estimate metric distance or answer the question. Return JSON with every pair exactly once: "
        '{"pairs":[{"pair_id":0,"moving_visible_original":false,"moving_visible_reconstruction":false,'
        '"same_physical_target":false,"background_consistent":false,"reason":"brief"}]}.'
    )
    alignment_frames = []
    for pair, frame in zip(pairs, baseline):
        alignment_frames.extend([original_by_pair[pair["pair_id"]], {**frame, "source": "baseline"}])
    alignment = judge(run, "alignment", model, base_url, prompt, alignment_frames)
    r.freeze(run / "alignment.json", alignment)
    evidence = split_evidence(alignment, pairs)
    r.freeze(run / "evidence_split.json", evidence)
    if len(evidence["train"]) < 2 or not evidence["holdouts"]:
        return finish("insufficient_aligned_target_evidence")
    for index in range(min(POLICY["max_rounds"], len(evidence["holdouts"]))):
        round_dir = run / f"round_{index}"
        train = evidence["train"]
        candidates = round_candidates(index)
        frames_to_send = [original_by_pair[p["pair_id"]] for p in train]
        for candidate in candidates:
            records = (baseline_pair_frames(baseline, train) if candidate["operation"] == "no_change" else
                       render_pair_frames(case, round_dir, blender, candidate["id"], train, candidate))
            frames_to_send.extend({**frame, "source": "candidate", "candidate_id": candidate["id"]}
                                  for frame in records)
        select_prompt = (
            f"Compare the position/size of {case['objects']['moving']['label']} in ORIGINAL and candidate pairs "
            "with the SAME pair_id. Reference: " + case["objects"]["reference"]["label"] + ". "
            "Choose a change only when visible target and stable background landmarks support improvement in "
            "BOTH training views. Never infer a desired metric distance. Camera is unchanged, original materials. "
            "Return JSON: " '{"candidate_id":"no_change","reason_code":"no_improvement|'
            'target_not_visible|alignment_mismatch|candidates_indistinguishable|improvement",'
            '"reason":"brief","supporting_pair_ids":[0,1]}. Candidate IDs: ' +
            json.dumps([candidate["id"] for candidate in candidates])
        )
        selected = judge(round_dir, "select", model, base_url, select_prompt, frames_to_send)
        by_id = {candidate["id"]: candidate for candidate in candidates}
        if selected.get("candidate_id") not in by_id:
            raise ValueError("Unknown candidate")
        candidate = by_id[selected["candidate_id"]]
        r.freeze(round_dir / "selection.json", selected)
        row = {"round": index, "selected": candidate["id"], "accepted": False,
               "before_gap_m": outcome["after_gap_m"], "after_gap_m": outcome["after_gap_m"]}
        outcome["rounds"].append(row)
        if candidate["operation"] == "no_change":
            return finish(selected.get("reason_code", "no_improvement"))
        if set(selected.get("supporting_pair_ids", [])) != {p["pair_id"] for p in train}:
            return finish("unsupported_selection")
        holdout = [evidence["holdouts"][index]]
        validation_frames = [original_by_pair[p["pair_id"]] for p in holdout]
        for name, option in (("baseline", NO_CHANGE), ("proposal", candidate)):
            records = (baseline_pair_frames(baseline, holdout) if name == "baseline" else
                       render_pair_frames(case, round_dir, blender, "heldout_" + name, holdout, option))
            validation_frames.extend({**frame, "source": name} for frame in records)
        validate_prompt = (
            f"Independently compare baseline/proposal against ORIGINAL for {case['objects']['moving']['label']} "
            "using same pair_id only. No previous preference is supplied. Do not estimate metric distances. "
            "Winner must be tie/uncertain unless the same visible target is better positioned or sized relative "
            "to matching stable landmarks without new floating/interpenetration/artifact evidence. Return JSON: "
            '{"pairs":[{"pair_id":0,"winner":"baseline|proposal|tie|uncertain",'
            '"moving_visible_both":false,"same_physical_target":false,"background_consistent":false,'
            '"position_or_size_better":false,"no_new_artifacts":false,"reason":"brief"}]}.'
        )
        validation = judge(round_dir, "validate", model, base_url, validate_prompt, validation_frames)
        r.freeze(round_dir / "validation.json", validation)
        if not accept_validation(validation, holdout):
            return finish("heldout_rejected")
        applied = geometry_stage(case, round_dir, blender, "accepted", {
            "action": "apply", "scene_frames": frames, "candidate": candidate,
        })
        before = applied["before_state"]["surface_distance_m"]
        after = applied["after_state"]["surface_distance_m"]
        if not all(math.isfinite(value) for value in (before, after)):
            raise ValueError("Nonfinite geometry")
        row.update({"accepted": True, "before_gap_m": before, "after_gap_m": after})
        outcome["after_gap_m"] = after
        case = copy.deepcopy(case)
        case["inputs"]["blend"] = applied["scene_after_blend"]
        r.freeze(round_dir / "accepted_case.json", r.blind_case(case))
        r.freeze(round_dir / "round_result.json", row)
        if index + 1 < min(POLICY["max_rounds"], len(evidence["holdouts"])):
            baseline = render_pair_frames(case, round_dir, blender, "next_baseline",
                                         pairs, NO_CHANGE)
    return finish("frozen_round_budget_exhausted")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--blender", required=True)
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if not args.base_url:
        parser.error("--base-url is required")
    run_case(args.case, args.run, args.blender, args.model, args.base_url, args.preflight_only)


if __name__ == "__main__":
    main()
