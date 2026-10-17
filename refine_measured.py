"""Opt-in numeric continuation of frozen no-change evidence runs. No QA or GT."""

import argparse
import copy
import json
from pathlib import Path
import subprocess

import refinement as r
import refine_iterative as it
import refine_measurement as m
import refine_alignment as alignment

PROTOCOL = "background_relative_box_residual_v1"


def parent_outcome_path(parent):
    return parent / ("outcome.json" if (parent / "outcome.json").exists() else "preflight.json")


def verify_parent(parent):
    config = r.read_json(parent / "config.json")
    outcome = r.read_json(parent_outcome_path(parent))
    if outcome["gt_used"] or outcome["qa_calls"] or outcome["accepted_edits"]:
        raise ValueError("Requires frozen blind no-edit parent")
    for name, digest in config["code_sha256"].items():
        if r.digest(r.ROOT / name) != digest:
            raise ValueError("Parent code changed: " + name)
    case = r.read_json(parent / "resolved_case.json")
    for key, digest in config["input_sha256"].items():
        if r.digest(case["inputs"][key]) != digest:
            raise ValueError("Parent source changed")
    return case


def split_measured(pairs, measurements):
    ids = [p["pair_id"] for p in pairs]
    if len(set(ids)) != len(ids) or set(ids) != set(measurements):
        raise ValueError("Measurements must cover each frozen pair exactly once")
    rows = [{"pair_id": p["pair_id"], **{key: measurements[p["pair_id"]] is not None
             for key in ("moving_visible_original", "moving_visible_reconstruction",
                         "same_physical_target", "background_consistent")}} for p in pairs]
    return it.split_evidence({"pairs": rows}, pairs)


def project(case, run, blender, pairs, candidates, observations=None, anchors=None):
    run.mkdir(parents=True, exist_ok=True)
    case_path, spec_path, output = run / "case.json", run / "spec.json", run / "result.json"
    r.freeze(case_path, r.blind_case(case))
    r.freeze(spec_path, {"scene_frames": [p["scene_frame"] for p in pairs],
                         "candidates": candidates, "source_sha256": r.digest(case["inputs"]["blend"]),
                         "observations": observations, "anchors": anchors})
    if not output.exists():
        command = [r.blender_command(blender), "--background", "--disable-autoexec",
                   case["inputs"]["blend"], "--threads", "2", "--python-exit-code", "1",
                   "--python", str(r.ROOT / "refine_project.py"), "--",
                   "--case", str(case_path), "--spec", str(spec_path), "--output", str(output)]
        with (run / "blender.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                           timeout=600, check=True, env=r.blender_environment())
    result = r.read_json(output)
    return {key: {pair["pair_id"]: row for pair, row in zip(pairs, rows)}
            for key, rows in result["candidates"].items()}


def measure_pair(case, run, model, base_url, pair, original, baseline, partial=False, align_background=False):
    prompt = (
        f"Measure the visible {case['objects']['moving']['label']} in this single ORIGINAL/BASELINE pair. "
        f"Reference context: {case['objects']['reference']['label']}. "
        "Do not answer a distance question, infer metric size, or recommend a modification. "
        "Coordinates must be normalized [0,1], origin TOP LEFT, x right, y down. "
        "Report tight xyxy boxes of the SAME entire physical target, not its supporting furniture or "
        "accessories. If cropped, occluded, ambiguous, or unmatched, use null. "
        "Identify at least three matching stationary background point landmarks, spread across both "
        "axes near the target, excluding the moving object and objects resting on it. "
        "Use exactly the same physical point in both images, not a category center. "
        "Do not compensate for camera differences yourself. Unavailable landmarks must be omitted, "
        "never invented. Return JSON only: "
        '{"target_original":null,"target_baseline":null,"landmarks":['
        '{"id":"unique physical point description","original":[0.1,0.2],'
        '"baseline":[0.1,0.2]}],"reason":"brief uncertainty statement"}.'
    )
    if partial:
        prompt = (
            f"Find corresponding visible LOCAL FEATURES of the same {case['objects']['moving']['label']} "
            f"in this ORIGINAL/BASELINE pair. Reference context: {case['objects']['reference']['label']}. "
            "The target may be cropped, occluded or incompletely reconstructed; this alone is NOT a rejection. "
            "Return 2-6 identifiable physical points visible in BOTH images on the target itself, such as "
            "surface details or corners. Do not use image borders, occlusion boundaries, accessories, or "
            "invent hidden geometry. Locate points just inside the target surface, not outside its silhouette. "
            "Each ID describes the same physical detail in both images. Use normalized [0,1] coordinates, "
            "TOP LEFT origin, x right, y down. Also report at least three spread-out matching stationary "
            "background points, excluding the target or objects resting on it. Do not compensate camera "
            "differences yourself. No metric distances, QA answer, desired motion or complete-object box. "
            "If identity or a feature cannot be matched, omit it and say why. Return JSON only: "
            '{"same_physical_target":false,"features":[{"id":"physical point description",'
            '"original":[0.3,0.4],"baseline":[0.3,0.4],"same_physical_point":true}],'
            '"landmarks":[{"id":"stationary point","original":[0.1,0.2],'
            '"baseline":[0.1,0.2]}],"reason":"brief"}'
        )
        if align_background:
            prompt += (
                " For background registration report 6-10 independently identifiable stationary points "
                "if available, spatially distributed around the target and near its depth. Do not reuse "
                "the same physical point under different IDs. Do not include any part of the target. "
                "Do not invent points to reach the count; return fewer with a reason if necessary."
            )
    return it.judge(run, "localize_" + str(pair["pair_id"]), model, base_url, prompt,
                    [{**original, "source": "original"}, {**baseline, "source": "baseline"}])


def run_case(parent, run, blender, model, base_url, measure_before_split=False, partial_features=False,
             align_background=False):
    if align_background and not partial_features:
        raise ValueError("Background alignment requires partial features")
    if partial_features and not measure_before_split:
        raise ValueError("Partial features require measurement before split")
    parent, run = Path(parent).resolve(), Path(run).resolve()
    case = verify_parent(parent)
    evidence = None if measure_before_split else r.read_json(parent / "evidence_split.json")
    if evidence is not None and (len(evidence["train"]) != 2 or not evidence["holdouts"]):
        raise ValueError("Parent has insufficient training/heldout evidence")
    protocol = "measurement_before_split_v2" if measure_before_split else PROTOCOL
    if partial_features:
        protocol = "partial_surface_features_v1"
    if align_background:
        protocol = "partial_background_alignment_v2"
    r.freeze(run / "config.json", {
        "protocol": protocol, "policy": m.POLICY, "candidate_policy": it.POLICY,
        "background_alignment_policy": alignment.POLICY if align_background else None,
        "parent": str(parent), "parent_config_sha256": r.digest(parent / "config.json"),
        "parent_outcome_file": parent_outcome_path(parent).name,
        "parent_outcome_sha256": r.digest(parent_outcome_path(parent)),
        "evidence_split_sha256": None if measure_before_split else r.digest(parent / "evidence_split.json"),
        "proposed_pairs_sha256": r.digest(parent / "proposed_pairs.json"),
        "model": model, "base_url": base_url, "qa_calls": 0, "gt_used": False,
        "code_sha256": {name: r.digest(r.ROOT / name) for name in (
            "refine_measured.py", "refine_measurement.py", "refine_project.py", "refine_alignment.py")},
    })
    pairs = r.read_json(parent / "proposed_pairs.json")
    if not pairs or len(pairs) > it.POLICY["max_alignment_pairs"]:
        raise ValueError("Pair count outside frozen measurement budget")
    if measure_before_split:
        originals = r.extract_frames(case["inputs"]["original_video"], run / "original",
                                      [p["original_frame_index_0based"] for p in pairs], 640)
        baseline_list = it.render_pair_frames(case, run, blender, "alignment_baseline", pairs, it.NO_CHANGE)
        baseline = {frame["pair_id"]: frame for frame in baseline_list}
    else:
        originals = r.read_json(parent / "original/manifest.json")["frames"]
        baseline_records = r.read_json(parent / "alignment_baseline/result.json")["frames"]
        by_frame = {frame["scene_frame"]: frame for frame in baseline_records}
        baseline = {p["pair_id"]: {**by_frame[p["scene_frame"]], "pair_id": p["pair_id"]} for p in pairs}
    if len(originals) != len(pairs) or set(baseline) != {p["pair_id"] for p in pairs}:
        raise ValueError("Incomplete frame evidence")
    original_by_pair = {p["pair_id"]: {**frame, "pair_id": p["pair_id"]}
                        for p, frame in zip(pairs, originals)}
    for frame in [*original_by_pair.values(), *baseline.values()]:
        if r.digest(frame["file"]) != frame["sha256"]:
            raise ValueError("Frozen parent frame changed")
    gap = r.read_json(parent_outcome_path(parent))["after_gap_m"]
    outcome = {"protocol": protocol, "case_id": case["case_id"], "gt_used": False,
               "qa_calls": 0, "before_gap_m": gap, "after_gap_m": gap, "rounds": []}

    def finish(reason):
        outcome.update({"stop_reason": reason, "final_blend": case["inputs"]["blend"],
                        "accepted_edits": sum(row["accepted"] for row in outcome["rounds"])})
        verify_parent(parent)
        r.freeze(run / "outcome.json", outcome)
        print(json.dumps(outcome), flush=True)
        return outcome

    measurements = {}
    anchors = {}
    if measure_before_split:
        projections = project(case, run / "screening_projection", blender, pairs, [it.NO_CHANGE])
        for pair in pairs:
            pid = pair["pair_id"]
            if partial_features:
                kwargs = {"partial": True}
                if align_background:
                    kwargs["align_background"] = True
                annotation = measure_pair(case, run, model, base_url, pair, original_by_pair[pid],
                                          baseline[pid], **kwargs)
                features = annotation.get("features", [])
                features = features if isinstance(features, list) and all(isinstance(f, dict) for f in features) else []
                bound = project(case, run / f"bind_{pid}", blender, [pair], [it.NO_CHANGE],
                                observations={str(pair["scene_frame"]): features})
                measured = m.partial_measurement(annotation, bound["no_change"][pid]["features"],
                                                 align_background=align_background)
                if align_background:
                    r.freeze(run / f"background_{pid}.json", alignment.background_alignment(annotation))
                if measured:
                    anchors[str(pair["scene_frame"])] = measured["features"]
            else:
                annotation = measure_pair(case, run, model, base_url, pair, original_by_pair[pid], baseline[pid])
                record = projections["no_change"][pid]
                measured = m.measurement(annotation, record["projected_bbox_xyxy"])
                if record["first_hit_group_samples"] < 2:
                    measured = None
            measurements[pid] = measured
            r.freeze(run / f"measurement_{pid}.json", {"annotation": annotation, "measurement": measured})
        evidence = split_measured(pairs, measurements)
        r.freeze(run / "evidence_split.json", evidence)
        outcome["measurable_pairs"] = sum(value is not None for value in measurements.values())
        if len(evidence["train"]) < 2 or not evidence["holdouts"]:
            return finish("insufficient_measurable_separated_evidence")
    train = evidence["train"]
    for index, holdout in enumerate(evidence["holdouts"][:it.POLICY["max_rounds"]]):
        directory = run / f"round_{index}"
        candidates = it.round_candidates(index)
        projections = (project(case, directory / "projections", blender, train, candidates, anchors=anchors)
                       if partial_features else project(case, directory / "projections", blender, train, candidates))
        if index == 0 and not measure_before_split:
            for pair in train:
                pid = pair["pair_id"]
                annotation = measure_pair(case, run, model, base_url, pair, original_by_pair[pid], baseline[pid])
                record = projections["no_change"][pid]
                measured = m.measurement(annotation, record["projected_bbox_xyxy"])
                r.freeze(run / f"measurement_{pid}.json", {"annotation": annotation, "measurement": measured})
                if measured is None:
                    return finish("unreliable_localization_or_background")
                measurements[pid] = measured
        scores = {}
        for candidate in candidates:
            records = projections[candidate["id"]]
            scores[candidate["id"]] = ([
                m.partial_loss(records[p["pair_id"]]["features"], measurements[p["pair_id"]]) for p in train
            ] if partial_features else [
                m.loss(records[p["pair_id"]]["projected_bbox_xyxy"], measurements[p["pair_id"]])
                if records[p["pair_id"]]["first_hit_group_samples"] >= 2 else 1.0 for p in train])
        selected = m.choose(scores)
        r.freeze(directory / "scores.json", {"scores": scores, "selected": selected})
        row = {"round": index, "selected": selected, "accepted": False,
               "before_gap_m": outcome["after_gap_m"], "after_gap_m": outcome["after_gap_m"]}
        outcome["rounds"].append(row)
        if selected == "no_change":
            return finish("no_measurable_training_improvement")
        candidate = next(c for c in candidates if c["id"] == selected)
        pid = holdout["pair_id"]
        # This view's coordinates never participate in search or step selection.
        held_projection = (project(case, directory / "heldout_projection", blender,
                                   [holdout], [it.NO_CHANGE, candidate], anchors=anchors)
                           if partial_features else project(case, directory / "heldout_projection", blender,
                                                             [holdout], [it.NO_CHANGE, candidate]))
        held_baseline = (baseline[pid] if index == 0 else it.render_pair_frames(
            case, directory, blender, "heldout_baseline", [holdout], it.NO_CHANGE)[0])
        if measure_before_split:
            held_measure = measurements[pid]
        else:
            annotation = measure_pair(case, directory, model, base_url, holdout,
                                      original_by_pair[pid], held_baseline)
            held_measure = m.measurement(annotation, held_projection["no_change"][pid]["projected_bbox_xyxy"])
            r.freeze(directory / "heldout_measurement.json", {"annotation": annotation, "measurement": held_measure})
        if held_measure is None:
            return finish("unreliable_heldout_measurement")
        held_scores = ({key: [m.partial_loss(records[pid]["features"], held_measure)]
                        for key, records in held_projection.items()} if partial_features else
                       {key: [m.loss(records[pid]["projected_bbox_xyxy"], held_measure)]
                       if records[pid]["first_hit_group_samples"] >= 2 else [1.0]
                       for key, records in held_projection.items()})
        r.freeze(directory / "heldout_scores.json", held_scores)
        if m.choose(held_scores) != selected:
            return finish("heldout_numeric_rejected")
        proposal = it.render_pair_frames(case, directory, blender, "heldout_proposal", [holdout], candidate)[0]
        validation = it.judge(directory, "validate", model, base_url,
            f"Compare ORIGINAL, baseline, proposal for {case['objects']['moving']['label']}. "
            "Do not estimate metric distance. Independently reject identity mismatch, floating, "
            "new intersections, or worse target position/size relative to stationary landmarks. "
            "Partial visibility is allowed: compare the same visible physical features; do not require "
            "complete-object boxes. Reject new separation of parts from the rest of the target. "
            "Return JSON: " '{"pairs":[{"pair_id":' + str(pid) +
            ',"winner":"baseline|proposal|tie|uncertain","moving_visible_both":false,'
            '"same_physical_target":false,"background_consistent":false,'
            '"position_or_size_better":false,"no_new_artifacts":false,"reason":"brief"}]}',
            [{**original_by_pair[pid], "source": "original"},
             {**held_baseline, "source": "baseline"}, {**proposal, "source": "proposal"}])
        r.freeze(directory / "validation.json", validation)
        if not it.accept_validation(validation, [holdout]):
            return finish("heldout_artifact_audit_rejected")
        applied = it.geometry_stage(case, directory, blender, "accepted", {
            "action": "apply", "scene_frames": [p["scene_frame"] for p in pairs], "candidate": candidate})
        row.update({"accepted": True, "after_gap_m": applied["after_state"]["surface_distance_m"]})
        outcome["after_gap_m"] = row["after_gap_m"]
        case = copy.deepcopy(case)
        case["inputs"]["blend"] = applied["scene_after_blend"]
        r.freeze(directory / "round_result.json", row)
    return finish("frozen_round_budget_exhausted")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--blender", required=True)
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--measure-before-split", action="store_true")
    parser.add_argument("--partial-features", action="store_true")
    parser.add_argument("--align-background", action="store_true")
    args = parser.parse_args()
    run_case(args.parent, args.run, args.blender, args.model, args.base_url,
             args.measure_before_split, args.partial_features, args.align_background)
