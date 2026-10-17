"""Simple same-time target-box refinement, with separate heldout measurement."""

import argparse
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import refinement as r
import refine_box as box
import refine_measured as measured
import refine_iterative as it
from refine_measurement import top_left_box


def geometry(case, run, blender, pairs, candidates, save=False):
    run = Path(run).resolve()
    spec = {"scene_frames": [p["scene_frame"] for p in pairs],
            "candidates": candidates, "save_preview": save}
    r.freeze(run / "case.json", r.blind_case(case))
    r.freeze(run / "spec.json", spec)
    output = run / "result.json"
    if output.exists():
        raise ValueError("Use a new geometry run directory")
    command = [r.blender_command(blender), "--background", "--disable-autoexec",
               case["inputs"]["blend"], "--threads", "2", "--python-exit-code", "1",
               "--python", str(r.ROOT / "scripts/refine_box_blender.py"),
               "--", "--case", str(run / "case.json"), "--spec", str(run / "spec.json"),
               "--output", str(output)]
    with (run / "blender.log").open("w", encoding="utf-8") as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True,
                       timeout=900, env=r.blender_environment())
    return r.read_json(output)


def locate(run, name, model, url, label, frames, fixed_original=None):
    prompt = (
        f"Locate the SAME physical {label} in ORIGINAL and RECONSTRUCTION. "
        "Return tight boxes around the visible portion of the target itself, excluding lamps, clocks "
        "and other objects resting on it. Cropping, occlusion or incomplete reconstruction is allowed: "
        "box the visible portion, do not invent hidden extents. Use null only if the target cannot be "
        "identified. Do not require matching background, estimate metric distance or answer a QA question. "
        "Coordinates normalized [0,1], TOP LEFT origin, box [left,top,right,bottom]. "
        'Return JSON {"same_physical_target":true,"original_box":[0.1,0.2,0.4,0.6],'
        '"baseline_box":[0.2,0.2,0.5,0.6],"partial":false,"reason":"brief"}.'
    )
    if fixed_original is not None:
        prompt += (
            f" The original target was already localized at {fixed_original}; this box is frozen. "
            "Use that region to identify the corresponding target in RECONSTRUCTION. "
            "Do not relocalize or reject the original region. Return that same original_box; "
            "set baseline_box to null and same_physical_target to false if the corresponding "
            "reconstruction target cannot be identified."
        )
    return it.judge(run, name, model, url, prompt, frames)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--blender", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--observations", type=Path, help="Reuse frozen before boxes, no new before calls")
    parser.add_argument("--parts", type=Path, help="Explicit body and carried part names for this case")
    parser.add_argument("--visible-surface", action="store_true",
                        help="Score occlusion-tested mesh samples instead of full AABB")
    args = parser.parse_args()
    args.parent, args.frames, args.run = [p.resolve() for p in (args.parent, args.frames, args.run)]
    case = measured.verify_parent(args.parent)
    if args.visible_surface:
        case["box_visible_surface"] = True
    extra_inputs = {}
    if args.parts:
        parts = r.read_json(args.parts)
        if parts["case_id"] != case["case_id"] or parts["blend_sha256"] != r.digest(case["inputs"]["blend"]):
            raise ValueError("Part manifest belongs to another source")
        case["objects"]["moving"] = {"label": case["objects"]["moving"]["label"], "names": parts["body"]}
        case["box_carried_names"] = parts["carried"]
        extra_inputs[str(args.parts.resolve())] = r.digest(args.parts)
    pairs = r.read_json(args.parent / "proposed_pairs.json")
    original = r.read_json(args.frames / "original/manifest.json")["frames"]
    baseline = {f["scene_frame"]: f for f in r.read_json(args.frames / "alignment_baseline/result.json")["frames"]}
    if len(pairs) != len(original):
        raise ValueError("Original pair count mismatch")
    for pair, frame in zip(pairs, original):
        if pair["original_frame_index_0based"] != frame["source_frame_index_0based"]:
            raise ValueError("Time pairing changed")
    for f in [*original, *baseline.values()]:
        if r.digest(f["file"]) != f["sha256"]:
            raise ValueError("Frozen frame changed")
    if args.observations:
        args.observations = args.observations.resolve()
        old_config = r.read_json(args.observations / "config.json")
        if Path(old_config["parent"]).resolve() != args.parent or old_config["model"] != args.model:
            raise ValueError("Observation source/model mismatch")
        for path, digest in old_config["input_sha256"].items():
            if r.digest(path) != digest:
                raise ValueError("Observation input changed")
        for pair, frame in zip(pairs, original):
            pid = pair["pair_id"]
            request_path = args.observations / "calls" / f"before_{pid}" / "request.json"
            request = r.read_json(request_path)
            expected = [frame, baseline[pair["scene_frame"]]]
            if [f["sha256"] for f in request["frames"]] != [f["sha256"] for f in expected]:
                raise ValueError("Observation frame mismatch")
            for path in (request_path, args.observations / f"before_{pid}.json"):
                extra_inputs[str(path)] = r.digest(path)
        extra_inputs[str(args.observations / "config.json")] = r.digest(args.observations / "config.json")
    r.freeze(args.run / "config.json", {
        "protocol": "simple_target_box_v2", "policy": box.POLICY,
        "parent": str(args.parent.resolve()), "model": args.model, "qa_calls": 0, "gt_used": False,
        "evaluation_type": "known-label-development-not-unseen-test",
        "pair_ids": [p["pair_id"] for p in pairs],
        "split": "first usable pair trains; next at most two usable pairs validate; no residual-based choice",
        "max_calls": 3 if args.observations else len(pairs) + 3, "camera_fixed": True,
        "extra_input_sha256": extra_inputs,
        "parts": str(args.parts.resolve()) if args.parts else None,
        "observations": str(args.observations) if args.observations else None,
        "validation": "Frozen original box; localize corresponding reconstruction only",
        "carried_policy": "Same body-pivot similarity transform; excluded from projection and gap",
        "projection": "occlusion_tested_surface_samples" if args.visible_surface else "aabb",
        "input_sha256": {str((args.parent / n).resolve()): r.digest(args.parent / n)
                         for n in ("resolved_case.json", "proposed_pairs.json", "config.json")},
        "code_sha256": {n: r.digest(r.ROOT / n) for n in (
            "refine_box.py", "scripts/run_box_refinement.py", "scripts/refine_box_blender.py",
            "refine_blender.py", "refine_probe.py", "refinement.py")},
    })
    rows, frames_by_id = {}, {}
    for pair, frame in zip(pairs, original):
        pid = pair["pair_id"]
        frames_by_id[pid] = [{**frame, "source": "original", "pair_id": pid},
                            {**baseline[pair["scene_frame"]], "source": "reconstruction", "pair_id": pid}]
        rows[pid] = (r.read_json(args.observations / f"before_{pid}.json") if args.observations else
                     locate(args.run, f"before_{pid}", args.model, args.base_url,
                            case["objects"]["moving"]["label"], frames_by_id[pid]))
        r.freeze(args.run / f"before_{pid}.json", rows[pid])
        print(f"localized pair {pid}: usable={box.usable(rows[pid])}", flush=True)
    usable = [p for p in pairs if box.usable(rows[p["pair_id"]])]
    if not usable:
        r.freeze(args.run / "outcome.json", {"stop": "no_visible_target_boxes", "accepted_edits": 0,
                                            "qa_calls": 0, "gt_used": False})
        return
    train, holdouts = usable[:1], usable[1:3]
    r.freeze(args.run / "split.json", {"train": train, "holdouts": holdouts})
    if args.observations and r.read_json(args.observations / "split.json") != {"train": train, "holdouts": holdouts}:
        raise ValueError("Reused observations changed split")
    initial = geometry(case, args.run / "initial", args.blender, train,
                       [{"id": "identity", "parameters": [0, 0, 1]}])
    initial_box = top_left_box(initial["candidates"]["identity"][0]["projected_bbox_xyxy"])
    if initial_box is None:
        raise ValueError("No projected target box in training frame")
    center = [0, 0, 1]
    pid = train[0]["pair_id"]
    for index in range(len(box.POLICY["steps"])):
        candidates = box.candidates(center, index)
        projection = geometry(case, args.run / f"search_{index}", args.blender, train, candidates)
        winner, scores = box.select([v[0] for v in projection["candidates"].values()], rows[pid], initial_box)
        center = winner["candidate"]["parameters"]
        r.freeze(args.run / f"scores_{index}.json", {"winner": winner, "scores": scores})
        print(f"search {index}: {center}, box loss={winner['loss']:.6f}", flush=True)
    selected = winner["candidate"]
    r.freeze(args.run / "selected.json", selected)
    checked = train + holdouts
    preview = geometry(case, args.run / "preview", args.blender, checked, [selected], save=True)
    checks = []
    by_frame = {f["scene_frame"]: f for f in preview["frames"]}
    for pair in checked:
        pid = pair["pair_id"]
        observation = locate(args.run, f"after_{pid}", args.model, args.base_url,
                             case["objects"]["moving"]["label"],
                             [frames_by_id[pid][0], {**by_frame[pair["scene_frame"]],
                                                   "source": "reconstruction", "pair_id": pid}],
                             fixed_original=rows[pid]["original_box"])
        r.freeze(args.run / f"after_{pid}.json", observation)
        before_loss = box.error(rows[pid]["baseline_box"], rows[pid]["original_box"])
        # Original target remains frozen; do not let repeated localization move the goal.
        after_loss = (box.error(observation["baseline_box"], rows[pid]["original_box"])
                      if box.usable(observation) else 1.0)
        checks.append({"pair_id": pid, "role": "train" if pair in train else "holdout",
                       "before": before_loss, "after": after_loss, "gain": before_loss - after_loss})
    changed = selected["parameters"] != [0, 0, 1]
    accepted = bool(changed and holdouts and all(p["gain"] >= box.POLICY["min_gain"] for p in checks))
    measured.verify_parent(args.parent)
    if any(r.digest(path) != digest for path, digest in extra_inputs.items()):
        raise ValueError("Frozen experiment inputs changed")
    r.freeze(args.run / "outcome.json", {
        "protocol": "simple_target_box_v2", "gt_used": False, "qa_calls": 0,
        "selected": selected, "checks": checks, "accepted_edits": int(accepted),
        "proposal_saved": preview["proposal_blend"], "proposal_sha256": r.digest(preview["proposal_blend"]),
        "final_blend": preview["proposal_blend"] if accepted else case["inputs"]["blend"],
        "before_gap_m": preview["before"]["surface_distance_m"],
        "proposal_gap_m": preview["after"]["surface_distance_m"],
        "after_gap_m": preview["after"]["surface_distance_m"] if accepted else preview["before"]["surface_distance_m"],
        "stop": "box_alignment_improved" if accepted else "proposal_not_confirmed_across_views",
        "limitations": "AABB projection proxy, approximate time pairing, partial visibility and unverified transform membership; not proven GT improvement.",
    })
    print(__import__("json").dumps({"accepted": accepted, "checks": checks}), flush=True)


if __name__ == "__main__":
    main()
