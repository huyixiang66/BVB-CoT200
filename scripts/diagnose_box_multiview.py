"""Frozen-box multiview geometry experiment. No renders, API calls, QA, or labels."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import refine_box as box
import refinement as r
import refine_measured as measured
from refine_measurement import top_left_box
from scripts.run_box_refinement import geometry
from scripts.measure_box_edges import validate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--parts", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--blender", required=True)
    parser.add_argument("--objective", choices=("mean", "worst_gain"), default="worst_gain")
    parser.add_argument("--translation-only", action="store_true",
                        help="Diagnostic ablation: freeze scale, optimize XY only")
    parser.add_argument("--metric", choices=("center_size", "censored_edges"), default="censored_edges")
    parser.add_argument("--edge-annotations", type=Path)
    args = parser.parse_args()
    source, run = args.source.resolve(), args.run.resolve()
    config = r.read_json(source / "config.json")
    parent = Path(config["parent"])
    case = measured.verify_parent(parent)
    parts = r.read_json(args.parts)
    if parts["case_id"] != case["case_id"] or parts["blend_sha256"] != r.digest(case["inputs"]["blend"]):
        raise ValueError("Part manifest/source mismatch")
    case["objects"]["moving"]["names"] = parts["body"]
    case["box_carried_names"] = parts["carried"]
    if args.metric == "censored_edges":
        case["box_censored_edges"] = True
    pairs = r.read_json(parent / "proposed_pairs.json")
    observations = {p["pair_id"]: r.read_json(source / f"before_{p['pair_id']}.json") for p in pairs}
    usable = [p for p in pairs if box.usable(observations[p["pair_id"]])]
    if len(usable) < 4:
        raise ValueError("Need three fitting views and a separate check")
    train, holdout = usable[:3], usable[3:4]
    checked = train + holdout
    inputs = [args.parts.resolve(), source / "config.json", parent / "proposed_pairs.json",
              parent / "resolved_case.json"]
    if args.edge_annotations:
        if args.metric != "censored_edges":
            raise ValueError("Edge annotations require censored_edges metric")
        annotation = r.read_json(args.edge_annotations)
        if Path(annotation["source"]).resolve() != source:
            raise ValueError("Edge annotation source mismatch")
        if any(r.digest(path) != digest for path, digest in annotation["frame_sha256"].items()):
            raise ValueError("Annotated frame changed")
        inputs.append(args.edge_annotations.resolve())
        for pair in checked:
            pid = pair["pair_id"]
            if r.digest(source / f"before_{pid}.json") != annotation["observation_sha256"][str(pid)]:
                raise ValueError("Annotated box changed")
            kinds = validate(annotation["pairs"][str(pid)])
            observations[pid].update({f"{key}_edge_types": value for key, value in kinds.items()})
    for pair in pairs:
        path = source / "calls" / f"before_{pair['pair_id']}" / "request.json"
        request = r.read_json(path)
        for frame in request["frames"]:
            if r.digest(frame["file"]) != frame["sha256"]:
                raise ValueError("Observation image changed")
        inputs.extend([path, source / f"before_{pair['pair_id']}.json"])
    hashes = {str(p): r.digest(p) for p in inputs}
    r.freeze(run / "config.json", {
        "protocol": "joint_box_geometry_v1", "case_id": case["case_id"],
        "train": train, "holdout": holdout, "policy": box.POLICY,
        "objective": args.objective,
        "translation_only": args.translation_only,
        "metric": args.metric, "edge_epsilon": box.EDGE_EPSILON,
        "edge_annotations": str(args.edge_annotations.resolve()) if args.edge_annotations else None,
        "edge_assumption": "Occluded/cropped edges are bounds; unknown edges are ignored" if args.edge_annotations else
                           "Only image-border cropping is censored; internal occlusion is unresolved",
        "projection": "aabb: same proxy for single/joint controlled comparison",
        "gt_used": False, "qa_calls": 0, "api_calls": 0, "renders": 0,
        "input_sha256": hashes,
        "code_sha256": {name: r.digest(r.ROOT / name) for name in (
            "refine_box.py", "scripts/diagnose_box_multiview.py", "scripts/run_box_refinement.py",
            "scripts/refine_box_blender.py", "scripts/measure_box_edges.py",
            "refine_blender.py", "refine_probe.py")},
        "evaluation_type": "known-label development; reused development views, not unseen validation",
    })
    identity = {"id": "identity", "parameters": [0, 0, 1]}
    baseline = geometry(case, run / "initial", args.blender, checked, [identity])
    initial = [(row["unclipped_bbox_top_left"] if args.metric == "censored_edges" else
                top_left_box(row["projected_bbox_xyxy"])) for row in baseline["candidates"]["identity"]]
    observed = [observations[p["pair_id"]] for p in checked]
    if any(b is None for b in initial):
        raise ValueError("Missing baseline projection")
    center, winners = [0, 0, 1], []
    for index in range(len(box.POLICY["steps"])):
        candidates = box.candidates(center, index)
        if args.translation_only:
            candidates = [c for c in candidates if c["parameters"][2] == 1]
        projection = geometry(case, run / f"search_{index}", args.blender, train,
                              candidates)
        records = list(projection["candidates"].values())
        winner, scores = box.select_multiview(records, observed[:3], initial[:3],
                                             objective=args.objective, metric=args.metric)
        mean_winner, _ = box.select_multiview(records, observed[:3], initial[:3], metric=args.metric)
        single, _ = box.select([rows[0] for rows in records], observed[0], initial[0], metric=args.metric)
        r.freeze(run / f"scores_{index}.json", {"winner": winner, "scores": scores,
                                              "mean_winner_same_candidates": mean_winner,
                                              "single_view_winner_same_candidates": single})
        center = winner["candidate"]["parameters"]
        winners.append(winner["candidate"])
        print(f"round {index}: joint={center}, loss={winner['loss']:.6f}, "
              f"worst_gain={winner['worst_gain']:.6f}; "
              f"single={single['candidate']['parameters']}", flush=True)
    r.freeze(run / "selected.json", winners[-1])
    final = geometry(case, run / "check", args.blender, checked, [identity, winners[-1]])
    checks = []
    for index, pair in enumerate(checked):
        row = final["candidates"][winners[-1]["id"]][index]
        score = box.select([row], observed[index], initial[index], metric=args.metric)[0]
        before_box = (box.predict_edges(initial[index], initial[index], observed[index]["baseline_box"],
                                       observed[index].get("baseline_edge_types"))
                      if args.metric == "censored_edges" else observed[index]["baseline_box"])
        before = (box.edge_error(before_box, observed[index]["original_box"],
                                 observed[index].get("original_edge_types"))
                  if args.metric == "censored_edges" else box.error(before_box, observed[index]["original_box"]))
        target = box.vector(observed[index]["original_box"])
        before_components = [abs(a-b) for a, b in zip(box.vector(observed[index]["baseline_box"]), target)]
        after_components = ([abs(a-b) for a, b in zip(box.vector(score["predicted_box"]), target)]
                            if score["predicted_box"] is not None and row["first_hit_group_samples"] else None)
        checks.append({"pair_id": pair["pair_id"], "role": "fit" if index < 3 else "holdout",
                       "before": before, "predicted_after": score["loss"],
                       "component_order": ["center_x", "center_y", "width", "height"],
                       "components_are_objective": args.metric == "center_size",
                       "objective_baseline_box": before_box, "objective_candidate_box": score["predicted_box"],
                       "before_components": before_components, "after_components": after_components,
                       "gain": before-score["loss"]})
    measured.verify_parent(parent)
    if any(r.digest(path) != digest for path, digest in hashes.items()):
        raise ValueError("Frozen inputs changed")
    r.freeze(run / "outcome.json", {
        "selected": winners[-1], "checks": checks, "gt_used": False, "qa_calls": 0,
        "objective": args.objective,
        "translation_only": args.translation_only,
        "metric": args.metric,
        "edge_annotations": str(args.edge_annotations.resolve()) if args.edge_annotations else None,
        "api_calls": 0, "renders": 0, "accepted_edits": 0,
        "before_gap_m": baseline["before"]["surface_distance_m"],
        "candidate_gap_m": final["candidate_states"][winners[-1]["id"]]["surface_distance_m"],
        "final_blend": case["inputs"]["blend"],
        "stop": "geometry_only_diagnostic_not_applied",
    })
    print(__import__("json").dumps(r.read_json(run / "outcome.json")), flush=True)


if __name__ == "__main__":
    main()
