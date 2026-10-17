"""Bounded development probe of geometry-anchored observations; no edits or QA."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import refinement as r
import refine_iterative as it
import refine_measured as measured
import refine_measurement as m
import refine_alignment as a
import refine_grounding as g


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--base-url", required=True)
    args = parser.parse_args()
    case = measured.verify_parent(args.parent)
    proposal_path = args.run / "proposals.json"
    proposal = r.read_json(proposal_path)
    pairs = r.read_json(args.parent / "proposed_pairs.json")
    if [{k: p[k] for k in pairs[0]} for p in proposal["pairs"]] != pairs:
        raise ValueError("Proposal pair identity changed")
    originals = r.read_json(args.frames / "original/manifest.json")["frames"]
    baseline = {p["scene_frame"]: p for p in r.read_json(args.frames / "alignment_baseline/result.json")["frames"]}
    if len(originals) != len(pairs):
        raise ValueError("Incomplete original frames")
    for frame in [*originals, *baseline.values()]:
        if r.digest(frame["file"]) != frame["sha256"]:
            raise ValueError("Frame changed")
    # Select by frozen pair order and source geometry only, not measured residuals.
    selected = [p for p in proposal["pairs"] if len(p["features"]) >= 2 and len(p["landmarks"]) >= 4][:3]
    r.freeze(args.run / "config.json", {
        "protocol": g.PROTOCOL, "evaluation_type": "known-label-development-not-unseen-test",
        "policy": g.POLICY, "alignment_policy": a.POLICY,
        "selected_pair_ids": [p["pair_id"] for p in selected], "max_calls": 3,
        "model": args.model, "qa_calls": 0, "gt_used": False, "edits_allowed": False,
        "proposal_sha256": r.digest(proposal_path),
        "parent_config_sha256": r.digest(args.parent / "config.json"),
        "code_sha256": {name: r.digest(r.ROOT / name) for name in (
            "refine_grounding.py", "refine_alignment.py", "refine_measurement.py",
            "scripts/propose_refinement_anchors.py", "scripts/measure_grounded_refinement.py")},
    })
    results = []
    for source in selected:
        pid = source["pair_id"]
        original = originals[next(i for i, p in enumerate(pairs) if p["pair_id"] == pid)]
        if original["source_frame_index_0based"] != source["original_frame_index_0based"]:
            raise ValueError("Original time changed")
        response = it.judge(args.run, f"grounded_{pid}", args.model, args.base_url,
                            g.prompt(case["objects"]["moving"]["label"], source),
                            [{**original, "source": "original", "pair_id": pid},
                             {**baseline[source["scene_frame"]], "source": "baseline", "pair_id": pid}])
        annotation = g.observation(response, source)
        evidence = m.partial_measurement(annotation, source["features"], align_background=True)
        row = {"pair_id": pid, "response": response, "annotation": annotation,
               "background_alignment": a.background_alignment(annotation), "measurement": evidence}
        r.freeze(args.run / f"measurement_{pid}.json", row)
        results.append({"pair_id": pid, "matched_features": len(annotation["features"]),
                        "matched_landmarks": len(annotation["landmarks"]), "measurable": evidence is not None,
                        "background_alignment": row["background_alignment"]})
        print(f"pair {pid}: {len(annotation['features'])} target, {len(annotation['landmarks'])} background, measurable={evidence is not None}", flush=True)
    measured.verify_parent(args.parent)
    responses = [r.read_json(p) for p in sorted((args.run / "calls").glob("*/response.json"))]
    r.freeze(args.run / "report.json", {
        "protocol": g.PROTOCOL, "evaluation_type": "known-label-development-not-unseen-test",
        "status": "complete", "pairs": results,
        "successful_calls": len(responses),
        "reported_tokens": sum(p.get("usage", {}).get("total_tokens", 0) for p in responses),
        "qa_calls": 0, "gt_used": False, "edits": 0, "candidate_search_run": False,
        "scope": "observation feasibility only; not a refinement outcome",
    })


if __name__ == "__main__":
    main()
