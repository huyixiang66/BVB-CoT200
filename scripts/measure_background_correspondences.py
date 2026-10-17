"""Background-task ablation on frozen source anchors and images. No edits or QA."""

import argparse
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import refinement as r
import refine_measured as measured
import refine_grounding as g
import refine_alignment as a
import refine_measurement as m

PROTOCOL = "explicit_background_correspondence_v1"


def replace_background(original, replacement):
    if replacement.get("background_exclusion") != "shared_target_collections_v1":
        raise ValueError("Requires explicitly filtered background proposals")
    if len(original["pairs"]) != len(replacement["pairs"]):
        raise ValueError("Proposal pairs changed")
    mutable = {"landmarks", "background_excluded_members"}
    for old, new in zip(original["pairs"], replacement["pairs"]):
        if ({k: v for k, v in old.items() if k not in mutable}
                != {k: v for k, v in new.items() if k not in mutable}):
            raise ValueError("Only background proposals may change")
        excluded = set(new["background_excluded_members"])
        if not set(old["background_excluded_members"]) <= excluded:
            raise ValueError("Background exclusions weakened")
        if any(p["object_name"] in excluded for p in new["landmarks"]):
            raise ValueError("Excluded object in background proposals")
    return replacement


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--background-proposals", type=Path)
    args = parser.parse_args()
    case = measured.verify_parent(args.parent)
    source_config = r.read_json(args.source / "config.json")
    proposals = r.read_json(args.source / "proposals.json")
    if r.digest(args.source / "proposals.json") != source_config["proposal_sha256"]:
        raise ValueError("Frozen source proposals changed")
    inputs = {}
    if args.background_proposals:
        report_path = args.background_proposals.parent / "report.json"
        report = r.read_json(report_path)
        if r.digest(args.background_proposals) != report["proposal_sha256"]:
            raise ValueError("Filtered proposals changed")
        for name, digest in report["code_sha256"].items():
            if r.digest(r.ROOT / name) != digest:
                raise ValueError("Filtered proposal code changed: " + name)
        proposals = replace_background(proposals, r.read_json(args.background_proposals))
        inputs[str(args.background_proposals.resolve())] = r.digest(args.background_proposals)
        inputs[str(report_path.resolve())] = r.digest(report_path)
    ids = source_config["selected_pair_ids"]
    sources = {p["pair_id"]: p for p in proposals["pairs"]}
    frames_by_id = {}
    for pid in ids:
        path = args.source / f"measurement_{pid}.json"
        request = args.source / "calls" / f"grounded_{pid}" / "request.json"
        inputs[str(path.resolve())] = r.digest(path)
        inputs[str(request.resolve())] = r.digest(request)
        frames_by_id[pid] = r.read_json(request)["frames"]
        for frame in frames_by_id[pid]:
            if r.digest(frame["file"]) != frame["sha256"]:
                raise ValueError("Source frame changed")
    r.freeze(args.run / "config.json", {
        "protocol": PROTOCOL,
        "background_variant": proposals.get("background_exclusion", "original_frozen"),
        "anchor_policy": proposals.get("anchor_policy"),
        "source_config_sha256": r.digest(args.source / "config.json"),
        "source_proposal_sha256": r.digest(args.source / "proposals.json"),
        "input_sha256": inputs, "selected_pair_ids": ids, "max_calls": len(ids),
        "model": source_config["model"], "alignment_policy": a.POLICY,
        "evaluation_type": "known-label-development-not-unseen-test",
        "edits_allowed": False, "qa_calls": 0, "gt_used": False,
        "code_sha256": {name: r.digest(r.ROOT / name) for name in (
            "scripts/measure_background_correspondences.py", "refine_grounding.py",
            "refine_alignment.py", "refine_measurement.py", "refinement.py")},
    })
    results = []
    for pid in ids:
        source = sources[pid]
        if proposals.get("anchor_policy") and len(source["landmarks"]) < a.POLICY["min_landmarks"]:
            results.append({"pair_id": pid, "status": "skipped_insufficient_source_landmarks",
                            "source_landmarks": len(source["landmarks"]), "measurable": False})
            print(f"pair {pid}: skipped, insufficient source landmarks", flush=True)
            continue
        result = r.call_model(args.run, f"background_{pid}", source_config["model"], args.base_url,
                              g.background_prompt(case["objects"]["moving"]["label"], source),
                              frames_by_id[pid], 6000, "high")
        if result.get("finish_reason") != "stop":
            raise ValueError("Incomplete background response")
        response = r.parse_json_response(result)
        landmarks = g.background_observation(response, source)
        # Reuse target observations exactly; no new target sampling or selection.
        annotation = r.read_json(args.source / f"measurement_{pid}.json")["annotation"]
        annotation["landmarks"] = landmarks
        alignment = a.background_alignment(annotation)
        evidence = m.partial_measurement(annotation, source["features"], align_background=True)
        r.freeze(args.run / f"measurement_{pid}.json", {
            "pair_id": pid, "response": response, "annotation": annotation,
            "background_alignment": alignment, "measurement": evidence,
        })
        counts = dict(Counter(p["status"] for p in response["matches"]))
        results.append({"pair_id": pid, "decisions": counts, "background_alignment": alignment,
                        "measurable": evidence is not None})
        print(f"pair {pid}: {counts}, measurable={evidence is not None}", flush=True)
    measured.verify_parent(args.parent)
    if any(r.digest(path) != digest for path, digest in inputs.items()):
        raise ValueError("Source measurements changed")
    responses = [r.read_json(p) for p in (args.run / "calls").glob("*/response.json")]
    r.freeze(args.run / "report.json", {
        "protocol": PROTOCOL, "status": "complete", "pairs": results,
        "background_variant": proposals.get("background_exclusion", "original_frozen"),
        "successful_calls": len(responses),
        "reported_tokens": sum(p.get("usage", {}).get("total_tokens", 0) for p in responses),
        "qa_calls": 0, "gt_used": False, "edits": 0, "candidate_search_run": False,
        "evaluation_type": "known-label-development-not-unseen-test",
    })


if __name__ == "__main__":
    main()
