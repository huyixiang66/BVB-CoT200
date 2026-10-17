"""Posthoc audit after all box proposals/outcomes are frozen. Never runs QA."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import refinement as r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--before-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frozen = {}
    results = []
    for run in args.runs:
        outcome_path = run / "outcome.json"
        frozen[str(outcome_path.resolve())] = r.digest(outcome_path)
        outcome = r.read_json(outcome_path)
        if r.digest(outcome["proposal_saved"]) != outcome["proposal_sha256"]:
            raise ValueError("Proposal changed")
        config = r.read_json(run / "config.json")
        inputs = config.get("extra_input_sha256", config.get("inputs", {}))
        if any(r.digest(path) != digest for path, digest in inputs.items()):
            raise ValueError("Frozen inputs changed")
        preview = r.read_json(run / "preview/result.json")
        calls = list((run / "calls").glob("*/response.json"))
        total = sum(r.read_json(path)["usage"]["total_tokens"] for path in calls)
        results.append({"run": str(run.resolve()), "outcome_sha256": frozen[str(outcome_path.resolve())],
                        "selected": outcome["selected"], "checks": outcome["checks"],
                        "accepted_edits": outcome["accepted_edits"],
                        "before_gap_m": outcome["before_gap_m"],
                        "proposal_gap_m": outcome["proposal_gap_m"],
                        "final_gap_m": outcome["after_gap_m"],
                        "transformed_members": preview.get("transformed_members"),
                        "observed_members": preview.get("observed_members"),
                        "successful_calls": len(calls), "reported_tokens": total})
    # Open only the requested development label after outcome hashes are captured.
    label = None
    with (r.ROOT / "data/labels.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if str(row["qa_id"]) == "4697":
                label = float(row["ground_truth"])
                break
    if label is None:
        raise ValueError("Missing requested label")
    for result in results:
        for stage in ("before", "proposal", "final"):
            result[f"{stage}_absolute_error_m"] = abs(result[f"{stage}_gap_m"] - label)
    before_calls = list((args.before_run / "calls").glob("before_*/response.json"))
    if any(r.digest(path) != digest for path, digest in frozen.items()):
        raise ValueError("Outcome changed during evaluation")
    r.freeze(args.output, {
        "case_id": "q4697", "gt_m": label, "gt_read_after_frozen_outcomes": True,
        "evaluation_type": "known-label-development-not-unseen-test",
        "metric": "Object-group AABB gap proxy, not exact mesh nearest-surface distance",
        "qa_calls": 0, "runs": results, "shared_before_successful_calls": len(before_calls),
        "shared_before_reported_tokens": sum(r.read_json(p)["usage"]["total_tokens"] for p in before_calls),
        "warning": "Group correction changes baseline; do not count this as refinement. "
                   "Development holdouts have been seen across protocol iterations, not a final unseen test.",
    })
    print(json.dumps({"gt_m": label, "results": results}, indent=2))


if __name__ == "__main__":
    main()
