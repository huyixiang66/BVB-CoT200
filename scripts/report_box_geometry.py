"""Posthoc geometry-only report using an already-revealed development label."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import refinement as r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--label-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for run in args.runs:
        path = run.resolve() / "outcome.json"
        outcome = r.read_json(path)
        config = r.read_json(run / "config.json")
        if any(r.digest(p) != h for p, h in config["input_sha256"].items()):
            raise ValueError("Frozen experiment input changed")
        rows.append({"run": str(run.resolve()), "case_id": config["case_id"],
                     "outcome_sha256": r.digest(path), **outcome})
    label = r.read_json(args.label_report)
    if any(row["case_id"] != label["case_id"] for row in rows):
        raise ValueError("Label case mismatch")
    for row in rows:
        row["before_absolute_error_m"] = abs(row["before_gap_m"] - label["gt_m"])
        row["candidate_absolute_error_m"] = abs(row["candidate_gap_m"] - label["gt_m"])
        if r.digest(Path(row["run"]) / "outcome.json") != row["outcome_sha256"]:
            raise ValueError("Outcome changed during posthoc evaluation")
    r.freeze(args.output, {
        "gt_m": label["gt_m"], "label_report_sha256": r.digest(args.label_report),
        "label_access": "Already-known development label; read only after outcomes frozen",
        "metric": "Object-group AABB gap, not exact surface distance",
        "runs": rows, "qa_calls": 0, "api_calls": 0, "renders": 0,
        "limitation": "Repeated development case; no unseen-test or real rendered improvement claim",
    })
    for row in rows:
        print(row["run"], row["selected"]["parameters"],
              row["before_absolute_error_m"], row["candidate_absolute_error_m"])


if __name__ == "__main__":
    main()
