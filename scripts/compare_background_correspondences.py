"""Compare frozen background-task responses by exact anchor identity; no model calls."""

import argparse
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import refinement as r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    left_config = r.read_json(args.before / "config.json")
    right_config = r.read_json(args.after / "config.json")
    for field in ("selected_pair_ids", "model", "alignment_policy", "source_proposal_sha256"):
        if left_config[field] != right_config[field]:
            raise ValueError("Comparison changed more than background catalog: " + field)
    rows, hashes = [], {}
    for pid in left_config["selected_pair_ids"]:
        before, after = [], []
        for directory, target in ((args.before, before), (args.after, after)):
            path = directory / f"measurement_{pid}.json"
            hashes[str(path.resolve())] = r.digest(path)
            target.append(r.read_json(path))
        before, after = before[0], after[0]
        if before["annotation"]["features"] != after["annotation"]["features"]:
            raise ValueError("Target measurements changed")
        old = {p["id"]: p for p in before["response"]["matches"]}
        new = {p["id"]: p for p in after["response"]["matches"]}
        common = sorted(old.keys() & new.keys())
        requests = [r.read_json(d / "calls" / f"background_{pid}" / "request.json")
                    for d in (args.before, args.after)]
        for field in ("frames", "max_completion_tokens", "temperature", "image_detail", "model"):
            if requests[0][field] != requests[1][field]:
                raise ValueError("Request controls changed: " + field)
        rows.append({
            "pair_id": pid, "common_anchor_count": len(common),
            "added_anchor_ids": sorted(new.keys() - old.keys()),
            "removed_anchor_ids": sorted(old.keys() - new.keys()),
            "status_changes": [{"id": key, "before": old[key]["status"], "after": new[key]["status"]}
                               for key in common if old[key]["status"] != new[key]["status"]],
            "repeated_matches": [{"id": key, "coordinate_distance": math.dist(
                old[key]["original"], new[key]["original"])} for key in common
                if old[key]["status"] == new[key]["status"] == "matched"],
            "before_matches": sum(p["status"] == "matched" for p in old.values()),
            "after_matches": sum(p["status"] == "matched" for p in new.values()),
        })
    r.freeze(args.output, {"pairs": rows, "input_sha256": hashes,
                          "code_sha256": r.digest(Path(__file__)),
                          "qa_calls": 0, "api_calls": 0, "gt_used": False,
                          "limitation": "Catalog context differs; status changes do not isolate stochasticity."})
    print(__import__("json").dumps(rows, indent=2))


if __name__ == "__main__":
    main()
