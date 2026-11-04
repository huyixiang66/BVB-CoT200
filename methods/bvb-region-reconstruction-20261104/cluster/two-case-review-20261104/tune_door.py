"""Reproduce the tuned 2607 result on the frozen 36-frame input."""

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from group_peak_regions import group_peaks, read, write


def run(question, *, appearance_weight, geometry_gate, minimum_pairs, threshold):
    for pair in question["pairs"]:
        geometry = pair["geometry"]
        bonus = appearance_weight * pair["semantic"] * (1 - geometry)
        pair["tuned"] = geometry + bonus if geometry >= geometry_gate else geometry
    groups, evidence = group_peaks(question, "tuned", threshold, minimum_pairs)
    return {
        "parameters": dict(appearance_weight=appearance_weight, geometry_gate=geometry_gate,
                           minimum_pairs=minimum_pairs, threshold=threshold),
        "groups": groups,
        "peak_pair_scores": evidence,
    }


def main():
    source = read(HERE.parent / "verified/results.json")["questions"]["2607"]
    base = run(source, appearance_weight=.35, geometry_gate=.35, minimum_pairs=2, threshold=.55)
    chosen = run(source, appearance_weight=.75, geometry_gate=0., minimum_pairs=1, threshold=.468)
    require_two = run(source, appearance_weight=.75, geometry_gate=0., minimum_pairs=2, threshold=.468)
    arms = {"previous": base, "tuned": chosen, "tuned_keep_two_pairs": require_two}
    write(HERE / "door-tuning.json", dict(question=source["question"], frame_count=source["frame_count"], arms=arms))
    for name, arm in arms.items():
        print(name, [group["peak_ids"] for group in arm["groups"]])


if __name__ == "__main__":
    main()
