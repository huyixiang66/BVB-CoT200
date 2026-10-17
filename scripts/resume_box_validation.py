"""Resume the frozen selected box proposal after local rendering failure, without relocalizing."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import refinement as r
import refine_measured as measured
import refine_box as box
from scripts.run_box_refinement import geometry, locate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--blender", required=True)
    parser.add_argument("--base-url", required=True)
    args = parser.parse_args()
    source, run = args.source.resolve(), args.run.resolve()
    config = r.read_json(source / "config.json")
    case = measured.verify_parent(Path(config["parent"]))
    selected = r.read_json(source / "selected.json")
    split = r.read_json(source / "split.json")
    checked = split["train"] + split["holdouts"]
    names = ["config.json", "selected.json", "split.json"] + [
        f"before_{p['pair_id']}.json" for p in checked]
    inputs = {str(source / name): r.digest(source / name) for name in names}
    r.freeze(run / "config.json", {
        "protocol": "simple_target_box_v1_render_resume", "inputs": inputs,
        "source_run": str(source), "reason": "absolute render path repair only; frozen candidate unchanged",
        "qa_calls": 0, "gt_used": False, "max_calls": len(checked),
        "code_sha256": {n: r.digest(r.ROOT / n) for n in (
            "scripts/resume_box_validation.py", "scripts/run_box_refinement.py",
            "scripts/refine_box_blender.py", "refine_box.py", "refine_blender.py", "refine_probe.py")},
    })
    preview = geometry(case, run / "preview", args.blender, checked, [selected], save=True)
    by_frame = {f["scene_frame"]: f for f in preview["frames"]}
    checks = []
    for pair in checked:
        pid = pair["pair_id"]
        before = r.read_json(source / f"before_{pid}.json")
        original = r.read_json(source / "calls" / f"before_{pid}" / "request.json")["frames"][0]
        if r.digest(original["file"]) != original["sha256"]:
            raise ValueError("Original frame changed")
        observation = locate(run, f"after_{pid}", config["model"], args.base_url,
                             case["objects"]["moving"]["label"],
                             [original, {**by_frame[pair["scene_frame"]],
                                         "source": "reconstruction", "pair_id": pid}])
        r.freeze(run / f"after_{pid}.json", observation)
        before_loss = box.error(before["baseline_box"], before["original_box"])
        after_loss = box.error(observation["baseline_box"], before["original_box"]) if box.usable(observation) else 1.
        checks.append({"pair_id": pid, "role": "train" if pair in split["train"] else "holdout",
                       "before": before_loss, "after": after_loss, "gain": before_loss-after_loss})
        print(f"pair {pid}: before={before_loss:.6f}, after={after_loss:.6f}", flush=True)
    accepted = bool(selected["parameters"] != [0, 0, 1] and split["holdouts"] and
                    all(p["gain"] >= box.POLICY["min_gain"] for p in checks))
    measured.verify_parent(Path(config["parent"]))
    if any(r.digest(p) != h for p, h in inputs.items()):
        raise ValueError("Frozen selection changed")
    r.freeze(run / "outcome.json", {
        "gt_used": False, "qa_calls": 0, "checks": checks, "selected": selected,
        "accepted_edits": int(accepted), "proposal_saved": preview["proposal_blend"],
        "proposal_sha256": r.digest(preview["proposal_blend"]),
        "final_blend": preview["proposal_blend"] if accepted else case["inputs"]["blend"],
        "before_gap_m": preview["before"]["surface_distance_m"],
        "proposal_gap_m": preview["after"]["surface_distance_m"],
        "after_gap_m": preview["after"]["surface_distance_m"] if accepted else preview["before"]["surface_distance_m"],
        "stop": "box_alignment_improved" if accepted else "proposal_not_confirmed_across_views",
    })


if __name__ == "__main__":
    main()
