"""Classify frozen visible box edges using existing frames; no relocalization or QA."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import refinement as r
import refine_iterative as it
import refine_box as box


def validate(annotation):
    for source in ("original", "baseline"):
        values = annotation.get(source)
        if not isinstance(values, list) or len(values) != 4 or any(
                v not in ("silhouette", "occluded", "cropped", "unknown") for v in values):
            raise ValueError("Need four explicit edge categories for each frame")
    return {source: annotation[source] for source in ("original", "baseline")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    args = parser.parse_args()
    source, run = args.source.resolve(), args.run.resolve()
    config = r.read_json(source / "config.json")
    parent = Path(config["parent"])
    case = r.read_json(parent / "resolved_case.json")
    pairs = r.read_json(parent / "proposed_pairs.json")
    pairs = [p for p in pairs if box.usable(r.read_json(source / f"before_{p['pair_id']}.json"))][:4]
    inputs = [source / "config.json", parent / "resolved_case.json", parent / "proposed_pairs.json"]
    for pair in pairs:
        inputs += [source / f"before_{pair['pair_id']}.json",
                   source / "calls" / f"before_{pair['pair_id']}" / "request.json"]
    hashes = {str(p): r.digest(p) for p in inputs}
    r.freeze(run / "config.json", {
        "protocol": "frozen_box_edge_types_v1", "source": str(source), "input_sha256": hashes,
        "model": config["model"], "max_calls": len(pairs), "gt_used": False, "qa_calls": 0,
        "renders": 0, "code_sha256": r.digest(__file__),
    })
    annotations, frame_hashes = {}, {}
    for pair in pairs:
        pid = pair["pair_id"]
        observation = r.read_json(source / f"before_{pid}.json")
        frames = r.read_json(source / "calls" / f"before_{pid}" / "request.json")["frames"]
        for frame in frames:
            if r.digest(frame["file"]) != frame["sha256"]:
                raise ValueError("Frozen frame changed")
            frame_hashes[frame["file"]] = frame["sha256"]
        prompt = (
            f"Classify the four edges of the already frozen visible {case['objects']['moving']['label']} "
            f"boxes: ORIGINAL {observation['original_box']}; RECONSTRUCTION {observation['baseline_box']}. "
            "Coordinates top-left normalized xyxy. Do not relocate, change the boxes, infer metric size, "
            "answer a distance question, or propose edits. Exclude accessories resting on the target. "
            "For each edge in LEFT, TOP, RIGHT, BOTTOM order return: silhouette if it is clearly the "
            "target's true outer contour; occluded if another object hides the continuation; cropped "
            "if the image border cuts off the continuation; unknown if ambiguous or the box does not "
            "clearly follow the target. Use unknown rather than guessing. Classify each frame separately. "
            'JSON only: {"original":["unknown","unknown","unknown","unknown"],'
            '"baseline":["unknown","unknown","unknown","unknown"],"reason":"brief"}.'
        )
        response = it.judge(run, f"edges_{pid}", config["model"], args.base_url, prompt, frames)
        r.freeze(run / f"edges_{pid}.json", response)
        annotations[str(pid)] = validate(response)
        print(f"pair {pid}: {annotations[str(pid)]}", flush=True)
    if any(r.digest(p) != h for p, h in {**hashes, **frame_hashes}.items()):
        raise ValueError("Inputs changed during annotation")
    r.freeze(run / "annotations.json", {
        "source": str(source), "observation_sha256": {
            str(p["pair_id"]): r.digest(source / f"before_{p['pair_id']}.json") for p in pairs},
        "frame_sha256": frame_hashes, "pairs": annotations,
        "gt_used": False, "qa_calls": 0, "renders": 0,
    })


if __name__ == "__main__":
    main()
