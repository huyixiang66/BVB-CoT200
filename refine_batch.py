"""Generate and run blind refinement cases for every object_abs_distance question."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import refinement as r


ROOT = Path(__file__).resolve().parent
QUESTION_TYPE = "object_abs_distance"


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def default_candidates():
    candidates = [{"id": "no_change", "operation": "no_change"}]
    for direction in ("toward", "away"):
        for distance in (0.15, 0.30, 0.45):
            candidates.append({
                "id": f"{direction}_{int(distance * 100):03d}",
                "operation": "translate",
                "direction": direction,
                "distance_m": distance,
                "horizontal_only": True,
            })
    candidates.extend([
        {"id": "scale_085", "operation": "scale", "factor": 0.85},
        {"id": "scale_115", "operation": "scale", "factor": 1.15},
    ])
    return candidates


def asset_path(root, template, question):
    name = template.format(dataset=question["dataset"], scene=question["scene_name"],
                           qa_id=question["qa_id"])
    return str((Path(root) / name).resolve())


def build_cases(original_videos, bvb_videos, blends,
                original_template="{scene}.mp4", bvb_template="{scene}.mp4",
                blend_template="{scene}.blend"):
    questions = [q for q in jsonl(ROOT / "data/questions.jsonl")
                 if q["question_type"] == QUESTION_TYPE]
    recipes = r.read_json(ROOT / "data/recipes.json")
    candidates = default_candidates()
    cases = []
    for question in questions:
        qaid = str(question["qa_id"])
        targets = [role for role in recipes[qaid]["roles"] if not role.get("anchor")]
        if len(targets) != 2:
            raise ValueError(f"q{qaid} must have exactly two non-anchor roles")
        cases.append({
            "case_id": "q" + qaid,
            "qa_id": qaid,
            "scene": question["scene_name"],
            "question_type": QUESTION_TYPE,
            "question": question["question"],
            "inputs": {
                "original_video": asset_path(original_videos, original_template, question),
                "bvb_video": asset_path(bvb_videos, bvb_template, question),
                "blend": asset_path(blends, blend_template, question),
            },
            "objects": {
                "moving": {"label": targets[0]["term"], "auto_match": True},
                "reference": {"label": targets[1]["term"], "auto_match": True},
            },
            "candidates": candidates,
            "candidate_policy": {
                "name": "fixed_symmetric_v1",
                "gt_informed": False,
                "shared_across_all_object_abs_distance": True,
            },
        })
    return cases


def generate_cases(args):
    cases = build_cases(args.original_videos, args.bvb_videos, args.blends,
                        args.original_template, args.bvb_template, args.blend_template)
    selected = set(args.only or [])
    if selected:
        cases = [case for case in cases if case["qa_id"] in selected or case["case_id"] in selected]
    args.cases.mkdir(parents=True, exist_ok=True)
    missing = []
    for case in cases:
        r.validate_case(case)
        path = args.cases / (case["case_id"] + ".json")
        r.freeze(path, case)
        absent = [key for key, value in case["inputs"].items() if not Path(value).is_file()]
        if absent:
            missing.append({"case_id": case["case_id"], "missing": absent})
    return {"question_type": QUESTION_TYPE, "cases": len(cases), "case_directory": str(args.cases.resolve()),
            "contains_ground_truth": False, "candidate_policy": "fixed_symmetric_v1",
            "missing_inputs": missing}


def case_paths(directory, only):
    selected = set(only or [])
    paths = sorted(Path(directory).glob("q*.json"),
                   key=lambda path: int(path.stem.removeprefix("q")))
    if selected:
        paths = [path for path in paths if path.stem in selected or path.stem.removeprefix("q") in selected]
    if not paths:
        raise ValueError("No generated cases selected")
    return paths


def targets():
    question_ids = {
        str(row["qa_id"]) for row in jsonl(ROOT / "data/questions.jsonl")
        if row["question_type"] == QUESTION_TYPE
    }
    return {
        str(row["qa_id"]): float(row["ground_truth"])
        for row in jsonl(ROOT / "data/labels.jsonl")
        if str(row["qa_id"]) in question_ids
    }


def execute(command, case, run, args, label_targets):
    if command == "doctor":
        return r.doctor(case, args.blender)
    if command == "resolve":
        resolved = r.resolve_case_objects(case, run, args.blender)
        return {"objects": resolved["objects"], "resolution": resolved.get("object_resolution"),
                "candidate_ids": [candidate["id"] for candidate in resolved["candidates"]]}
    if command == "prepare":
        return r.prepare(case, run, args.blender)
    if command == "inspect":
        return r.inspect(case, run, args.blender)
    if command == "retrieve":
        return r.retrieve_targets(case, run, args.audit_model, args.base_url)
    if command == "candidates":
        return r.render_candidates(case, run, args.blender)
    if command == "select":
        return r.select_candidate(case, run, args.audit_model, args.base_url)
    if command == "finalize":
        return r.finalize(case, run, args.blender)
    if command == "answer":
        return r.answer(case, run, args.qa_model, args.base_url, args.samples, args.condition)
    scored = dict(case)
    scored["ground_truth_m"] = label_targets[str(case["qa_id"])]
    return r.report(scored, run)


def run_stage(args):
    if args.command in {"retrieve", "select"} and not args.audit_model:
        raise ValueError(f"{args.command} requires --audit-model or REFINE_AUDIT_MODEL")
    if args.command == "answer" and not args.qa_model:
        raise ValueError("answer requires --qa-model or OPENAI_MODEL")
    label_targets = targets() if args.command == "report" else {}
    results = []
    for path in case_paths(args.cases, args.only):
        case = r.load_case(path)
        run = (args.run_root / case["case_id"]).resolve()
        try:
            outcome = execute(args.command, case, run, args, label_targets)
            row = {"case_id": case["case_id"], "scene": case["scene"], "status": "ok"}
            if args.command == "resolve":
                resolution = outcome["resolution"]
                row.update({"resolution_complete": resolution["complete"],
                            "unresolved_roles": resolution["unresolved_roles"],
                            "candidate_ids": outcome["candidate_ids"]})
            results.append(row)
        except Exception as error:
            results.append({"case_id": case["case_id"], "scene": case["scene"], "status": "error",
                            "error_type": type(error).__name__, "error": str(error)})
    summary = {
        "command": args.command,
        "question_type": QUESTION_TYPE,
        "gt_available_to_stage": args.command == "report",
        "total": len(results),
        "succeeded": sum(row["status"] == "ok" for row in results),
        "failed": sum(row["status"] == "error" for row in results),
        "results": results,
    }
    if args.command == "resolve":
        summary["resolution_complete"] = sum(row.get("resolution_complete") is True for row in results)
        summary["no_change_only"] = sum(row.get("candidate_ids") == ["no_change"] for row in results)
    args.run_root.mkdir(parents=True, exist_ok=True)
    r.save_json(args.run_root / f"batch_{args.command}.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "doctor", "resolve", "prepare", "inspect", "retrieve",
                                             "candidates", "select", "finalize", "answer", "report"))
    parser.add_argument("--cases", type=Path, default=Path("runs/refine-object-abs-distance/cases"))
    parser.add_argument("--run-root", type=Path, default=Path("runs/refine-object-abs-distance/items"))
    parser.add_argument("--original-videos", type=Path)
    parser.add_argument("--bvb-videos", type=Path)
    parser.add_argument("--blends", type=Path)
    parser.add_argument("--original-template", default="{scene}.mp4")
    parser.add_argument("--bvb-template", default="{scene}.mp4")
    parser.add_argument("--blend-template", default="{scene}.blend")
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--blender", default=os.getenv("BLENDER_BIN", "blender"))
    parser.add_argument("--audit-model", default=os.getenv("REFINE_AUDIT_MODEL"))
    parser.add_argument("--qa-model", default=os.getenv("OPENAI_MODEL"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--condition", choices=("before", "after", "both"), default="both")
    args = parser.parse_args()
    if args.command == "generate":
        if not all((args.original_videos, args.bvb_videos, args.blends)):
            parser.error("generate requires --original-videos, --bvb-videos, and --blends")
        result = generate_cases(args)
    else:
        result = run_stage(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
