"""CLI for blind local geometry refinement and explicitly separated GT diagnostics."""

import argparse
import json
import os
from pathlib import Path

import refinement as r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("doctor", "resolve", "prepare", "inspect", "retrieve", "candidates",
                                             "select", "finalize", "answer", "report"))
    parser.add_argument("--case", type=Path, required=True, help="Case JSON; paths are relative to this file")
    parser.add_argument("--run", type=Path, required=True, help="Dedicated output directory")
    parser.add_argument("--blender", default=os.getenv("BLENDER_BIN", "blender"))
    parser.add_argument("--audit-model", default=os.getenv("REFINE_AUDIT_MODEL"))
    parser.add_argument("--qa-model", default=os.getenv("OPENAI_MODEL"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--condition", choices=("before", "after", "both"), default="both")
    parser.add_argument("--candidate-id", help="Override blind selection only with --diagnostic")
    parser.add_argument("--diagnostic", action="store_true", help="Mark candidate override as label-informed")
    args = parser.parse_args()
    case = r.load_case(args.case)
    run = args.run.resolve()
    if args.command in {"retrieve", "select"} and not args.audit_model:
        parser.error(f"{args.command} requires --audit-model or REFINE_AUDIT_MODEL")
    if args.command == "answer" and not args.qa_model:
        parser.error("answer requires --qa-model or OPENAI_MODEL")
    if args.command == "doctor":
        result = r.doctor(case, args.blender)
    elif args.command == "resolve":
        resolved = r.resolve_case_objects(case, run, args.blender)
        result = {"case_id": resolved["case_id"], "scene": resolved["scene"],
                  "objects": resolved["objects"], "resolution": resolved.get("object_resolution"),
                  "candidate_ids": [candidate["id"] for candidate in resolved["candidates"]]}
    elif args.command == "prepare":
        result = r.prepare(case, run, args.blender)
    elif args.command == "inspect":
        result = r.inspect(case, run, args.blender)
    elif args.command == "retrieve":
        result = r.retrieve_targets(case, run, args.audit_model, args.base_url)
    elif args.command == "candidates":
        result = r.render_candidates(case, run, args.blender)
    elif args.command == "select":
        result = r.select_candidate(case, run, args.audit_model, args.base_url)
    elif args.command == "finalize":
        result = r.finalize(case, run, args.blender, args.candidate_id, args.diagnostic)
    elif args.command == "answer":
        result = r.answer(case, run, args.qa_model, args.base_url, args.samples, args.condition)
    else:
        result = r.report(case, run)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
