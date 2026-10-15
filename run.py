"""CLI: frozen BVB blends -> per-type frames and CoT -> one answer."""

import argparse
import importlib.metadata
import os
from pathlib import Path

import pipeline as p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("doctor", "download", "render", "validate", "answer", "report"))
    parser.add_argument("--smoke", action="store_true", help="Use eight fixed questions, one per type")
    parser.add_argument("--only", help="Comma-separated question IDs within the selected set")
    parser.add_argument("--run", type=Path, default=Path("runs/main"))
    parser.add_argument("--blends", type=Path, default=Path("assets/blends"))
    parser.add_argument("--blender", default=os.getenv("BLENDER_BIN", "blender"))
    parser.add_argument("--hf-endpoint", default=os.getenv("HF_ENDPOINT", "https://huggingface.co"))
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-6-luna"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=4, help="Concurrent single-answer requests, not repeated samples")
    args = parser.parse_args()
    bundle = p.load_bundle()
    questions = p.select_questions(bundle, args.smoke, args.only)
    run, blends = args.run.resolve(), args.blends.resolve()
    if args.command == "doctor":
        print(f"DATA OK: {len(questions)} questions, {len({q['scene_name'] for q in questions})} scenes")
        _, version = p.check_blender(args.blender)
        print(version)
        for package in ("openai", "Pillow"):
            print(package + " " + importlib.metadata.version(package))
        print("OPENAI_API_KEY: " + ("set" if os.getenv("OPENAI_API_KEY") else "not set; required only for answer"))
        print("No network/API calls or rendering performed.")
    elif args.command == "download":
        p.download(bundle, questions, blends, args.hf_endpoint)
    elif args.command == "render":
        p.render(bundle, questions, blends, run, args.blender)
    elif args.command == "validate":
        p.validate(bundle, questions, run)
    elif args.command == "answer":
        p.answer(bundle, questions, run, args.model, args.base_url, args.max_tokens, args.workers)
    else:
        p.report(bundle, questions, run)


if __name__ == "__main__":
    main()
