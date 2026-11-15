"""Run the geometry-only Blender view selector for one user-specified question.

Question parsing and semantic-to-mesh grounding are intentionally explicit JSON
inputs. No model or image API is used. The source video is recorded for
provenance; the current geometric selector does not use video pixels.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_TYPES = {
    "object_counting", "object_abs_distance", "object_rel_distance",
    "object_size_estimation", "room_size_estimation",
    "object_rel_direction_easy", "object_rel_direction_medium",
    "object_rel_direction_hard",
}
ROLE_FIELDS = {
    "object_abs_distance": ("endpoints",),
    "object_rel_distance": ("anchor", "candidates"),
    "object_size_estimation": ("target",),
    "object_rel_direction_easy": ("standing", "facing", "query"),
    "object_rel_direction_medium": ("standing", "facing", "query"),
    "object_rel_direction_hard": ("standing", "facing", "query"),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blend", required=True, type=Path,
                        help="Input Blender scene (.blend). It is opened read-only.")
    parser.add_argument("--video", required=True, type=Path,
                        help="Source video, recorded in output provenance only.")
    parser.add_argument("--question", required=True, type=Path,
                        help="JSON with question type, target terms, and mesh-name groups.")
    parser.add_argument("--out", required=True, type=Path,
                        help="New or existing output directory for poses and renders.")
    parser.add_argument("--blender", default=os.environ.get("BLENDER_BIN"),
                        help="Blender executable (defaults to BLENDER_BIN).")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Square render resolution (default: 512).")
    parser.add_argument('--render-diagnostics', action='store_true',
                        help='Also render rejected attempts separately for inspection.')
    parser.add_argument('--role-colors', action='store_true', help='Color direction/distance targets by semantic role; no source scene changes.')
    return parser.parse_args(argv)


def load_question(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    qa_id = str(data.get("qa_id", "question"))
    question_type = data.get("question_type")
    question = data.get("question")
    targets = data.get("targets")
    if not question_type or not question or not isinstance(targets, list):
        raise ValueError("question JSON needs question_type, question, and targets[]")
    if question_type not in SUPPORTED_TYPES:
        raise ValueError("unsupported question_type: %s" % question_type)
    if question_type != "room_size_estimation" and not targets:
        raise ValueError("targets[] must contain at least one target for this question type")

    group_map = {}
    target_terms = []
    for target in targets:
        term = target.get("term")
        instances = target.get("instances")
        if not term or not isinstance(instances, list) or not instances:
            raise ValueError("Each target needs a term and non-empty instances[]")
        if term in group_map:
            raise ValueError("Target terms must be unique: %s" % term)
        cleaned = []
        for instance in instances:
            names = instance if isinstance(instance, list) else [instance]
            if not names or not all(isinstance(name, str) and name for name in names):
                raise ValueError("Each instance must be a mesh name or a list of mesh names")
            cleaned.append(names)
        group_map[term] = cleaned
        target_terms.append(term)

    room = data.get("room", {})
    if not room.get("floor"):
        raise ValueError("room.floor must list the Blender mesh names for the floor")
    roles = data.get("roles", {})
    for field in ROLE_FIELDS.get(question_type, ()):
        if not roles.get(field):
            raise ValueError("roles.%s is required for %s" % (field, question_type))
    expected_counts = {"endpoints": 2, "anchor": 1, "standing": 1, "target": 1,
                       "facing": 1, "query": 1}
    for field in ROLE_FIELDS.get(question_type, ()):
        value = roles[field]
        if field in expected_counts and len(value) != expected_counts[field]:
            raise ValueError("roles.%s must contain %d term(s)" %
                             (field, expected_counts[field]))
        if any(term not in group_map for term in value):
            raise ValueError("roles.%s contains a term missing from targets[]" % field)
    selected_room = {"selection": "user_input"}
    if room.get("floor"):
        selected_room["floor"] = room["floor"]
    if room.get("ceiling"):
        selected_room["ceiling"] = room["ceiling"]
    return (
        {"scene": data.get("scene", "user_scene"), "question_count": 1,
         "questions": [{"qa_id": qa_id, "question_type": question_type,
                        "question": question, "targets": target_terms,
                        "roles": roles, "room": selected_room}]},
        {"scene": data.get("scene", "user_scene"), "groups": group_map,
         "room": {"floor": room.get("floor", []),
                  "ceiling": room.get("ceiling", []),
                  "walls": room.get("walls", []),
                  "remove_walls": bool(room.get("remove_walls", False))}},
    )


def run_blender(blender, blend, script, args):
    command = [str(blender), "--background", "--disable-autoexec", str(blend),
               "--threads", "1", "--python-exit-code", "1", "--python", str(script),
               "--", *[str(arg) for arg in args]]
    subprocess.run(command, check=True, cwd=str(ROOT))


def attach_provenance(path, video, question_path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["inputs"] = {
        "source_video": str(video.resolve()),
        "source_video_bytes": video.stat().st_size,
        "source_video_used_for_selection": False,
        "question_json": str(question_path.resolve()),
        "selector": "Blender scene geometry only; no AI or image inference",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    if not args.blender:
        raise SystemExit("Pass --blender or set BLENDER_BIN to a Blender executable.")
    args.blend = args.blend.resolve()
    args.video = args.video.resolve()
    args.question = args.question.resolve()
    for path, label in ((args.blend, "blend"), (args.video, "video"),
                        (args.question, "question JSON")):
        if not path.is_file():
            raise SystemExit("Input %s does not exist: %s" % (label, path))
    if args.blend.suffix.lower() != ".blend":
        raise SystemExit("--blend must point to a .blend file")
    if args.resolution < 64:
        raise SystemExit("--resolution must be at least 64")

    spec, groups = load_question(args.question)
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit("Output directory is not empty; choose a fresh --out: %s" % out)
    out.mkdir(parents=True, exist_ok=True)
    spec_path, groups_path = out / "question_spec.json", out / "object_groups.json"
    geometry_path, summary_path = out / "selected_views.json", out / "summary.md"
    render_root, render_manifest = out / "renders", out / "render_manifest.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    groups_path.write_text(json.dumps(groups, ensure_ascii=False, indent=2), encoding="utf-8")

    run_blender(args.blender, args.blend, ROOT / "scripts" / "direct_views" / "select_views.py", [
        "--scene", spec["scene"], "--spec", spec_path, "--groups", groups_path,
        "--out", geometry_path,
    ])
    render_args = [
        "--geometry", geometry_path, "--groups", groups_path, "--out-root", render_root,
        "--manifest", render_manifest, "--res", args.resolution, "--scene", spec["scene"],
    ]
    if args.render_diagnostics: render_args.append('--render-diagnostics')
    if args.role_colors: render_args.append('--role-colors')
    run_blender(args.blender, args.blend, ROOT / "scripts" / "direct_views" / "render_selected.py", render_args)
    for path in (geometry_path, render_manifest):
        attach_provenance(path, args.video, args.question)
    result = json.loads(geometry_path.read_text(encoding="utf-8"))
    lines = ["# Direct geometric view selection", "",
             "Continuous geometry optimization; no AI or video pixels used.", ""]
    for item in result.get("questions", []):
        ids = ", ".join(view.get("id", "") for view in item.get("chosen_readable", []))
        lines.append("- %s (%s): %s accepted view(s): %s; status=%s; qa_ready=%s" % (
            item.get("qa_id", ""), item.get("question_type", ""),
            len(item.get("chosen_readable", [])), ids,
            item['selection_status'], item['qa_ready']))
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("SELECT_VIEWS_OK", geometry_path)
    print("RENDERS", render_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
