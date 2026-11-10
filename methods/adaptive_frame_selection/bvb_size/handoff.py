"""Prepare portable object-size inputs. No model calls or generated-code execution."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def public_question(question):
    allowed = ("id", "scene", "scene_name", "question_type", "question", "objects", "roles")
    return {key: question[key] for key in allowed if key in question}


def prepare_joint(evidence_path, decision_path, base_prompt_path, out, runtime_root):
    """Package all frames of the reviewed group, including original shared poses."""
    evidence_path, out = Path(evidence_path), Path(out)
    evidence, decision = read(evidence_path), read(decision_path)
    if not (decision.get("target_confirmed") and decision.get("boundary_visible")
            and decision.get("reference_category")):
        raise ValueError("No confirmed target/reference with visible comparison boundaries")
    frames = [f for f in evidence["frames"] if f["group"] == decision["selected_group_id"]]
    if not frames or not decision.get("boundary_evidence"):
        raise ValueError("Selected group or boundary evidence missing")
    times = {f["timestamp_s"] for f in frames}
    if any(x["timestamp_s"] not in times for x in decision["boundary_evidence"]):
        raise ValueError("Boundary evidence must refer to supplied group frames")
    question = public_question(evidence["question"])
    target, reference = question["objects"][0], decision["reference_category"]
    if target == reference:
        raise ValueError("This two-category prototype requires distinct target and reference")
    question.update(objects=[target, reference], roles={"target": target, "reference": reference})
    prepared = []
    for i, frame in enumerate(sorted(frames, key=lambda x: x["timestamp_s"])):
        src = evidence_path.parent / frame["file"]
        if digest(src) != frame["sha256"]:
            raise ValueError("Frame hash mismatch")
        for key in ("c2w_blender", "intrinsics_640x480", "compact_camera"):
            if key not in frame:
                raise ValueError("Missing shared pose field: " + key)
        name = f"{i:04d}" + src.suffix
        row = {key: frame[key] for key in ("sha256", "timestamp_s", "source_frame_index",
                                          "c2w_blender", "intrinsics_640x480", "compact_camera")}
        row.update(path=runtime_root.rstrip("/") + "/frames/" + name,
                   file="frames/" + name, groups=[decision["selected_group_id"]],
                   observations=[{"object": target, "status": "reviewed group; provisional detection"}])
        prepared.append((src, name, row))
    out.mkdir(parents=True, exist_ok=False)
    (out / "frames").mkdir()
    for src, name, _ in prepared:
        shutil.copy2(src, out / "frames" / name)
    colors = {target: [.85, .08, .055, 1], reference: [.05, .48, .15, 1]}
    limits = {target: 1, reference: 1}
    payload = dict(version="bvb_v10_sameframe_extension", question=question,
                   groups=[dict(group_id=decision["selected_group_id"], primary_objects=[target, reference],
                                timestamps_s=[row["timestamp_s"] for _, _, row in prepared])],
                   frames=[row for _, _, row in prepared], colors=colors, instance_limits=limits,
                   shared_pose_sources=evidence.get("shared_pose_sources", []),
                   architecture_inventory_required=True, reference_selection=decision,
                   semantic_review_receipt_sha256=digest(decision_path))
    write(out / "input.json", payload)
    # Caller supplies the clean upstream system template, not a previous case prompt.
    prompt = Path(base_prompt_path).read_text(encoding="utf-8")
    prompt += "\n" + (Path(__file__).parent / "prompts/joint_extension.txt").read_text(encoding="utf-8")
    for key, value in [("PUBLIC QUESTION", question), ("REFERENCE SELECTION", decision),
                       ("FINAL INSTANCE LIMITS", limits), ("TARGET COLORS", colors)]:
        prompt += "\n" + key + " = " + json.dumps(value, ensure_ascii=False)
    prompt += "\nOUTPUT_BLEND_PATH = " + runtime_root.rstrip("/") + "/scene.blend\n"
    prompt += "CAMERA ENCODING = [px,py,pz,ux,uy,uz,fx,fy,fz,focal_x,focal_y]; p=position,u=up,f=forward,right=f cross u.\n"
    (out / "system-prompt.txt").write_text(prompt, encoding="utf-8")
    write(out / "experiment.json", dict(input_sha256=digest(out / "input.json"),
          prompt_sha256=digest(out / "system-prompt.txt"), ground_truth_provided=False,
          adaptation="cached reviewed group; same-frame reference extension; not a complete V10 frontend rerun"))
    return payload


def prepare_qa(joint_path, views_path, out, include_frames=False):
    joint_path, views_path = Path(joint_path), Path(views_path)
    joint, views = read(joint_path), read(views_path)
    mode = views["mode"]
    if mode not in ("single", "pair"):
        raise ValueError("Unresolved views are diagnostic; do not send to QA")
    expected = 1 if mode == "single" else 2
    if len(views["images"]) != expected:
        raise ValueError("Unexpected accepted image count")
    target, reference = joint["question"]["objects"]
    description = ("Image 1 shows both target and reference." if mode == "single" else
                   "Image 1 is centered on the target; image 2 is centered on the reference. "
                   "The cameras share focal length, orientation and distance to their respective group centers.")
    prompt = ("Estimate the target object's longest dimension in centimeters. " + description +
              f" Red is target {target}; green is reference {reference}; white is context. "
              "These are diagnostic colors, not original colors. Objects remain at their jointly "
              "reconstructed positions and relative scale, but geometry may be inaccurate. "
              "No measured metric scale or reference length is supplied. Reference sizes vary; "
              "use visible proportions and justified size priors, and state uncertainty. ")
    images = [views_path.parent / name for name in views["images"]]
    if include_frames:
        frames = sorted(joint["frames"], key=lambda f: f["timestamp_s"])
        for f in frames:
            p = joint_path.parent / f["file"]
            if digest(p) != f["sha256"]:
                raise ValueError("Original frame hash mismatch")
            images.append(p)
        prompt += (f"Images {expected+1}–{len(images)} are chronological original reconstruction frames. "
                   "Use them to check identities, extents, reference plausibility and reconstruction errors. ")
    prompt += ('Return JSON with answer (nonnegative number in centimeters), reason (concise Chinese evidence), '
               'and sufficient_evidence (boolean). Give a best estimate if uncertain. Question: ' +
               joint["question"]["question"])
    request = dict(prompt=prompt, images=[dict(path=str(p.resolve()), sha256=digest(p)) for p in images],
                   model="gpt-6-astra", effort="high", gt_in_input=False, automatic_retries=0,
                   visual_review_required=True, qa_run=False)
    write(out, request)
    return request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    joint = sub.add_parser("joint")
    for name in ("evidence", "decision", "base-prompt", "out", "runtime-root"):
        joint.add_argument("--" + name, required=True)
    qa = sub.add_parser("qa")
    for name in ("joint", "views", "out"):
        qa.add_argument("--" + name, required=True)
    qa.add_argument("--include-frames", action="store_true")
    args = parser.parse_args()
    if args.command == "joint":
        prepare_joint(args.evidence, args.decision, args.base_prompt, args.out, args.runtime_root)
    else:
        prepare_qa(args.joint, args.views, args.out, args.include_frames)


if __name__ == "__main__":
    main()
