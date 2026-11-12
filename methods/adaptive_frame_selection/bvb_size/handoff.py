"""Prepare portable object-size inputs. No model calls or generated-code execution."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from .policy import reference_requirement, load_priors


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
    question = public_question(evidence["question"])
    target = question["objects"][0]
    required, policy_source = reference_requirement(target, load_priors())
    if not decision.get("target_confirmed"):
        raise ValueError("Target not confirmed")
    reference = decision.get("reference_category") if required else None
    fallback_reason = None
    if reference:
        priors = load_priors()
        if reference.lower() in priors['poor_reference']:
            fallback_reason = 'ineligible_reference_category'
        elif not decision.get("boundary_visible") or not decision.get("boundary_evidence"):
            fallback_reason = 'reference_boundary_unconfirmed'
        elif target.lower() == reference.lower():
            fallback_reason = 'same_category_unsupported'
        if fallback_reason: reference = None
    frames = [f for f in evidence["frames"] if f["group"] == decision["selected_group_id"]]
    if not frames:
        raise ValueError("Selected target group missing")
    times = {f["timestamp_s"] for f in frames}
    if reference and any(x["timestamp_s"] not in times for x in decision["boundary_evidence"]):
        raise ValueError("Boundary evidence must refer to supplied group frames")
    objects = [target] + ([reference] if reference else [])
    roles = {"target": target}
    if reference: roles["reference"] = reference
    question.update(objects=objects, roles=roles)
    policy = {"required": required, "source": policy_source, "reference": reference,
              "status": "selected" if reference else "missing_eligible_reference" if required else "target_only",
              "fallback_reason": fallback_reason}
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
    colors = {target: [.85, .08, .055, 1]}
    if reference: colors[reference] = [.05, .48, .15, 1]
    limits = {obj: 1 for obj in objects}
    payload = dict(version="bvb_v10_sameframe_extension", question=question,
                   groups=[dict(group_id=decision["selected_group_id"], primary_objects=objects,
                                timestamps_s=[row["timestamp_s"] for _, _, row in prepared])],
                   frames=[row for _, _, row in prepared], colors=colors, instance_limits=limits,
                   shared_pose_sources=evidence.get("shared_pose_sources", []),
                   architecture_inventory_required=True, reference_selection=decision, reference_policy=policy,
                   semantic_review_receipt_sha256=digest(decision_path))
    write(out / "input.json", payload)
    # Caller supplies the clean upstream system template, not a previous case prompt.
    prompt = Path(base_prompt_path).read_text(encoding="utf-8")
    extension = "joint_extension.txt" if reference else "target_only_extension.txt"
    prompt += "\n" + (Path(__file__).parent / "prompts" / extension).read_text(encoding="utf-8")
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
    if mode not in ("single", "pair", "target_only"):
        raise ValueError("Unresolved views are diagnostic; do not send to QA")
    expected = len(views["images"]) if mode == "target_only" else 1 if mode == "single" else 2
    if expected not in (1, 2) or len(views["images"]) != expected:
        raise ValueError("Unexpected accepted image count")
    objects = joint["question"]["objects"]
    target = objects[0]
    if len(objects) == 1:
        if mode not in ("single", "target_only"): raise ValueError("Target-only views expected")
        prompt = (f"Estimate the longest dimension of the target {target} in centimeters. "
                  "The depicted object is the target. "
                  "The object is reconstructed and may be inaccurate. No measured scale is supplied. ")
    else:
        if mode == "target_only": raise ValueError("Reference views required")
        reference = objects[1]
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


def prepare_reconstruction(joint_path, out):
    joint_path=Path(joint_path);joint=read(joint_path)
    prompt=(joint_path.parent/'system-prompt.txt').read_text(encoding='utf8')
    images=[];content=[]
    for frame in joint['frames']:
        path=joint_path.parent/frame['file']
        if digest(path)!=frame['sha256']: raise ValueError('Reconstruction frame hash mismatch')
        metadata={k:frame[k] for k in ('timestamp_s','groups','compact_camera','observations')}
        content.append({'type':'text','text':json.dumps(metadata,ensure_ascii=False)})
        content.append({'type':'image','index':len(images)})
        images.append({'path':str(path.resolve()),'sha256':frame['sha256']})
    request={'prompt':prompt,'images':images,'content':content,'model':'gpt-6-astra',
             'effort':'high','automatic_retries':0,'gt_in_input':False}
    write(out,request)
    return request


def target_views(manifest_path, out):
    manifest_path = Path(manifest_path)
    manifest = read(manifest_path)
    if manifest.get('reference') is not None:
        raise ValueError('Expected target-only render manifest')
    images = manifest.get('images', [])
    if len(images) not in (1, 2): raise ValueError('Expected one or two images')
    for name in images:
        if Path(name).name != name or not (manifest_path.parent / name).is_file():
            raise ValueError('Target-only image missing or unsafe')
    if Path(out).resolve().parent != manifest_path.resolve().parent:
        raise ValueError('Place views.json beside rendered images')
    result = {'mode':'target_only', 'images':images, 'visual_review_required':True}
    write(out,result)
    return result


def prepare_review(evidence_path, out):
    evidence = read(evidence_path)
    target = evidence['question']['objects'][0]
    required, source = reference_requirement(target, load_priors())
    priors = load_priors()
    prompt = (Path(__file__).parent / 'prompts/reference_review.txt').read_text(encoding='utf8')
    prompt += ('\nPolicy: reference required=' + str(required) + '; target=' + target +
               '; preferred reference categories=' + json.dumps([x for x in priors['reliable'] if x not in priors['poor_reference']]) + '; excluded reference categories=' + json.dumps(priors['poor_reference']) +
               '. If not required, set reference_category=null and select the clearest target group. '
               'If required, choose a reliable co-visible reference with visible endpoints; '
               'unknown physical size alone is not grounds for rejection; never use excluded categories; if none exists set reference_category=null. Never invent one.')
    groups = {}
    for frame in evidence['frames']:
        groups.setdefault(frame['group'], []).append(frame)
    representatives = []
    for frames in groups.values():
        frames.sort(key=lambda f:f['timestamp_s'])
        representatives.extend([frames[0], frames[len(frames)//2]] if len(frames)>1 else frames)
    if not representatives: raise ValueError('No candidate frames')
    images=[]; order=[]
    for frame in representatives:
        path=Path(evidence_path).parent/frame['file']
        if digest(path)!=frame['sha256']: raise ValueError('Candidate frame hash mismatch')
        images.append({'path':str(path.resolve()),'sha256':frame['sha256']})
        order.append({'group':frame['group'],'timestamp_s':frame['timestamp_s']})
    prompt += '\nPublic question: '+json.dumps(public_question(evidence['question']),ensure_ascii=False)
    prompt += '\nImage order: '+json.dumps(order,ensure_ascii=False)
    request={'prompt':prompt, 'target':target, 'reference_required':required, 'policy_source':source,
             'question':public_question(evidence['question']), 'images':images,
             'model':'gpt-6-astra','effort':'high','qa_run':False}
    write(out, request)
    return request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    recon = sub.add_parser("reconstruction-request")
    recon.add_argument("--joint", required=True)
    recon.add_argument("--out", required=True)
    views = sub.add_parser("target-views")
    views.add_argument("--manifest", required=True)
    views.add_argument("--out", required=True)
    review = sub.add_parser("review")
    review.add_argument("--evidence", required=True)
    review.add_argument("--out", required=True)
    joint = sub.add_parser("joint")
    for name in ("evidence", "decision", "base-prompt", "out", "runtime-root"):
        joint.add_argument("--" + name, required=True)
    qa = sub.add_parser("qa")
    for name in ("joint", "views", "out"):
        qa.add_argument("--" + name, required=True)
    qa.add_argument("--include-frames", action="store_true")
    args = parser.parse_args()
    if args.command == "reconstruction-request":
        prepare_reconstruction(args.joint, args.out)
    elif args.command == "target-views":
        target_views(args.manifest, args.out)
    elif args.command == "review":
        prepare_review(args.evidence, args.out)
    elif args.command == "joint":
        prepare_joint(args.evidence, args.decision, args.base_prompt, args.out, args.runtime_root)
    else:
        prepare_qa(args.joint, args.views, args.out, args.include_frames)


if __name__ == "__main__":
    main()
