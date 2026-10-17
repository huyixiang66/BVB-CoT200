"""Question-driven local geometry refinement with frozen, auditable stages."""

import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import cv2
from openai import OpenAI


ROOT = Path(__file__).resolve().parent
PAIRING_PROTOCOL = "normalized_time_pairs_v4"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def data_hash(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def resolve_path(value, base):
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return (path if path.is_absolute() else Path(base) / path).resolve()


def number(text):
    answers = re.findall(r"(?im)^\s*ANSWER\s*:\s*([-+]?\d*\.?\d+)", text or "")
    if answers:
        return float(answers[-1])
    match = re.search(r"[-+]?\d*\.?\d+", text or "")
    return float(match.group()) if match else None


def mra(value, target):
    if value is None or target in (None, 0):
        return 0.0
    error = abs(value - target) / target
    thresholds = [0.5 + index * 0.045 for index in range(11)]
    return sum(error <= 1 - threshold for threshold in thresholds) / 11


def validate_case(case):
    required = {"case_id", "scene", "question_type", "question", "inputs", "objects", "candidates"}
    missing = required - set(case)
    if missing:
        raise ValueError("Missing case fields: " + ", ".join(sorted(missing)))
    if case["question_type"] != "object_abs_distance":
        raise ValueError("This first release supports object_abs_distance only")
    if set(case["objects"]) != {"moving", "reference"}:
        raise ValueError("objects must contain moving and reference")
    for role, obj in case["objects"].items():
        if not obj.get("label"):
            raise ValueError(f"{role} needs label")
        if not (obj.get("names") or obj.get("prefixes") or obj.get("auto_match") is True):
            raise ValueError(f"{role} needs names, prefixes, or auto_match=true")
    ids = [candidate.get("id") for candidate in case["candidates"]]
    if len(ids) != len(set(ids)) or "no_change" not in ids:
        raise ValueError("Candidate IDs must be unique and include no_change")
    for candidate in case["candidates"]:
        operation = candidate.get("operation")
        if operation not in {"no_change", "translate", "scale"}:
            raise ValueError("Unsupported candidate operation")
        if operation == "translate":
            if float(candidate.get("distance_m", 0)) <= 0:
                raise ValueError("Translate candidates need positive distance_m")
            if candidate.get("direction", "toward") not in {"toward", "away"}:
                raise ValueError("Translate direction must be toward or away")
        if operation == "scale" and float(candidate.get("factor", 0)) <= 0:
            raise ValueError("Scale candidates need positive factor")
    return case


def load_case(path):
    path = Path(path).resolve()
    case = validate_case(read_json(path))
    case["_spec_path"] = str(path)
    case["_spec_sha256"] = digest(path)
    base = path.parent
    case["inputs"] = {key: str(resolve_path(value, base)) for key, value in case["inputs"].items()}
    return case


def blind_case(case):
    hidden = {"ground_truth_m", "historical_answers", "diagnostic_candidate_id"}
    return {key: value for key, value in case.items() if key not in hidden and not key.startswith("_")}


def freeze(path, value):
    path = Path(path)
    if path.exists() and read_json(path) != value:
        raise ValueError(f"Frozen configuration changed: {path}")
    save_json(path, value)


def video_metadata(path):
    path = Path(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("Cannot open video: " + str(path))
    result = {
        "path": str(path),
        "sha256": digest(path),
        "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    capture.release()
    if result["frame_count"] <= 0 or result["fps"] <= 0:
        raise ValueError("Invalid video metadata: " + str(path))
    return result


def uniform_indices(count, samples):
    if count <= 0 or samples <= 0:
        raise ValueError("count and samples must be positive")
    return [round(index * (count - 1) / (samples - 1)) for index in range(samples)] if samples > 1 else [0]


def extract_frames(video, directory, indices, max_side):
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    metadata = video_metadata(video)
    expected = {"source": metadata, "indices": indices, "max_side": max_side}
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if {key: manifest[key] for key in expected} != expected:
            raise ValueError("Frozen frame extraction changed")
        for frame in manifest["frames"]:
            if digest(frame["file"]) != frame["sha256"]:
                raise ValueError("Extracted frame hash changed")
        return manifest["frames"]
    directory.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    records = []
    try:
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise ValueError("Cannot decode frame " + str(index))
            height, width = frame.shape[:2]
            scale = min(1.0, max_side / max(height, width))
            if scale < 1:
                frame = cv2.resize(frame, (round(width * scale), round(height * scale)))
            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if not ok:
                raise ValueError("Cannot encode frame")
            target = directory / f"frame_{index:06d}.jpg"
            target.write_bytes(encoded.tobytes())
            records.append({
                "source_frame_index_0based": index,
                "seconds": index / metadata["fps"],
                "file": str(target.resolve()),
                "sha256": digest(target),
            })
    finally:
        capture.release()
    save_json(manifest_path, {**expected, "frames": records})
    return records


def needs_object_resolution(case):
    return any(not (obj.get("names") or obj.get("prefixes")) for obj in case["objects"].values())


def resolve_case_objects(case, run, blender):
    if not needs_object_resolution(case):
        return case
    run = Path(run).resolve()
    run.mkdir(parents=True, exist_ok=True)
    provisional_path = run / "object_resolution_case.json"
    provisional = blind_case(case)
    freeze(provisional_path, provisional)
    config = {"case_sha256": data_hash(provisional),
              "blend_sha256": digest(case["inputs"]["blend"])}
    freeze(run / "object_resolution_config.json", config)
    output = run / "object_resolution.json"
    if not output.exists():
        run_blender(case, run, blender, "resolve", output, case_path=provisional_path)
    result = read_json(output)
    if result.get("blend_sha256") != config["blend_sha256"]:
        raise ValueError("Object resolution belongs to another blend")
    if result.get("case_id") != case["case_id"] or result.get("scene") != case["scene"]:
        raise ValueError("Object resolution belongs to another case")
    resolved = copy.deepcopy(case)
    complete = True
    for role in ("moving", "reference"):
        group = result["objects"].get(role, {})
        if group.get("label") != case["objects"][role]["label"]:
            raise ValueError("Invalid automatic object resolution for " + role)
        if group.get("names"):
            resolved["objects"][role] = group
        else:
            resolved["objects"][role] = {"label": group["label"], "auto_match": True}
            complete = False
    resolved["object_resolution"] = result["resolution"]
    if not complete:
        resolved["candidates"] = [candidate for candidate in resolved["candidates"]
                                  if candidate["operation"] == "no_change"]
        resolved["candidate_policy"] = {
            **resolved.get("candidate_policy", {}),
            "resolution_gate": "no_change_only",
        }
    return validate_case(resolved)


def prepare(case, run, blender=None, retrieval_samples=24):
    run = Path(run).resolve()
    run.mkdir(parents=True, exist_ok=True)
    for key in ("original_video", "bvb_video", "blend"):
        if not Path(case["inputs"][key]).is_file():
            raise ValueError(f"Missing {key}: {case['inputs'][key]}")
    if needs_object_resolution(case):
        if not blender:
            raise ValueError("Automatic object matching requires Blender")
        case = resolve_case_objects(case, run, blender)
    full = {key: value for key, value in case.items() if not key.startswith("_")}
    full["input_sha256"] = {key: digest(value) for key, value in case["inputs"].items()}
    full["spec_sha256"] = case["_spec_sha256"]
    freeze(run / "case.json", full)
    freeze(run / "blind_case.json", blind_case(case))
    sources = {}
    for source, key in (("original", "original_video"), ("bvb", "bvb_video")):
        metadata = video_metadata(case["inputs"][key])
        sources[source] = metadata
        indices = uniform_indices(metadata["frame_count"], retrieval_samples)
        extract_frames(case["inputs"][key], run / "frames" / "retrieval" / source, indices, 640)
    save_json(run / "sources.json", sources)
    return {"case_id": case["case_id"], "run": str(run), "sources": sources}


def blender_command(blender):
    command = shutil.which(str(blender)) or (str(Path(blender).resolve()) if Path(blender).is_file() else None)
    if not command:
        raise ValueError("Set BLENDER_BIN or --blender to a Blender 5.2.2 executable")
    return command


def blender_environment():
    return {key: value for key, value in os.environ.items()
            if not re.search(r"(API_KEY|TOKEN|SECRET|PASSWORD)", key, re.I)}


def check_blender(blender):
    command = blender_command(blender)
    result = subprocess.run([command, "--version"], capture_output=True, text=True, timeout=30,
                            check=True, env=blender_environment())
    version = result.stdout.splitlines()[0]
    if not re.search(r"\b5\.2\.2\b", version):
        raise ValueError("Refinement pipeline requires Blender 5.2.2, found: " + version)
    return version


def run_blender(case, run, blender, action, output, stage_spec=None, timeout=1800, case_path=None):
    blend = Path(case["inputs"]["blend"])
    before_hash = digest(blend)
    command = [blender_command(blender), "--background", "--disable-autoexec", str(blend), "--threads", "2",
               "--python-exit-code", "1", "--python", str(ROOT / "refine_blender.py"), "--",
               action, "--case", str(case_path or Path(run) / "blind_case.json"), "--output", str(output)]
    if stage_spec:
        command += ["--spec", str(stage_spec)]
    log = Path(run) / "logs" / f"blender_{action}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, timeout=timeout,
                                env=blender_environment())
    if result.returncode:
        raise RuntimeError(f"Blender {action} failed; see {log}")
    if digest(blend) != before_hash:
        raise RuntimeError("Source blend changed")


def inspect(case, run, blender):
    output = Path(run) / "scene_before.json"
    run_blender(case, run, blender, "inspect", output)
    return read_json(output)


def model_client(base_url, timeout, client=None):
    if client is not None:
        return client
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY is required for model stages")
    return OpenAI(api_key=key, base_url=base_url, max_retries=0, timeout=timeout)


def model_temperature(model):
    return 1 if str(model).startswith("gpt-6-") else 0


def encode_frame(frame, detail):
    path = Path(frame["file"])
    if digest(path) != frame["sha256"]:
        raise ValueError("Frame hash changed: " + str(path))
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return {"type": "image_url", "image_url": {
        "url": f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode(),
        "detail": detail,
    }}


def call_model(run, name, model, base_url, prompt, frames, max_tokens, detail, client=None, timeout=300):
    directory = Path(run) / "calls" / name
    response_path = directory / "response.json"
    request_path = directory / "request.json"
    if response_path.exists() and not request_path.exists():
        raise ValueError("Cached response has no frozen request: " + name)
    freeze(request_path, {
        "model": model,
        "base_url_origin": re.sub(r"(https?://[^/]+).*", r"\1", base_url),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "frames": frames,
        "max_completion_tokens": max_tokens,
        "temperature": model_temperature(model),
        "image_detail": detail,
    })
    if response_path.exists():
        return read_json(response_path)
    if (directory / "started.json").exists():
        raise ValueError("Call may have been billed and will not be resubmitted: " + name)
    (directory / "prompt.txt").write_text(prompt, encoding="utf-8")
    content = [{"type": "text", "text": prompt}]
    for frame in frames:
        metadata = {key: value for key, value in frame.items() if key not in {"file", "sha256", "camera"}}
        content.append({"type": "text", "text": "Frame metadata: " + json.dumps(metadata)})
        content.append(encode_frame(frame, detail))
    save_json(directory / "started.json", {"started_unix": time.time(), "pid": os.getpid()})
    start = time.monotonic()
    try:
        api = model_client(base_url, timeout, client)
        response = api.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_completion_tokens=max_tokens,
            temperature=model_temperature(model),
        )
        result = {
            "requested_model": model,
            "response_model": response.model,
            "text": response.choices[0].message.content,
            "finish_reason": response.choices[0].finish_reason,
            "usage": response.usage.model_dump() if response.usage else {},
            "seconds": time.monotonic() - start,
        }
        save_json(response_path, result)
        return result
    except Exception as error:
        save_json(directory / "error.json", {"type": type(error).__name__,
                                              "http_status": getattr(error, "status_code", None)})
        raise RuntimeError("Model stage failed: " + name + "; " + type(error).__name__) from None


def parse_json_response(result):
    text = (result.get("text") or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(text)


def usable_evidence_pairs(observations):
    return [
        item["pair_id"] for item in observations
        if item.get("visible_target") in {"moving", "both"}
        and item.get("alignment_quality") in {"high", "medium"}
    ]


def retrieve_targets(case, run, model, base_url, client=None):
    output_path = Path(run) / "target_retrieval.json"
    if output_path.exists() and read_json(output_path).get("pairing_protocol") != PAIRING_PROTOCOL:
        raise ValueError("Legacy retrieval must be preserved; use a new run directory")
    blind = read_json(Path(run) / "blind_case.json")
    source_frames = {
        source: read_json(Path(run) / "frames" / "retrieval" / source / "manifest.json")["frames"]
        for source in ("original", "bvb")
    }
    pair_count = min(8, min(len(frames) for frames in source_frames.values()))
    if not pair_count:
        raise ValueError("Paired retrieval requires frames from both sources")
    pairs = []
    frames = []
    for pair_id in range(pair_count):
        progress = pair_id / (pair_count - 1) if pair_count > 1 else 0.0
        pair = {"pair_id": pair_id, "normalized_time": progress}
        for source in ("original", "bvb"):
            position = round(progress * (len(source_frames[source]) - 1))
            frame = source_frames[source][position]
            pair[source] = frame
            frames.append({**frame, "source": source, "pair_id": pair_id,
                           "normalized_time": pair["normalized_time"]})
        pairs.append(pair)
    prompt = (
        f"Compare normalized-time frame pairs from an ORIGINAL indoor video and a BVB reconstruction video. "
        f"Find up to four pair_ids that best support alignment of the same "
        f"{blind['objects']['moving']['label']} and {blind['objects']['reference']['label']}. Prefer pairs where "
        "a target object and stable nearby landmarks are visible in both sources. The cameras are only approximately "
        "reconstructed, so do not require pixel alignment. Do not estimate their distance, answer the QA, or infer a "
        "desired correction. Return only JSON: "
        '{"selected_pair_ids":[integer],"rationale":"...","pair_observations":['
        '{"pair_id":integer,"visible_target":"moving|reference|both|none",'
        '"stable_landmarks":"...","alignment_quality":"high|medium|low","observed_mismatch":"..."}]}.'
    )
    freeze(Path(run) / "retrieval_pairs.json", {
        "pairing_protocol": PAIRING_PROTOCOL, "pairs": pairs, "prompt": prompt, "model": model,
        "base_url_origin": re.sub(r"(https?://[^/]+).*", r"\1", base_url),
    })
    result = parse_json_response(call_model(run, "retrieve_paired_compact_v4", model, base_url, prompt,
                                            frames, 2000, "low", client))
    chosen = result.get("selected_pair_ids", [])
    allowed = {pair["pair_id"] for pair in pairs}
    if not chosen or len(chosen) > 4 or len(set(chosen)) != len(chosen) or not set(chosen) <= allowed:
        raise ValueError("Invalid paired retrieval selection")
    by_id = {pair["pair_id"]: pair for pair in pairs}
    output = {"paired": result, "pairing_protocol": PAIRING_PROTOCOL}
    for source in ("original", "bvb"):
        output[source] = {
            "selected_frame_indices_0based": [
                by_id[pair_id][source]["source_frame_index_0based"] for pair_id in chosen
            ],
            "selected_pair_ids": chosen,
            "rationale": result.get("rationale", ""),
            "observations": result.get("pair_observations", []),
        }
    save_json(output_path, output)
    return output


def candidate_frame_pairs(run):
    retrieval = read_json(Path(run) / "target_retrieval.json")
    original, bvb = retrieval["original"], retrieval["bvb"]
    pair_ids = original.get("selected_pair_ids", [])
    if not pair_ids or len(set(pair_ids)) != len(pair_ids) or pair_ids != bvb.get("selected_pair_ids"):
        raise ValueError("Original and BVB pair IDs must match in selection order")
    original_indices = original["selected_frame_indices_0based"]
    bvb_indices = bvb["selected_frame_indices_0based"]
    if len(original_indices) != len(pair_ids) or len(bvb_indices) != len(pair_ids):
        raise ValueError("Selected pair IDs and frame indices must have equal lengths")
    sources = read_json(Path(run) / "sources.json")
    scene = read_json(Path(run) / "scene_before.json")
    start, end = scene["timeline"]
    count = sources["bvb"]["frame_count"]
    pairs = []
    for pair_id, original_index, bvb_index in zip(pair_ids, original_indices, bvb_indices):
        if not (0 <= original_index < sources["original"]["frame_count"] and 0 <= bvb_index < count):
            raise ValueError("Selected pair frame index is outside its source video")
        progress = bvb_index / (count - 1) if count > 1 else 0.0
        pairs.append({
            "pair_id": pair_id,
            "original_frame_index_0based": original_index,
            "bvb_frame_index_0based": bvb_index,
            "scene_frame": round(start + progress * (end - start)),
            "mapping_method": "uniform_full_timeline_approximation",
        })
    return pairs


def render_candidates(case, run, blender):
    blind = read_json(Path(run) / "blind_case.json")
    pairs = candidate_frame_pairs(run)
    spec = {"candidates": blind["candidates"], "frame_pairs": pairs,
            "scene_frames": sorted({pair["scene_frame"] for pair in pairs}),
            "pairing_protocol": PAIRING_PROTOCOL}
    spec_path = Path(run) / "candidate_spec.json"
    freeze(spec_path, spec)
    output = Path(run) / "candidate_renders"
    run_blender(case, run, blender, "render_candidates", output, spec_path)
    return read_json(output / "manifest.json")


def frame_lookup(run, source, indices):
    frames = read_json(Path(run) / "frames" / "retrieval" / source / "manifest.json")["frames"]
    by_index = {frame["source_frame_index_0based"]: frame for frame in frames}
    return [by_index[index] for index in indices]


def select_candidate(case, run, model, base_url, client=None):
    output_path = Path(run) / "selection.json"
    if output_path.exists() and read_json(output_path).get("pairing_protocol") != PAIRING_PROTOCOL:
        raise ValueError("Legacy selection must be preserved; use a new run directory")
    pairs = candidate_frame_pairs(run)[:2]
    original_indices = [pair["original_frame_index_0based"] for pair in pairs]
    frames = [
        {**frame, "source": "original", "pair_id": pair["pair_id"],
         "paired_scene_frame": pair["scene_frame"]}
        for pair, frame in zip(pairs, frame_lookup(run, "original", original_indices))
    ]
    manifest = read_json(Path(run) / "candidate_renders" / "manifest.json")
    candidates = []
    for item in manifest["candidates"]:
        candidates.append(item["candidate"])
        by_frame = {frame["scene_frame"]: frame for frame in item["frames"]}
        if len(by_frame) != len(item["frames"]):
            raise ValueError("Duplicate candidate scene frames make paired frames ambiguous")
        for pair in pairs:
            if pair["scene_frame"] not in by_frame:
                raise ValueError("Missing paired frame for candidate " + item["candidate"]["id"])
            frames.append({
                **by_frame[pair["scene_frame"]], "source": "candidate",
                "candidate_id": item["candidate"]["id"], "pair_id": pair["pair_id"],
                "paired_original_frame_index_0based": pair["original_frame_index_0based"],
            })
    prompt = (
        "Choose the single candidate whose moving-object position or size best aligns with the ORIGINAL frames "
        "relative to the reference object and stable landmarks. Compare ORIGINAL and candidate frames with the "
        "same pair_id only. Each pair retains the same normalized-time correspondence; do not match frames "
        "from different pairs. All candidates keep the BVB camera unchanged. "
        "Time correspondence does not guarantee identical camera poses, so choose no_change "
        "when evidence is inconsistent. Do not estimate the QA distance. Return only JSON: "
        '{"candidate_id":"...","supporting_original_frame_indices_0based":[integer],'
        '"observed_alignment_change":"...","reason":"...","uncertainties":["..."]}.\n\nCANDIDATES:\n' +
        json.dumps(candidates, ensure_ascii=False)
    )
    freeze(Path(run) / "selection_pairs.json", {
        "pairing_protocol": PAIRING_PROTOCOL, "frame_pairs": pairs, "frames": frames,
        "prompt": prompt, "model": model,
        "base_url_origin": re.sub(r"(https?://[^/]+).*", r"\1", base_url),
    })
    result = parse_json_response(call_model(run, "select_candidate_paired_v4", model, base_url, prompt,
                                            frames, 3000, "high", client))
    by_id = {candidate["id"]: candidate for candidate in candidates}
    if result.get("candidate_id") not in by_id:
        raise ValueError("Unknown selected candidate")
    result["candidate"] = by_id[result["candidate_id"]]
    result["pairing_protocol"] = PAIRING_PROTOCOL
    result["frame_pairs"] = pairs
    save_json(output_path, result)
    return result


def bvb_qa_sampling(run):
    start, end = read_json(Path(run) / "scene_before.json")["timeline"]
    count = read_json(Path(run) / "sources.json")["bvb"]["frame_count"]
    if count > end - start + 1:
        raise ValueError("BVB frame count exceeds the scene timeline; exact sampling metadata is required")
    return {
        "protocol": "bvb_source_count_full_timeline_v1",
        "bvb_frame_count": count,
        "scene_frames": [start + value for value in uniform_indices(end - start + 1, count)],
        "mapping_method": "uniform_full_timeline_approximation",
    }


def validate_qa_frames(manifest, sampling=None):
    before = [frame["scene_frame"] for frame in manifest["before_qa"]]
    after = [frame["scene_frame"] for frame in manifest["after_qa"]]
    if not before or before != after or before != sorted(set(before)):
        raise ValueError("QA frames must be complete, chronological, unique, and paired")
    if sampling and before != sampling["scene_frames"]:
        raise ValueError("QA frames do not match the frozen BVB sampling")
    for left, right in zip(manifest["before_qa"], manifest["after_qa"]):
        if left.get("camera") != right.get("camera"):
            raise ValueError("QA camera changed between paired frames")


def finalize(case, run, blender, candidate_id=None, diagnostic=False):
    selected = read_json(Path(run) / "selection.json")
    if candidate_id is None:
        candidate = selected["candidate"]
        mode = "blind"
    else:
        if not diagnostic:
            raise ValueError("Candidate override requires diagnostic=True")
        candidates = {item["id"]: item for item in case["candidates"]}
        if candidate_id not in candidates:
            raise ValueError("Unknown diagnostic candidate")
        candidate = {**candidates[candidate_id], "diagnostic_only": True, "uses_label_after_blind_selection": True}
        mode = "diagnostic"
    sampling = bvb_qa_sampling(run)
    spec = {"candidate": candidate, "qa_scene_frames": sampling["scene_frames"],
            "qa_sampling": sampling, "mode": mode}
    spec_path = Path(run) / "finalize_spec.json"
    freeze(spec_path, spec)
    output = Path(run) / "final"
    run_blender(case, run, blender, "finalize", output, spec_path)
    manifest = read_json(output / "manifest.json")
    manifest["mode"] = mode
    manifest["qa_sampling"] = sampling
    validate_qa_frames(manifest, sampling)
    save_json(output / "manifest.json", manifest)
    return manifest


def qa_prompt(case, frame_count=None):
    if case.get("qa_prompt"):
        return case["qa_prompt"]
    moving = case["objects"]["moving"]["label"]
    reference = case["objects"]["reference"]["label"]
    frame_label = f"Frames 1-{frame_count}" if frame_count else "The supplied frames"
    return f"""Answer a spatial question about the reconstructed indoor scene using the supplied frames.

{frame_label} are chronological observations from the BVB scene's original camera. Only the two QA objects retain special colours; all other rendered meshes are grayscale.

Colour guide:
RED: {moving} (first object)
BLUE: {reference} (second object)

Highlights are name-matched candidate parts, not verified instance IDs or answers. Confirm each complete physical object from shape and context. Do not interpret grayscale as absence or use highlight colour as a distance cue.

Question: {case['question']}

Reasoning procedure:
1. Identify the two complete objects across the frames and separate them from neighbouring parts or repeated instances.
2. Locate their nearest surfaces, not their centres. Prefer views containing both objects; otherwise combine consistent views into one layout.
3. Estimate the surface gap using a visible grayscale object or fixture as an uncertain size reference. Correct for perspective rather than comparing raw pixels at different depths.
4. Check whether vertical separation contributes to the shortest three-dimensional surface distance.
5. Cross-check another view and report the shortest straight surface-to-surface distance, not a walking distance.

Give a brief evidence summary of at most 120 words and distinguish visible evidence from uncertain size assumptions.

End with exactly one line: ANSWER: <number in metres, one decimal>"""


def answer(case, run, model, base_url, samples, condition, max_tokens=8000, client=None):
    if samples < 1:
        raise ValueError("samples must be positive")
    manifest = read_json(Path(run) / "final" / "manifest.json")
    sampling = manifest.get("qa_sampling")
    if sampling and sampling != bvb_qa_sampling(run):
        raise ValueError("Frozen BVB frame sampling changed")
    validate_qa_frames(manifest, sampling)
    conditions = ("before", "after") if condition == "both" else (condition,)
    if not set(conditions) <= {"before", "after"}:
        raise ValueError("condition must be before, after, or both")
    outputs = []
    for name in conditions:
        frames = [{**frame, "source": "reconstruction_" + name} for frame in manifest[name + "_qa"]]
        for sample in range(samples):
            call_name = f"answer_{name}_{sample:02d}"
            result = call_model(run, call_name, model, base_url, qa_prompt(case, len(frames)), frames,
                                max_tokens, "high", client)
            outputs.append({"condition": name, "sample": sample, **result})
    return outputs


def report(case, run):
    manifest = read_json(Path(run) / "final" / "manifest.json")
    target = case.get("ground_truth_m")
    answers = {"before": [], "after": []}
    calls = Path(run) / "calls"
    for condition in answers:
        for directory in sorted(calls.glob(f"answer_{condition}_*")):
            response_path = directory / "response.json"
            if not response_path.exists():
                continue
            response = read_json(response_path)
            value = number(response["text"])
            row = {"sample": int(directory.name.rsplit("_", 1)[1]), "raw": response["text"],
                   "value_m": value, "response_model": response["response_model"],
                   "usage": response.get("usage", {})}
            if target is not None:
                row["mra"] = mra(value, float(target))
            answers[condition].append(row)
    summary = {}
    for condition, rows in answers.items():
        values = [row["value_m"] for row in rows if row["value_m"] is not None]
        summary[condition] = {"samples": len(rows), "values_m": values,
                              "unique_values_m": sorted(set(values))}
        if target is not None and rows:
            scores = [row["mra"] for row in rows]
            errors = [abs(row["value_m"] - float(target)) for row in rows if row["value_m"] is not None]
            summary[condition].update({"mra_values": scores, "mean_mra": sum(scores) / len(scores),
                                       "best_mra": max(scores), "any_full_mra": any(score == 1 for score in scores),
                                       "mean_abs_error_m": sum(errors) / len(errors) if errors else None})
    payload = {
        "case": {"case_id": case["case_id"], "scene": case["scene"], "question": case["question"],
                 "ground_truth_m": target},
        "mode": manifest.get("mode"),
        "blind_selection": read_json(Path(run) / "selection.json"),
        "final_candidate": manifest["candidate"],
        "geometry": {"before_surface_distance_m": manifest["before_state"]["surface_distance_m"],
                     "after_surface_distance_m": manifest["after_state"]["surface_distance_m"],
                     "distance_method": "object_group_aabb_gap_not_exact_mesh_surface"},
        "qa_sampling": manifest.get("qa_sampling"),
        "source_camera_unchanged": manifest.get("source_camera_unchanged"),
        "qa_frame_counts": {name: len(manifest[name + "_qa"]) for name in ("before", "after")},
        "qa_frames_byte_identical": manifest.get("qa_frames_byte_identical", False),
        "visual_ablation": manifest.get("visual_ablation"),
        "answers": answers,
        "summary": summary,
        "interpretation_guardrail": (
            "Repeated answers measure output variability. A single improved sample is not stable evidence of "
            "geometry-refinement gain; compare distributions under frozen inputs."
        ),
    }
    save_json(Path(run) / "report.json", payload)
    return payload


def doctor(case, blender):
    for key, path in case["inputs"].items():
        if not Path(path).is_file():
            raise ValueError(f"Missing {key}: {path}")
    return {"status": "ok", "blender": check_blender(blender), "case_id": case["case_id"],
            "source_blend_sha256": digest(case["inputs"]["blend"]), "api_calls": 0,
            "visual_inspection": False}
