"""Question-driven local geometry refinement with frozen, auditable stages."""

import base64
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
        if not obj.get("label") or not (obj.get("names") or obj.get("prefixes")):
            raise ValueError(f"{role} needs label and names or prefixes")
    ids = [candidate.get("id") for candidate in case["candidates"]]
    if len(ids) != len(set(ids)) or "no_change" not in ids:
        raise ValueError("Candidate IDs must be unique and include no_change")
    for candidate in case["candidates"]:
        if candidate.get("operation") not in {"no_change", "translate", "scale"}:
            raise ValueError("Unsupported candidate operation")
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


def prepare(case, run, retrieval_samples=24):
    run = Path(run).resolve()
    run.mkdir(parents=True, exist_ok=True)
    for key in ("original_video", "bvb_video", "blend"):
        if not Path(case["inputs"][key]).is_file():
            raise ValueError(f"Missing {key}: {case['inputs'][key]}")
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


def run_blender(case, run, blender, action, output, stage_spec=None, timeout=1800):
    blend = Path(case["inputs"]["blend"])
    before_hash = digest(blend)
    command = [blender_command(blender), "--background", "--disable-autoexec", str(blend), "--threads", "2",
               "--python-exit-code", "1", "--python", str(ROOT / "refine_blender.py"), "--",
               action, "--case", str(Path(run) / "blind_case.json"), "--output", str(output)]
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
    if response_path.exists():
        return read_json(response_path)
    if (directory / "started.json").exists():
        raise ValueError("Call may have been billed and will not be resubmitted: " + name)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "prompt.txt").write_text(prompt, encoding="utf-8")
    save_json(directory / "request.json", {
        "model": model,
        "base_url_origin": re.sub(r"(https?://[^/]+).*", r"\1", base_url),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "frames": frames,
        "max_completion_tokens": max_tokens,
    })
    content = [{"type": "text", "text": prompt}]
    for frame in frames:
        metadata = {key: value for key, value in frame.items() if key not in {"file", "sha256"}}
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
            temperature=0,
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


def retrieve_targets(case, run, model, base_url, client=None):
    blind = read_json(Path(run) / "blind_case.json")
    output = {}
    for source in ("original", "bvb"):
        frames = read_json(Path(run) / "frames" / "retrieval" / source / "manifest.json")["frames"]
        prompt = (
            f"Find frames that best show the {blind['objects']['moving']['label']} and "
            f"{blind['objects']['reference']['label']} for geometric alignment. Do not estimate their distance and "
            "do not answer the QA. Select at most four supplied frame indices. Return only JSON: "
            '{"selected_frame_indices_0based":[integer],"rationale":"...","observations":['
            '{"frame_index_0based":integer,"visible_target":"moving|reference|both",'
            '"stable_landmarks":"...","uncertainty":"..."}]}.'
        )
        result = parse_json_response(call_model(run, "retrieve_" + source, model, base_url, prompt,
                                                frames, 3000, "high", client))
        allowed = {frame["source_frame_index_0based"] for frame in frames}
        chosen = result.get("selected_frame_indices_0based", [])
        if not chosen or len(chosen) > 4 or not set(chosen) <= allowed:
            raise ValueError("Invalid retrieval selection for " + source)
        output[source] = result
    save_json(Path(run) / "target_retrieval.json", output)
    return output


def candidate_scene_frames(run):
    retrieval = read_json(Path(run) / "target_retrieval.json")["bvb"]
    sources = read_json(Path(run) / "sources.json")
    scene = read_json(Path(run) / "scene_before.json")
    start, end = scene["timeline"]
    count = sources["bvb"]["frame_count"]
    return sorted({round(start + index / (count - 1) * (end - start))
                   for index in retrieval["selected_frame_indices_0based"]})


def render_candidates(case, run, blender):
    spec = {"candidates": case["candidates"], "scene_frames": candidate_scene_frames(run)}
    spec_path = Path(run) / "candidate_spec.json"
    save_json(spec_path, spec)
    output = Path(run) / "candidate_renders"
    run_blender(case, run, blender, "render_candidates", output, spec_path)
    return read_json(output / "manifest.json")


def frame_lookup(run, source, indices):
    frames = read_json(Path(run) / "frames" / "retrieval" / source / "manifest.json")["frames"]
    by_index = {frame["source_frame_index_0based"]: frame for frame in frames}
    return [by_index[index] for index in indices]


def select_candidate(case, run, model, base_url, client=None):
    retrieval = read_json(Path(run) / "target_retrieval.json")
    original_indices = retrieval["original"]["selected_frame_indices_0based"][:2]
    frames = [{**frame, "source": "original"} for frame in frame_lookup(run, "original", original_indices)]
    manifest = read_json(Path(run) / "candidate_renders" / "manifest.json")
    candidates = []
    for item in manifest["candidates"]:
        candidates.append(item["candidate"])
        selected = item["frames"] if len(item["frames"]) <= 2 else [item["frames"][0], item["frames"][-1]]
        frames.extend({**frame, "source": "candidate", "candidate_id": item["candidate"]["id"]}
                      for frame in selected)
    prompt = (
        "Choose the single candidate whose moving-object position or size best aligns with the ORIGINAL frames "
        "relative to the reference object and stable landmarks. Camera views are approximate, so choose no_change "
        "when evidence is inconsistent. Do not estimate the QA distance. Return only JSON: "
        '{"candidate_id":"...","supporting_original_frame_indices_0based":[integer],'
        '"observed_alignment_change":"...","reason":"...","uncertainties":["..."]}.\n\nCANDIDATES:\n' +
        json.dumps(candidates, ensure_ascii=False)
    )
    result = parse_json_response(call_model(run, "select_candidate", model, base_url, prompt,
                                            frames, 3000, "high", client))
    by_id = {candidate["id"]: candidate for candidate in candidates}
    if result.get("candidate_id") not in by_id:
        raise ValueError("Unknown selected candidate")
    result["candidate"] = by_id[result["candidate_id"]]
    save_json(Path(run) / "selection.json", result)
    return result


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
    start, end = read_json(Path(run) / "scene_before.json")["timeline"]
    spec = {"candidate": candidate, "qa_scene_frames": uniform_indices(end - start + 1, 16)}
    spec["qa_scene_frames"] = [start + value for value in spec["qa_scene_frames"]]
    spec["mode"] = mode
    spec_path = Path(run) / "finalize_spec.json"
    save_json(spec_path, spec)
    output = Path(run) / "final"
    run_blender(case, run, blender, "finalize", output, spec_path)
    manifest = read_json(output / "manifest.json")
    manifest["mode"] = mode
    save_json(output / "manifest.json", manifest)
    return manifest


def qa_prompt(case):
    return case.get("qa_prompt") or (
        "You are answering a spatial video question from egocentric indoor video. Use only the provided frames. "
        "Reply with one numeric distance in metres, with no explanation.\n\nQuestion: " + case["question"]
    )


def answer(case, run, model, base_url, samples, condition, max_tokens=64, client=None):
    if samples < 1:
        raise ValueError("samples must be positive")
    manifest = read_json(Path(run) / "final" / "manifest.json")
    conditions = ("before", "after") if condition == "both" else (condition,)
    if not set(conditions) <= {"before", "after"}:
        raise ValueError("condition must be before, after, or both")
    outputs = []
    for name in conditions:
        frames = [{**frame, "source": "reconstruction_" + name} for frame in manifest[name + "_qa"]]
        for sample in range(samples):
            call_name = f"answer_{name}_{sample:02d}"
            result = call_model(run, call_name, model, base_url, qa_prompt(case), frames,
                                max_tokens, "low", client)
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
            summary[condition].update({"mra_values": scores, "mean_mra": sum(scores) / len(scores),
                                       "best_mra": max(scores), "any_full_mra": any(score == 1 for score in scores)})
    payload = {
        "case": {"case_id": case["case_id"], "scene": case["scene"], "question": case["question"],
                 "ground_truth_m": target},
        "mode": manifest.get("mode"),
        "blind_selection": read_json(Path(run) / "selection.json"),
        "final_candidate": manifest["candidate"],
        "geometry": {"before_surface_distance_m": manifest["before_state"]["surface_distance_m"],
                     "after_surface_distance_m": manifest["after_state"]["surface_distance_m"]},
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
