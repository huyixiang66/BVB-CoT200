"""Portable, single-answer execution of the frozen 200-question release."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

from fixed_protocol import build_prompt, extract_answer
from scoring import semantic_correct

ROOT = Path(__file__).resolve().parent
RELEASE = "bvb-cot200-single-v1"
FIXED = "fixed-camera-C-2026-10-14"
SOURCE_FILES = ("run.py", "pipeline.py", "fixed_protocol.py", "name_resolver.py",
                "render_fixed.py", "render_legacy.py", "scoring.py")
FIVE_TYPES = {"object_abs_distance", "room_size_estimation", "route_planning",
              "object_rel_distance", "object_size_estimation"}
EIGHT_TYPES = FIVE_TYPES | {"object_counting", "object_rel_direction", "obj_appearance_order"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()] if path.exists() else []


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def data_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def category(qa):
    kind = qa["question_type"]
    return "object_rel_direction" if kind.startswith("object_rel_direction") else kind


def load_bundle(root=ROOT):
    directory = Path(root) / "data"
    provenance = read_json(directory / "provenance.json")
    for name, expected in provenance["files"].items():
        path = (Path(root) / name).resolve()
        if not path.is_relative_to(directory.resolve()) or digest(path) != expected:
            raise ValueError("Frozen data changed: " + name)
    # Labels are deliberately excluded from all render/answer inputs.
    bundle = {
        "questions": read_jsonl(directory / "questions.jsonl"),
        "recipes": read_json(directory / "recipes.json"),
        "prompts": read_json(directory / "prompts.json"),
        "assets": read_json(directory / "scenes.json"),
        "smoke": read_json(directory / "smoke.json"),
        "data_hash": data_hash(provenance["files"]),
    }
    questions = bundle["questions"]
    ids = [q["qa_id"] for q in questions]
    if len(ids) != 200 or len(set(ids)) != 200:
        raise ValueError("Expected 200 unique question IDs")
    if Counter(category(q) for q in questions) != Counter({kind: 25 for kind in EIGHT_TYPES}):
        raise ValueError("Expected eight types, 25 questions per type")
    if set(ids) != set(bundle["recipes"]) or set(ids) != set(bundle["prompts"]):
        raise ValueError("Question/recipe/prompt IDs differ")
    scenes = bundle["assets"]["scenes"]
    if len(scenes) != 126 or set(scenes) != {q["scene_name"] for q in questions}:
        raise ValueError("Expected 126 scenes covering all questions")
    for qa in questions:
        if set(qa) != {"qa_id", "dataset", "scene_name", "question_type", "question", "options"}:
            raise ValueError("Unexpected fields in question input")
        if not re.fullmatch(r"\d+", qa["qa_id"]) or not re.fullmatch(r"[A-Za-z0-9_]+", qa["scene_name"]):
            raise ValueError("Invalid question or scene identifier")
        recipe = bundle["recipes"][qa["qa_id"]]
        prompt = bundle["prompts"][qa["qa_id"]]
        if recipe["scene"] != qa["scene_name"]:
            raise ValueError("Recipe scene mismatch")
        if hashlib.sha256(prompt.encode()).hexdigest() != recipe["prompt_sha256"]:
            raise ValueError("Prompt hash mismatch")
        if category(qa) in FIVE_TYPES:
            if recipe["protocol"] != FIXED or recipe["views"] != ["original_camera"] * 16:
                raise ValueError("Five types must use exactly 16 fixed-camera C frames")
        elif recipe["protocol"] != "legacy-v2-render-single-answer":
            raise ValueError("The other three types must use frozen v2 rendering")
    return bundle


def select_questions(bundle, smoke=False, only=None):
    ids = set(bundle["smoke"]) if smoke else {q["qa_id"] for q in bundle["questions"]}
    if only:
        requested = set(only.split(","))
        if not requested <= ids:
            raise ValueError("--only contains an unknown or out-of-scope ID")
        ids = requested
    return [q for q in bundle["questions"] if q["qa_id"] in ids]


def identity(bundle, questions):
    return {
        "release": RELEASE, "samples_per_question": 1, "data_hash": bundle["data_hash"],
        "qa_ids": [q["qa_id"] for q in questions],
        "source_sha256": {name: digest(ROOT / name) for name in SOURCE_FILES},
    }


def freeze(path, expected):
    if path.exists() and read_json(path) != expected:
        raise ValueError("Run configuration changed; use a different --run directory: " + str(path))
    if not path.exists():
        save_json(path, expected)


@contextmanager
def run_lock(run):
    run.mkdir(parents=True, exist_ok=True)
    path = run / ".lock"
    try:
        handle = path.open("x", encoding="utf-8")
    except FileExistsError:
        raise ValueError("Run is locked. Check the recorded PID before removing " + str(path)) from None
    try:
        with handle:
            handle.write(str(os.getpid()) + "\n")
        yield
    finally:
        path.unlink(missing_ok=True)


def blend_path(blends, scene):
    return Path(blends) / (scene + ".blend")


def check_blends(bundle, questions, blends):
    for scene in sorted({q["scene_name"] for q in questions}):
        path = blend_path(blends, scene)
        expected = bundle["assets"]["scenes"][scene]
        if not path.is_file() or path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            raise ValueError("Missing or changed blend: " + str(path))


def download(bundle, questions, blends, endpoint):
    parsed = urlparse(endpoint)
    if parsed.scheme != "https" or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("HF endpoint must be a credential-free HTTPS URL")
    blends.mkdir(parents=True, exist_ok=True)
    for scene in sorted({q["scene_name"] for q in questions}):
        asset = bundle["assets"]["scenes"][scene]
        path = blend_path(blends, scene)
        if path.exists():
            if path.stat().st_size != asset["bytes"] or digest(path) != asset["sha256"]:
                raise ValueError("Existing blend is not the frozen source: " + str(path))
            print("CACHED " + scene, flush=True)
            continue
        url = (endpoint.rstrip("/") + "/datasets/" + bundle["assets"]["dataset"] + "/resolve/" +
               bundle["assets"]["revision"] + "/" + asset["remote_path"])
        partial = path.with_suffix(".blend.part")
        try:
            with urlopen(url, timeout=120) as response, partial.open("wb") as target:
                shutil.copyfileobj(response, target)
            if partial.stat().st_size != asset["bytes"] or digest(partial) != asset["sha256"]:
                raise ValueError("Downloaded blend hash mismatch: " + scene)
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)
        print("DOWNLOADED " + scene, flush=True)


def blender_command(blender):
    command = shutil.which(blender) or (str(Path(blender).resolve()) if Path(blender).is_file() else None)
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
        raise ValueError("Frozen experiment requires Blender 5.2.2, found: " + version)
    return command, version


def frame_paths(manifest):
    return [Path(frame["file"]) for frame in manifest["frames"]]


def verify_render(bundle, qa, run):
    qid = qa["qa_id"]
    recipe = bundle["recipes"][qid]
    directory = run / "renders" / qid
    manifest = read_json(directory / "manifest.json")
    if (manifest["qa_id"] != qid or manifest["scene"] != qa["scene_name"] or
            manifest["recipe_sha256"] != data_hash(recipe) or
            manifest["blend_sha256"] != bundle["assets"]["scenes"][qa["scene_name"]]["sha256"]):
        raise ValueError("Render provenance mismatch: " + qid)
    if [f["view"] for f in manifest["frames"]] != recipe["views"]:
        raise ValueError("Missing or reordered frames: " + qid)
    for frame in manifest["frames"]:
        path = Path(frame["file"]).resolve()
        if not path.is_relative_to(run.resolve()) or not path.is_file() or path.stat().st_size < 100:
            raise ValueError("Frame outside run or incomplete: " + qid)
        if digest(path) != frame["sha256"]:
            raise ValueError("Frame hash mismatch: " + qid)
    if (directory / "prompt.txt").read_text(encoding="utf-8") != bundle["prompts"][qid]:
        raise ValueError("Render prompt mismatch: " + qid)
    return manifest


def normalize_render(bundle, qa, run, raw):
    qid = qa["qa_id"]
    recipe = bundle["recipes"][qid]
    original = read_json(raw)
    frames = []
    for frame in original["frames"]:
        path = Path(frame["file"])
        path = path if path.is_absolute() else raw.parent / path
        frames.append({**frame, "file": str(path.resolve()), "sha256": digest(path)})
    manifest = {
        "qa_id": qid, "scene": qa["scene_name"], "recipe_sha256": data_hash(recipe),
        "blend_sha256": bundle["assets"]["scenes"][qa["scene_name"]]["sha256"],
        "frames": frames, "source_manifest_sha256": digest(raw),
    }
    if recipe["protocol"] == FIXED:
        base = read_json(run / "raw-fixed/shared" / qa["scene_name"] / "manifest.json")
        if [f["camera"] for f in frames] != [f["camera"] for f in base["frames"]]:
            raise ValueError("Fixed camera parameters changed: " + qid)
        if original["timeline_samples"] != recipe["timeline_samples"]:
            raise ValueError("Fixed-camera timeline changed: " + qid)
        if any(original["qc"].get(key) is not True for key in (
                "original_camera_unchanged", "restored_materials", "restored_geometry_visibility", "question_isolation")):
            raise ValueError("Fixed-camera QC failed: " + qid)
        roles = original["roles"]
        if qa["question_type"] == "object_size_estimation":
            roles = [dict(role, roles=[label for label in role["roles"]
                                      if "reference" not in label or "target object" not in role["roles"]])
                     for role in roles]
        if build_prompt(qa, "C", roles) != bundle["prompts"][qid]:
            raise ValueError("Resolved highlights no longer match the frozen C prompt: " + qid)
        manifest["unresolved_roles"] = [r["term"] for r in roles if not r["objects"]]
    directory = run / "renders" / qid
    save_json(directory / "manifest.json", manifest)
    (directory / "prompt.txt").write_text(bundle["prompts"][qid], encoding="utf-8")
    verify_render(bundle, qa, run)


def render(bundle, questions, blends, run, blender, timeout=1800):
    command, version = check_blender(blender)
    check_blends(bundle, questions, blends)
    config = {**identity(bundle, questions), "blender": version, "blends": str(blends.resolve())}
    with run_lock(run):
        freeze(run / "render_config.json", config)
        groups = {}
        for qa in questions:
            if (run / "renders" / qa["qa_id"] / "manifest.json").exists():
                verify_render(bundle, qa, run)
            key = ("fixed", qa["scene_name"]) if bundle["recipes"][qa["qa_id"]]["protocol"] == FIXED else ("legacy", qa["qa_id"])
            groups.setdefault(key, []).append(qa)
        for (kind, key), group in groups.items():
            if all((run / "renders" / q["qa_id"] / "manifest.json").exists() for q in group):
                print("RENDER_CACHED " + key, flush=True)
                continue
            scene = group[0]["scene_name"]
            if kind == "fixed":
                recipe = bundle["recipes"][group[0]["qa_id"]]
                spec = {
                    "scene": scene, "blend_sha256": bundle["assets"]["scenes"][scene]["sha256"],
                    "video_frames": recipe["video_frames"], "output": str((run / "raw-fixed").resolve()),
                    "arms": ["C"], "questions": [
                        {**q, "roles": bundle["recipes"][q["qa_id"]]["roles"]} for q in group],
                }
                spec["render_hash"] = data_hash({"spec": spec, "config": config})
                script = ROOT / "render_fixed.py"
            else:
                spec = dict(bundle["recipes"][key]["spec"], out_dir=str((run / "raw-legacy" / key).resolve()))
                script = ROOT / "render_legacy.py"
            spec_path = run / "specs" / (kind + "_" + key + ".json")
            save_json(spec_path, spec)
            log_path = run / "logs" / (kind + "_" + key + ".log")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            print("RENDER " + key + " -> " + str(log_path), flush=True)
            with log_path.open("w", encoding="utf-8") as log:
                subprocess.run([
                    command, "--background", "--disable-autoexec", str(blend_path(blends, scene).resolve()),
                    "--threads", "2", "--python-exit-code", "1", "--python", str(script),
                    "--", "--spec", str(spec_path.resolve()),
                ], stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=True,
                   env=blender_environment())
            for qa in group:
                raw = (run / "raw-fixed/renders" / scene / qa["qa_id"] / "manifest.json"
                       if kind == "fixed" else run / "raw-legacy" / key / "manifest.json")
                normalize_render(bundle, qa, run, raw)
            if digest(blend_path(blends, scene)) != bundle["assets"]["scenes"][scene]["sha256"]:
                raise ValueError("Source blend changed")


def validate(bundle, questions, run):
    for qa in questions:
        verify_render(bundle, qa, run)
    print("VALIDATED questions=" + str(len(questions)) + " (metadata and hashes; no visual inspection)")


def encode_frames(paths, quality):
    from PIL import Image

    encoded = []
    for path in paths:
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.thumbnail((512, 512), Image.Resampling.LANCZOS)
            stream = io.BytesIO()
            image.save(stream, format="JPEG", quality=quality)
        encoded.append(base64.b64encode(stream.getvalue()).decode("ascii"))
    return encoded


def request_answer(client, model, prompt, images, max_tokens):
    content = [{"type": "text", "text": prompt}]
    content.extend({"type": "image_url", "image_url": {
        "url": "data:image/jpeg;base64," + image, "detail": "high"}} for image in images)
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": content}], max_completion_tokens=max_tokens)
    choice = response.choices[0]
    return {
        "answer_raw": choice.message.content or "", "finish_reason": choice.finish_reason,
        "response_model": response.model, "response_id": response.id,
        "usage": response.usage.model_dump() if response.usage else {},
    }


def append_jsonl(path, row):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def pending_questions(questions, starts, answers):
    ids = {q["qa_id"] for q in questions}
    started = [r["qa_id"] for r in starts]
    finished = [r["qa_id"] for r in answers]
    if len(set(started)) != len(started) or len(set(finished)) != len(finished):
        raise ValueError("Duplicate request IDs in journal")
    if not set(started) <= ids or not set(finished) <= set(started):
        raise ValueError("Journal does not match this run")
    uncertain = set(started) - set(finished)
    if uncertain:
        raise ValueError("Interrupted requests may already be billed; will not resubmit: " + ",".join(sorted(uncertain)))
    return [q for q in questions if q["qa_id"] not in set(finished)]


def answer(bundle, questions, run, model, base_url, max_tokens, workers=4, client=None):
    parsed = urlparse(base_url)
    if parsed.scheme not in {"https", "http"} or not parsed.netloc or parsed.username or parsed.password or parsed.query:
        raise ValueError("API base URL must not contain credentials or a query")
    if not model or workers < 1 or max_tokens < 1:
        raise ValueError("Invalid model, workers or token limit")
    if parsed.scheme == "http":
        print("WARNING: HTTP sends your key and frames without TLS; use a trusted tunnel.")
    config = {
        **identity(bundle, questions), "model": model, "base_url": base_url,
        "max_completion_tokens": max_tokens, "timeout_seconds": 300, "automatic_retries": 0,
        "image_detail": "high", "image_max_side": 512, "jpeg_quality": {"fixed": 80, "legacy": 85},
    }
    with run_lock(run):
        validate(bundle, questions, run)
        freeze(run / "answer_config.json", config)
        starts_path, answers_path = run / "requests.jsonl", run / "answers.jsonl"
        previous = read_jsonl(answers_path)
        pending = pending_questions(questions, read_jsonl(starts_path), previous)
        failures = sum(r["status"] != "ok" for r in previous)
        if failures >= 3:
            raise ValueError("Three recorded failures; investigate before running another experiment")
        if not pending:
            print("ANSWER_CACHED " + str(len(questions)))
            return
        owned = client is None
        if owned:
            from openai import OpenAI

            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                raise ValueError("Set OPENAI_API_KEY (never put it in a tracked file)")
            client = OpenAI(api_key=key, base_url=base_url, timeout=300, max_retries=0)

        def work(qa):
            qid = qa["qa_id"]
            started = time.monotonic()
            record = {"qa_id": qid, "model": model, "sample": 0,
                      "prompt_sha256": bundle["recipes"][qid]["prompt_sha256"]}
            try:
                manifest = verify_render(bundle, qa, run)
                record["frame_sha256"] = [f["sha256"] for f in manifest["frames"]]
                images = encode_frames(frame_paths(manifest), bundle["recipes"][qid]["jpeg_quality"])
                reply = request_answer(client, model, bundle["prompts"][qid], images, max_tokens)
                result = extract_answer(reply["answer_raw"])
                record.update(reply, answer=result,
                              status="ok" if result and reply["finish_reason"] == "stop" else "invalid_response")
            except Exception as exc:
                record.update(status="error", answer="", error_type=type(exc).__name__,
                              http_status=getattr(exc, "status_code", None))
            record["seconds"] = round(time.monotonic() - started, 3)
            return record

        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for offset in range(0, len(pending), workers):
                    futures = []
                    for qa in pending[offset:offset + workers]:
                        append_jsonl(starts_path, {"qa_id": qa["qa_id"], "sample": 0,
                                                  "started_at": datetime.now(timezone.utc).isoformat()})
                        futures.append(pool.submit(work, qa))
                    for future in as_completed(futures):
                        record = future.result()
                        append_jsonl(answers_path, record)
                        failures += record["status"] != "ok"
                        print("ANSWER q" + record["qa_id"] + " " + record["status"], flush=True)
                    if failures >= 3:
                        raise RuntimeError("Three failures; stopping before the next batch (no automatic retries)")
        finally:
            if owned:
                client.close()
    if failures:
        raise RuntimeError("Failed answers are retained, not retried. Run report to score them as incorrect.")


def report(bundle, questions, run):
    config = read_json(run / "answer_config.json")
    if any(config.get(k) != v for k, v in identity(bundle, questions).items()):
        raise ValueError("Report selection or source differs from the frozen answer run")
    rows = read_jsonl(run / "answers.jsonl")
    if len({r["qa_id"] for r in rows}) != len(rows) or not {r["qa_id"] for r in rows} <= set(config["qa_ids"]):
        raise ValueError("Unexpected or duplicate answer IDs")
    # This is the only stage that reads ground-truth values.
    labels = {r["qa_id"]: r["ground_truth"] for r in read_jsonl(ROOT / "data/labels.jsonl")}
    answers = {r["qa_id"]: r for r in rows}
    detailed = []
    for qa in questions:
        row = answers.get(qa["qa_id"], {})
        text = row.get("answer", "")
        ok = row.get("status") == "ok" and bool(text)
        scored = {**qa, "ground_truth": labels[qa["qa_id"]]}
        detailed.append({
            "qa_id": qa["qa_id"], "scene": qa["scene_name"], "type": category(qa),
            "protocol": bundle["recipes"][qa["qa_id"]]["protocol"],
            "status": row.get("status", "missing"), "answer": text,
            "ground_truth": scored["ground_truth"], "correct": bool(ok and semantic_correct(text, scored)),
        })
    summary = {}
    for kind in sorted(EIGHT_TYPES) + ["ALL"]:
        selected = [r for r in detailed if kind == "ALL" or r["type"] == kind]
        n = len(selected)
        summary[kind] = {"n": n, "correct": sum(r["correct"] for r in selected),
                         "valid": sum(r["status"] == "ok" for r in selected)}
        summary[kind]["accuracy"] = summary[kind]["correct"] / n if n else None
    save_json(run / "summary.json", {
        "release": RELEASE, "samples_per_question": 1, "by_type": summary,
        "metric": "BVB pilot semantic accuracy (20% numeric tolerance), not VSI MRA or dual-VQA",
        "reported_tokens": sum((r.get("usage") or {}).get("total_tokens", 0) or 0 for r in rows),
        "complete": len(rows) == len(questions), "missing_and_failed_count_as_incorrect": True,
    })
    with (run / "per_question.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detailed[0]))
        writer.writeheader()
        writer.writerows(detailed)
    lines = ["# BVB-CoT200 Single-Answer Results", "",
             "One request per question; missing/failed answers remain in the denominator.",
             "Development set; not held-out evidence. Not a controlled comparison to historical runs.", "",
             "| Type | Correct | Valid |", "| --- | ---: | ---: |"]
    lines += [f"| {kind} | {s['correct']}/{s['n']} | {s['valid']}/{s['n']} |"
              for kind, s in summary.items() if s["n"]]
    lines += ["", "Metric: BVB pilot semantic accuracy, not official VSI-Bench MRA or dual-VQA.", ""]
    (run / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary["ALL"]))
