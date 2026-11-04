"""Prepare and submit bounded regional reconstructions with the V5 asset protocol."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[2]
SENIOR = ROOT / "senior"
CACHE = ROOT.parent / "code-selected-reconstruction-20261103" / "inputs"
sys.path[:0] = [str(SENIOR / "scripts"), str(SENIOR / "scripts" / "assets")]
from compact_pose_input import build_transport
from local_context_scene import expand, prompt_for


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shared_pose(poses, reference_cameras):
    pose_path = poses / "camera-poses.npz"
    size_path = poses / "geometry-info.json"
    width, height = read(size_path)["image_size_wh"] if size_path.is_file() else (518, 392)
    with np.load(pose_path) as data:
        c2w = data["c2w"].copy()
        intrinsics = np.diag([640 / width, 480 / height, 1]) @ data["intrinsics"]
    up = -c2w[:, :3, 1].mean(axis=0)
    up /= np.linalg.norm(up)
    right = c2w[0, :3, 0] - up * np.dot(c2w[0, :3, 0], up)
    right /= np.linalg.norm(right)
    rotation = np.stack([right, np.cross(up, right), up])
    scale = 1.
    translation = -rotation @ c2w[0, :3, 3]
    source = {"camera-poses.npz": sha(pose_path), "sampling.json": sha(poses / "sampling.json"),
              "coordinate_policy": "one common Z-up LingBot frame, first camera origin, native scale"}
    if reference_cameras:
        samples = read(poses / "sampling.json")["frames"]
        times = np.array([row["timestamp_s"] for row in samples])
        target = [row for row in read(reference_cameras) if times[0] <= row["t"] <= times[-1]]
        indices = np.argmin(abs(times[:, None] - [row["t"] for row in target]), axis=0)
        assert np.max(abs(times[indices] - [row["t"] for row in target])) <= .15
        a, b = c2w[indices, :3, 3], np.asarray([row["p"] for row in target])
        ac, bc = a - a.mean(axis=0), b - b.mean(axis=0)
        u, s, vt = np.linalg.svd(ac.T @ bc)
        sign = np.diag([1., 1., np.linalg.det(vt.T @ u.T)])
        rotation = vt.T @ sign @ u.T
        scale = float((s * np.diag(sign)).sum() / np.square(ac).sum())
        translation = b.mean(axis=0) - scale * (rotation @ a.mean(axis=0))
        errors = (scale * a @ rotation.T + translation) - b
        source.update(coordinate_policy="single Sim(3) fit to existing BVB camera centers",
                      reference_sha256=sha(reference_cameras), reference_samples=len(target),
                      scale=scale, rotation=rotation.tolist(), translation=translation.tolist(),
                      center_rmse=float(np.sqrt(np.square(errors).sum(axis=1).mean())))
    c2w[:, :3, 3] = scale * c2w[:, :3, 3] @ rotation.T + translation
    c2w[:, :3, :3] = rotation @ c2w[:, :3, :3] @ np.diag([1, -1, -1])
    return c2w, intrinsics, source


def observations(row):
    return row.get("observations") or ([{"object": row["object"], "box_xyxy": row.get("box_xyxy"),
                                         "status": row.get("status")}]
                                       if row.get("object") else [])


def pick(rows, candidates, c2w, budget):
    if len(rows) <= budget:
        return rows
    positions = np.array([c2w[candidates[r["source_frame_index"]]["index"], :3, 3] for r in rows])
    directions = np.array([c2w[candidates[r["source_frame_index"]]["index"], :3, 2] for r in rows])
    distances = np.linalg.norm(positions[:, None] - positions, axis=-1)
    distances /= max(float(distances.max()), 1e-8)
    distances += np.arccos(np.clip(directions @ directions.T, -1, 1)) / np.pi
    classes = [{d["object"] for d in observations(row) if d.get("box_xyxy")} for row in rows]
    chosen, covered = [], set()
    for _ in range(budget):
        i = max((i for i in range(len(rows)) if i not in chosen),
                key=lambda i: (len(classes[i] - covered),
                               min(distances[i, chosen]) if chosen else float(bool(classes[i])),
                               candidates[rows[i]["source_frame_index"]]["sharpness"]))
        chosen.append(i)
        covered.update(classes[i])
    return [rows[i] for i in sorted(chosen)]


def prepare(out, question, groups, inputs, frame_cap, reference_cameras, *, cache=None, poses=None):
    cache = cache or inputs / "cache"
    poses = poses or inputs / "lingbot"
    candidates = {r["source_frame_index"]: r for r in read(cache / "candidates.json")}
    sampling = {r["source_frame_index"]: r for r in read(poses / "sampling.json")["frames"]}
    c2w, intrinsics, source = shared_pose(poses, reference_cameras)
    palette = [[.85, .08, .055, 1], [.05, .48, .15, 1], [.06, .30, .90, 1], [.95, .43, .04, 1]]
    colors = {obj: palette[i % len(palette)] for i, obj in enumerate(question["objects"])}
    colors["scene context"] = [.60, .60, .60, 1]
    out.mkdir(parents=True, exist_ok=False)
    catalog = []
    for group in groups:
        gid = group["group_id"]
        classes = list(group["objects"] if "objects" in group else [group["object"]])
        scoped = dict(question, objects=classes, roles=question.get("roles", {}))
        folder = out / gid
        folder.mkdir()
        frames, refs = [], []
        selected = [dict(r, object=r.get("object", group.get("object"))) for r in group["selected"]]
        for row in sorted(pick(selected, candidates, c2w, frame_cap),
                          key=lambda row: row["source_frame_index"]):
            fid = row["source_frame_index"]
            candidate, sample = candidates[fid], sampling[fid]
            path = (cache / candidate["frame"]).resolve()
            assert sha(path) == candidate["frame_sha256"] == sample["sha256"] == row["frame_sha256"]
            i = candidate["index"]
            assert i == sample["sample_index"] and row["timestamp_s"] == sample["timestamp_s"]
            frames.append({"path": str(path), "sha256": sample["sha256"],
                           "source_frame_index": fid, "timestamp_s": sample["timestamp_s"],
                           "official_query_frame": bool(row.get("official_query_frame")),
                           "observations": observations(row)})
            refs.append(dict(source_frame_index=fid, timestamp_s=sample["timestamp_s"],
                             c2w_blender=c2w[i].tolist(), intrinsics_640x480=intrinsics[i].tolist()))
        reference = dict(rows=refs, source_hashes=source)
        transport = build_transport(reference)
        prompt = prompt_for(scoped, walls_only=True)
        prompt = prompt.replace("Use only the observed first 90 seconds.",
                                "Use only the supplied timestamped frames.")
        prompt = prompt.replace("Coordinates: meters", "Coordinates: approximate scene units")
        if not reference_cameras:
            prompt = prompt.replace("Use z=0 as the implicit floor reference",
                                    "Infer the floor level from the inputs")
        prompt += ("\nReconstruct this region in the shared Z-up world coordinates supplied with every image. "
                   "Do not independently recenter, rotate or mirror the scene. Poses are predicted and the "
                   "units are approximate. Use the visible evidence to place targets and walls. "
                   "Detector boxes are provisional hints. Reject false boxes, deduplicate repeated "
                   "observations, and leave out target categories you cannot see. Return only the complete "
                   "structure, targets, context JSON. Keep the final JSON compact.\n")
        prompt += transport["explanation"]
        write(folder / "question.json", scoped)
        write(folder / "input-frames.json", frames)
        write(folder / "pose-reference.json", reference)
        write(folder / "pose-reference-vlm.json", transport)
        write(folder / "experiment.json", dict(camera_pose_reference=True, pose_input_format="compact",
              source="senior V5 prompt/schema/asset compiler, Chat Completions gateway",
              alignment=source["coordinate_policy"]))
        (folder / "system-prompt.txt").write_text(prompt, encoding="utf-8")
        catalog.append(dict(group_id=gid, object=classes[0] if classes else "scene context",
                            objects=classes, folder=gid, frames=len(frames)))
    write(out / "manifest.json", dict(question=question, groups=catalog, colors=colors, frame_cap=frame_cap))


def submit(folder, key, max_tokens, attempt, api_alias, reasoning):
    from openai import OpenAI

    frames = read(folder / "input-frames.json")
    pose = read(folder / "pose-reference-vlm.json")
    prompt = (folder / "system-prompt.txt").read_text(encoding="utf-8")
    content = [{"type": "text", "text": prompt}]
    for frame, camera, mapping in zip(frames, pose["rows"], pose["frame_mapping"], strict=True):
        path = Path(frame["path"])
        assert sha(path) == frame["sha256"]
        url = "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")
        content += [{"type": "text", "text": f'Frame {mapping["timestamp_s"]:.2f}s'},
                    {"type": "image_url", "image_url": {"detail": "high", "url": url}},
                    {"type": "text", "text": json.dumps(camera, separators=(",", ":"))}]
        hints = [{k: d.get(k) for k in ("object", "box_xyxy", "status")}
                 for d in frame["observations"] if d.get("box_xyxy")]
        if hints:
            content.append({"type": "text", "text": "Detector hints: " + json.dumps(hints, separators=(",", ":"))})
    assets = folder / "assets"
    assets.mkdir(exist_ok=True)
    ledger = assets / f"request{attempt}.json"
    record = dict(model="gpt-6-astra", reasoning_effort=reasoning, max_completion_tokens=max_tokens,
                  api_alias=api_alias,
                  frame_count=len(frames), status="started", usage=None,
                  prompt_sha256=sha(folder / "system-prompt.txt"),
                  pose_sha256=sha(folder / "pose-reference-vlm.json"))
    with ledger.open("x", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    started = time.monotonic()
    try:
        client = OpenAI(api_key=key, base_url="http://35.220.164.252:3888/v1", timeout=150, max_retries=0)
        response = client.chat.completions.create(model=record["model"], reasoning_effort=record["reasoning_effort"],
                  max_completion_tokens=max_tokens, messages=[{"role": "user", "content": content}])
        choice = response.choices[0]
        reply = (choice.message.content or "").strip()
        response_path = assets / f"response{attempt}.txt"
        response_path.write_text(reply, encoding="utf-8")
        record.update(status="received", usage=response.usage.model_dump() if response.usage else None,
                      returned_model=response.model, finish_reason=choice.finish_reason,
                      response_file=response_path.name)
        if reply.startswith("```") and reply.endswith("```"):
            reply = reply.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        spec, audit = expand(json.loads(reply), read(folder / "question.json"), walls_only=True)
        write(assets / "scene-spec.json", spec)
        write(assets / "question-audit.json", audit)
        record.update(status="schema_valid", missing_keywords=audit["missing_keywords"])
    except Exception as exc:
        record.update(status="invalid_reply" if record["status"] == "received" else "request_failed",
                      error_type=type(exc).__name__, http_status=getattr(exc, "status_code", None))
        if isinstance(getattr(exc, "body", None), dict):
            detail = exc.body.get("error", exc.body)
            if isinstance(detail, dict) and "message" in detail:
                record["provider_message"] = str(detail["message"]).replace(key, "[REDACTED]")[:300]
        if record["status"] == "invalid_reply":
            record["validation_error"] = str(exc)
    finally:
        record["elapsed_s"] = time.monotonic() - started
        write(ledger, record)
    return {"group": folder.name, "status": record["status"], "usage": record["usage"],
            "http_status": record.get("http_status"),
            "missing_keywords": record.get("missing_keywords")}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--qid", type=str)
    p.add_argument("--inputs", type=Path, default=CACHE)
    p.add_argument("--groups", type=Path)
    p.add_argument("--arm", default="geometry")
    p.add_argument("--frame-cap", type=int, default=8)
    p.add_argument("--reference-cameras", type=Path)
    p.add_argument("--submit", action="store_true")
    p.add_argument("--limit", type=int, default=3)
    p.add_argument("--workers", type=int, choices=[1, 2], default=1)
    p.add_argument("--max-tokens", type=int, default=3600)
    p.add_argument("--api", choices=["API1", "API3"], default="API3")
    p.add_argument("--reasoning", choices=["low", "medium", "high"], default="low")
    p.add_argument("--retry-403", action="store_true")
    p.add_argument("--attempt-id", default="403-retry", help="New ledger suffix for a confirmed retry")
    p.add_argument("--group")
    a = p.parse_args()
    if not a.submit:
        row = read(a.groups)["questions"][a.qid]
        prepare(a.out, row["question"], row["arms"][a.arm], a.inputs, a.frame_cap, a.reference_cameras)
        return
    key = next(line.partition("=")[2].strip().strip("'\"")
               for line in (WORKSPACE / ".env").read_text(encoding="utf-8").splitlines()
               if line.startswith(a.api + "="))
    folders = [a.out / group["folder"] for group in read(a.out / "manifest.json")["groups"]]
    if a.group:
        folders = [f for f in folders if f.name == a.group]
    if not re.fullmatch(r"[a-zA-Z0-9-]+", a.attempt_id):
        p.error("--attempt-id must contain only letters, numbers and hyphens")
    attempt = "-" + a.attempt_id if a.retry_403 else ""
    folders = [f for f in folders if not (f / "assets" / f"request{attempt}.json").exists()]
    folders = [f for f in folders if not (f / "assets" / "scene-spec.json").exists()]
    if a.retry_403:
        folders = [f for f in folders if (f / "assets" / "request.json").exists() and
                   read(f / "assets" / "request.json").get("http_status") == 403]
    if len(folders) > a.limit:
        p.error(f"{len(folders)} pending calls exceed explicit limit {a.limit}")
    with ThreadPoolExecutor(a.workers) as pool:
        for start in range(0, len(folders), a.workers):
            batch = folders[start:start + a.workers]
            rows = list(pool.map(lambda folder: submit(folder, key, a.max_tokens, attempt, a.api, a.reasoning), batch))
            for row in rows:
                print(json.dumps(row, ensure_ascii=False), flush=True)
            if any(row["http_status"] in (401, 403, 429) for row in rows):
                break


if __name__ == "__main__":
    main()
