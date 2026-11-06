"""Run independent image-question forks with a signed-in Codex app-server."""

import argparse
import hashlib
import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path


def load_manifest(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    images = [(path.parent / image).resolve() for image in data["images"]]
    if not 1 <= len(images) <= 6 or any(not image.is_file() for image in images):
        raise ValueError("Provide one to six existing images")
    questions = data["questions"]
    if not questions or any(not isinstance(q.get("id"), (str, int)) or not isinstance(q.get("question"), str)
                            or not q["question"].strip() for q in questions):
        raise ValueError("Each question needs an id and a nonempty question")
    ids = [str(q["id"]) for q in questions]
    if len(set(ids)) != len(ids):
        raise ValueError("Question ids must be unique")
    return images, questions


class AppServer:
    def __init__(self, executable, model, cwd):
        command = codex_command(executable)
        self.process = subprocess.Popen(command + ["app-server", "-c", f'model="{model}"'],
            cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", errors="replace", bufsize=1)
        self.events = queue.Queue()
        self.pending = []
        self.sequence = 0
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.process.stdout:
            try:
                self.events.put(json.loads(line))
            except json.JSONDecodeError:
                continue

    def _send(self, message):
        self.process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        self.process.stdin.flush()

    def wait(self, predicate, timeout=120):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for item in list(self.pending):
                if predicate(item):
                    self.pending.remove(item)
                    return item
            try:
                item = self.events.get(timeout=1)
            except queue.Empty:
                if self.process.poll() is not None:
                    raise RuntimeError("app_server_exited")
                continue
            if predicate(item):
                return item
            self.pending.append(item)
        raise TimeoutError("app_server_response_timeout")

    def call(self, method, params):
        self.sequence += 1
        self._send({"jsonrpc": "2.0", "id": self.sequence, "method": method, "params": params})
        response = self.wait(lambda item: item.get("id") == self.sequence)
        if "error" in response:
            raise RuntimeError(f"{method}_error_{response['error'].get('code', 'unknown')}")
        return response["result"]

    def turn(self, thread_id, content):
        start = time.monotonic()
        response = self.call("turn/start", {"threadId": thread_id, "input": content, "effort": "low"})
        turn_id = response["turn"]["id"]
        done = self.wait(lambda x: x.get("method") == "turn/completed" and
                         x.get("params", {}).get("turn", {}).get("id") == turn_id)
        status = done["params"]["turn"].get("status")
        try:
            usage_event = self.wait(lambda x: x.get("method") == "thread/tokenUsage/updated" and
                                    x.get("params", {}).get("turnId") == turn_id, timeout=4)
            usage = usage_event["params"]["tokenUsage"]["last"]
        except TimeoutError:
            usage = None
        answers = []
        for event in self.pending:
            params = event.get("params", {})
            item = params.get("item", {})
            if event.get("method") == "item/completed" and params.get("turnId") == turn_id and item.get("type") == "agentMessage":
                answers.append(item.get("text", ""))
        return {"status": status, "answer": "\n".join(answers), "usage": usage,
                "seconds": round(time.monotonic() - start, 2)}

    def close(self):
        if self.process.stdin:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=5)


def codex_command(executable):
    if executable.endswith(".js"):
        node = shutil.which("node")
        if not node:
            raise RuntimeError("node_not_found")
        return [node, executable]
    return [executable]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="JSON file with images and questions; do not include ground truth")
    parser.add_argument("--output", type=Path, required=True, help="New result JSON path")
    parser.add_argument("--codex", default="codex", help="Codex executable (0.167.0 was tested)")
    parser.add_argument("--model", default="gpt-6.1-sol")
    parser.add_argument("--limit", type=int, default=2, help="Maximum number of questions; default 2")
    parser.add_argument("--validate-only", action="store_true", help="Check inputs without starting Codex")
    args = parser.parse_args()
    if args.limit < 1 or args.output.exists() or not re.fullmatch(r"[A-Za-z0-9._-]+", args.model):
        parser.error("--limit must be positive, --output must be new, and --model must be a model ID")
    manifest = args.manifest.resolve()
    images, questions = load_manifest(manifest)
    if args.validate_only:
        print(f"Valid: {len(images)} images, {len(questions)} questions; no model calls")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"model_requested": args.model, "manifest": str(manifest), "mode": "prediction_fork_experimental",
              "images": [{"path": str(image), "sha256": hashlib.sha256(image.read_bytes()).hexdigest()} for image in images],
              "root": None, "questions": [], "errors": []}

    def save():
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, args.output)

    server = None
    try:
        version = subprocess.run(codex_command(args.codex) + ["--version"], capture_output=True,
                                 text=True, encoding="utf-8", errors="replace", timeout=10)
        match = re.search(r"(\d+)\.(\d+)\.(\d+)", version.stdout)
        if version.returncode != 0 or not match or tuple(map(int, match.groups())) < (0, 167, 0):
            raise RuntimeError("codex_0.167.0_or_newer_required")
        result["codex_version"] = match.group(0)
        server = AppServer(args.codex, args.model, str(manifest.parent))
        server.call("initialize", {"clientInfo": {"name": "subscription_fork_qa", "version": "0.1"},
                                   "capabilities": {"experimentalApi": True}})
        server._send({"jsonrpc": "2.0", "method": "initialized", "params": {}})
        parent = server.call("thread/start", {"model": args.model, "cwd": str(manifest.parent),
            "approvalPolicy": "never", "sandbox": "read-only",
            "baseInstructions": "Answer the user's question. Do not use tools."})["thread"]["id"]
        content = [{"type": "localImage", "path": str(image)} for image in images]
        content.append({"type": "text", "text": "Remember these images for later questions. Reply READY."})
        result["root"] = server.turn(parent, content)
        save()
        if result["root"]["status"] != "completed":
            raise RuntimeError("parent_turn_failed")
        for question in questions[:args.limit]:
            record = {"id": str(question["id"]), "question": question["question"], "status": "started"}
            result["questions"].append(record)
            save()
            child = server.call("thread/fork", {"threadId": parent, "experimentalPredictionMode": True,
                "ephemeral": True, "excludeTurns": True})["thread"]["id"]
            record.update(server.turn(child, [{"type": "text", "text": question["question"]}]))
            save()
            if record["status"] != "completed":
                raise RuntimeError("question_turn_failed")
    except (KeyError, OSError, RuntimeError, TimeoutError, ValueError, subprocess.TimeoutExpired) as exc:
        result["errors"].append({"type": type(exc).__name__, "message": str(exc)[:200], "usage": "unknown"})
        save()
        print("Stopped; see result JSON for status. No automatic retry.")
        return 1
    finally:
        if server:
            server.close()
    print(f"Completed {len(result['questions'])} questions; results saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
