"""Local, deliberately limited Chat Completions adapter for Codex subscription access."""

import argparse
import base64
import binascii
import hmac
import json
import os
import re
import secrets
import tempfile
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from .run_fork_qa import AppServer, check_codex_version

MAX_BODY = 32 * 1024 * 1024
ALLOWED = {"model", "messages", "reasoning_effort", "response_format", "max_completion_tokens", "max_tokens", "stream", "store", "n"}


class InputError(ValueError):
    pass


def convert_messages(messages, image_dir, allow_system=True):
    if not isinstance(messages, list) or not messages:
        raise InputError("messages must be a nonempty array")
    system, user, seen_user = [], [], False
    for message in messages:
        if not isinstance(message, dict) or set(message) - {"role", "content"}:
            raise InputError("unsupported message field")
        role, content = message.get("role"), message.get("content")
        if role in ("system", "developer"):
            if seen_user or not allow_system or not isinstance(content, str):
                raise InputError("system messages must be text before the user message")
            system.append(content)
            continue
        if role != "user" or seen_user:
            raise InputError("exactly one user message is supported")
        seen_user = True
        parts = [{"type": "text", "text": content}] if isinstance(content, str) else content
        if not isinstance(parts, list) or not parts:
            raise InputError("user content must contain text or images")
        for part in parts:
            if not isinstance(part, dict):
                raise InputError("invalid content part")
            if part.get("type") == "text" and set(part) == {"type", "text"} and isinstance(part["text"], str):
                user.append({"type": "text", "text": part["text"]})
            elif part.get("type") == "image_url" and set(part) == {"type", "image_url"}:
                obj = part["image_url"]
                if not isinstance(obj, dict) or set(obj) - {"url", "detail"}:
                    raise InputError("invalid image_url")
                url = obj.get("url", "")
                match = re.fullmatch(r"data:(image/(?:png|jpeg|webp|gif));base64,([A-Za-z0-9+/=]+)", url)
                if not match:
                    raise InputError("only base64 image data URLs are supported")
                try:
                    decoded = base64.b64decode(match.group(2), validate=True)
                except (ValueError, binascii.Error):
                    raise InputError("invalid image base64") from None
                if not decoded or len(decoded) > 16 * 1024 * 1024:
                    raise InputError("image must be between 1 byte and 16 MiB")
                detail = obj.get("detail", "auto")
                if detail not in ("auto", "low", "high", "original"):
                    raise InputError("unsupported image detail")
                suffix = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp", "image/gif": ".gif"}[match.group(1)]
                image_path = Path(image_dir) / (secrets.token_hex(16) + suffix)
                image_path.write_bytes(decoded)
                user.append({"type": "localImage", "path": str(image_path), "detail": detail})
            else:
                raise InputError("unsupported content part")
    if not seen_user:
        raise InputError("a user message is required")
    return "\n\n".join(system), user


def validate_request(body, allow_uncapped):
    if not isinstance(body, dict) or set(body) - ALLOWED:
        raise InputError("request includes unsupported parameters")
    model = body.get("model")
    if not isinstance(model, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", model):
        raise InputError("model must be a model ID")
    if body.get("stream", False) is not False or body.get("store", False) is not False or body.get("n", 1) != 1:
        raise InputError("streaming, store=true and n != 1 are unsupported")
    effort = body.get("reasoning_effort") or "low"
    if effort not in ("minimal", "low", "medium", "high", "xhigh"):
        raise InputError("unsupported reasoning_effort")
    limit_present = "max_tokens" in body or "max_completion_tokens" in body
    if limit_present and not allow_uncapped:
        raise InputError("Codex cannot enforce the requested output-token cap; restart with --allow-uncapped-output to accept")
    for field in ("max_tokens", "max_completion_tokens"):
        if field in body and (type(body[field]) is not int or body[field] < 1):
            raise InputError("output-token limits must be positive integers")
    response_format = body.get("response_format") or {"type": "text"}
    if not isinstance(response_format, dict):
        raise InputError("invalid response_format")
    kind = response_format.get("type")
    if kind == "text" and set(response_format) == {"type"}:
        schema = None
    elif kind == "json_schema" and set(response_format) == {"type", "json_schema"}:
        spec = response_format.get("json_schema")
        if not isinstance(spec, dict) or not isinstance(spec.get("schema"), dict):
            raise InputError("json_schema.schema must be an object")
        schema = spec["schema"]
    else:
        raise InputError("only text and json_schema response formats are supported")
    return model, effort, schema, limit_present


def usage_object(usage):
    if not usage:
        return None
    return {"prompt_tokens": usage.get("inputTokens"), "completion_tokens": usage.get("outputTokens"),
            "total_tokens": usage.get("totalTokens"),
            "prompt_tokens_details": {"cached_tokens": usage.get("cachedInputTokens")}}


class Proxy:
    def __init__(self, backend, model_turn_limit, allow_uncapped):
        self.backend, self.limit, self.allow_uncapped = backend, model_turn_limit, allow_uncapped
        self.turns = 0
        self.sessions = {}

    def close(self):
        for session in self.sessions.values():
            session["images"].cleanup()
        self.sessions.clear()

    def turn(self, thread_id, content, effort, schema):
        if self.turns >= self.limit:
            raise InputError("configured model-turn limit reached")
        self.turns += 1  # Count attempts, including failures with unknown consumption.
        result = self.backend.turn(thread_id, content, effort=effort, output_schema=schema)
        if result.get("status") != "completed" or not result.get("answer", "").strip():
            raise RuntimeError("model_turn_failed_or_empty; usage may be unknown")
        if schema is not None:
            try:
                json.loads(result["answer"])
            except ValueError:
                raise RuntimeError("model_returned_invalid_json; usage was consumed") from None
        return result

    def start(self, model, system):
        result = self.backend.call("thread/start", {"model": model, "approvalPolicy": "never", "sandbox": "read-only",
            "baseInstructions": system or "Answer the user. Do not use tools."})
        return result["thread"]["id"]

    def complete(self, body, session_id=None):
        model, effort, schema, uncapped = validate_request(body, self.allow_uncapped)
        with tempfile.TemporaryDirectory(prefix="subscription-input-") as image_dir:
            system, content = convert_messages(body.get("messages"), image_dir, allow_system=session_id is None)
            if session_id is None:
                thread = self.start(model, system)
            else:
                session = self.sessions.get(session_id)
                if session is None:
                    raise KeyError("session_not_found")
                if model != session["model"]:
                    raise InputError("fork model must match its parent")
                thread = self.backend.call("thread/fork", {"threadId": session["parent"],
                    "experimentalPredictionMode": True, "ephemeral": True, "excludeTurns": True})["thread"]["id"]
            answer = self.turn(thread, content, effort, schema)
        return {"id": "chatcmpl-local-" + secrets.token_hex(12), "object": "chat.completion", "created": int(time.time()),
                "model": model, "choices": [{"index": 0, "message": {"role": "assistant", "content": answer["answer"]},
                                               "finish_reason": "stop"}], "usage": usage_object(answer.get("usage")),
                "bvb_meta": {"output_token_cap_enforced": False if uncapped else None,
                             "returned_model_verified": False, "mode": "fork" if session_id else "standalone"}}

    def create_session(self, body):
        model, effort, schema, uncapped = validate_request(body, self.allow_uncapped)
        if schema is not None:
            raise InputError("parent context must use text response format")
        images = tempfile.TemporaryDirectory(prefix="subscription-parent-")
        try:
            system, content = convert_messages(body.get("messages"), images.name)
            thread = self.start(model, system)
            content.append({"type": "text", "text": "Remember the shared context above for independent later questions. Reply READY."})
            answer = self.turn(thread, content, effort, None)
            ident = secrets.token_urlsafe(24)
            self.sessions[ident] = {"parent": thread, "model": model, "images": images}
        except Exception:
            images.cleanup()
            raise
        return {"id": ident, "object": "bvb.fork_session", "model": model,
                "parent_usage": usage_object(answer.get("usage")),
                "bvb_meta": {"output_token_cap_enforced": False if uncapped else None}}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, status, payload):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def authenticated(self):
        supplied = self.headers.get("Authorization", "")
        if not hmac.compare_digest(supplied, "Bearer " + self.server.local_key):
            self.reply(401, {"error": {"message": "Unauthorized", "type": "authentication_error"}})
            return False
        return True

    def do_GET(self):
        if not self.authenticated():
            return
        if self.path == "/v1/models":
            self.reply(200, {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "requested-not-verified"}
                                                         for m in self.server.models]})
        else:
            self.reply(404, {"error": {"message": "Not found", "type": "invalid_request_error"}})

    def do_POST(self):
        if not self.authenticated():
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= MAX_BODY:
                raise InputError("invalid request size")
            body = json.loads(self.rfile.read(size))
            match = re.fullmatch(r"/v1/fork/sessions/([A-Za-z0-9_-]+)/chat/completions", self.path)
            if self.path == "/v1/chat/completions":
                result = self.server.proxy.complete(body)
            elif self.path == "/v1/fork/sessions":
                result = self.server.proxy.create_session(body)
            elif match:
                result = self.server.proxy.complete(body, match.group(1))
            else:
                self.reply(404, {"error": {"message": "Not found", "type": "invalid_request_error"}})
                return
            self.reply(200, result)
        except KeyError:
            self.reply(404, {"error": {"message": "Session not found", "type": "invalid_request_error"}})
        except InputError as exc:
            self.reply(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
        except (ValueError, TypeError):
            self.reply(400, {"error": {"message": "Invalid or unsupported request; check proxy documentation", "type": "invalid_request_error"}})
        except (RuntimeError, TimeoutError, OSError):
            self.reply(502, {"error": {"message": "Model operation failed; usage may be unknown; no automatic retry", "type": "server_error"}})

    def do_DELETE(self):
        if not self.authenticated():
            return
        match = re.fullmatch(r"/v1/fork/sessions/([A-Za-z0-9_-]+)", self.path)
        session = self.server.proxy.sessions.pop(match.group(1), None) if match else None
        if session is None:
            self.reply(404, {"error": {"message": "Session not found", "type": "invalid_request_error"}})
            return
        session["images"].cleanup()
        self.reply(200, {"deleted": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18317)
    parser.add_argument("--codex", default="codex")
    parser.add_argument("--models", nargs="+", default=["gpt-6.1-sol", "gpt-6-astra"])
    parser.add_argument("--max-model-turns", type=int, default=10)
    parser.add_argument("--allow-uncapped-output", action="store_true")
    args = parser.parse_args()
    key = os.environ.get("BVB_PROXY_API_KEY", "")
    if len(key) < 24 or args.max_model_turns < 1:
        parser.error("BVB_PROXY_API_KEY must contain at least 24 characters and --max-model-turns must be positive")
    check_codex_version(args.codex)
    with tempfile.TemporaryDirectory(prefix="subscription-proxy-") as cwd:
        backend = AppServer(args.codex, args.models[0], cwd)
        proxy = Proxy(backend, args.max_model_turns, args.allow_uncapped_output)
        try:
            backend.call("initialize", {"clientInfo": {"name": "subscription_fork_proxy", "version": "0.1"},
                "capabilities": {"experimentalApi": True}})
            backend._send({"jsonrpc": "2.0", "method": "initialized", "params": {}})
            with HTTPServer(("127.0.0.1", args.port), Handler) as server:
                server.local_key, server.models = key, args.models
                server.proxy = proxy
                print(f"Local subscription proxy listening on 127.0.0.1:{args.port}; no external access", flush=True)
                try:
                    server.serve_forever()
                except KeyboardInterrupt:
                    pass
        finally:
            proxy.close()
            backend.close()


if __name__ == "__main__":
    main()
