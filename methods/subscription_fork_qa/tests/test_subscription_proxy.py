import base64
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import HTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from subscription_proxy import Handler, Proxy


class FakeBackend:
    def __init__(self):
        self.calls = []
        self.threads = 0

    def call(self, method, params):
        self.calls.append((method, params))
        self.threads += 1
        return {"thread": {"id": "thread-" + str(self.threads)}}

    def turn(self, thread_id, content, effort="low", output_schema=None):
        self.calls.append(("turn", {"thread": thread_id, "content": content, "effort": effort, "schema": output_schema}))
        return {"status": "completed", "answer": '{"ok":true}' if output_schema else "READY",
                "usage": {"inputTokens": 20, "outputTokens": 3, "totalTokens": 23, "cachedInputTokens": 10}}


class ProxyEndpointsTest(unittest.TestCase):
    def setUp(self):
        self.backend = FakeBackend()
        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.server.local_key = "test-secret-at-least-24-characters"
        self.server.models = ["gpt-6.1-sol"]
        self.server.proxy = Proxy(self.backend, 10, allow_uncapped=False)
        self.base = "http://127.0.0.1:" + str(self.server.server_port)
        self.worker = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.worker.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.server.proxy.close()
        self.worker.join()

    def request(self, method, path, body=None, key=None):
        content = None if body is None else json.dumps(body).encode()
        headers = {"Authorization": "Bearer " + (key or self.server.local_key), "Content-Type": "application/json"}
        try:
            with urlopen(Request(self.base + path, data=content, headers=headers, method=method), timeout=5) as response:
                return response.status, json.load(response)
        except HTTPError as exc:
            return exc.code, json.load(exc)

    def message(self, text="question"):
        return {"model": "gpt-6.1-sol", "messages": [{"role": "user", "content": text}]}

    def test_models_requires_auth_and_lists_configured_models(self):
        self.assertEqual(self.request("GET", "/v1/models", key="wrong")[0], 401)
        status, result = self.request("GET", "/v1/models")
        self.assertEqual((status, result["data"][0]["id"]), (200, "gpt-6.1-sol"))

    def test_standalone_chat_preserves_order_and_reports_usage(self):
        image = "data:image/png;base64," + base64.b64encode(b"image").decode()
        body = {"model": "gpt-6.1-sol", "messages": [{"role": "system", "content": "system"},
            {"role": "user", "content": [{"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": image, "detail": "high"}}]}]}
        status, result = self.request("POST", "/v1/chat/completions", body)
        self.assertEqual(status, 200)
        self.assertEqual(result["usage"]["prompt_tokens_details"]["cached_tokens"], 10)
        turn = self.backend.calls[-1][1]
        self.assertEqual([item["type"] for item in turn["content"]], ["text", "localImage"])
        self.assertEqual(turn["content"][1]["detail"], "high")

    def test_independent_forks_and_delete(self):
        status, parent = self.request("POST", "/v1/fork/sessions", self.message("shared"))
        self.assertEqual(status, 200)
        ident = parent["id"]
        path = "/v1/fork/sessions/" + ident + "/chat/completions"
        self.assertEqual(self.request("POST", path, self.message("first"))[0], 200)
        self.assertEqual(self.request("POST", path, self.message("second"))[0], 200)
        forks = [params for method, params in self.backend.calls if method == "thread/fork"]
        self.assertEqual([x["threadId"] for x in forks], ["thread-1", "thread-1"])
        self.assertEqual(self.request("DELETE", "/v1/fork/sessions/" + ident)[0], 200)
        self.assertEqual(self.request("POST", path, self.message())[0], 404)

    def test_unsupported_options_fail_before_inference(self):
        for option in ({"stream": True}, {"tools": []}, {"max_completion_tokens": 10}):
            body = self.message()
            body.update(option)
            self.assertEqual(self.request("POST", "/v1/chat/completions", body)[0], 400)
        self.assertEqual(self.backend.calls, [])

    def test_output_schema_and_explicit_uncapped_opt_in(self):
        self.server.proxy.allow_uncapped = True
        body = self.message()
        body.update(max_completion_tokens=10, response_format={"type": "json_schema", "json_schema":
            {"name": "result", "strict": True, "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}}}})
        status, result = self.request("POST", "/v1/chat/completions", body)
        self.assertEqual(status, 200)
        self.assertFalse(result["bvb_meta"]["output_token_cap_enforced"])
        self.assertEqual(self.backend.calls[-1][1]["schema"]["type"], "object")

    def test_openai_sdk_can_call_standalone_and_fork(self):
        try:
            from openai import OpenAI
        except ImportError:
            self.skipTest("OpenAI SDK not installed")
        client = OpenAI(api_key=self.server.local_key, base_url=self.base + "/v1", max_retries=0)
        answer = client.chat.completions.create(model="gpt-6.1-sol", messages=[{"role": "user", "content": "hello"}])
        self.assertEqual(answer.choices[0].message.content, "READY")
        _, session = self.request("POST", "/v1/fork/sessions", self.message("shared"))
        fork_client = OpenAI(api_key=self.server.local_key, base_url=self.base + "/v1/fork/sessions/" + session["id"], max_retries=0)
        answer = fork_client.chat.completions.create(model="gpt-6.1-sol", messages=[{"role": "user", "content": "question"}])
        self.assertEqual(answer.choices[0].message.content, "READY")

    def test_end_to_end_openai_sdk_to_app_server_protocol(self):
        try:
            from openai import OpenAI
        except ImportError:
            self.skipTest("OpenAI SDK not installed")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        with tempfile.TemporaryDirectory() as directory:
            env = os.environ.copy()
            env["BVB_PROXY_API_KEY"] = self.server.local_key
            env["BVB_TEST_TRACE"] = str(Path(directory) / "trace.jsonl")
            script = Path(__file__).resolve().parents[1] / "subscription_proxy.py"
            fake = Path(__file__).resolve().parent / "fake_app_server.js"
            process = subprocess.Popen([sys.executable, str(script), "--codex", str(fake), "--port", str(port),
                "--allow-uncapped-output"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                base = f"http://127.0.0.1:{port}/v1"
                for _ in range(100):
                    try:
                        with urlopen(Request(base + "/models", headers={"Authorization": "Bearer " + self.server.local_key}), timeout=1):
                            break
                    except OSError:
                        if process.poll() is not None:
                            self.fail("proxy exited before it was ready")
                        time.sleep(.1)
                else:
                    self.fail("proxy did not start")
                client = OpenAI(api_key=self.server.local_key, base_url=base, max_retries=0)
                response = client.chat.completions.create(model="gpt-6.1-sol", max_completion_tokens=20,
                    response_format={"type": "json_schema", "json_schema": {"name": "result", "strict": True,
                        "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}}}},
                    messages=[{"role": "user", "content": "Reply JSON"}])
                self.assertEqual(json.loads(response.choices[0].message.content), {"ok": True})
                self.assertEqual(response.usage.prompt_tokens, 20)
                parent_body = {"model": "gpt-6.1-sol", "messages": [{"role": "user", "content": "shared"}]}
                request = Request(base + "/fork/sessions", data=json.dumps(parent_body).encode(),
                    headers={"Authorization": "Bearer " + self.server.local_key, "Content-Type": "application/json"}, method="POST")
                with urlopen(request, timeout=5) as parent_response:
                    session = json.load(parent_response)
                fork = OpenAI(api_key=self.server.local_key, base_url=base + "/fork/sessions/" + session["id"], max_retries=0)
                for question in ("first", "second"):
                    result = fork.chat.completions.create(model="gpt-6.1-sol", messages=[{"role": "user", "content": question}])
                    self.assertTrue(result.choices[0].message.content.startswith("answer-"))
                trace = [json.loads(line) for line in Path(env["BVB_TEST_TRACE"]).read_text(encoding="utf-8").splitlines()]
                forks = [r["params"] for r in trace if r["method"] == "thread/fork"]
                self.assertEqual(len(forks), 2)
                self.assertEqual(forks[0]["threadId"], forks[1]["threadId"])
            finally:
                process.terminate()
                process.wait(timeout=10)


if __name__ == "__main__":
    unittest.main()
