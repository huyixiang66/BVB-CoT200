import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
RUNNER = PACKAGE / "run_fork_qa.py"
MOCK = PACKAGE / "tests" / "fake_app_server.js"


class ForkQATest(unittest.TestCase):
    def execute(self, fail=False):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        base = Path(directory.name)
        (base / "frame.jpg").write_bytes(b"fake-image")
        manifest = base / "input.json"
        manifest.write_text(json.dumps({"images": ["frame.jpg"], "ground_truth": "NEVER_SEND_GT",
            "questions": [{"id": i, "question": f"question {i}"} for i in range(3)]}), encoding="utf-8")
        output, trace = base / "out.json", base / "trace.jsonl"
        env = os.environ.copy()
        env["BVB_TEST_TRACE"] = str(trace)
        if fail:
            env["BVB_TEST_FAIL_CHILD"] = "1"
        result = subprocess.run([sys.executable, str(RUNNER), str(manifest), "--output", str(output),
                                 "--codex", str(MOCK)], env=env, capture_output=True, text=True, timeout=20)
        records = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
        return result, json.loads(output.read_text(encoding="utf-8")), records

    def test_shared_parent_and_independent_questions(self):
        result, output, records = self.execute()
        self.assertEqual(result.returncode, 0, result.stderr)
        forks = [r for r in records if r["method"] == "thread/fork"]
        self.assertEqual([r["params"]["threadId"] for r in forks], ["parent", "parent"])
        turns = [r["params"] for r in records if r["method"] == "turn/start"]
        self.assertEqual([r["threadId"] for r in turns], ["parent", "child-1", "child-2"])
        self.assertEqual(sum(item["type"] == "localImage" for turn in turns for item in turn["input"]), 1)
        self.assertNotIn("NEVER_SEND_GT", json.dumps(records))
        self.assertEqual([q["answer"] for q in output["questions"]], ["answer-2", "answer-3"])
        self.assertEqual(output["questions"][0]["usage"]["cachedInputTokens"], 10)

    def test_failure_is_saved_and_stops_later_questions(self):
        result, output, records = self.execute(fail=True)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(output["questions"][0]["status"], "failed")
        self.assertEqual(len([r for r in records if r["method"] == "thread/fork"]), 1)
        self.assertTrue(output["errors"])


if __name__ == "__main__":
    unittest.main()
