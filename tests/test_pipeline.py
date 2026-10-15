import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from contextlib import redirect_stdout

import pipeline as p
from fixed_protocol import bvb_sample_frames


class BundleTests(unittest.TestCase):
    def test_complete_frozen_release(self):
        bundle = p.load_bundle()
        self.assertEqual(len(bundle["questions"]), 200)
        self.assertEqual(len(bundle["assets"]["scenes"]), 126)
        self.assertEqual(sum(r["protocol"] == p.FIXED for r in bundle["recipes"].values()), 125)
        self.assertEqual(sum(len(r["views"]) for r in bundle["recipes"].values()), 3217)

    def test_unmatched_counting_case_describes_its_actual_six_frames(self):
        bundle = p.load_bundle()
        self.assertEqual(len(bundle["recipes"]["2583"]["views"]), 6)
        self.assertIn("frame 6 = the plan again", bundle["prompts"]["2583"])
        self.assertIn("no target is colour-highlighted", bundle["prompts"]["2583"])
        self.assertNotIn("frame 14", bundle["prompts"]["2583"])

    def test_smoke_covers_eight_types(self):
        bundle = p.load_bundle()
        questions = p.select_questions(bundle, smoke=True)
        self.assertEqual({p.category(q) for q in questions}, p.EIGHT_TYPES)
        self.assertEqual(len(questions), 8)

    def test_unknown_question_is_rejected(self):
        with self.assertRaises(ValueError):
            p.select_questions(p.load_bundle(), only="not-a-question")

    def test_no_label_values_in_question_inputs(self):
        bundle = p.load_bundle()
        for qa in bundle["questions"]:
            self.assertNotIn("ground_truth", qa)
            self.assertNotIn("bvb", qa)
            self.assertNotIn("original", qa)

    def test_fixed_camera_has_no_extra_views(self):
        bundle = p.load_bundle()
        for q in bundle["questions"]:
            recipe = bundle["recipes"][q["qa_id"]]
            if p.category(q) in p.FIVE_TYPES:
                self.assertEqual(recipe["views"], ["original_camera"] * 16)
                self.assertIn("No overhead", bundle["prompts"][q["qa_id"]])
                self.assertEqual(recipe["jpeg_quality"], 80)
            else:
                self.assertEqual(recipe["jpeg_quality"], 85)

    def test_changed_data_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            for source in (p.ROOT / "data").iterdir():
                (root / "data" / source.name).write_bytes(source.read_bytes())
            (root / "data/prompts.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Frozen data changed"):
                p.load_bundle(root)

    def test_cli_has_no_samples_or_voting_switch(self):
        result = subprocess.run([sys.executable, str(p.ROOT / "run.py"), "--help"],
                                capture_output=True, text=True, check=True)
        self.assertNotIn("--samples", result.stdout)
        self.assertNotIn("--vote", result.stdout)

    def test_official_two_stage_sampling(self):
        expected64 = [1 + round(i * 299 / 63) for i in range(64)]
        expected16 = [expected64[int(i * 63 / 15)] for i in range(16)]
        self.assertEqual(bvb_sample_frames(1, 300, 64), expected16)
        with self.assertRaises(ValueError):
            bvb_sample_frames(1, 300, 63)


class JournalTests(unittest.TestCase):
    def setUp(self):
        self.questions = [{"qa_id": "1"}, {"qa_id": "2"}]

    def test_completed_questions_never_repeat(self):
        result = p.pending_questions(self.questions, [{"qa_id": "1"}],
                                     [{"qa_id": "1", "status": "ok"}])
        self.assertEqual(result, [{"qa_id": "2"}])

    def test_failed_questions_never_repeat(self):
        result = p.pending_questions(self.questions, [{"qa_id": "1"}],
                                     [{"qa_id": "1", "status": "error"}])
        self.assertEqual(result, [{"qa_id": "2"}])

    def test_interrupted_request_is_not_reissued(self):
        with self.assertRaisesRegex(ValueError, "will not resubmit"):
            p.pending_questions(self.questions, [{"qa_id": "1"}], [])

    def test_duplicate_journal_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            p.pending_questions(self.questions, [{"qa_id": "1"}, {"qa_id": "1"}], [])

    def test_unknown_result_is_rejected(self):
        with self.assertRaises(ValueError):
            p.pending_questions(self.questions, [], [{"qa_id": "2"}])

    def test_lock_rejects_another_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with p.run_lock(root):
                with self.assertRaisesRegex(ValueError, "locked"):
                    with p.run_lock(root):
                        pass
            self.assertFalse((root / ".lock").exists())

    def test_configuration_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            p.freeze(path, {"model": "first"})
            p.freeze(path, {"model": "first"})
            with self.assertRaises(ValueError):
                p.freeze(path, {"model": "second"})


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.run = Path(self.temporary.name)
        full = p.load_bundle()
        self.questions = [q for q in full["questions"] if q["qa_id"] in {"2", "983", "2750"}]
        self.bundle = full
        for qa in self.questions:
            qid = qa["qa_id"]
            recipe = full["recipes"][qid]
            directory = self.run / "renders" / qid
            directory.mkdir(parents=True)
            frames = []
            for i, view in enumerate(recipe["views"]):
                path = directory / f"fixture_{i}.bin"
                path.write_bytes(b"not an image; metadata test only\n" * 8)
                frames.append({"file": str(path), "view": view, "sha256": p.digest(path)})
            p.save_json(directory / "manifest.json", {
                "qa_id": qid, "scene": qa["scene_name"], "recipe_sha256": p.data_hash(recipe),
                "blend_sha256": full["assets"]["scenes"][qa["scene_name"]]["sha256"], "frames": frames,
            })
            (directory / "prompt.txt").write_text(full["prompts"][qid], encoding="utf-8")
        self.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=Mock())))
        self.client.chat.completions.create.return_value = SimpleNamespace(
            id="mock-response", model="test-model", usage=SimpleNamespace(model_dump=lambda: {"total_tokens": 100}),
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="Evidence.\nANSWER: 2"))],
        )

    def tearDown(self):
        self.temporary.cleanup()

    def answer(self, questions=None):
        with patch.object(p, "encode_frames", return_value=["synthetic"]):
            p.answer(self.bundle, questions or self.questions, self.run, "test-model",
                     "https://api.example.test/v1", 8000, 1, self.client)

    def test_frame_order_mismatch_fails(self):
        qa = self.questions[0]
        path = self.run / "renders" / qa["qa_id"] / "manifest.json"
        manifest = p.read_json(path)
        manifest["frames"].reverse()
        p.save_json(path, manifest)
        with self.assertRaisesRegex(ValueError, "reordered"):
            p.verify_render(self.bundle, qa, self.run)

    def test_frame_hash_mismatch_fails(self):
        qa = self.questions[0]
        manifest = p.read_json(self.run / "renders" / qa["qa_id"] / "manifest.json")
        Path(manifest["frames"][0]["file"]).write_bytes(b"changed" * 100)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            p.verify_render(self.bundle, qa, self.run)

    def test_frame_path_outside_run_fails(self):
        qa = self.questions[0]
        path = self.run / "renders" / qa["qa_id"] / "manifest.json"
        manifest = p.read_json(path)
        manifest["frames"][0]["file"] = str(p.ROOT / "pipeline.py")
        p.save_json(path, manifest)
        with self.assertRaisesRegex(ValueError, "outside run"):
            p.verify_render(self.bundle, qa, self.run)

    def test_exactly_one_call_per_question_and_resume(self):
        self.answer()
        self.answer()
        self.assertEqual(self.client.chat.completions.create.call_count, 3)
        rows = p.read_jsonl(self.run / "answers.jsonl")
        self.assertEqual(len(rows), 3)
        self.assertEqual({r["sample"] for r in rows}, {0})
        self.assertTrue(all(r["status"] == "ok" for r in rows))
        for call in self.client.chat.completions.create.call_args_list:
            self.assertEqual(call.kwargs["model"], "test-model")
            self.assertEqual(call.kwargs["max_completion_tokens"], 8000)
            self.assertEqual(len(call.kwargs["messages"]), 1)
            self.assertEqual(call.kwargs["messages"][0]["content"][1]["image_url"]["detail"], "high")

    def test_no_gt_in_requests_and_labels_not_parsed(self):
        original = p.read_jsonl

        def guarded(path):
            if Path(path).name == "labels.jsonl":
                self.fail("Labels were parsed during answer")
            return original(path)

        with patch.object(p, "read_jsonl", side_effect=guarded):
            self.answer()
        calls = self.client.chat.completions.create.call_args_list
        for qa, call in zip(self.questions, calls):
            prompt = call.kwargs["messages"][0]["content"][0]["text"]
            self.assertEqual(prompt, self.bundle["prompts"][qa["qa_id"]])
            self.assertNotIn("ground_truth", prompt)

    def test_failed_api_not_retried_and_exception_redacted(self):
        self.client.chat.completions.create.side_effect = RuntimeError("secret request details")
        with self.assertRaisesRegex(RuntimeError, "Three failures"):
            self.answer()
        self.assertEqual(self.client.chat.completions.create.call_count, 3)
        text = (self.run / "answers.jsonl").read_text()
        self.assertNotIn("secret request details", text)
        with self.assertRaisesRegex(ValueError, "Three recorded failures"):
            self.answer()
        self.assertEqual(self.client.chat.completions.create.call_count, 3)

    def test_length_response_is_invalid(self):
        self.client.chat.completions.create.return_value.choices[0].finish_reason = "length"
        with self.assertRaisesRegex(RuntimeError, "Failed answers"):
            self.answer(self.questions[:1])
        self.assertEqual(p.read_jsonl(self.run / "answers.jsonl")[0]["status"], "invalid_response")

    def test_empty_answer_is_invalid(self):
        self.client.chat.completions.create.return_value.choices[0].message.content = ""
        with self.assertRaisesRegex(RuntimeError, "Failed answers"):
            self.answer(self.questions[:1])
        self.assertEqual(p.read_jsonl(self.run / "answers.jsonl")[0]["status"], "invalid_response")

    def test_report_counts_missing_and_failed_in_denominator(self):
        p.save_json(self.run / "answer_config.json", p.identity(self.bundle, self.questions))
        qa = next(q for q in self.questions if q["qa_id"] == "2")
        p.append_jsonl(self.run / "answers.jsonl", {"qa_id": qa["qa_id"], "status": "ok", "answer": "2"})
        p.report(self.bundle, self.questions, self.run)
        summary = p.read_json(self.run / "summary.json")
        self.assertEqual(summary["by_type"]["ALL"]["n"], 3)
        self.assertEqual(summary["by_type"]["ALL"]["correct"], 1)
        self.assertFalse(summary["complete"])

    def test_model_change_cannot_mix_results(self):
        self.answer()
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            p.answer(self.bundle, self.questions, self.run, "different-model",
                     "https://api.example.test/v1", 8000, 1, self.client)

    def test_invalid_url_rejected(self):
        with self.assertRaises(ValueError):
            p.answer(self.bundle, self.questions, self.run, "test-model",
                     "https://user:secret@example.test/v1", 8000, client=self.client)

    def test_sdk_disables_automatic_retries(self):
        with patch("openai.OpenAI", return_value=self.client) as constructor, \
             patch.dict(p.os.environ, {"OPENAI_API_KEY": "test-not-a-real-key"}), \
             patch.object(p, "encode_frames", return_value=["synthetic"]):
            self.client.close = Mock()
            p.answer(self.bundle, self.questions[:1], self.run, "test-model",
                     "https://api.example.test/v1", 8000, 1)
            self.assertEqual(constructor.call_args.kwargs["max_retries"], 0)
            self.assertEqual(constructor.call_args.kwargs["timeout"], 300)
            self.client.close.assert_called_once()

    def test_full_200_single_answer_run_and_report(self):
        questions = self.bundle["questions"]
        for qa in questions:
            directory = self.run / "renders" / qa["qa_id"]
            if directory.exists():
                continue
            directory.mkdir(parents=True)
            fixture = directory / "synthetic.bin"
            fixture.write_bytes(b"metadata-only test; no image decoding\n" * 8)
            recipe = self.bundle["recipes"][qa["qa_id"]]
            p.save_json(directory / "manifest.json", {
                "qa_id": qa["qa_id"], "scene": qa["scene_name"], "recipe_sha256": p.data_hash(recipe),
                "blend_sha256": self.bundle["assets"]["scenes"][qa["scene_name"]]["sha256"],
                "frames": [{"file": str(fixture), "view": view, "sha256": p.digest(fixture)}
                           for view in recipe["views"]],
            })
            (directory / "prompt.txt").write_text(self.bundle["prompts"][qa["qa_id"]], encoding="utf-8")
        with patch.object(p, "encode_frames", return_value=["synthetic"]), redirect_stdout(io.StringIO()):
            p.answer(self.bundle, questions, self.run, "test-model", "https://api.example.test/v1",
                     8000, 8, self.client)
            p.answer(self.bundle, questions, self.run, "test-model", "https://api.example.test/v1",
                     8000, 8, self.client)
            p.report(self.bundle, questions, self.run)
        self.assertEqual(self.client.chat.completions.create.call_count, 200)
        self.assertEqual(len(p.read_jsonl(self.run / "requests.jsonl")), 200)
        self.assertEqual(len(p.read_jsonl(self.run / "answers.jsonl")), 200)
        self.assertEqual(p.read_json(self.run / "summary.json")["by_type"]["ALL"]["n"], 200)


class DownloadAndRenderTests(unittest.TestCase):
    def test_blender_does_not_inherit_api_secrets(self):
        with patch.dict(p.os.environ, {"OPENAI_API_KEY": "test", "GH_TOKEN": "test", "PATH": "keep"}):
            env = p.blender_environment()
            self.assertNotIn("OPENAI_API_KEY", env)
            self.assertNotIn("GH_TOKEN", env)
            self.assertEqual(env["PATH"], "keep")

    def test_download_is_pinned_and_verifies_hash(self):
        payload = b"synthetic blend bytes"
        asset = {"file": "s.blend", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
                 "remote_path": "run/blends/s.blend"}
        bundle = {"assets": {"dataset": "owner/data", "revision": "f" * 40, "scenes": {"s": asset}}}
        with tempfile.TemporaryDirectory() as directory, patch.object(p, "urlopen") as request:
            request.return_value.__enter__.return_value = __import__("io").BytesIO(payload)
            p.download(bundle, [{"scene_name": "s"}], Path(directory), "https://example.test")
            self.assertIn("/resolve/" + "f" * 40 + "/", request.call_args.args[0].full_url)
            p.download(bundle, [{"scene_name": "s"}], Path(directory), "https://example.test")
            self.assertEqual(request.call_count, 1)

    def test_bad_download_never_becomes_blend(self):
        bundle = {"assets": {"dataset": "o/d", "revision": "a" * 40, "scenes": {
            "s": {"bytes": 50, "sha256": "b" * 64, "remote_path": "s.blend"}}}}
        with tempfile.TemporaryDirectory() as directory, patch.object(p, "urlopen") as request:
            request.return_value.__enter__.return_value = __import__("io").BytesIO(b"error")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                p.download(bundle, [{"scene_name": "s"}], Path(directory), "https://example.test")
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_blender_version_gate(self):
        with patch.object(p, "blender_command", return_value="blender"), patch.object(p.subprocess, "run") as run:
            run.return_value = SimpleNamespace(stdout="Blender 5.2.2 LTS\n")
            self.assertEqual(p.check_blender("blender"), ("blender", "Blender 5.2.2 LTS"))
            run.return_value = SimpleNamespace(stdout="Blender 5.3.0 Beta\n")
            with self.assertRaisesRegex(ValueError, "requires Blender 5.2.2"):
                p.check_blender("blender")

    def test_fixed_render_dispatch_has_only_c_and_no_video_dependency(self):
        bundle = p.load_bundle()
        questions = p.select_questions(bundle, only="720")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = root / "source.blend"
            fake.write_bytes(b"fake")
            with patch.object(p, "check_blender", return_value=("blender", "Blender 5.2.2")), \
                 patch.object(p, "check_blends"), patch.object(p, "normalize_render"), \
                 patch.object(p, "digest", return_value=bundle["assets"]["scenes"][questions[0]["scene_name"]]["sha256"]), \
                 patch.object(p.subprocess, "run") as call:
                p.render(bundle, questions, root, root / "run", "blender")
                args = call.call_args.args[0]
                self.assertIn("--disable-autoexec", args)
                self.assertIn("--python-exit-code", args)
                self.assertTrue(any(arg.endswith("render_fixed.py") for arg in args))
                spec_path = Path(args[-1])
                spec = p.read_json(spec_path)
                self.assertEqual(spec["arms"], ["C"])
                self.assertEqual(len(spec["questions"]), 1)
                self.assertNotIn("ground_truth", spec["questions"][0])


class ScoreTests(unittest.TestCase):
    def test_numeric_boundary_and_counting(self):
        self.assertTrue(p.semantic_correct("120", {"question_type": "object_size_estimation", "ground_truth": "100"}))
        self.assertFalse(p.semantic_correct("121", {"question_type": "object_size_estimation", "ground_truth": "100"}))
        self.assertTrue(p.semantic_correct("3", {"question_type": "object_counting", "ground_truth": "3"}))
        self.assertFalse(p.semantic_correct("2", {"question_type": "object_counting", "ground_truth": "3"}))

    def test_option_matching(self):
        qa = {"question_type": "object_rel_direction", "ground_truth": "B", "options": ["A. left", "B. right"]}
        self.assertTrue(p.semantic_correct("B", qa))
        self.assertTrue(p.semantic_correct("right", qa))
        self.assertFalse(p.semantic_correct("A", qa))


if __name__ == "__main__":
    unittest.main()
