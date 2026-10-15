import hashlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.request import HTTPRedirectHandler

import pipeline as p
from fixed_protocol import bvb_sample_frames, resolve_roles
from name_resolver import COMPOUND_BLOCK, SYN, head_term
from scene_protocol import legacy_roles, legacy_views, render_prompt, resolve_legacy_spec


class SceneSelectionTests(unittest.TestCase):
    def test_catalogs_have_same_scenes_but_different_assets(self):
        astra, luna = p.load_bundle(), p.load_bundle(scene_model="luna")
        self.assertEqual(astra["assets"]["revision"], luna["assets"]["revision"])
        self.assertEqual(set(astra["assets"]["scenes"]), set(luna["assets"]["scenes"]))
        for scene, asset in luna["assets"]["scenes"].items():
            self.assertIn("luna-reasoning-xhigh", asset["remote_path"])
            self.assertRegex(asset["sha256"], r"^[0-9a-f]{64}$")
            self.assertGreater(asset["bytes"], 0)
            self.assertNotEqual(asset["sha256"], astra["assets"]["scenes"][scene]["sha256"])
        with self.assertRaises(ValueError):
            p.load_bundle(scene_model="unknown")

    def test_source_switch_cannot_reuse_run_identity(self):
        astra, luna = p.load_bundle(), p.load_bundle(scene_model="luna")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            p.freeze(path, p.identity(astra, astra["questions"]))
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                p.freeze(path, p.identity(luna, luna["questions"]))

    def test_luna_has_no_astra_prompts_names_or_sampling(self):
        bundle = p.load_bundle(scene_model="luna")
        self.assertEqual(bundle["prompts"], {})
        for qa in bundle["questions"]:
            recipe = bundle["recipes"][qa["qa_id"]]
            self.assertNotIn("video_frames", recipe)
            self.assertNotIn("timeline_samples", recipe)
            self.assertNotIn("prompt_sha256", recipe)
            if p.category(qa) in p.FIVE_TYPES:
                self.assertEqual(recipe["views"], ["original_camera"] * 16)
            else:
                self.assertNotIn("highlights", recipe["spec"])
                self.assertNotIn("startcam", recipe["spec"]["views"])
                self.assertTrue(legacy_roles(qa))

    def test_missing_direction_reference_does_not_invent_view(self):
        bundle = p.load_bundle(scene_model="luna")
        qa = p.select_questions(bundle, only="983")[0]
        spec = resolve_legacy_spec(bundle["recipes"]["983"]["spec"], ["Bed base", "Toilet bowl"])
        self.assertNotIn("startcam", spec["views"])
        views = legacy_views(qa["question_type"], spec["resolved_roles"], [1, 24])
        self.assertEqual(len(views), 10)
        self.assertNotIn("startcam", views)
        prompt = render_prompt(qa, spec["resolved_roles"], views)
        self.assertNotIn("BLUE:", prompt)
        self.assertIn("GREEN: bed", prompt)
        self.assertNotIn("Frame 11", prompt)

    def test_role_matching_is_exact_after_resolution(self):
        bundle = p.load_bundle(scene_model="luna")
        spec = resolve_legacy_spec(bundle["recipes"]["2627"]["spec"],
                                   ["chair", "chair.001", "chairman", "floor"])
        self.assertTrue(spec["exact_names"])
        self.assertEqual(spec["highlights"][0]["match"], ["chair", "chair.001"])

    def test_unmatched_counting_and_short_appearance_prompts(self):
        bundle = p.load_bundle(scene_model="luna")
        for qid, timeline, expected in (("2627", [1, 24], 6), ("2750", [9, 13], 5)):
            qa = p.select_questions(bundle, only=qid)[0]
            spec = resolve_legacy_spec(bundle["recipes"][qid]["spec"], ["unrelated"])
            views = legacy_views(qa["question_type"], spec["resolved_roles"], timeline)
            self.assertEqual(len(views), expected)
            prompt = render_prompt(qa, spec["resolved_roles"], views)
            self.assertIn(f"The {expected} frames", prompt)
            self.assertIn("No reliable colour guide", prompt)
            self.assertNotIn("RED:", prompt)

    def test_cli_requires_explicit_qa_model(self):
        env = dict(os.environ)
        env.pop("OPENAI_MODEL", None)
        result = subprocess.run([sys.executable, str(p.ROOT / "run.py"), "answer"],
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("requires --model", result.stderr)

    def test_cli_source_paths_and_qa_model_are_independent(self):
        import run

        for source in ("astra", "luna"):
            with patch.object(sys, "argv", ["run.py", "answer", "--scene-model", source, "--smoke",
                                           "--model", "my-vision-model"]), patch.object(p, "answer") as call:
                run.main()
                args = call.call_args.args
                self.assertEqual(args[0]["scene_model"], source)
                self.assertEqual(args[2], (Path("runs") / source / "smoke").resolve())
                self.assertEqual(args[3], "my-vision-model")
            with patch.object(sys, "argv", ["run.py", "download", "--scene-model", source]), \
                    patch.object(p, "download") as call:
                run.main()
                self.assertEqual(call.call_args.args[2], (Path("assets") / source / "blends").resolve())

    def test_download_auth_is_optional_and_not_forwarded_on_redirect(self):
        payload = b"synthetic blend"
        bundle = {"assets": {"dataset": "owner/data", "revision": "a" * 40, "scenes": {
            "s": {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
                  "remote_path": "luna/blends/s.blend"}}}}
        with tempfile.TemporaryDirectory() as directory, patch.object(p, "urlopen") as call, \
                patch.dict(os.environ, {"HF_TOKEN": "test-only"}):
            call.return_value.__enter__.return_value = io.BytesIO(payload)
            p.download(bundle, [{"scene_name": "s"}], Path(directory), "https://example.test")
            request = call.call_args.args[0]
            self.assertEqual(request.get_header("Authorization"), "Bearer test-only")
            redirected = HTTPRedirectHandler().redirect_request(
                request, None, 302, "Found", {}, "https://cdn.example.test/file")
            self.assertFalse(redirected.has_header("Authorization"))


class LunaRenderTests(unittest.TestCase):
    def setUp(self):
        self.bundle = p.load_bundle(scene_model="luna")
        self.temporary = tempfile.TemporaryDirectory()
        self.run = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def normalize(self, qa, fixed=False):
        recipe = self.bundle["recipes"][qa["qa_id"]]
        timeline = [11, 42]
        roles = resolve_roles(recipe["roles"] if fixed else recipe["spec"]["roles"],
                              ["table", "door", "chair", "bed", "tv", "toilet", "laptop"],
                              SYN, head_term, COMPOUND_BLOCK)
        views = (["original_camera"] * 16 if fixed else
                 legacy_views(qa["question_type"], roles, timeline))
        directory = self.run / "raw" / qa["qa_id"]
        directory.mkdir(parents=True)
        file = directory / "metadata.bin"
        file.write_bytes(b"synthetic metadata only; no image content\n" * 8)
        frames = [{"file": str(file), "view": view, "camera": {"frame": i}}
                  for i, view in enumerate(views)]
        raw = {"roles": roles, "frames": frames, "source_timeline": timeline}
        if fixed:
            raw["timeline_samples"] = bvb_sample_frames(*timeline, 32)
            raw["qc"] = dict.fromkeys(("original_camera_unchanged", "restored_materials",
                                      "restored_geometry_visibility", "question_isolation"), True)
            p.save_json(self.run / "raw-fixed/shared" / qa["scene_name"] / "manifest.json",
                        {"frames": frames})
        path = directory / "manifest.json"
        p.save_json(path, raw)
        p.normalize_render(self.bundle, qa, self.run, path)
        return raw

    def test_full_200_normalize_and_answer_once(self):
        questions = self.bundle["questions"]
        for qa in questions:
            self.normalize(qa, p.category(qa) in p.FIVE_TYPES)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=Mock())))
        client.chat.completions.create.return_value = SimpleNamespace(
            model="chosen-vision-model", id="mock", usage=None,
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="ANSWER: 2"))])
        with patch.object(p, "encode_frames", return_value=["synthetic"]), redirect_stdout(io.StringIO()):
            for _ in range(2):
                p.answer(self.bundle, questions, self.run, "chosen-vision-model",
                         "https://api.example.test/v1", 8000, 1, client)
            p.report(self.bundle, questions, self.run)
        self.assertEqual(client.chat.completions.create.call_count, 200)
        self.assertEqual(len(p.read_jsonl(self.run / "requests.jsonl")), 200)
        self.assertEqual(len(p.read_jsonl(self.run / "answers.jsonl")), 200)
        for qa, call in zip(questions, client.chat.completions.create.call_args_list):
            manifest = p.verify_render(self.bundle, qa, self.run)
            self.assertEqual(call.kwargs["model"], "chosen-vision-model")
            self.assertEqual(call.kwargs["messages"][0]["content"][0]["text"],
                             p.expected_prompt(self.bundle, qa, manifest))
        summary = p.read_json(self.run / "summary.json")
        self.assertEqual(summary["scene_model"], "luna")
        self.assertEqual(summary["qa_model"], "chosen-vision-model")

    def test_wrong_source_prompt_and_timeline_rejected(self):
        qa = p.select_questions(self.bundle, only="720")[0]
        self.normalize(qa, fixed=True)
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            p.verify_render(p.load_bundle(), qa, self.run)
        directory = self.run / "renders" / "720"
        (directory / "prompt.txt").write_text("wrong source prompt", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "prompt mismatch"):
            p.verify_render(self.bundle, qa, self.run)
        manifest = p.read_json(directory / "manifest.json")
        manifest["timeline_samples"] = [1] * 16
        p.save_json(directory / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "timeline mismatch"):
            p.verify_render(self.bundle, qa, self.run)

    def test_fixed_camera_qc_still_enforced_for_luna(self):
        qa = p.select_questions(self.bundle, only="720")[0]
        raw = self.normalize(qa, fixed=True)
        raw["qc"]["original_camera_unchanged"] = False
        path = self.run / "raw" / "720" / "manifest.json"
        p.save_json(path, raw)
        with self.assertRaisesRegex(ValueError, "QC failed"):
            p.normalize_render(self.bundle, qa, self.run, path)

    def test_luna_fixed_dispatch_uses_own_timeline(self):
        qa = p.select_questions(self.bundle, only="720")
        with patch.object(p, "check_blender", return_value=("blender", "Blender 5.2.2")), \
             patch.object(p, "check_blends"), patch.object(p, "normalize_render"), \
             patch.object(p, "digest", return_value=self.bundle["assets"]["scenes"][qa[0]["scene_name"]]["sha256"]), \
             patch.object(p.subprocess, "run") as call:
            p.render(self.bundle, qa, self.run, self.run / "dispatch", "blender")
            spec = p.read_json(Path(call.call_args.args[0][-1]))
            self.assertIsNone(spec["video_frames"])
            self.assertEqual(spec["scene_model"], "luna")
            self.assertEqual(spec["arms"], ["C"])


if __name__ == "__main__":
    unittest.main()
