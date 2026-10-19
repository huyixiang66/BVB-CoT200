import json
import tempfile
import unittest
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import refinement as r


def fixture_case(root):
    return {
        "case_id": "q-test",
        "scene": "fixture",
        "question_type": "object_abs_distance",
        "question": "What is the distance between the table and chair?",
        "inputs": {
            "original_video": str(root / "original.mp4"),
            "bvb_video": str(root / "bvb.mp4"),
            "blend": str(root / "scene.blend"),
        },
        "objects": {
            "moving": {"label": "table", "names": ["table"]},
            "reference": {"label": "chair", "names": ["chair"]},
        },
        "candidates": [
            {"id": "no_change", "operation": "no_change"},
            {"id": "toward_050", "operation": "translate", "distance_m": 0.5},
        ],
        "ground_truth_m": 1.2,
        "historical_answers": {"original_m": 2.0, "bvb_m": 1.5},
    }


class CaseTests(unittest.TestCase):
    def test_blind_case_excludes_labels_and_history(self):
        case = fixture_case(Path("."))
        blind = r.blind_case(case)
        self.assertNotIn("ground_truth_m", blind)
        self.assertNotIn("historical_answers", blind)
        self.assertEqual(blind["question"], case["question"])

    def test_case_requires_unique_candidates_and_no_change(self):
        case = fixture_case(Path("."))
        case["candidates"] = [{"id": "move", "operation": "translate", "distance_m": 0.5}]
        with self.assertRaisesRegex(ValueError, "no_change"):
            r.validate_case(case)

    def test_case_can_defer_object_names_to_blender_resolution(self):
        case = fixture_case(Path("."))
        case["objects"] = {
            "moving": {"label": "table", "auto_match": True},
            "reference": {"label": "chair", "auto_match": True},
        }
        self.assertTrue(r.needs_object_resolution(r.validate_case(case)))

    def test_translate_candidates_require_positive_distance(self):
        case = fixture_case(Path("."))
        case["candidates"][1]["distance_m"] = 0
        with self.assertRaisesRegex(ValueError, "positive distance"):
            r.validate_case(case)

    def test_unresolved_auto_match_restricts_case_to_no_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blend = root / "scene.blend"
            blend.write_bytes(b"synthetic blend")
            case = fixture_case(root)
            case["objects"] = {
                "moving": {"label": "table", "auto_match": True},
                "reference": {"label": "chair", "auto_match": True},
            }
            run = root / "run"
            run.mkdir()
            r.save_json(run / "object_resolution.json", {
                "case_id": case["case_id"],
                "scene": case["scene"],
                "blend_sha256": r.digest(blend),
                "objects": {
                    "moving": {"label": "table", "names": ["table top"]},
                    "reference": {"label": "chair", "names": []},
                },
                "resolution": {"complete": False, "unresolved_roles": ["reference"]},
            })
            resolved = r.resolve_case_objects(case, run, "unused")
            self.assertEqual(resolved["candidates"], [{"id": "no_change", "operation": "no_change"}])
            self.assertEqual(resolved["candidate_policy"]["resolution_gate"], "no_change_only")

    def test_candidate_render_uses_frozen_blind_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            r.save_json(run / "blind_case.json", {
                "candidates": [{"id": "no_change", "operation": "no_change"}],
            })
            case = fixture_case(root)

            def fake_blender(_case, _run, _blender, _action, output, stage_spec):
                spec = r.read_json(stage_spec)
                r.save_json(Path(output) / "manifest.json", {"candidates": spec["candidates"]})

            with patch.object(r, "candidate_frame_pairs", return_value=[{"pair_id": 0, "scene_frame": 1}]), \
                    patch.object(r, "run_blender", side_effect=fake_blender):
                manifest = r.render_candidates(case, run, "unused")
            self.assertEqual(manifest["candidates"], [{"id": "no_change", "operation": "no_change"}])

    def test_mra_matches_partial_credit_example(self):
        self.assertEqual(r.mra(1.5, 1.2), 6 / 11)
        self.assertEqual(r.mra(1.3, 1.2), 10 / 11)
        self.assertEqual(r.mra(1.8, 1.2), 0.0)

    def test_answer_line_wins_over_cot_numbers(self):
        self.assertEqual(r.number("Use a 2.1-2.4 m range and check 3D.\nANSWER: 2.3"), 2.3)
        self.assertEqual(r.number("plain response 1.7"), 1.7)

    def test_uniform_indices_include_both_ends(self):
        self.assertEqual(r.uniform_indices(10, 3), [0, 4, 9])

    def test_reasoning_model_uses_supported_frozen_temperature(self):
        self.assertEqual(r.model_temperature("gpt-6-luna"), 1)
        self.assertEqual(r.model_temperature("gpt-5.4-mini"), 0)

    def test_default_qa_prompt_matches_target_colour_only_protocol(self):
        prompt = r.qa_prompt(fixture_case(Path(".")))
        self.assertIn("RED: table (first object)", prompt)
        self.assertIn("BLUE: chair (second object)", prompt)
        self.assertIn("all other rendered meshes are grayscale", prompt)
        self.assertNotIn("GREEN:", prompt)
        self.assertIn("nearest surfaces, not their centres", prompt)

    def test_blender_environment_drops_secrets(self):
        original = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "must-not-reach-blender"
        try:
            self.assertNotIn("OPENAI_API_KEY", r.blender_environment())
        finally:
            if original is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = original


class ModelClientTests(unittest.TestCase):
    def test_model_client_reads_openai_key_from_environment(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "local-test-key"}, clear=True), \
                patch.object(r, "OpenAI") as openai:
            r.model_client("https://api.openai.com/v1", 10)
        self.assertEqual(openai.call_args.kwargs["api_key"], "local-test-key")

    def test_direct_json_call_uses_completion_budget_and_json_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "frame.jpg"
            image.write_bytes(b"synthetic image bytes")
            frame = {"file": str(image), "sha256": r.digest(image)}
            response = SimpleNamespace(
                model="gpt-test-vision",
                choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="{}"))],
                usage=SimpleNamespace(model_dump=lambda: {"total_tokens": 3}),
            )
            create = Mock(return_value=response)
            client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
            r.call_model(root / "run", "direct", "gpt-test-vision", "https://api.openai.com/v1",
                         "Return JSON", [frame], 128, "low", client=client, json_mode=True)
            request = create.call_args.kwargs
            self.assertEqual(request["max_completion_tokens"], 128)
            self.assertEqual(request["response_format"], {"type": "json_object"})


class AnswerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run = self.root / "run"
        frames = []
        for index in range(2):
            path = self.run / "final" / f"frame_{index}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((b"synthetic-frame" + bytes([index])) * 20)
            frames.append({"scene_frame": index + 1, "file": str(path), "sha256": r.digest(path)})
        r.save_json(self.run / "selection.json", {
            "candidate_id": "no_change", "candidate": {"id": "no_change", "operation": "no_change"}
        })
        r.save_json(self.run / "final" / "manifest.json", {
            "mode": "diagnostic",
            "candidate": {"id": "toward_050", "operation": "translate", "distance_m": 0.5},
            "before_state": {"surface_distance_m": 1.5},
            "after_state": {"surface_distance_m": 1.2},
            "before_qa": frames,
            "after_qa": frames,
            "qa_frames_byte_identical": True,
            "visual_ablation": {
                "target_colours_only": True,
                "non_target_meshes_grayscale": True,
            },
        })
        self.case = fixture_case(self.root)
        responses = []
        for answer in ("1.8", "1.3", "1.8"):
            responses.append(SimpleNamespace(
                model="test-model",
                usage=SimpleNamespace(model_dump=lambda: {"total_tokens": 10}),
                choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=answer))],
            ))
        create = Mock(side_effect=responses)
        self.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    def tearDown(self):
        self.temp.cleanup()

    def test_three_frozen_samples_are_journaled_and_reported(self):
        r.answer(self.case, self.run, "test-model", "https://api.example.test/v1",
                 3, "after", client=self.client)
        self.assertEqual(self.client.chat.completions.create.call_count, 3)
        requests = [r.read_json(path) for path in sorted((self.run / "calls").glob("*/request.json"))]
        self.assertEqual(len({request["prompt_sha256"] for request in requests}), 1)
        self.assertEqual(len({json.dumps(request["frames"], sort_keys=True) for request in requests}), 1)
        self.assertEqual({request["max_completion_tokens"] for request in requests}, {8000})
        self.assertEqual({request["image_detail"] for request in requests}, {"high"})
        report = r.report(self.case, self.run)
        self.assertEqual(report["summary"]["after"]["values_m"], [1.8, 1.3, 1.8])
        self.assertEqual(report["summary"]["after"]["best_mra"], 10 / 11)
        self.assertFalse(report["summary"]["after"]["any_full_mra"])
        self.assertEqual(report["summary"]["after"]["unique_values_m"], [1.3, 1.8])
        self.assertAlmostEqual(report["summary"]["after"]["mean_abs_error_m"], 1.3 / 3)
        self.assertTrue(report["qa_frames_byte_identical"])
        self.assertTrue(report["visual_ablation"]["target_colours_only"])

    def test_completed_samples_are_not_called_again(self):
        r.answer(self.case, self.run, "test-model", "https://api.example.test/v1",
                 1, "after", client=self.client)
        r.answer(self.case, self.run, "test-model", "https://api.example.test/v1",
                 1, "after", client=self.client)
        self.assertEqual(self.client.chat.completions.create.call_count, 1)

    def test_changed_input_cannot_reuse_completed_answer(self):
        r.answer(self.case, self.run, "test-model", "https://api.example.test/v1",
                 1, "after", client=self.client)
        self.case["question"] += " Changed question."
        with self.assertRaisesRegex(ValueError, "Frozen configuration"):
            r.answer(self.case, self.run, "test-model", "https://api.example.test/v1",
                     1, "after", client=self.client)
        self.assertEqual(self.client.chat.completions.create.call_count, 1)

    def test_started_call_is_never_retried(self):
        r.save_json(self.run / "calls/answer_after_00/started.json", {"pid": 1})
        with self.assertRaisesRegex(ValueError, "will not be resubmitted"):
            r.answer(self.case, self.run, "test-model", "https://api.example.test/v1",
                     1, "after", client=self.client)
        self.client.chat.completions.create.assert_not_called()

    def test_qa_uses_all_64_frames_and_dynamic_prompt(self):
        path = self.run / "final/manifest.json"
        manifest = r.read_json(path)
        frames = [{**manifest["before_qa"][0], "scene_frame": index + 1} for index in range(64)]
        manifest.update({"before_qa": frames, "after_qa": frames})
        r.save_json(path, manifest)
        r.answer(self.case, self.run, "test-model", "https://api.example.test/v1",
                 1, "both", client=self.client)
        calls = self.client.chat.completions.create.call_args_list
        self.assertEqual(len(calls), 2)
        for call in calls:
            content = call.kwargs["messages"][0]["content"]
            self.assertIn("Frames 1-64", content[0]["text"])
            self.assertEqual(sum(item["type"] == "image_url" for item in content), 64)

    def test_incomplete_after_frames_stops_before_any_qa_call(self):
        path = self.run / "final/manifest.json"
        manifest = r.read_json(path)
        manifest["after_qa"] = manifest["after_qa"][:1]
        r.save_json(path, manifest)
        with self.assertRaisesRegex(ValueError, "frame"):
            r.answer(self.case, self.run, "test-model", "https://api.example.test/v1",
                     1, "both", client=self.client)
        self.client.chat.completions.create.assert_not_called()

    def test_candidate_override_requires_diagnostic_flag(self):
        with self.assertRaisesRegex(ValueError, "diagnostic"):
            r.finalize(self.case, self.run, "blender", "toward_050", diagnostic=False)


class FinalizeSamplingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.run = Path(self.temp.name)
        self.case = fixture_case(self.run)
        r.save_json(self.run / "selection.json", {"candidate": self.case["candidates"][0]})
        r.save_json(self.run / "scene_before.json", {"timeline": [1, 4628]})
        r.save_json(self.run / "sources.json", {"bvb": {"frame_count": 64}})

    def tearDown(self):
        self.temp.cleanup()

    def fake_blender(self, _case, _run, _blender, _action, output, spec_path):
        spec = r.read_json(spec_path)
        frames = [{"scene_frame": frame} for frame in spec["qa_scene_frames"]]
        r.save_json(Path(output) / "manifest.json", {"before_qa": frames, "after_qa": frames})

    def test_finalize_matches_source_count_and_full_timeline(self):
        for count in (1, 32, 64):
            with self.subTest(count=count):
                run = self.run / str(count)
                for name in ("selection", "scene_before"):
                    r.save_json(run / (name + ".json"), r.read_json(self.run / (name + ".json")))
                r.save_json(run / "sources.json", {"bvb": {"frame_count": count}})
                with patch.object(r, "run_blender", side_effect=self.fake_blender):
                    manifest = r.finalize(self.case, run, "unused")
                expected = [1 + round(i * 4627 / (count - 1)) for i in range(count)] if count > 1 else [1]
                self.assertEqual([frame["scene_frame"] for frame in manifest["before_qa"]], expected)
                self.assertEqual(manifest["qa_sampling"]["bvb_frame_count"], count)

    def test_old_16_frame_spec_is_not_overwritten(self):
        old = {"candidate": self.case["candidates"][0], "qa_scene_frames": list(range(16))}
        r.save_json(self.run / "finalize_spec.json", old)
        with patch.object(r, "run_blender") as render:
            with self.assertRaisesRegex(ValueError, "Frozen configuration"):
                r.finalize(self.case, self.run, "unused")
            render.assert_not_called()
        self.assertEqual(r.read_json(self.run / "finalize_spec.json"), old)

    def test_truncated_bvb_matched_manifest_is_rejected_before_api(self):
        with patch.object(r, "run_blender", side_effect=self.fake_blender):
            r.finalize(self.case, self.run, "unused")
        path = self.run / "final/manifest.json"
        manifest = r.read_json(path)
        for name in ("before_qa", "after_qa"):
            manifest[name] = manifest[name][:16]
        r.save_json(path, manifest)
        with patch.object(r, "call_model") as call:
            with self.assertRaisesRegex(ValueError, "frame"):
                r.answer(self.case, self.run, "test-model", "https://api.example.test/v1", 1, "both")
            call.assert_not_called()


class ModelMessageTests(unittest.TestCase):
    def test_model_receives_only_frame_identity_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "frame.png"
            image.write_bytes(b"synthetic image bytes")
            frame = {
                "file": str(image),
                "sha256": r.digest(image),
                "source": "reconstruction",
                "pair_id": 2,
                "source_frame_index_0based": 80,
                "scene_frame": 82,
                "seconds": 3.333,
                "source_pose": [[1, 0, 0, 4], [0, 1, 0, 5], [0, 0, 1, 6], [0, 0, 0, 1]],
                "bvb_camera_matrix": [[1, 0, 0, 4], [0, 1, 0, 5], [0, 0, 1, 6], [0, 0, 0, 1]],
                "target_world_bbox": {"min_xyz": [0, 0, 0], "max_xyz": [1, 1, 1]},
                "reference_world_bbox": {"min_xyz": [2, 2, 2], "max_xyz": [3, 3, 3]},
                "target_projection": {"projected_bbox_xyxy": [0.1, 0.2, 0.3, 0.4]},
                "reference_projection": {"projected_bbox_xyxy": [0.5, 0.6, 0.7, 0.8]},
                "candidate_id": "servo_030",
            }
            captured = {}

            class Completions:
                def create(self, **kwargs):
                    captured.update(kwargs)
                    return SimpleNamespace(
                        model="fake-model",
                        choices=[SimpleNamespace(
                            message=SimpleNamespace(content="{}"), finish_reason="stop")],
                        usage=None,
                    )

            client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
            r.call_model(root / "run", "metadata_check", "fake-model", "https://api.example.test/v1",
                         "Locate the target.", [frame], 100, "low", client=client)
            texts = [part["text"] for part in captured["messages"][0]["content"]
                     if part["type"] == "text"]
            metadata = [text for text in texts if text.startswith("Frame metadata: ")]
            self.assertEqual(len(metadata), 1)
            self.assertEqual(json.loads(metadata[0].split(": ", 1)[1]), {
                "source": "reconstruction",
                "pair_id": 2,
                "source_frame_index_0based": 80,
                "scene_frame": 82,
            })
            message = "\n".join(texts)
            for forbidden in ("source_pose", "bvb_camera_matrix", "target_world_bbox",
                              "reference_world_bbox", "target_projection",
                              "reference_projection", "candidate_id"):
                self.assertNotIn(forbidden, message)


class RetrievalTests(unittest.TestCase):
    def test_paired_retrieval_maps_pair_ids_back_to_each_video(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            case = fixture_case(run)
            r.save_json(run / "blind_case.json", r.blind_case(case))
            for source, offset in (("original", 0), ("bvb", 100)):
                frames = []
                for index in range(24):
                    frames.append({
                        "source_frame_index_0based": offset + index,
                        "file": str(run / f"{source}_{index}.jpg"),
                        "sha256": "unused",
                    })
                r.save_json(run / "frames" / "retrieval" / source / "manifest.json", {"frames": frames})
            response = {
                "text": json.dumps({
                    "selected_pair_ids": [1, 6],
                    "rationale": "fixture",
                    "pair_observations": [],
                })
            }
            with patch.object(r, "call_model", return_value=response):
                result = r.retrieve_targets(case, run, "test-model", "https://api.example.test/v1")
            self.assertEqual(result["original"]["selected_frame_indices_0based"], [3, 20])
            self.assertEqual(result["bvb"]["selected_frame_indices_0based"], [103, 120])
            self.assertEqual(result["paired"]["selected_pair_ids"], [1, 6])

    def test_different_sample_counts_still_pair_full_timelines(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            case = fixture_case(run)
            r.save_json(run / "blind_case.json", r.blind_case(case))
            for source, count in (("original", 24), ("bvb", 8)):
                r.save_json(run / "frames" / "retrieval" / source / "manifest.json", {
                    "frames": [{"source_frame_index_0based": index}
                               for index in range(count)],
                })
            response = {"text": json.dumps({"selected_pair_ids": [0, 7]})}
            with patch.object(r, "call_model", return_value=response):
                result = r.retrieve_targets(case, run, "test-model", "https://api.example.test/v1")
            self.assertEqual(result["original"]["selected_frame_indices_0based"], [0, 23])
            self.assertEqual(result["bvb"]["selected_frame_indices_0based"], [0, 7])

    def test_single_frame_pair_has_zero_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            case = fixture_case(run)
            r.save_json(run / "blind_case.json", r.blind_case(case))
            for source in ("original", "bvb"):
                r.save_json(run / "frames" / "retrieval" / source / "manifest.json", {
                    "frames": [{"source_frame_index_0based": 0}],
                })
            response = {"text": json.dumps({"selected_pair_ids": [0]})}
            with patch.object(r, "call_model", return_value=response) as call:
                r.retrieve_targets(case, run, "test-model", "https://api.example.test/v1")
            self.assertEqual([frame["normalized_time"] for frame in call.call_args.args[5]], [0.0, 0.0])

    def test_legacy_retrieval_is_preserved_without_calling_model(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            old = {"original": {"selected_frame_indices_0based": [9]}}
            r.save_json(run / "target_retrieval.json", old)
            with patch.object(r, "call_model") as call:
                with self.assertRaisesRegex(ValueError, "Legacy retrieval"):
                    r.retrieve_targets(fixture_case(run), run, "test-model", "https://api.example.test/v1")
                call.assert_not_called()
            self.assertEqual(r.read_json(run / "target_retrieval.json"), old)


class SelectionPairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.run = Path(self.temp.name)
        self.case = fixture_case(self.run)
        self.ids = [7, 3, 2, 0]
        self.original_indices = [5784, 2515, 1760, 0]
        self.bvb_indices = [63, 27, 19, 0]
        r.save_json(self.run / "blind_case.json", r.blind_case(self.case))
        r.save_json(self.run / "sources.json", {
            "original": {"frame_count": 5785}, "bvb": {"frame_count": 64},
        })
        r.save_json(self.run / "scene_before.json", {"timeline": [1, 4628]})
        r.save_json(self.run / "target_retrieval.json", {
            source: {"selected_pair_ids": self.ids, "selected_frame_indices_0based": indices}
            for source, indices in (("original", self.original_indices), ("bvb", self.bvb_indices))
        })
        r.save_json(self.run / "frames" / "retrieval" / "original" / "manifest.json", {
            "frames": [{"source_frame_index_0based": index, "file": f"original_{index}.jpg",
                        "sha256": f"original-hash-{index}"} for index in self.original_indices],
        })
        self.manifest_path = self.run / "candidate_renders" / "manifest.json"
        r.save_json(self.manifest_path, {
            "candidates": [{
                "candidate": candidate,
                "frames": [{"scene_frame": frame, "file": f"{candidate['id']}_{frame}.png",
                            "sha256": f"{candidate['id']}-hash-{frame}"}
                           for frame in [1, 1396, 1984, 4628]],
            } for candidate in self.case["candidates"]],
        })
        self.response = {"text": json.dumps({
            "candidate_id": "no_change",
            "supporting_original_frame_indices_0based": [5784, 2515],
        })}

    def tearDown(self):
        self.temp.cleanup()

    def select(self):
        return r.select_candidate(self.case, self.run, "test-model", "https://api.example.test/v1")

    def test_q819_candidates_use_the_same_two_pairs_as_original(self):
        with patch.object(r, "call_model", return_value=self.response) as call:
            self.select()
        frames = call.call_args.args[5]
        originals = [frame for frame in frames if frame["source"] == "original"]
        self.assertEqual([frame["source_frame_index_0based"] for frame in originals], [5784, 2515])
        for candidate in self.case["candidates"]:
            with self.subTest(candidate=candidate["id"]):
                selected = [frame for frame in frames if frame.get("candidate_id") == candidate["id"]]
                self.assertEqual([frame["scene_frame"] for frame in selected], [4628, 1984])
                self.assertEqual([frame["pair_id"] for frame in selected], [7, 3])
        self.assertEqual([frame["pair_id"] for frame in originals], [7, 3])

    def test_missing_paired_frame_stops_before_model_call(self):
        manifest = r.read_json(self.manifest_path)
        manifest["candidates"][0]["frames"] = [
            frame for frame in manifest["candidates"][0]["frames"] if frame["scene_frame"] != 1984
        ]
        r.save_json(self.manifest_path, manifest)
        with patch.object(r, "call_model", return_value=self.response) as call:
            with self.assertRaisesRegex(ValueError, "paired frame"):
                self.select()
            call.assert_not_called()

    def test_pair_ids_must_match_between_sources(self):
        retrieval = r.read_json(self.run / "target_retrieval.json")
        retrieval["bvb"]["selected_pair_ids"] = [0, 2, 3, 7]
        r.save_json(self.run / "target_retrieval.json", retrieval)
        with patch.object(r, "call_model", return_value=self.response) as call:
            with self.assertRaisesRegex(ValueError, "pair"):
                self.select()
            call.assert_not_called()

    def test_pair_mapping_keeps_retrieval_order_and_marks_approximation(self):
        pairs = r.candidate_frame_pairs(self.run)
        self.assertEqual([pair["pair_id"] for pair in pairs], self.ids)
        self.assertEqual([pair["scene_frame"] for pair in pairs], [4628, 1984, 1396, 1])
        self.assertEqual([pair["bvb_frame_index_0based"] for pair in pairs], self.bvb_indices)
        self.assertTrue(all(pair["mapping_method"] == "uniform_full_timeline_approximation"
                            for pair in pairs))

    def test_single_bvb_frame_maps_to_timeline_start(self):
        r.save_json(self.run / "sources.json", {
            "original": {"frame_count": 1}, "bvb": {"frame_count": 1},
        })
        r.save_json(self.run / "target_retrieval.json", {
            source: {"selected_pair_ids": [0], "selected_frame_indices_0based": [0]}
            for source in ("original", "bvb")
        })
        r.save_json(self.run / "scene_before.json", {"timeline": [12, 90]})
        self.assertEqual(r.candidate_frame_pairs(self.run)[0]["scene_frame"], 12)

    def test_candidate_render_spec_preserves_pairs_and_freezes_changes(self):
        with patch.object(r, "run_blender") as render:
            r.render_candidates(self.case, self.run, "unused")
            render.assert_called_once()
        spec = r.read_json(self.run / "candidate_spec.json")
        self.assertEqual([pair["pair_id"] for pair in spec["frame_pairs"]], self.ids)
        self.assertEqual(spec["scene_frames"], [1, 1396, 1984, 4628])
        self.assertEqual(spec["candidates"], self.case["candidates"])
        r.save_json(self.run / "scene_before.json", {"timeline": [2, 4629]})
        with patch.object(r, "run_blender") as render:
            with self.assertRaisesRegex(ValueError, "Frozen configuration"):
                r.render_candidates(self.case, self.run, "unused")
            render.assert_not_called()
        self.assertEqual(r.read_json(self.run / "candidate_spec.json"), spec)

    def test_candidate_file_order_does_not_change_selected_pairs(self):
        manifest = r.read_json(self.manifest_path)
        for item in manifest["candidates"]:
            item["frames"].reverse()
        r.save_json(self.manifest_path, manifest)
        self.test_q819_candidates_use_the_same_two_pairs_as_original()

    def test_changed_candidate_input_cannot_reuse_selection_cache(self):
        with patch.object(r, "call_model", return_value=self.response):
            self.select()
        old = r.read_json(self.run / "selection.json")
        manifest = r.read_json(self.manifest_path)
        manifest["candidates"][0]["frames"][-1]["sha256"] = "changed-frame"
        r.save_json(self.manifest_path, manifest)
        with patch.object(r, "call_model") as call:
            with self.assertRaisesRegex(ValueError, "Frozen configuration"):
                self.select()
            call.assert_not_called()
        self.assertEqual(r.read_json(self.run / "selection.json"), old)

    def test_legacy_selection_is_preserved_without_model_call(self):
        old = {"candidate_id": "no_change"}
        r.save_json(self.run / "selection.json", old)
        with patch.object(r, "call_model") as call:
            with self.assertRaisesRegex(ValueError, "Legacy selection"):
                self.select()
            call.assert_not_called()
        self.assertEqual(r.read_json(self.run / "selection.json"), old)

    def test_paired_selection_uses_new_cache_and_resumes_without_second_request(self):
        r.save_json(self.run / "calls" / "select_candidate" / "response.json", {
            "text": json.dumps({"candidate_id": "toward_050"}),
        })
        create = Mock(return_value=SimpleNamespace(
            model="test-model",
            usage=SimpleNamespace(model_dump=lambda: {"total_tokens": 10}),
            choices=[SimpleNamespace(finish_reason="stop",
                                     message=SimpleNamespace(content=self.response["text"]))],
        ))
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with patch.object(r, "encode_frame", return_value={"type": "text", "text": "synthetic frame"}):
            first = r.select_candidate(self.case, self.run, "test-model",
                                       "https://api.example.test/v1", client=client)
            second = r.select_candidate(self.case, self.run, "test-model",
                                        "https://api.example.test/v1", client=client)
        self.assertEqual(create.call_count, 1)
        self.assertEqual(first, second)
        self.assertEqual(first["candidate_id"], "no_change")
        self.assertEqual([pair["pair_id"] for pair in first["frame_pairs"]], [7, 3])
        request = r.read_json(self.run / "calls" / "select_candidate_paired_v4" / "request.json")
        self.assertEqual(len(request["frames"]), 6)
        self.assertEqual(request["max_completion_tokens"], 3000)


if __name__ == "__main__":
    unittest.main()
