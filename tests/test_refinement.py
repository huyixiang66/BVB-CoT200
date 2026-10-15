import json
import tempfile
import unittest
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

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

    def test_mra_matches_partial_credit_example(self):
        self.assertEqual(r.mra(1.5, 1.2), 6 / 11)
        self.assertEqual(r.mra(1.3, 1.2), 10 / 11)
        self.assertEqual(r.mra(1.8, 1.2), 0.0)

    def test_uniform_indices_include_both_ends(self):
        self.assertEqual(r.uniform_indices(10, 3), [0, 4, 9])

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
        report = r.report(self.case, self.run)
        self.assertEqual(report["summary"]["after"]["values_m"], [1.8, 1.3, 1.8])
        self.assertEqual(report["summary"]["after"]["best_mra"], 10 / 11)
        self.assertFalse(report["summary"]["after"]["any_full_mra"])
        self.assertEqual(report["summary"]["after"]["unique_values_m"], [1.3, 1.8])

    def test_completed_samples_are_not_called_again(self):
        r.answer(self.case, self.run, "test-model", "https://api.example.test/v1",
                 1, "after", client=self.client)
        r.answer(self.case, self.run, "test-model", "https://api.example.test/v1",
                 1, "after", client=self.client)
        self.assertEqual(self.client.chat.completions.create.call_count, 1)

    def test_candidate_override_requires_diagnostic_flag(self):
        with self.assertRaisesRegex(ValueError, "diagnostic"):
            r.finalize(self.case, self.run, "blender", "toward_050", diagnostic=False)


if __name__ == "__main__":
    unittest.main()
