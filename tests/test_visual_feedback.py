import json
import math
import unittest
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import refinement as r
from scripts import run_visual_feedback as runner
from scripts.run_visual_feedback import parse_action
from visual_feedback import validate_action


class VisualFeedbackTests(unittest.TestCase):
    def action(self, **kwargs):
        return {"action": "adjust", "subject": "camera", "translation_local": [0, 0, .3],
                "rotation_local_degrees": [0, 0, 0], "scale_ratio": 1, **kwargs}

    def test_direct_camera_retreat_is_preserved(self):
        value = self.action()
        self.assertEqual(parse_action({"finish_reason": "stop", "text": json.dumps(value)}), value)

    def test_stop_needs_no_geometry_or_confidence(self):
        self.assertEqual(validate_action({"action": "stop", "reason": "Cannot determine"})["action"], "stop")

    def test_invalid_values_and_oversized_changes_are_rejected(self):
        for changes in ({"translation_local": [math.nan, 0, 0]},
                        {"translation_local": [.4, .4, 0]},
                        {"rotation_local_degrees": [9, 0, 0]},
                        {"scale_ratio": 1.1}, {"translation_local": [True, 0, 0]}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_action(self.action(**changes))

    def test_target_scale_and_translation_are_supported(self):
        value = self.action(subject="target", translation_local=[.1, 0, 0], scale_ratio=1.1)
        self.assertEqual(validate_action(value), value)

    def test_target_cannot_receive_camera_rotation(self):
        with self.assertRaises(ValueError):
            validate_action(self.action(subject="target", rotation_local_degrees=[1, 0, 0]))

    def test_joint_camera_and_target_changes_are_supported(self):
        value = {"action": "adjust", "subject": "both",
                 "camera": {"translation_local": [0, 0, .3], "rotation_local_degrees": [0, 0, 6]},
                 "target": {"translation_local": [.1, .08, 0], "scale_ratio": .9}}
        self.assertEqual(parse_action({"finish_reason": "stop", "text": json.dumps(value)}), value)

    def test_joint_changes_enforce_each_subjects_bounds(self):
        for camera, target in (({}, None),
                               ({"translation_local": [0, 0, .6]}, {"translation_local": [0, 0, 0]}),
                               ({"translation_local": [0, 0, 0]}, {"translation_local": [.3, 0, 0]}),
                               ({"translation_local": [0, 0, 0]},
                                {"translation_local": [0, 0, 0], "scale_ratio": .7})):
            with self.subTest(camera=camera, target=target), self.assertRaises(ValueError):
                validate_action({"action": "adjust", "subject": "both", "camera": camera, "target": target})

    def test_loop_sends_the_new_render_to_the_second_comparison(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, before, blend = [root / name for name in ("source.jpg", "before.png", "original.blend")]
            for path in (source, before, blend):
                path.write_bytes(path.name.encode())
            pair = {"pair_id": 3, "source": {"file": str(source), "sha256": r.digest(source)},
                    "render": {"file": str(before), "sha256": r.digest(before)}}
            case = {"inputs": {"blend": str(blend)}, "objects": {"moving": {"label": "monitor"}}}
            r.save_json(root / "case.json", case)
            r.save_json(root / "pairs.json", [pair])
            seen, scenes = [], []

            def model_call(run, name, model, url, prompt, frames, *args, **kwargs):
                seen.append(frames[1]["file"])
                action = {"action": "adjust", "subject": "both",
                          "camera": {"translation_local": [0, 0, .3]},
                          "target": {"translation_local": [.1, 0, 0], "scale_ratio": 1.1}}
                return {"finish_reason": "stop", "text": json.dumps(action)}

            def blender_call(command, **kwargs):
                spec_path = Path(command[-1])
                spec = r.read_json(spec_path)
                self.assertEqual(spec["action"]["subject"], "both")
                scenes.append(command[command.index("--disable-autoexec")+1])
                output = spec_path.parent / "render.png"
                output.write_bytes(b"updated")
                saved = spec_path.parent / "scene.blend"
                saved.write_bytes(b"corrected scene")
                updated = {**spec["pair"], "render": {"file": str(output), "sha256": r.digest(output)}}
                r.save_json(spec_path.parent / "state.json", {"pair": updated, "blend": str(saved)})

            argv = ["runner", "--case", str(root / "case.json"), "--pairs", str(root / "pairs.json"),
                    "--pair-id", "3", "--run", str(root / "run"), "--blender", "blender",
                    "--model", "vision-model", "--rounds", "3"]
            with patch.object(sys, "argv", argv), patch.object(runner.r, "call_model", model_call), \
                    patch.object(runner.r, "blender_command", return_value="blender"), \
                    patch.object(runner.subprocess, "run", blender_call):
                runner.main()
            self.assertEqual(seen[0], str(before))
            self.assertEqual(seen[1], str(root / "run/edit_01/render.png"))
            self.assertEqual(seen[2], str(root / "run/edit_02/render.png"))
            self.assertEqual(scenes, [str(blend), str(root / "run/edit_01/scene.blend"),
                                      str(root / "run/edit_02/scene.blend")])
            self.assertEqual(len(seen), 3)
            result = r.read_json(root / "run/result.json")
            self.assertFalse(result["model_stopped"])
            self.assertEqual(result["applied_edit_rounds"], 3)
            self.assertEqual(blend.read_bytes(), b"original.blend")

    def test_stop_does_not_launch_blender_or_claim_an_edit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, blend = root / "source.jpg", root / "original.blend"
            source.write_bytes(b"frame")
            blend.write_bytes(b"scene")
            frame = {"file": str(source), "sha256": r.digest(source)}
            r.save_json(root / "pairs.json", [{"pair_id": 0, "source": frame, "render": frame}])
            r.save_json(root / "case.json", {"inputs": {"blend": str(blend)},
                                           "objects": {"moving": {"label": "monitor"}}})
            argv = ["runner", "--case", str(root / "case.json"), "--pairs", str(root / "pairs.json"),
                    "--pair-id", "0", "--run", str(root / "run"), "--blender", "blender",
                    "--model", "vision-model", "--rounds", "5"]
            response = {"finish_reason": "stop", "text": '{"action":"stop","reason":"Done"}'}
            with patch.object(sys, "argv", argv), patch.object(runner.r, "call_model", return_value=response), \
                    patch.object(runner.subprocess, "run") as render:
                runner.main()
            render.assert_not_called()
            result = r.read_json(root / "run/result.json")
            self.assertTrue(result["model_stopped"])
            self.assertEqual(result["applied_edit_rounds"], 0)
            self.assertEqual(result["result_blend"], str(blend))


if __name__ == "__main__":
    unittest.main()
