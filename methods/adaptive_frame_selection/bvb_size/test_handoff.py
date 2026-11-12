import json
import tempfile
import unittest
from pathlib import Path
from .handoff import digest, prepare_joint, prepare_qa, prepare_review, prepare_reconstruction, target_views, write


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        frame = self.root / "original.jpg"
        frame.write_bytes(b"fixture image bytes; no API")
        self.evidence = dict(question=dict(id=1, question="Longest size?", objects=["refrigerator"],
                                          ground_truth=999, answer=999), frames=[dict(
            group="g1", file="original.jpg", sha256=digest(frame), timestamp_s=1,
            source_frame_index=30, c2w_blender=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
            intrinsics_640x480=[[1,0,320],[0,1,240],[0,0,1]], compact_camera=[0]*11)])
        self.decision = dict(target_confirmed=True, boundary_visible=True,
                             reference_category="laptop", selected_group_id="g1",
                             boundary_evidence=[dict(timestamp_s=1, description="Visible edges")])
        write(self.root / "evidence.json", self.evidence)
        write(self.root / "decision.json", self.decision)
        (self.root / "base.txt").write_text("Reconstruct observed objects.")

    def joint(self):
        return prepare_joint(self.root / "evidence.json", self.root / "decision.json",
                             self.root / "base.txt", self.root / "joint", "/runs/size-case")

    def test_joint_preserves_pose_and_excludes_answer(self):
        result = self.joint()
        self.assertNotIn("ground_truth", result["question"])
        self.assertNotIn("answer", result["question"])
        self.assertEqual(result["frames"][0]["c2w_blender"], self.evidence["frames"][0]["c2w_blender"])
        self.assertEqual(digest(self.root / "joint/frames/0000.jpg"), self.evidence["frames"][0]["sha256"])
        self.assertEqual(result["instance_limits"], {"refrigerator":1, "laptop":1})

    def test_unverified_boundary_falls_back_and_bad_timestamp_rejected(self):
        write(self.root / "decision.json", dict(self.decision, boundary_visible=False))
        result = self.joint()
        self.assertEqual(result["reference_policy"]["fallback_reason"], "reference_boundary_unconfirmed")
        import shutil
        shutil.rmtree(self.root / "joint")
        write(self.root / "decision.json", dict(self.decision, boundary_evidence=[{"timestamp_s":99}]))
        with self.assertRaises(ValueError): self.joint()

    def test_reject_hash_mismatch(self):
        (self.root / "original.jpg").write_bytes(b"changed")
        with self.assertRaises(ValueError):
            self.joint()

    def test_pair_prompt_and_original_order(self):
        self.joint()
        for name in ("a.png", "b.png"):
            (self.root / name).write_bytes(name.encode())
        write(self.root / "views.json", dict(mode="pair", images=["a.png", "b.png"]))
        req = prepare_qa(self.root / "joint/input.json", self.root / "views.json",
                         self.root / "request.json", True)
        self.assertEqual(len(req["images"]), 3)
        self.assertIn("Image 2", req["prompt"].replace("image 2", "Image 2"))
        self.assertIn("Images 3–3", req["prompt"])
        self.assertEqual(req["images"][-1]["sha256"], self.evidence["frames"][0]["sha256"])
        self.assertNotIn("999", req["prompt"])

    def test_no_reference_policy_ignores_inapplicable_candidate(self):
        self.evidence["question"]["objects"] = ["table"]
        write(self.root / "evidence.json", self.evidence)
        result = self.joint()
        self.assertEqual(result["question"]["objects"], ["table"])
        self.assertEqual(result["reference_policy"]["status"], "target_only")
        self.assertNotIn("laptop", result["colors"])

    def test_required_but_missing_reference_falls_back_without_inventing(self):
        self.decision.update(reference_category=None, boundary_visible=False, boundary_evidence=[])
        write(self.root / "decision.json", self.decision)
        result = self.joint()
        self.assertEqual(result["reference_policy"]["status"], "missing_eligible_reference")
        self.assertEqual(result["question"]["objects"], ["refrigerator"])
        (self.root / "a.png").write_bytes(b"image")
        write(self.root / "views.json", dict(mode="target_only", images=["a.png"]))
        req=prepare_qa(self.root / "joint/input.json", self.root / "views.json", self.root / "request.json")
        self.assertNotIn("green is reference", req["prompt"])

    def test_ineligible_reference_rejected(self):
        self.decision["reference_category"] = "printer"
        write(self.root / "decision.json", self.decision)
        self.assertEqual(self.joint()["reference_policy"]["fallback_reason"], "ineligible_reference_category")

    def test_review_and_target_views_adapter(self):
        review = prepare_review(self.root / "evidence.json", self.root / "review.json")
        self.assertTrue(review["reference_required"])
        self.assertNotIn("ground_truth", review["question"])
        (self.root / "main.png").write_bytes(b"image")
        write(self.root / "manifest.json", {"reference":None,"images":["main.png"]})
        views = target_views(self.root / "manifest.json", self.root / "views.json")
        self.assertEqual(views["mode"], "target_only")
        write(self.root / "manifest.json", {"reference":"laptop","images":["main.png"]})
        with self.assertRaises(ValueError): target_views(self.root / "manifest.json", self.root / "views.json")

    def test_model_requests_bind_images_and_pose(self):
        review=prepare_review(self.root / "evidence.json", self.root / "review.json")
        self.assertEqual(len(review["images"]),1)
        self.assertIn('Image order:',review['prompt'])
        self.joint()
        request=prepare_reconstruction(self.root/'joint/input.json',self.root/'reconstruct.json')
        self.assertEqual([x['type'] for x in request['content']],['text','image'])
        self.assertIn('compact_camera',request['content'][0]['text'])
        self.assertNotIn('999',request['prompt'])
        (self.root/'joint/frames/0000.jpg').write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):
            prepare_reconstruction(self.root/'joint/input.json',self.root/'other.json')

    def test_unresolved_does_not_become_qa(self):
        self.joint()
        write(self.root / "views.json", dict(mode="unresolved", images=["diagnostic.png"]))
        with self.assertRaises(ValueError):
            prepare_qa(self.root / "joint/input.json", self.root / "views.json", self.root / "request.json")
        self.assertFalse((self.root / "request.json").exists())


if __name__ == "__main__":
    unittest.main()
