import unittest

import refine_structure as s


class StructuralAnchorTests(unittest.TestCase):
    def test_structural_names_are_candidates_not_substring_aliases(self):
        for name in ("round finial.001", "headboard arched crown", "frame rail", "door frame"):
            self.assertTrue(s.structural_name(name))
        for name in ("side skirt pleat", "postcard", "window curtain", "headboard pillow", "soft duvet"):
            self.assertFalse(s.structural_name(name))

    def test_description_uses_world_bounds_and_reports_uncertainty(self):
        bounds = {"min_xyz": [0, 0, 1], "max_xyz": [2, 2, 3]}
        result = s.point_description([1, 1, 3], bounds, ["Bed 2", "Detail"], "topmost_vertex")
        self.assertEqual(result["world_height_band"], "upper")
        self.assertEqual(result["collections"], ["Bed 2", "Detail"])
        self.assertIn("reject", result["warning"])
        self.assertEqual(s.point_description([1, 1, 2], bounds, [], "corner")["world_height_band"],
                         "middle-height")

    def test_flat_bounds_do_not_invent_top_or_bottom(self):
        bounds = {"min_xyz": [0, 0, 0], "max_xyz": [0, 1, 0]}
        self.assertEqual(s.point_description([0, 0, 0], bounds, [], "corner")["world_height_band"],
                         "middle-height")

    def test_background_prompt_retains_geometry_hint_without_target_catalog(self):
        import refine_grounding as g
        proposal = {"landmarks": [{"id": "post:top", "object_name": "post", "xy": [0.4, 0.2],
                                   "point_description": {"kind": "topmost_mesh_vertex"}}]}
        self.assertIn("topmost_mesh_vertex", g.background_prompt("nightstand", proposal))


if __name__ == "__main__":
    unittest.main()
