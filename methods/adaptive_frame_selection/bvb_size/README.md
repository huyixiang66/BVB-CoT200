# Size-question reference and view prototype

**Latest checkpoint:** [2026-11-11 joint reference reconstruction and in-situ camera handoff](HANDOFF_20261111.md). The older extracted-object route below remains available; the new route is experimental and does not replace it.

This experimental module extends the question-conditioned reconstruction workflow for single-object size estimation. It accepts an existing Blender scene and independently verified mesh groups; it does **not** perform frame selection, reconstruction, object verification, scale calibration, V8 integration or QA. It is a portable extraction of the geometry and decisions explored on 2026-11-07, not an exact replay of all earlier ad hoc experiment scripts.

## Decision

`priors.json` records user judgments about whether model size knowledge is reliable. A reliable target is shown without a reference. For an unreliable target, `choose_reference` chooses among independently verified candidates whose categories have reliable priors and are not in `poor_reference`. It ranks the ratio of reconstructed maximum extents, accepting ratios from 0.25 to 4.0. Kettle, blanket and printer are excluded as references; kettle and printer can still be reliable **targets**. Unknown categories and missing candidates produce a target-only fallback with a reason.

All candidates must be in the same calibrated scene, have `verified: true`, and contain complete non-overlapping mesh groups. `verified` is an assertion from the caller, not an automatic visual validation. An object being large or a VLM prior being reliable does **not** establish the actual object size or a correct shared 3D scale.

## Render

Use Blender 4+ with an existing scene. Provide one JSON file such as:

```json
{
  "question": "What is the bowl's longest dimension in centimeters?",
  "target": {"id": "bowl-1", "category": "bowl", "meshes": ["bowl_body"], "verified": true},
  "candidates": [{"id": "pan-1", "category": "pan", "meshes": ["pan_body", "pan_handle"], "verified": true}]
}
```

Run from the `methods/adaptive_frame_selection` directory:

```sh
blender -b /path/to/scene.blend --python bvb_size/render_size.py -- /path/to/groups.json /path/to/output
```

`main.png` shows the target and an optional reference after moving the reference **in memory**, without scaling; the input blend is never saved. Output `manifest.json` records the selection, suggested question prompt, image paths and diagnostics. A second view is rendered if the target projection is under 200 pixels or its estimated longest axis retains under 80% of its length. A small-target detail or high-angle complement has a different scale from the main image; do not compare pixel lengths between images. The second image is explicitly listed in `manifest.json`; callers must pass all listed images to QA.

The view uses an oblique geometric heuristic to expose multiple faces. It is not a semantic recognition model, does not guarantee absence of occlusion, and does not automatically compare original video against the generated scene. If VLM identification, reconstruction, or shared scale is unreliable, inspect the original frames and treat the output as diagnostic. The bounding-box direction is an approximation and can be wrong for rotated objects or deformable objects such as blankets.

## QA and experiments

`reference_prompt` supplies the experimental English prompt and optionally identifies the target on the left and reference on the right. It does not call an API or require a subscription. Use the caller's approved inference interface, record the actual input images, model, tokens and failed attempts, and score VSI numeric questions using `evaluation.vsi_mean_relative_accuracy` on parsed numeric values. An unparseable answer should be recorded separately, not silently assigned a score. Prior experiments on 24 available questions included unequal input budgets (original-frame baseline versus one reconstruction image); they do not establish the performance of this extracted implementation. Five of 29 targets lacked verified geometry in the old scenes. No benchmark frames, predictions, credentials, blend files or weights are distributed here.

## Test

```sh
python -m unittest bvb_size.test_policy
```
