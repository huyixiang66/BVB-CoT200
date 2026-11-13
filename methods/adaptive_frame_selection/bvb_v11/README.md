# BVB V11

V11 packages the current local V10_2 task adapters behind one versioned entry point.
It adds portable paths, a seven-task dispatcher, a source fingerprint, and a pinned
room-size backend. It does not publish local videos, model weights, credentials,
historical answers, or experiment output directories.

## Architecture

```text
video → 2 fps frame / DINO / optical-flow / LingBot caches
      → question-specific peaks → appearance and pose clusters
      → Astra question policy, clean-image identity review, instance comparison
      → reject/uncertain/missing category: 1 fps full-video search + clean re-review
      → admitted clusters + task-specific structural context
      → one joint Astra bpy reconstruction with soft trajectory/surface anchors
      → execution / category validation → render candidates → select views → QA
```

Trajectory and sparse ROI anchors are uncertain spatial evidence, not room
boundaries or metric calibration. Reconstruction can reject or move anchor
hypotheses. There is no post-generation coordinate clamping or registration.
Public questions and visual evidence enter inference; reference answers do not.

| Task | Current adapter |
|---|---|
| Relative direction | Soft anchors, 9 candidate views, select 1, imagined standing/facing direction |
| Relative distance | Multiple instances, category coverage and colors, 13 candidates, 3 complementary views |
| Absolute distance | Stable physical dimension selected from an endpoint or third cluster; 2-second temporal bridge plus uncovered camera-position representatives; 3 views; metres |
| Counting | Counting-only 0.25 peak threshold, no singleton cap, nearest-neighbour instance colors, 3 views; recovery before zero-evidence handling |
| Object size | Longest complete physical dimension, 3 views; centimetres |
| Route planning | 16 uniform layout frames, keep walls and openings, top + 2 oblique views; validate action order |
| Room size | Existing `.blend` + explicit floor/ceiling mapping → geometry-only top-view selection → fresh Astra area QA |

The six reconstruction adapters remove detected higher non-target occluders for
display, while preserving targets, walls required for navigation, object geometry,
and the original saved scene. Visibility is restored between views. Finite ray
sampling is not a proof of complete visibility. V11 also connects this pass to the
legacy 9-view relative-direction route, which previously bypassed the shared
reliability renderer. The room backend keeps its upstream ceiling-cutaway policy.

Absolute distance defaults to **stable dimensions**, with width, length, height or
diameter eligible. `--scale-policy width-largest` is an explicit optional variant:
only stable natural front-facing widths are eligible, then prefer the largest
reliable physical width. It does not resume the paused historical width experiment.
Reference dimensions are common-sense priors with uncertainty, never measured
calibration. Unavailable/invalid references stop the run instead of silently using
an unstable ruler. Camera-position groups do not denote rooms; temporal bridge
coverage does not guarantee reconstruction stability or a direct spatial corridor.

## Installation and verification

From `methods/adaptive_frame_selection`:

```sh
python -m pip install -r requirements-v11.txt
python -m bvb_v11 verify
python -m unittest discover -s tests_v11 -p 'test_*.py'
```

Python 3.11+ is expected. GPU preprocessing needs the existing GPU requirements,
cached detector/appearance weights and a LingBot installation; see the inherited
feature documentation. Generation uses the configured subscription-direct client
with `gpt-6-astra`, high reasoning, and no paid-key fallback. Authorize locally with
`scripts/test_subscription_direct.py`; no authorization state is shipped.

The generated-code Blender executor currently uses **macOS `sandbox-exec`**.
On other operating systems, supply a separately isolated executor; this release
deliberately fails rather than running generated code without isolation. The room
selector uses known vendored code and requires SciPy in Blender's Python environment.
Its source is pinned unchanged to
[`pure-blender-view-selector`, d4bc0c8](https://github.com/huyixiang66/BVB-CoT200/tree/d4bc0c876917c0e2b8620fbea025475859548c83).

## Six reconstruction task families

Use one video/scene and one task family per workflow. Public question JSON is a
list, or `{ "questions": [...] }`. Each item needs `id`, `scene_name`, `question`,
`question_type`, `objects`, and task-specific `roles`; categorical tasks also need
`options`. Preserve the public template's role schema (for example, abs dist uses
`first` and `second`). Keep answer keys in a separate file. The detector vocabulary
must include every queried category and any separately desired reference candidates.

```sh
# Optional when valid 2 fps caches already exist.
python -m bvb_v11 features --video /data/scene.mp4 \
  --questions /data/detector-questions.json --lingbot-root /opt/LingBot \
  --out /data/features --keyframe-interval auto

python -m bvb_v11 sample --cache /data/features/detector/local-cache \
  --poses /data/features/poses --flow /data/features/native-flow.csv \
  --questions /data/public-questions.json --out /data/sampling --device cuda

python -m bvb_v11 audit --selection /data/sampling/clusters/selection.json \
  --out /data/audit

python -m bvb_v11 workflow \
  --selection /data/audit/input/review/reviewed-selection.json \
  --cache /data/features/detector/local-cache --poses /data/features/poses \
  --video /data/scene.mp4 --out /data/v11-run --blender /path/to/blender
```

For counting, add `--counting-mode` to **both sample and audit**; do not reuse a
0.5-threshold selection as a 0.25 experiment. For abs dist, optionally add
`--reference-selection /data/scene-candidates/reviewed-selection.json` to workflow.
These must be same-scene positively reviewed clusters with exact cached source
frames. Without this extra catalog, references come from the supplied admitted
clusters, including the question endpoints. Do not assume a nonexistent all-scene
object detector is run implicitly.

Recovery needs the full exact 1 fps subset of the canonical 2 fps cache, with
hashes matching LingBot sampling. Missing images raise an error listing paths;
there is no hardcoded SSH host or timestamp approximation. Context and reference
frames also retain their original hashes. A search proposal only enters a scene
after positive clean-image review. Failed recovery remains explicit; counting's
completed no-evidence path reports low confidence rather than treating a question
as proof that an instance exists.

Workflow outputs include `run-config.json`, `scenes/`, `semantic-recovery/`,
`references/`, `contexts/`, `prepared/`, and `questions/<id>/`. Each question retains
input manifests, prompts, model request/usage receipts, generated scripts, validation,
renders, view selections and QA. Existing successful calls are resumed. Potentially
live or incomplete requests are not silently resubmitted. Preserve failed receipts
before a deliberate retry; geometry repairs are bounded to three generated scripts.

For already prepared packets:

```sh
python -m bvb_v11 prepare --source /data/prepared/42 \
  --poses /data/features/poses --out /data/v11-question
python -m bvb_v11 run --out /data/v11-question --blender /path/to/blender
```

Abs dist additionally requires `source/scale-reference.json` and
`--context /data/v11-run/contexts/42.json`; the workflow creates both automatically.

## Room-size backend

Room size has a different input boundary: **the scene already exists**. This GitHub
branch does not reconstruct a room or select original video frames. Its selector
makes zero model calls, followed by fresh Astra QA. Historical scene generation
cost is excluded; do not compare this backend's tokens with end-to-end costs for
the six other task families.

Provide explicit mesh names in selector JSON:

```json
{
  "qa_id": "room-1",
  "scene": "scene-id",
  "question_type": "room_size_estimation",
  "question": "What is the floor area of this room in square metres?",
  "targets": [],
  "room": {
    "floor": ["Main floor"],
    "ceiling": ["Ceiling"],
    "walls": [],
    "remove_walls": false
  }
}
```

```sh
python -m bvb_v11 room --blend /data/scene.blend --video /data/scene.mp4 \
  --question /data/room-question.json --out /data/v11-room \
  --blender /path/to/blender
```

Only accepted `qa_ready` views enter QA; rejected diagnostics do not. The original
blend hash is checked. See [upstream geometry-selector details](vendor/room_selector/UPSTREAM.md).

## Token and time receipts

Add global `--measure` **before the subcommand** to record every new model attempt
in a sibling `<out>-usage/cost-calls/` directory. Input includes cached input;
output includes reasoning, so neither subset is added twice. Missing usage remains
unknown. Calls and queue times are separate; parallel API durations can exceed
wall-clock time. Successful cached receipts create no new usage entry. The current
35-question cost study and its HTML remain local and separate from the main table.

`source-provenance.json` identifies the local modules and pinned room revision;
`method-lock.json` fingerprints this release's executable sources. Packaging and
adapter tests are distinct from a fresh 35-question V11 accuracy evaluation.
