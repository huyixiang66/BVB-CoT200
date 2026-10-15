# Sources and Scope

This repository packages an existing development experiment, not a new method evaluation.
It does not include the source videos, Blender assets, rendered frames, credentials, or
private repository contents.

- Questions/labels: 200 selected annotations from `nyu-visionx/VSI-Bench`,
  eight categories with 25 questions each. The dataset card declares Apache-2.0.
  Selection, separation of labels, and render/prompt recipes are local modifications.
  Source: https://huggingface.co/datasets/nyu-visionx/VSI-Bench
- Assets: 126 scenes each for Astra xhigh and Luna xhigh from `yunlong10/BVB-results`.
  `data/scenes.json` (Astra) and `data/scenes_luna.json` pin the same dataset
  revision, filenames, byte counts and SHA-256. Luna hashes come from authenticated
  upstream LFS metadata; 20 locally cached Luna files were also hash-verified.
  One authenticated Luna download was verified against the pinned hash during
  packaging. This is not a full 126-scene download or rendering validation.
  The downloader retrieves only those files. No new asset redistribution license is asserted.
  Source: https://huggingface.co/datasets/yunlong10/BVB-results
- `scoring.py` contains the `semantic_correct` helper and its dependencies from BVB.
  Copyright (c) 2026 Yolo Y. Tang, MIT License; see `licenses/BVB-MIT.txt`.
  Source: https://github.com/yunlong10/BVB/blob/main/eval/dual_vqa_scoring.py
- The local renderers, name matching and prompt text are frozen from the existing
  development pipeline. Source hashes and the migration scope are in `data/provenance.json`.
  No license is assigned to unrelated assets or to the whole repository by this packaging step.

The five newer types use C: original **BVB-generated** camera animation, 16 highlighted
frames, no extra overhead/elevation cameras, no numeric answer annotations.
This is not ground-truth source-video camera alignment.

Counting, relative direction and appearance order use the presentation page's v2
rendering recipes and CoT, but **one answer instead of three samples/voting**.
The Astra v2 recipes contain development-time name matching and manual overrides.
They are not a general, held-out spec generator. Name matches are not verified instances.
Missing highlights are preserved, not repaired using ground truth.

The archived v2 calls did not save prompt hashes. Published v2 prompts are generated
from the archived harness; the three displayed examples match the archived webpage
prompts. This does not prove byte-for-byte identity with every historical API request.
New requests record their actual prompt and frame hashes.

One packaging correction is explicit: counting q2583 has no matched highlight
and only six frames, not fourteen. Its prompt now describes those six frames and
the missing highlights; no geometry, label, target match, or counting strategy
was changed. All 200 questions remain in the denominator (3,217 Astra frames total).

Luna uses the same question set, five-type C steps and three legacy view families,
but resolves names against its own visible scene meshes, without Astra's manual
object-name overrides. Its fixed-camera sampling uses its own blend timeline.
Legacy close-ups require matched targets; standing-position views require both
standing and facing references; trajectory samples can be fewer than 24 on a short
timeline. Its actual view list and colour legend are generated after rendering.
The adapted legacy prompt retains the per-type reasoning procedure but removes
assumptions about universally available highlights and hardcoded frame numbers.
This source option is not a controlled, identical-input Astra/Luna comparison.

Scene source selection is independent of the QA model. The selected API must
support vision Chat Completions and the configured output token parameter.
No QA model is silently selected or substituted. Source, requested QA model and
provider-reported model are recorded separately. The scene's xhigh generation
setting does not impose an xhigh reasoning setting on the answering model.

The five-type C result (51/125), historical v1 (91/200), and the presentation page's
mixed aggregate (87/200) are **not results of this all-single-answer release**.
No 200-question accuracy is claimed until this release is actually run.

Scoring follows the BVB pilot helper: counting tolerance below 0.5, other numeric
answers within 20%, and option/semantic matching. It is not official VSI-Bench
numeric MRA and not the dual-VQA retention metric. Missing, invalid and failed
answers count as incorrect in the fixed selected denominator.

The scoring stage is the only stage that reads label values. Asset-derived geometry
in the legacy views is not ground-truth geometry. Additional renderer access and
prompt changes prevent attributing differences to CoT alone.
