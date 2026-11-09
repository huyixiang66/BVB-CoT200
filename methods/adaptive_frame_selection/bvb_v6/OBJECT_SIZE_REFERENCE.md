# V8 object-size reference reconstruction

This branch packages the method used in the 2026-11-09 local 19-question experiment. It starts from upstream `011ccf10`, preserving that experiment’s code baseline rather than silently incorporating later upstream changes.

## Model inputs and switch

`prepare --object-size-reference-mode` is opt-in and defaults off. It accepts one object-size question, one target class, and exactly one selected cluster. The experiment runner asks Astra to compare every candidate cluster using original video frames and provisional detector boxes, then retains one cluster. Reconstruction receives that cluster’s original frames, public question, and shared pose evidence. It must produce one target instance and a separate visible scale reference, or explicitly report that no reference is available.

The reference length is an estimated visual prior, never the benchmark answer. The Blender renderer attaches a numbered meter ruler to the reference and records its assumed scale and uncertainty. QA receives two renders (target/reference overview and reference close-up) plus all 4–8 original V8 frames selected for that cluster. QA does not receive benchmark answers, prior answers, target dimensions, or scene coordinates.

## Running from this checkout

Run commands from `methods/adaptive_frame_selection`. Install the existing requirements plus `requirements-review19.txt`. Blender and the externally prepared V8 detection/pose/selection inputs are required. The scripts are an experiment replay harness; video, caches, input manifests and generated assets stay external to Git.

```sh
python scripts/test_v8_size_reference.py
python scripts/test_v8_final_instances.py
python -m bvb_v6 prepare --help
python scripts/run_vsi_review19_v8_reference.py --root /path/to/experiment --object-size-reference-mode
```

The experiment root must contain `questions.public.json` and `inputs/<scene>/detector/local-cache`, `inputs/<scene>/poses`, and `inputs/<scene>/clusters/selection.json`. Original frame paths and hashes must resolve on the running machine. Keep outputs from enabled and disabled modes separate. The runner uses explicitly authorized direct subscription credentials; `scripts/test_subscription_direct.py` performs interactive local OAuth authorization and a small smoke request. Credentials live in ignored `.local/siwc-direct/`; no credentials are packaged. Generic preparation/submission also supports the existing explicit providers.

## QA and local HTML comparison

The following replay scripts expect the original experiment layout beneath the method directory:

- `results/vsi-review19-v8-reference-20261109/`: public questions, reconstruction artifacts and QA receipts.
- `exports/vsi-review19-uniform-20261109/questions.json`: evaluation metadata; scoring only, not model inputs.
- `results/vsi-review19-uniform-{1,0.5,0.25,0.125}fps-20261109/answers.json`: uniform original-frame baselines.
- `upstream/VSI-Bench/scannetpp/<scene>.mp4`: local original videos.

```sh
python scripts/qa_vsi_review19_v8_reference.py --help
python scripts/run_vsi_review19_rates.py --help
python scripts/report_vsi_review19_v8_reference.py
python scripts/validate_vsi_review19_reference_qa.py
```

The HTML defaults to the 1 FPS uniform baseline and switches among all four rates. Each switch synchronizes original answer, original evidence explanation (`reason`), MRA, frame count, and improvement/decrease markers. The earlier review remains separately labelled history. The report compares two different input protocols, not a single-variable ablation. Its MRA follows this experiment’s explicit 11 thresholds (`linspace(0.5, 0.95, 11)`), not the standard ten-threshold variant.

All generated reports remain local. This source branch contains no generated HTML, frames, videos, Blender scenes, request receipts, credentials, or benchmark results.
