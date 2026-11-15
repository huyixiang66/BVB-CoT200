# BVB V13 release

Published source snapshot: 2026-11-15, version 13.0.0, branch `v_13`.
Based on the published `v_11` repository commit `e16ff5652a5d2cb8841bc4b949b81fa9234a8467`; the V13 package is independently derived from the local V12 light 2 implementation and bundles its own runtime dependencies. Historical packages retain their existing behavior.

## Setup and offline verification

From the repository root, with Python 3.11 or newer:

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r methods/adaptive_frame_selection/bvb_v13/_runtime/requirements-v12.txt
cd methods/adaptive_frame_selection
python -m bvb_v13 --version
python -m bvb_v13 verify
python -m bvb_v13 --help
python -m unittest discover -s bvb_v13/_runtime/tests_v12
```

The `requirements-v12.txt` and `tests_v12` names are retained from the bundled runtime. The source lock checks 236 files. `_runtime/bvb_v13` is a relative symlink to `..`; preserve Git symlinks when checking out this package. No local experiment directories are required for these checks.

For actual runs, prepare your own videos, questions, features and poses and configure the model provider and Blender environment as described in the [V13 guide](methods/adaptive_frame_selection/bvb_v13/README.md) and [bundled runtime guide](methods/adaptive_frame_selection/bvb_v13/_runtime/bvb_v10/README.md). Raw-video feature extraction additionally requires the GPU dependencies and external models described in the bundled runtime. Local Blender defaults to the macOS application path and can be selected with `--blender` where supported. Some inherited rendering/sandbox paths are platform specific; this release does not claim cross-platform validation.

## Scope

This snapshot preserves the local V13 implementation and its method lock, including compact-scene-v2, task-essential-geometry-v1, tabular scene evidence, answer-only QA with NEED_MORE_VIEWS, 0.5-second reconstruction-frame deduplication, full counting category palettes and object-size original-image evidence.

Included: method source, bundled runtime source, tests, dependency manifests, source provenance and historical validation metadata. Provenance and validation records may name local source or experiment paths; those paths are references, not bundled datasets. The method README also describes local experiments that are not part of this release.

Excluded: experiment results, generated HTML reports, media, model weights, caches, account configuration and credentials. Publishing checks make no model calls and do not establish new accuracy results. See `V13-VALIDATION.json` for checks performed on the exported Git contents.
