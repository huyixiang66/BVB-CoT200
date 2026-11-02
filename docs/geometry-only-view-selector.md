# Geometry-only Blender view selector

This entry point selects and renders one or a small set of question-conditioned views from a BVB-style Blender scene. Relationship, direction, distance, and size views use bounded Powell optimization over camera position and horizontal look-at offset; camera framing is computed from target geometry at each evaluated pose. Counting and room-size views use a geometry-derived orthographic top view. Blender BVH ray casts estimate target-surface visibility and occlusion. No AI model or image API is called.

## What each input does

- `.blend`: geometry and object names used to calculate and render views. The file is loaded in a separate background Blender process and is not saved or modified.
- Question JSON: question type and explicit semantic-term-to-mesh-name mapping. This is necessary because pure computation cannot reliably infer which Blender meshes a free-form question refers to.
- Video: recorded in output provenance only. It does not affect camera selection; this selector does not estimate video poses or inspect video frames. The Blender scene is the geometric source of truth for this version.

The supported question types are counting, absolute distance, relative distance, size estimation, room-size estimation, and the three VSI relative-direction levels. `appearance_order` and other temporal questions are unsupported. The continuous optimizer needs SciPy in Blender's Python environment; install it with Blender's bundled `python.exe -m pip install scipy`.

## Run

Install Blender and set `BLENDER_BIN`, or pass its executable directly. Then run:

```powershell
python scripts\select_views.py `
  --blend C:\data\scene.blend `
  --video C:\data\source.mp4 `
  --question data\envelope_views\question.example.json `
  --out runs\user-view-001 `
  --blender "C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"
```

Use a fresh output directory for each run. The output contains:

- `selected_views.json`: candidate scores, chosen camera positions and look-at targets.
- Each question records `selection_status` and `qa_ready`. A partial set of accepted views can still leave the question incomplete; consumers must not silently evaluate only the successful subset.
- `renders/<qa_id>/*.png`: Blender renders from the selected cameras.
- `render_manifest.json`: image paths, hashes, camera parameters, and input provenance.
- `summary.md`: a compact text summary.

The optimizer records geometry-accepted solutions separately from failed attempts. Every view in `chosen_readable` must pass its declared role's framing (7% border), projected area, minimum short side (10px at a 512px reference resolution), surface-ray visibility, collision/free-space, and any silhouette gate. Each required relation must independently retain at least 0.55 of its direction in projection. A higher utility score cannot compensate for a failed gate. Visibility and area thresholds remain geometric proxies, not guarantees of recognition.

The minimum relationship area is 0.0008 of the image (the previous clear-context floor), while the continuous utility still rewards a larger visible-area proxy up to 0.003. Minimum visibility is 0.20; size views use area 0.006 and visibility 0.55. Two analytic, relation-derived framing poses and one room-centre pose initialize continuous optimization. They are initial states, not a fixed angle grid of output views.

Room-area views have an explicit `room_footprint_layout` scope: the full floor footprint must fit with the same margin/size/free-space gates, but furniture covering bare floor is not treated as a visibility failure. Raw floor visibility is still recorded. This exception does not apply to target-object or counting views.

Relative-distance questions use a continuous overview solve plus two shared-anchor comparison groups, then solve missing pairs up to the six-view limit. Counting starts with an analytic overhead view and, when that fails its gates, solves group or instance context views. A direction reference must show the facing object from the valid standing region; a query object behind the observer is covered by a separate accepted layout/relation view. `standing_joint_view_complete` explicitly records whether the standing image itself shows both.

If no solution passes, `chosen_readable` can be empty and the manifest still retains that question. Failed attempts are stored in `diagnostic_views`. Add `--render-diagnostics` to render them under a separate `diagnostics/` directory; these images are not in the manifest's QA `views` list. No failed target view is replaced with a room image and labelled complete.

The question file schema is illustrated in [`data/envelope_views/question.example.json`](../data/envelope_views/question.example.json). Each semantic term contains one or more instances; an instance can consist of multiple mesh names. Chair seat, back, and legs can be one instance; separate chairs should be separate instances. Mesh names must match the `.blend` exactly. `room.floor` is required and `room.ceiling` is optional. Distance questions require endpoint or anchor roles; direction questions require standing, facing, and query roles.

## Limits

This is a geometry-based local optimizer, not a learned quality metric or guaranteed global optimum. It optimizes projected visibility and readability proxies; it does not guarantee that a VLM will recognize every object or answer correctly. Mesh-name mapping is user supplied. The video is not aligned to the `.blend`, so changing the video while keeping the scene and question fixed produces the same selected poses.
