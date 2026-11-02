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
- `renders/<qa_id>/*.png`: Blender renders from the selected cameras.
- `render_manifest.json`: image paths, hashes, camera parameters, and input provenance.
- `summary.md`: a compact text summary.

The question file schema is illustrated in [`data/envelope_views/question.example.json`](../data/envelope_views/question.example.json). Each semantic term contains one or more instances; an instance can consist of multiple mesh names. Chair seat, back, and legs can be one instance; separate chairs should be separate instances. Mesh names must match the `.blend` exactly. `room.floor` is required and `room.ceiling` is optional. Distance questions require endpoint or anchor roles; direction questions require standing, facing, and query roles.

## Limits

This is a geometry-based local optimizer, not a learned quality metric or guaranteed global optimum. It optimizes projected visibility and readability proxies; it does not guarantee that a VLM will recognize every object or answer correctly. Mesh-name mapping is user supplied. The video is not aligned to the `.blend`, so changing the video while keeping the scene and question fixed produces the same selected poses.
