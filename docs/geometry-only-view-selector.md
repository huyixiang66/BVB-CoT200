# Geometry-only Blender view selector

This entry point selects and renders one or a small set of question-conditioned views from a BVB-style Blender scene. It uses deterministic geometry calculations in Blender: target-object bounds, candidate camera positions, frustum coverage, and ray-cast occlusion. It does not call an AI model or image API.

## What each input does

- `.blend`: geometry and object names used to calculate and render views. The file is loaded in a separate background Blender process and is not saved or modified.
- Question JSON: question type and explicit semantic-term-to-mesh-name mapping. This is necessary because pure computation cannot reliably infer which Blender meshes a free-form question refers to.
- Video: recorded in output provenance only. It does not affect camera selection; this selector does not estimate video poses or inspect video frames. The Blender scene is the geometric source of truth for this version.

The supported question types are counting, absolute distance, relative distance, size estimation, room-size estimation, and the three VSI relative-direction levels. `appearance_order` and other temporal questions are unsupported.

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

The question file schema is illustrated in [`data/envelope_views/question.example.json`](../data/envelope_views/question.example.json). Each semantic term contains one or more instances; an instance can consist of multiple mesh names. For example, chair seat, chair back, and chair legs can be listed as one instance, while two separate chairs should be listed as two instances. Mesh names must match the `.blend` exactly. Optional `room.floor` and `room.ceiling` list the Blender meshes that define the room bounds. If omitted, the selector uses detected floor/ceiling names or scene bounds.

## Limits

This is a geometry-based baseline, not a learned quality metric. It optimizes projected visibility and readability proxies; it does not guarantee that a VLM will recognize every object or answer correctly. Mesh-name mapping is user supplied. The video is not aligned to the `.blend`, so changing the video while keeping the scene and question fixed produces the same selected poses.
