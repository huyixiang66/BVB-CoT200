"""Read scene names, hierarchy and bounds only; no rendering or image access."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import refine_blender as b

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
if args.output.exists():
    raise ValueError("Refusing to overwrite a frozen audit")
source = Path(bpy.data.filepath)
records = []
for obj in sorted(bpy.data.objects, key=lambda obj: obj.name):
    if obj.type != "MESH":
        continue
    records.append({
        "name": obj.name, "mesh_data": obj.data.name,
        "parent": obj.parent.name if obj.parent else None,
        "collections": sorted(group.name for group in obj.users_collection),
        "hide_render": obj.hide_render, "world_bbox": b.bbox([obj]),
        "custom_property_keys": sorted(obj.keys()),
    })
payload = {
    "source_blend": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "objects": records,
    "text_blocks": [{"name": text.name, "line_count": len(text.lines)} for text in bpy.data.texts],
    "no_image_access": True, "rendered": False, "source_modified": False,
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps({"meshes": len(records), "text_blocks": payload["text_blocks"]}))
