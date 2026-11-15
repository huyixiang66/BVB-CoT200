"""Blender: optional target-category colors after the standard oblique preview.

Run after render_fps_ablation.py in the same Blender process. Does not save blend.
"""
import bpy
import hashlib
import json
import math
from pathlib import Path
import sys

folder=Path(sys.argv[sys.argv.index('--')+1]).resolve()
audit=json.loads((folder/'question-audit.json').read_text())
palette=[(.85,.08,.055,1),(.05,.48,.15,1),(.06,.30,.90,1),(.95,.43,.04,1),(.55,.10,.70,1)]
colors={label:palette[i%len(palette)] for i,label in enumerate(audit['keywords'])}
if (folder/'target-colors-input.json').exists():
    supplied=json.loads((folder/'target-colors-input.json').read_text())
    colors={label:supplied[label] for label in audit['keywords']}
for o in bpy.context.scene.objects:
    if o.type=='MESH':o.color=o.data.materials[0].diffuse_color if o.data.materials else (.85,.85,.82,1)
for target in audit['targets']:
    for o in bpy.data.collections[target['id']].objects:
        if o.type=='MESH':o.color=colors[target['label']]
scene=bpy.context.scene;scene.display.shading.color_type='OBJECT'
scene.render.filepath=str(folder/'targets-highlight.png');bpy.ops.render.render(write_still=True)
top=json.loads((folder/'scene-inspection.json').read_text())['preview_views']['top']
scene.camera.rotation_euler=[math.radians(v) for v in top['rotation_degrees']]
scene.camera.location=top['location'];scene.camera.data.ortho_scale=top['ortho_scale']
scene.render.filepath=str(folder/'targets-highlight-top.png');bpy.ops.render.render(write_still=True)
(folder/'target-colors.json').write_text(json.dumps(colors,indent=2)+'\n')
expected=json.loads((folder/'scene-inspection.json').read_text())['source_sha256']
assert hashlib.sha256(Path(bpy.data.filepath).read_bytes()).hexdigest()==expected
print('QUESTION_TARGET_HIGHLIGHT_OK')
