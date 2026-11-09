"""Refresh presentation-only ruler views after renderer changes; no model calls."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/vsi-review19-v8-reference-20261109'
SCRIPT=ROOT/'bvb_v6/runtime/annotate_scale_reference.py'
digest=hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
for p in sorted((BASE/'questions').glob('*/result.json')):
    folder=p.parent;merged=folder/'build/merged';meta=merged/'scale-reference.json'
    old=json.loads(meta.read_text())
    if old.get('renderer_sha256')!=digest:
        with (folder/'build/reference-ruler-refresh.log').open('w') as log:
            subprocess.run(['/Applications/Blender.app/Contents/MacOS/Blender','--background','--disable-autoexec',
                str(merged/'merged-scene.blend'),'--python-exit-code','1','--python',str(SCRIPT),'--',
                '--source',str(folder/'build/fragments'),'--out',str(merged)],stdout=log,stderr=subprocess.STDOUT,check=True)
    result=json.loads(p.read_text());result['ruler']=json.loads(meta.read_text())
    p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('RULER_VERIFIED',folder.name,flush=True)
