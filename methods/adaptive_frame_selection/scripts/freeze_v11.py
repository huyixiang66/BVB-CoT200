"""Release-maintainer command: intentionally update source fingerprints after review."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def freeze():
    for path in [ROOT/'bvb_v10/method-lock.json',ROOT/'bvb_v10_2/method-lock.json']:
        data=json.loads(path.read_text())
        data['files']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in data['files']}
        path.write_text(json.dumps(data,indent=2)+'\n')
    files=[p for name in ['bvb_v11','bvb_v10','bvb_v6','bvb_v5','bvb_v3','scripts']
        for p in (ROOT/name).rglob('*') if p.is_file() and p.suffix in {'.py','.json','.txt'}
        and '__pycache__' not in p.parts and p.name!='method-lock.json']
    record=dict(version='11.0.0',files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)})
    (ROOT/'bvb_v11/method-lock.json').write_text(json.dumps(record,indent=2)+'\n')
    print('Locked',len(files),'source files')

if __name__=='__main__':freeze()
