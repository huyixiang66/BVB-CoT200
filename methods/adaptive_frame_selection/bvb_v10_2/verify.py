import json,hashlib
from pathlib import Path

def verify():
    from bvb_v10.__main__ import verify as verify_v10
    verify_v10()
    root=Path(__file__).resolve().parents[1]
    lock=json.loads((root/'bvb_v10_2/method-lock.json').read_text())
    changed=[p for p,h in lock['files'].items() if not (root/p).is_file() or hashlib.sha256((root/p).read_bytes()).hexdigest()!=h]
    if changed:raise RuntimeError('V10_2 sources changed: '+', '.join(changed))
    return len(lock['files'])
