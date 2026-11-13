import hashlib
import json
from pathlib import Path

def verify():
    root=Path(__file__).resolve().parents[1]
    lock=json.loads((root/'bvb_v11/method-lock.json').read_text())
    changed=[p for p,h in lock['files'].items() if not (root/p).is_file() or hashlib.sha256((root/p).read_bytes()).hexdigest()!=h]
    if changed:raise RuntimeError('V11 source fingerprint mismatch: '+', '.join(changed))
    return len(lock['files'])
