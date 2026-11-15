"""Shared JSON reconstruction helpers; cost rates are historical estimates."""
import hashlib,json,os
from pathlib import Path
from vendor.bvb_sandbox import Sandbox
ROOT=Path(__file__).resolve().parents[1]
MODEL=os.environ.get('BVB_MODEL','gpt-6-astra')
CAP=int(os.environ.get('BVB_MAX_OUTPUT_TOKENS','32768'))
COST_LIMIT=float(os.environ.get('BVB_COST_LIMIT_USD','5.0'))
def _truncate(text,limit=8000):
    return text if len(text)<=limit else text[:limit]+'\n[truncated]'

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def usage_row(u):
    i = u['input_tokens']; o = u['output_tokens']
    d = u.get('input_tokens_details') or {}
    cached = d.get('cached_tokens', 0) or 0
    written = d.get('cache_write_tokens', 0) or 0
    ordinary = i - cached - written
    assert ordinary >= 0
    long = i > 272000
    cost = (ordinary * (20 if long else 10) + cached * (2 if long else 1)
            + written * (25 if long else 12.5) + o * (75 if long else 50)) / 1e6
    return dict(input_tokens=i, output_tokens=o, total_tokens=i+o,
                cached_input_tokens=cached, cache_write_tokens=written,
                ordinary_input_tokens=ordinary,
                reasoning_tokens=(u.get('output_tokens_details') or {}).get('reasoning_tokens', 0) or 0,
                estimated_cost_usd=cost)
