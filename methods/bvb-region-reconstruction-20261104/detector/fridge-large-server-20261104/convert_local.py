import _codecs
from collections import Counter
import json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoConfig
from convert_mappings import convert_mm_to_hf_state

torch.set_num_threads(2)

class HistoryBuffer:
    pass

HistoryBuffer.__module__ = 'mmengine.logging.history_buffer'

def unused_history_statistic(*args):
    raise RuntimeError('Training statistics are outside this conversion.')

def history_getattr(value, name):
    if value is not HistoryBuffer or name not in {'min', 'max', 'mean', 'current'}:
        raise ValueError(f'Unexpected checkpoint metadata attribute {name!r}')
    return unused_history_statistic

history_getattr.__module__ = 'builtins'
history_getattr.__name__ = 'getattr'
numpy_metadata = [np.core.multiarray._reconstruct, np.core.multiarray.scalar, np.ndarray, np.dtype,
                  type(np.dtype('float64')), type(np.dtype('int64')), type(np.dtype('float32'))]
torch.serialization.add_safe_globals([HistoryBuffer, history_getattr, Counter, _codecs.encode, *numpy_metadata])
checkpoint = torch.load('openmmlab-large.pth', map_location='cpu', weights_only=True)
config = AutoConfig.from_pretrained('checkpoint', local_files_only=True)
state = convert_mm_to_hf_state(checkpoint['state_dict'], config)
keys = json.loads(Path('checkpoint-keys.json').read_text())
missing = set(keys) - state.keys()
assert not missing, sorted(missing)
weights = {key: state[key].contiguous().clone() for key in keys}
out = Path('checkpoint/model.safetensors')
assert not out.exists()
save_file(weights, str(out.with_suffix('.part')), metadata={'format':'pt'})
out.with_suffix('.part').replace(out)
print('CONVERSION_DONE', len(weights), out.stat().st_size, flush=True)
