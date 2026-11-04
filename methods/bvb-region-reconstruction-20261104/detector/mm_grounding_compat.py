"""Restore per-decoder-layer checkpoint heads in Transformers 5.16.1 MM-GDINO.

This changes weight-alias metadata and explicitly reloads each checkpoint head
in this process only. The installed library, checkpoint tensors, model
architecture, preprocessing and scoring are untouched.
The 5.16.1 class erroneously aliases all bbox/class heads to layer zero even when
its MM config explicitly disables sharing. Always audit the loaded head tensors.
"""
from pathlib import Path

def restore_checkpoint_heads(model,repo,revision,info):
 """Load each unshared head explicitly; bypass loader alias rewriting entirely."""
 if not info['applied']:return
 import copy,torch
 from huggingface_hub import hf_hub_download
 from safetensors import safe_open
 path=hf_hub_download(repo,'model.safetensors',revision=revision,local_files_only=True)
 restored=[]
 with safe_open(path,framework='pt',device='cpu') as f:
  for name in ['bbox_embed','class_embed']:
   heads=torch.nn.ModuleList([copy.deepcopy(head) for head in getattr(model,name)])
   prefix='model.decoder.'+name+'.'
   weights={key[len(prefix):]:f.get_tensor(key) for key in f.keys() if key.startswith(prefix)}
   weights.update({key[len(name)+1:]:f.get_tensor(key) for key in f.keys() if key.startswith(name+'.')})
   heads.load_state_dict(weights,strict=True)
   setattr(model,name,heads)
   setattr(model.model.decoder,name,heads)
   restored.extend(name+'.'+key for key in weights)
 info['explicitly_restored_checkpoint_heads']=restored

def prepare(config):
 import transformers
 info={'applied':False,'transformers':transformers.__version__}
 if config.model_type!='mm-grounding-dino':return info
 if transformers.__version__=='5.16.1':
  from transformers import MMGroundingDinoForObjectDetection
  assert config.decoder_bbox_embed_share is False and config.decoder_cls_embed_share is False
  original=dict(MMGroundingDinoForObjectDetection._tied_weights_keys)
  expected={'model.decoder.bbox_embed':'bbox_embed','model.decoder.class_embed':'class_embed'}
  MMGroundingDinoForObjectDetection._tied_weights_keys=expected
  info.update(applied=True,original_aliases=original,corrected_aliases=expected,reason='Respect non-shared decoder heads; retain same-layer module aliases only')
 return info

def audit(model,repo,revision):
 from huggingface_hub import hf_hub_download
 from safetensors import safe_open
 import torch
 if model.config.model_type!='mm-grounding-dino':return {'applicable':False}
 path=Path(hf_hub_download(repo,'model.safetensors',revision=revision,local_files_only=True))
 state=model.state_dict();checked=[]
 with safe_open(str(path),framework='pt',device='cpu') as f:
  for key in f.keys():
   if key.startswith(('model.decoder.bbox_embed.','model.decoder.class_embed.','bbox_embed.','class_embed.')):
    assert key in state,key
    assert torch.equal(f.get_tensor(key),state[key].cpu()),'Checkpoint head mismatch: '+key
    checked.append(key)
 assert len(checked)==42,checked
 for heads in [model.bbox_embed,model.class_embed]:
  assert len({id(head) for head in heads})==len(heads)
  params=[next(head.parameters()) for head in heads]
  assert len({p.data_ptr() for p in params})==len(params),'Unexpected shared decoder head parameters'
 return {'applicable':True,'status':'passed','exact_checkpoint_head_tensors':checked,'distinct_decoder_head_parameters':True}
