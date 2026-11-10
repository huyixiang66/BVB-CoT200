"""Download pinned weights; no credentials or checkpoint files are committed."""
import argparse,hashlib,json,os,shutil
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--lingbot-root',type=Path,required=True);a=p.parse_args()
 os.environ.setdefault('HF_ENDPOINT','https://hf-mirror.com');os.environ.setdefault('HF_HUB_DISABLE_XET','1')
 from huggingface_hub import snapshot_download,hf_hub_download
 manifest=[]
 for repo,revision in [('openmmlab-community/mm_grounding_dino_base_all','b32293f50a56f7218d9f73ff940ffb9eba84d63c'),('facebook/dinov2-base','f9e44c814b77203eaa57a6bdbbd535f21ede1415')]:
  path=snapshot_download(repo,revision=revision,allow_patterns=['*.json','*.safetensors','*.txt','*.model']);manifest.append(dict(repo=repo,revision=revision,path=path))
 revision='e8984048a940fbdfa19909f8cc0c0ad834c1ee5a';repo='robbyant/lingbot-map';path=Path(hf_hub_download(repo,'lingbot-map.pt',revision=revision))
 modeldir=a.lingbot_root.resolve()/'models';modeldir.mkdir(parents=True,exist_ok=True);target=modeldir/'lingbot-map.pt'
 if target.exists() and target.resolve()!=path.resolve():raise FileExistsError('Preserve existing checkpoint: '+str(target))
 if not target.exists():target.symlink_to(path)
 manifest.append(dict(repo=repo,revision=revision,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
 (modeldir/'v10-models.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
