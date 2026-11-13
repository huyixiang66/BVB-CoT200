"""Shared helpers without experiment-specific scheduling or report side effects."""
import json
import os
import time
from pathlib import Path
from bvb_v10.pipeline import read,save,sha

BLENDER=os.environ.get('BLENDER','/Applications/Blender.app/Contents/MacOS/Blender')

def state(out,stage,**extra):
    save(Path(out)/'status.json',dict(stage=stage,updated=time.time(),**extra))

def parse(text):
    text=text.strip()
    if text.startswith('```'):text=text.split('\n',1)[1].rsplit('```',1)[0]
    return json.loads(text)

def video_path(base,sid):
    config=read(Path(base)/'run-config.json')
    path=config.get('videos',{}).get(sid)
    if not path:raise ValueError('Set run-config.json videos[scene_id] to the original video path')
    result=Path(path).expanduser().resolve()
    if not result.is_file():raise FileNotFoundError(result)
    return result
