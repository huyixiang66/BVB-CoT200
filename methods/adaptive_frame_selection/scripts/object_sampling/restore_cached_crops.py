"""Recreate local ROI crops from identical server frames; verify exact RGB pixels."""
import argparse,hashlib,json
from pathlib import Path
from PIL import Image

p=argparse.ArgumentParser()
p.add_argument('--appearance',type=Path,required=True)
p.add_argument('--cache',type=Path,required=True)
a=p.parse_args()
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
rows=json.loads((a.appearance/'rois.json').read_text())['rows']
frames={r['source_frame_index']:r for r in json.loads((a.cache/'candidates.json').read_text())}
for row in rows:
 source=a.cache/frames[row['source_frame_index']]['frame']
 assert sha(source)==row['source_sha256']
 with Image.open(source) as im:crop=im.convert('RGB').crop(row['crop_box_xyxy'])
 assert hashlib.sha256(crop.tobytes()).hexdigest()==row['crop_pixels_sha256']
 dest=a.appearance/row['crop'];dest.parent.mkdir(parents=True,exist_ok=True)
 crop.save(dest)
print('VERIFIED_CROPS',len(rows),flush=True)
