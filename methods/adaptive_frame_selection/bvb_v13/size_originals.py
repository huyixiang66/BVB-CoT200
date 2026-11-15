"""Select a few unchanged original frames from existing reconstruction evidence."""
from pathlib import Path
from PIL import Image
from bvb_v10.pipeline import read,sha
from .temporal_frames import select

POLICY='object-size-originals-largest-target-time-diverse-v1'

def candidates(out,limit=3):
    out=Path(out);packet=read(out/'input.json')
    if packet['question']['question_type']!='object_size_estimation':raise ValueError('Original QA frames are only enabled for object size')
    model,selection=select(packet)
    recorded=out/'reconstruction-frame-selection.json'
    if recorded.exists():
        previous=read(recorded)
        if previous['source_input_sha256']!=sha(out/'input.json') or previous['kept_source_packet_indices']!=selection['kept_source_packet_indices']:
            raise ValueError('Original QA frame selection differs from reconstruction')
    targets=set(packet['question'].get('objects',[]));rows=[];seen=set()
    for index,f in zip(selection['kept_source_packet_indices'],model['frames']):
        path=Path(f['path']);digest=sha(path)
        if digest!=f['sha256']:raise ValueError('Original QA image changed')
        if digest in seen:continue
        seen.add(digest)
        with Image.open(path) as im:size=list(im.size)
        quality=0.
        for o in f.get('observations',[]):
            box=o.get('box_xyxy')
            if o.get('object') in targets and box and len(box)==4:
                area=max(0,box[2]-box[0])*max(0,box[3]-box[1])/(size[0]*size[1])
                quality=max(quality,area*max(0,float(o.get('dino_score') or 0)))
        rows.append(dict(id=f'original-frame-{index}',path=str(path.resolve()),sha256=digest,
                         kind='original',timestamp_s=f['timestamp_s'],source_packet_index=index,
                         original_size=size,sent_size=size,scale=1,quality=quality))
    chosen=[]
    while rows and len(chosen)<limit:
        # First favor a large, confidently detected target; then temporal diversity.
        def priority(r):
            separation=min(abs(r['timestamp_s']-c['timestamp_s']) for c in chosen) if chosen else 0
            return (separation,r['quality'],-r['timestamp_s']) if chosen else (r['quality'],-r['timestamp_s'])
        row=max(rows,key=priority);rows.remove(row);chosen.append(row)
    if not chosen:raise ValueError('Object-size QA needs an original evidence frame')
    return chosen
