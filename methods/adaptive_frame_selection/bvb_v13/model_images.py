"""Smaller model-facing copies of generated images; originals remain authoritative."""
from pathlib import Path
from PIL import Image
from bvb_v10.pipeline import read,save,sha

SCALE=.5

def prepare(source,folder):
    source,folder=Path(source),Path(folder);digest=sha(source)
    dest=folder/(digest[:20]+'-'+source.stem+'.png');receipt=dest.with_suffix('.json')
    with Image.open(source) as im:
        original=list(im.size);size=[max(1,n//2) for n in original]
        identity=dict(source=str(source.resolve()),source_sha256=digest,original_size=original,sent_size=size,scale=SCALE)
        if receipt.exists():
            previous=read(receipt)
            if any(previous.get(k)!=v for k,v in identity.items()) or sha(dest)!=previous['sha256']:
                raise ValueError('Generated image transport changed')
            return previous
        folder.mkdir(parents=True,exist_ok=True)
        im.resize(size,Image.Resampling.LANCZOS).save(dest,format='PNG')
    result=dict(identity,path=str(dest.resolve()),sha256=sha(dest));save(receipt,result)
    return result

def image_item(source,folder):
    from subscription_direct_client import image_item as raw_image_item
    return raw_image_item(prepare(source,folder)['path'])
