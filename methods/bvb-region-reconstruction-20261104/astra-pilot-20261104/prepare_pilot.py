"""Freeze six pilot questions and prepare the senior's per-group reconstruction inputs."""
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageOps

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
WORKSPACE = ROOT.parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'cluster'), str(ROOT / 'cluster/two-case-review-20261104'), str(ROOT / 'senior')]
from reconstruct_regions import CACHE, prepare, read, sha, write
from group_regions import describe_view, geometric_affinity, semantic_affinity
from tune_door import run
from bvb_v5.dinov2_appearance import crop_bounds

PARAMS = read(WORKSPACE / 'docs/experiments/bvb-geometry-appearance-075-preset.json')['parameters']
VSI = read(ROOT / 'cluster/verified/results.json')


def frozen_rows(question):
    return {r['source_frame_index']: r for g in question['arms']['geometry'] for r in g['selected']}


def scoped(question, label):
    groups = []
    for group in question['arms']['geometry']:
        rows = []
        for row in group['selected']:
            obs = [h for h in row['observations'] if h['object'] == label]
            if obs:
                rows.append(dict(row, observations=obs))
        if rows:
            groups.append(dict(group, selected=rows))
    return dict(question, arms={'geometry': groups})


def grouped(question, labels):
    out = []
    for label in labels:
        subset = scoped(question, label)
        if not subset['arms']['geometry']:
            continue
        for i, group in enumerate(run(subset, **PARAMS)['groups'], 1):
            group['group_id'] = f"{label.replace(' ', '-')}-{i:02d}"
            out.append(group)
    return out


def rescore(rows, cache, poses, fixed_scale=None):
    import torch
    from transformers import AutoImageProcessor, AutoModel
    ckpt = ROOT.parent / 'q1585-reconstruction-pilot-20261103/model'
    processor = AutoImageProcessor.from_pretrained(ckpt, local_files_only=True)
    model = AutoModel.from_pretrained(ckpt, local_files_only=True).eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(2)
    model.to(device)
    jobs = []
    for fid, row in rows.items():
        path = cache / row['image']
        assert sha(path) == row['frame_sha256']
        with Image.open(path) as im:
            for h in row['observations']:
                box = h.get('box_xyxy')
                if box is not None:
                    bounds, _ = crop_bounds(box, *im.size, 'adaptive-context')
                    if bounds[2] > bounds[0] and bounds[3] > bounds[1]:
                        crop = ImageOps.pad(im.convert('RGB').crop(bounds), (224,224),
                                           method=Image.Resampling.BICUBIC, color=(124,116,104))
                        jobs.append((fid,h['object'],crop))
    vectors = defaultdict(lambda: defaultdict(list))
    with torch.inference_mode():
        for i in range(0, len(jobs), 8):
            batch = jobs[i:i+8]
            pixels = processor(images=[j[2] for j in batch],do_resize=False,do_center_crop=False,
                               return_tensors='pt')['pixel_values'].to(device)
            result = torch.nn.functional.normalize(model(pixel_values=pixels).last_hidden_state[:,0].float(), dim=-1)
            for (fid,label,_), vector in zip(batch,result.cpu().numpy(),strict=True):
                vectors[fid][label].append(vector)
    with np.load(poses/'camera-poses.npz') as file:
        matrices, k = file['c2w'], file['intrinsics']
    with np.load(poses/'sparse-geometry-corrected.npz') as file:
        cloud = dict(file)
    meta=poses/'geometry-info.json'
    w,h=read(meta)['image_size_wh'] if meta.exists() else (518,392)
    resize=np.diag([640/w,480/h,1])
    views={fid:describe_view(row['frame'],matrices[row['frame']],resize@k[row['frame']],cloud,np.array([640,480]))
           for fid,row in rows.items()}
    scale=fixed_scale or float(np.median([x['depth'] for x in views.values() if x]))
    pairs=[]
    for a,b in combinations(sorted(rows),2):
        geo,_,_=geometric_affinity(views[a],views[b],scale)
        pairs.append(dict(frames=[a,b],geometry=geo,semantic=semantic_affinity(vectors[a],vectors[b])))
    return dict(arms={'geometry':[dict(selected=list(rows.values()))]},pairs=pairs)


def vsi_case(qid):
    data = VSI['questions'][qid]
    question = data['question']
    if qid == '1585':
        # Bounded visual pilot: preserve the original peaks and supply the Tiny built-in prompt candidate
        # from the already investigated kitchen interval. All boxes are detector output.
        rows=frozen_rows(data)
        index={r['source_frame_index']:r for r in read(CACHE/'cache/candidates.json')}
        for line in (ROOT/'detector/tiny-built-in/aliases.jsonl').read_text(encoding='utf-8').splitlines():
            d=json.loads(line)
            if not 113.5 <= d['timestamp_s'] <= 115.75:
                continue
            boxes=[x for x in d['normal'] if x['object']=='refrigerator']
            if not boxes:continue
            best=max(boxes,key=lambda x:x['score'])
            fid=d['source_frame_index']; c=index[fid]
            hint=dict(object='refrigerator',peak_id='tiny-alias-kitchen',box_xyxy=best['box_xyxy'],
                      dino_score=best['score'],status='tiny_alias_pilot')
            if fid not in rows:
                rows[fid]=dict(source_frame_index=fid,timestamp_s=c['timestamp_s'],frame=c['index'],
                               image=c['frame'],frame_sha256=c['frame_sha256'],observations=[])
            rows[fid]['observations'].append(hint)
        data=rescore(rows,CACHE/'cache',CACHE/'lingbot',VSI['depth_scale'])
    groups=grouped(data,question['objects'])
    prepare(OUT/'vsi'/qid,question,groups,CACHE,8,ROOT/'reference/centers.json')


def transfer_case(bench,qid):
    old=ROOT/'prepared-transfer'/bench/qid
    manifest=read(old/'manifest.json');question=manifest['question'];dest=OUT/bench/qid
    cache=ROOT/'transfer/pose-inputs'/bench;poses=ROOT/'transfer/derived'/bench
    index={r['source_frame_index']:r for r in read(cache/'candidates.json')}
    rows={};context=[]
    for group in manifest['groups']:
        previous=None;track=0
        for f in read(old/group['folder']/'input-frames.json'):
            fid=f['source_frame_index'];c=index[fid];t=f['timestamp_s']
            if previous is None or t-previous>2:track+=1
            previous=t
            row=rows.setdefault(fid,dict(source_frame_index=fid,timestamp_s=t,frame=c['index'],image=c['frame'],
                frame_sha256=f['sha256'],official_query_frame=f.get('official_query_frame',False),observations=[]))
            row['observations'].extend(dict(h,peak_id=f"{group['group_id']}-segment-{track}",
                                            dino_score=h.get('dino_score',h.get('score')))
                                       for h in f['observations'])
            if not group['objects']:context.append(row)
    groups=(grouped(rescore(rows,cache,poses),question['objects']) if question['objects'] else
            [dict(group_id='trajectory-context',objects=[],selected=context)])
    for group in groups:
        selected=group['selected']
        if len(selected)>=4:
            continue
        times=[row['timestamp_s'] for row in selected]
        used={row['source_frame_index'] for row in selected}
        clip=question.get('official_clip')
        neighbors=sorted((row for row in index.values() if row['source_frame_index'] not in used
                          and (not clip or clip['start']<=row['timestamp_s']<=clip['end'])),
                         key=lambda row:min(abs(t-row['timestamp_s']) for t in times))
        for row in neighbors[:4-len(selected)]:
            selected.append(dict(source_frame_index=row['source_frame_index'],timestamp_s=row['timestamp_s'],
                frame=row['index'],image=row['frame'],frame_sha256=row['frame_sha256'],object=None,observations=[]))
    prepare(dest,question,groups,None,8,None,cache=cache,poses=poses)
    source=read(old/'trajectory-reference.json')
    if bench=='sti' and qid=='8':
        write(dest/'trajectory-reference.json',source)
    if bench=='mmsi' and qid=='question_0734':
        source['highlight_time_s']=126.0
        write(dest/'trajectory-reference.json',source)


if __name__=='__main__':
    for case in sys.argv[1:]:
        bench,qid=case.split('/')
        vsi_case(qid) if bench=='vsi' else transfer_case(bench,qid)
        print(case,[(x['group_id'],x['frames']) for x in read(OUT/bench/qid/'manifest.json')['groups']],flush=True)
