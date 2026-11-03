"""DINOv2 ROI appearance cache and cross-peak evidence, without asserting IDs.

prepare/report run locally; extract can run independently on the GPU server.
No SAM masks, LLM calls, or changes to existing frame selections.
"""
import argparse
import hashlib
import html
import json
import math
import re
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def read(p):
    return json.loads(Path(p).read_text())


def save(p, x):
    Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2) + '\n')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def roi_key(obj, fid, box):
    # Full detector coordinates: do not collapse distinct boxes in the same frame.
    return hashlib.sha256(json.dumps([obj, fid, box], sort_keys=True).encode()).hexdigest()[:20]


def crop_bounds(box, width, height, policy):
    x0, y0, x1, y1 = box
    fraction = 0.
    if policy == "adaptive-context":
        fraction = .75 if (x1 - x0) * (y1 - y0) <= .02 * width * height else .20
    dx, dy = fraction * (x1 - x0), fraction * (y1 - y0)
    return ([max(0, math.floor(x0-dx)), max(0, math.floor(y0-dy)),
             min(width, math.ceil(x1+dx)), min(height, math.ceil(y1+dy))], fraction)


def prepare(a):
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / 'crops').mkdir()
    groups = read(a.result / 'groups.json')
    peaks = read(a.peaks)
    cache = read(a.cache / 'candidates.json')
    rows, skipped = {}, []
    crop_policy = getattr(a, "crop_policy", "tight")

    def add(obj, r, peak_id=None):
        box = r.get('box_xyxy')
        if box is None:
            skipped.append({'object': obj, 'source_frame_index': r['source_frame_index'],
                            'reason': 'no verified ROI; context frame is not an object embedding'})
            return
        fid = r['source_frame_index']
        key = roi_key(obj, fid, box)
        if key not in rows:
            c = cache[r['frame']]
            assert c['source_frame_index'] == fid
            src = a.cache / c['frame']
            assert sha(src) == c['frame_sha256']
            with Image.open(src) as image:
                image = image.convert('RGB')
                w, h = image.size
                b, fraction = crop_bounds(box, w, h, crop_policy)
                if b[2] <= b[0] or b[3] <= b[1]:
                    raise ValueError(f'Invalid ROI: {key}')
                image.crop(b).save(a.out / 'crops' / (key + '.png'))
            rows[key] = {'id': key, 'object': obj, 'source_frame_index': fid,
                         'timestamp_s': c['timestamp_s'], 'box_xyxy': box, 'crop_box_xyxy': b,
                         'crop_expansion_each_side': fraction,
                         'crop': f'crops/{key}.png', 'source_sha256': c['frame_sha256'],
                         'crop_sha256': sha(a.out / 'crops' / (key + '.png')),
                         'crop_pixels_sha256': hashlib.sha256(Image.open(a.out/'crops'/(key+'.png')).convert('RGB').tobytes()).hexdigest(),
                         'dino_score': r.get('dino_score', r.get('score')),
                         'status': r['status'], 'anchor_for': [], 'candidate_for': []}
        if peak_id and peak_id not in rows[key]['anchor_for']:
            rows[key]['anchor_for'].append(peak_id)
        rows[key]['candidate_for'] = sorted(set(rows[key]['candidate_for'] + r.get('source_peaks', [])))

    for g in groups:
        for r in g['candidates']:
            add(g['object'], r)
    for p in peaks:
        c = cache[p['anchor']['frame']]
        add(p['object'], {**p['anchor'], 'source_frame_index': c['source_frame_index'],
                         'status': 'low_score_anchor' if p['fallback'] else 'anchor'}, p['peak_id'])
    manifest = {'schema_version': 1, 'crop_policy': crop_policy,
                'rows': sorted(rows.values(), key=lambda r: (r['object'], r['source_frame_index'], r['id'])),
                'skipped_context_slots': skipped, 'video_sha256': read(a.cache / 'config.json')['video_sha256'],
                'provenance': {'groups_sha256': sha(a.result / 'groups.json'), 'peaks_sha256': sha(a.peaks),
                               'geometry_sha256': sha(a.result / 'anchor-geometry.json')},
                'purpose': 'appearance evidence only; not a calibrated same-instance probability'}
    save(a.out / 'rois.json', manifest)
    print(json.dumps({'unique_rois': len(rows), 'anchors': sum(len(r['anchor_for']) for r in rows.values()),
                      'skipped_context_slots': len(skipped)}))


def extract(a):
    import torch
    import transformers
    from transformers import AutoImageProcessor, AutoModel
    torch.set_num_threads(4)
    start = time.perf_counter()
    if a.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    model = AutoModel.from_pretrained(a.model, revision=a.revision).eval().to(a.device)
    revision = model.config._commit_hash
    processor = AutoImageProcessor.from_pretrained(a.model, revision=revision or a.revision)
    loaded = time.perf_counter()
    rows = read(a.out / 'rois.json')['rows']
    cls, patch = [], []
    def sync():
        if a.device.startswith('cuda'):
            torch.cuda.synchronize()
    # Full ROI preserved with aspect-ratio-preserving letterboxing, no center crop.
    def pixels(batch):
        images = []
        for r in batch:
            path = a.out / r['crop']
            with Image.open(path) as im:
                if 'crop_pixels_sha256' in r:
                    assert hashlib.sha256(im.convert('RGB').tobytes()).hexdigest() == r['crop_pixels_sha256']
                else:
                    assert sha(path) == r['crop_sha256']
                images.append(ImageOps.pad(im.convert('RGB'), (224, 224),
                                          method=Image.Resampling.BICUBIC, color=(124, 116, 104)))
        return processor(images=images, do_resize=False, do_center_crop=False,
                         return_tensors='pt')['pixel_values'].to(a.device)
    with torch.inference_mode():
        model(pixel_values=pixels(rows[:min(a.batch_size, len(rows))]))
        sync()
        warm = time.perf_counter()
        forward_s = 0.
        for i in range(0, len(rows), a.batch_size):
            x = pixels(rows[i:i + a.batch_size])
            sync(); t = time.perf_counter()
            y = model(pixel_values=x).last_hidden_state
            v = torch.nn.functional.normalize(y[:, 0].float(), dim=-1)
            p = torch.nn.functional.normalize(y[:, 1:].float().mean(dim=1), dim=-1)
            sync(); forward_s += time.perf_counter() - t
            cls.append(v.cpu().numpy()); patch.append(p.cpu().numpy())
            print(f'ROI {min(i+a.batch_size,len(rows))}/{len(rows)}', flush=True)
    end = time.perf_counter()
    x = np.concatenate(cls); p = np.concatenate(patch)
    assert np.isfinite(x).all() and np.allclose(np.linalg.norm(x, axis=1), 1, atol=1e-5)
    np.savez_compressed(a.out / 'features.npz', ids=np.array([r['id'] for r in rows]), cls=x, patch_mean=p)
    save(a.out / 'extraction.json', {'model': a.model, 'revision': revision,
         'model_source': 'https://huggingface.co/' + a.model, 'torch': torch.__version__,
         'transformers': transformers.__version__, 'device': a.device,
         'gpu': torch.cuda.get_device_name() if a.device.startswith('cuda') else None,
         'batch_size': a.batch_size, 'dtype': 'float32', 'roi_count': len(rows),
         'dimension': x.shape[1], 'rois_sha256': sha(a.out / 'rois.json'),
         'features_sha256': sha(a.out / 'features.npz'),
         'preprocessing': 'RGB ROI; aspect-preserving 224x224 bicubic pad with RGB(124,116,104); checkpoint normalization; no center crop',
         'feature': 'L2 normalized final CLS; auxiliary L2 normalized mean patch tokens',
         'load_seconds': loaded-start, 'warmup_seconds': warm-loaded,
         'extraction_seconds_including_io': end-warm, 'forward_seconds': forward_s,
         'total_seconds_excluding_save': end-start})


def pair_evidence(a, b, score, patch_score, peaks, geometry):
    def iou(x, y):
        lo=np.maximum(x[:2],y[:2]); hi=np.minimum(x[2:],y[2:]); inter=np.prod(np.maximum(hi-lo,0))
        return float(inter/max(np.prod(np.subtract(x[2:],x[:2]))+np.prod(np.subtract(y[2:],y[:2]))-inter,1e-9))
    pa, pb = peaks[a], peaks[b]
    oa = {r['frame']: r['box_xyxy'] for r in pa['observations'] if r['score'] >= .5}
    ob = {r['frame']: r['box_xyxy'] for r in pb['observations'] if r['score'] >= .5}
    simultaneous = any(iou(oa[k], ob[k]) < .1 for k in oa.keys() & ob.keys())
    ga, gb = geometry.get(a), geometry.get(b)
    ratio = None
    if ga and gb:
        ratio = float(np.linalg.norm(np.array(ga['center'])-gb['center']) / max(ga['radius']+gb['radius'],1e-6))
    return {'peak_a': a, 'peak_b': b, 'object': pa['object'], 'cosine_cls': float(score),
            'cosine_patch_mean': float(patch_score), 'same_frame_distinct_evidence': simultaneous,
            'sparse_geometry_separation_ratio': ratio,
            'geometry_conflict_heuristic': ratio is not None and ratio > 1.5,
            'automatic_merge': False,
            'decision': 'do_not_merge_same_frame_distinct' if simultaneous else
                        ('review_geometry_conflict' if ratio is not None and ratio > 1.5 else 'appearance_evidence_only')}


def report(a):
    meta = read(a.out / 'rois.json'); rows = meta['rows']; features = np.load(a.out / 'features.npz')
    run = read(a.out / 'extraction.json')
    assert sha(a.out/'rois.json') == run['rois_sha256']
    assert sha(a.out/'features.npz') == run['features_sha256']
    assert list(features['ids']) == [r['id'] for r in rows]
    assert sha(a.peaks) == meta['provenance']['peaks_sha256']
    assert sha(a.result/'groups.json') == meta['provenance']['groups_sha256']
    if 'geometry_sha256' in meta['provenance']:
        assert sha(a.result/'anchor-geometry.json') == meta['provenance']['geometry_sha256']
    peaks = {r['peak_id']: r for r in read(a.peaks)}
    geometry = read(a.result/'anchor-geometry.json')
    anchors = {p: i for i,r in enumerate(rows) for p in r['anchor_for']}
    assert set(anchors) == set(peaks)
    cosine = np.clip(features['cls'] @ features['cls'].T, -1, 1)
    patches = np.clip(features['patch_mean'] @ features['patch_mean'].T, -1, 1)
    pairs=[]; names=sorted(anchors)
    for k,p in enumerate(names):
        for q in names[k+1:]:
            if peaks[p]['object']==peaks[q]['object']:
                pairs.append(pair_evidence(p,q,cosine[anchors[p],anchors[q]],patches[anchors[p],anchors[q]],peaks,geometry))
    save(a.out/'peak-pairs.json',pairs)
    # All ROI similarity matrices allow downstream tools to query any candidate.
    np.savez_compressed(a.out/'similarities.npz',ids=features['ids'],cls=cosine,patch_mean=patches)
    neighbors=[]
    for i,r in enumerate(rows):
        eligible=[j for j,s in enumerate(rows) if s['object']==r['object'] and s['source_frame_index']!=r['source_frame_index']]
        best=sorted(eligible,key=lambda j:-cosine[i,j])[:5]
        neighbors.append({'id':r['id'],'matches':[{'id':rows[j]['id'],'cosine_cls':float(cosine[i,j])} for j in best]})
    save(a.out/'roi-neighbors.json',neighbors)
    cards=[]
    for p in names:
        r=rows[anchors[p]]
        candidates=sorted([x for x in pairs if p in (x['peak_a'],x['peak_b']) and
                           peaks[x['peak_b'] if x['peak_a']==p else x['peak_a']]['anchor']['frame'] != peaks[p]['anchor']['frame']],
                          key=lambda x:-x['cosine_cls'])[:3]
        def thumb(row, label):
            return f'<figure><a href="{row["crop"]}"><img loading="lazy" src="{row["crop"]}"></a><figcaption>{html.escape(label)}<br>{row["timestamp_s"]:.2f}s · DINO检测分数 {row["dino_score"]:.3f}</figcaption></figure>'
        body=thumb(r,p+' · 查询框')
        for x in candidates:
            q=x['peak_b'] if x['peak_a']==p else x['peak_a']; s=rows[anchors[q]]
            label=f'{q} · 外观 {x["cosine_cls"]:.3f}'
            label+=' · 同帧独立，禁止合并' if x['same_frame_distinct_evidence'] else (' · 几何冲突待核查' if x['geometry_conflict_heuristic'] else ' · 身份未确认')
            body+=thumb(s,label)
        cards.append(f'<section><h3>{html.escape(r["object"])} · {p}</h3><div class="row">{body}</div></section>')
    summary={'rois':len(rows),'anchor_count':len(anchors),'object_categories':len(set(r['object'] for r in rows)),
             'same_category_peak_pairs':len(pairs),'same_frame_distinct_pairs':sum(x['same_frame_distinct_evidence'] for x in pairs),
             'heuristic_geometry_conflicts':sum(x['geometry_conflict_heuristic'] for x in pairs),
             'selection_changed':False,'instance_accuracy_measured':False,'extraction':run}
    save(a.out/'summary.json',summary)
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>DINOv2 跨帧外观对照</title>
<style>body{font:16px system-ui;max-width:1300px;margin:35px auto;padding:0 20px;background:#f5f7fa;color:#18212f}section{background:white;padding:18px;margin:18px 0;border-radius:12px}.row{display:flex;flex-wrap:wrap;gap:16px}figure{margin:0;width:260px}img{width:260px;height:200px;object-fit:contain;background:#eee}figcaption{line-height:1.6}a{color:#2459bc}</style>
<h1>DINOv2：跨帧物体外观对照</h1>
<p>每个波峰展示检测框裁剪图，以及同类别其他时刻波峰的三个最相似候选。分数是归一化 CLS 特征的余弦相似度，范围 −1～1，越大越像，<b>不是同一实例的概率</b>。</p>
<p>同帧高分独立框禁止合并；LingBot 稀疏几何冲突只提示复核。相似家具、错误检测、遮挡和背景均可能干扰。本次提取外观证据，不改变 V4 抽帧，也不自动确认物体身份。</p>'''
    page+=f'<p>{len(rows)} 个 ROI · {len(anchors)} 个波峰 · {len(pairs)} 对同类跨波峰比较 · {run["dimension"]} 维特征</p>'
    page+=f'<p>A100：批量提取（含裁剪读取）{run["extraction_seconds_including_io"]:.2f} 秒；含模型加载与预热 {run["total_seconds_excluding_save"]:.2f} 秒。不含首次权重下载、网络传输和报告制作。</p>'
    page+='<p><a href="peak-pairs.json">完整波峰比较</a> · <a href="roi-neighbors.json">每个候选框近邻</a> · <a href="summary.json">运行统计</a> · <a href="features.npz">特征文件</a></p>'
    if (a.out/'review-montage.jpg').exists():
        page+='<details><summary>展开人工抽查：相似匹配及检测类别误差</summary><p>椅子跨时刻外观接近；bed 框仍是披布物体，table 框仍是白色钢琴。高外观相似度不能纠正类别，也不等于确认实例。最后一行是同帧两个独立椅子框。</p><a href="review-montage.jpg"><img src="review-montage.jpg" style="width:min(800px,100%);height:auto"></a></details>'
    page+=''.join(cards)+'</html>'
    (a.out/'index.html').write_text(page)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


def publish(a):
    """Attach the reproducible appearance audit to the existing V4 delivery."""
    if not a.delivery:
        raise ValueError('--delivery required')
    before = {str(p): sha(p) for p in a.delivery.glob('03-per-question/*/input-frames.json')}
    target = a.delivery / '05-dinov2-appearance'
    shutil.copytree(a.out, target, dirs_exist_ok=True)
    shutil.copy2(Path(__file__), target / 'dinov2_appearance.py')
    summary = read(a.out / 'summary.json')
    block = f'''<!-- dinov2-appearance-start -->
<section id="dinov2-appearance"><h2>DINOv2 跨帧外观比较</h2>
<p>新增 {summary['rois']} 个框内特征，覆盖 {summary['anchor_count']} 个波峰。比较同类物体跨帧的外观；身份仍为待验证假设，不改变本页已有抽帧。</p>
<p><a href="05-dinov2-appearance/index.html">独立打开外观对照</a> · <a href="05-dinov2-appearance/peak-pairs.json">全部相似度及几何约束</a></p>
<details><summary>在本页展开所有波峰的相似匹配图</summary><iframe title="DINOv2 跨帧外观对照" src="05-dinov2-appearance/index.html" loading="lazy" style="width:100%;height:800px;border:1px solid #ddd"></iframe></details>
</section><!-- dinov2-appearance-end -->'''
    path = a.delivery / 'index.html'
    page = re.sub(r'<!-- dinov2-appearance-start -->.*?<!-- dinov2-appearance-end -->', '', path.read_text(), flags=re.S)
    page = page.replace('</main>', block+'</main>') if '</main>' in page else page.replace('</body>',block+'</body>')
    assert 'id="dinov2-appearance"' in page
    path.write_text(page)
    manifest = read(a.delivery / 'manifest.json')
    manifest['dinov2_appearance'] = {'path':'05-dinov2-appearance/index.html',**summary}
    save(a.delivery / 'manifest.json', manifest)
    assert all(sha(p)==h for p,h in before.items())
    save(target/'delivery-validation.json',{'input_manifests_unchanged':True,'input_manifest_count':len(before)})
    print(str(path))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','extract','report','publish'])
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--result',type=Path)
    parser.add_argument('--peaks',type=Path)
    parser.add_argument('--cache',type=Path)
    parser.add_argument('--delivery',type=Path)
    parser.add_argument('--model',default='facebook/dinov2-base')
    parser.add_argument('--revision',default='main')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--crop-policy', choices=('tight', 'adaptive-context'), default='tight')
    a=parser.parse_args()
    if a.batch_size<1:parser.error('--batch-size must be positive')
    if a.action in ['prepare','report'] and (not a.result or not a.peaks):parser.error('--result and --peaks required')
    if a.action=='prepare' and not a.cache:parser.error('--cache required')
    globals()[a.action](a)


if __name__=='__main__':
    main()
