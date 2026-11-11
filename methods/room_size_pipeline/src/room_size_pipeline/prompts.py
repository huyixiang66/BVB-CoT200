import json

REGIONS = '''Inspect the chronologically ordered video frames. Identify observed connected physical spaces and scale references. Do not answer total room area. Return JSON only:
{"regions":[{"id":"stable-id","space_id":"physical-space-id","size_m":[length,width],"evidence_times_s":[0]}],
"connections":[{"regions":["id-a","id-b"],"kind":"door or open"}],
"references":[{"id":"object-id","region_id":"region-id","label":"short recognizable object name, no numbers or area","size_m":[local_x,local_y,height],"size_ranges_m":[[xmin,xmax],[ymin,ymax],[zmin,zmax]],"qa_axes":[0,1],"source":"brief original-frame evidence","evidence_times_s":[0]}],
"missing_evidence":[]}
Use one or several non-overlapping rectangles for an irregular room; accounting subdivisions of the same physical space share space_id and open connections, not invented walls. Estimate dimensions from multiple visible references when possible, retain uncertainty in size ranges. Each physical space must have at least one clearly identifiable reference. Reference size_m is the adopted estimate inside its ranges, not measured truth. Use object-local axes consistently (e.g. bed length x, width y). Include occupied floor under furniture. Do not invent rooms beyond closed doors. Use the original region-based, multiple-reference estimation procedure: check whether dimensions accommodate observed furnishings and passage clearances. Cross-check with another visible reference when available; do not invent a second one. If boundaries are hidden, report uncertainty rather than treating missing floor as zero. If references conflict, describe the conflict and choose the estimate best supported by visual evidence, not a mechanical average or systematic upward correction. Give your best numerical region dimensions even if uncertain; record limitations in missing_evidence. These are estimates, not verified measurements. Do not invent unseen rooms. No question IDs, GT, historical answers, total-area fields or labels.'''

HARD_DIMENSIONS = """HARD REGIONAL DIMENSIONS -- OVERRIDES EARLIER COORDINATE/ARCHITECTURE/OUTPUT RULES:
Keep V10_2 joint reconstruction: inspect ALL original frames, camera encodings, full trajectory diagrams and soft surface evidence together. Return ONLY a complete executable Blender Python script, not JSON. Do not answer total area.
This room_context group is spatial coverage, NOT a reviewed object-instance cluster. Repeated detections need not be one instance. Boxes/anchors are provisional; reject incorrect detections using original images. Null limits permit distinct visible instances, not invented objects.
The supplied regional dimensions and reference ranges are previous ORIGINAL-FRAME ESTIMATES, not GT. Work in metric world XY with floor top Z=0. You may rotate/translate relative to native cameras; do not claim retained camera alignment. Use trajectory for topology, not a competing hard metric constraint.
The harness will construct rectangular floors and walls with exactly the supplied regional dimensions. Set scene['region_layout']=json.dumps({region_id:{'origin':[xmin,ymin],'swap':False},...}); swap exchanges length/width axes. Do not create floor/wall meshes yourself. Set scene['openings']=json.dumps([{'region':id,'side':'north','start':0.2,'end':1.1},...]); north/south offsets from xmin, east/west from ymin. Matching interior openings must share a world interval at least0.7m. For accounting subdivisions of one open space remove the entire shared boundary with matching full-length openings; do not invent partitions. No overlap or floating/disconnected regions. Do not add unseen rooms. Include occupied floor under furniture.
Build recognizable natural-color furniture/fixtures using the original V10 primitive-assembly method, not a new JSON shape vocabulary. Preserve original metadata role/category/instance_id; every furniture mesh also needs region_id. Keep parent-free parts fully inside their region. The supplied reference labels are retained targets, even outside the question's original target list. Other context follows V10 scope. Retain full reference objects with plausible sizes inside the estimated ranges; do not distort furniture to fit rooms.
For each supplied reference id use that exact instance_id on ALL its components. Set each component's reference_yaw to the same object-local XY yaw in radians (local x/y match size_m and size_ranges_m). This is measurement metadata only, not a restriction on primitive shapes or rotations. The program measures the complete assembly, never a single mesh. No text labels or area/dimension labels.
Set scene['layout_rationale'] with evidence/uncertainties. If dimensions cannot fit evidence, report the conflict without silently resizing. Include a temporary overview camera/light, but no rendering. Save ONLY to OUTPUT_BLEND_PATH from the upstream prompt. The harness runs fixed-dimension enforcement and checks after your script.
"""

QA_BASE = '这是根据真实房间重建的虚拟场景俯视图。请估计图中房间的总地板面积，单位为平方米。如果包含多个房间，请估计它们的合计面积。'
QA_UPPER = '请先根据图像和参照尺寸确定房间面积的合理区间。本次要求输出该区间的上沿估计，而不是最可能值、中位数或保守值。只要较大的面积仍与图像证据相容，就选择较大的值。不要额外添加未展示的空间。'
QA_BEST = '给出一个最佳估计数值，并简要说明依据。'


def frame_context(manifest):
    # Explicit allowlist, never forwarding a dataset question record or scoring fields.
    return json.dumps({'frames':[{'timestamp_s':f['timestamp_s'], **({'c2w_blender':f['c2w_blender']} if 'c2w_blender' in f else {})}
                                 for f in manifest['frames']],
                       'soft_anchors':manifest.get('soft_anchors', []),
                       'coordinate_note':'Native units are arbitrary; poses are soft evidence.'}, ensure_ascii=False)


def qa_prompt(reference, mode='upper', question=None):
    if mode not in ['upper','best']: raise ValueError('Unknown QA mode')
    # reference text is generated from validated geometry, never copying region estimates.
    task=QA_BASE if not question or question==QA_BASE else '这是根据真实房间重建的虚拟场景俯视图。'+question
    return task + (QA_UPPER if mode=='upper' else QA_BEST) + '\n补充尺度参考：' + reference + '。上述尺寸是依据此前原帧估计、重建时采用的值，并非实测值。'
