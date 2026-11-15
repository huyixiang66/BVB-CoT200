"""Pinned geometry-only room selector + fresh QA on an existing Blender scene."""
import importlib.util
import math
import re
import subprocess
import sys
from pathlib import Path
from bvb_v10.pipeline import ROOT,read,save,sha
from .common import state

VENDOR=Path(__file__).parent/'vendor/room_selector'
COMMIT='d4bc0c876917c0e2b8620fbea025475859548c83'

def run(blend,video,question,out,blender):
    blend,video,question,out=[p.resolve() for p in [blend,video,question,out]]
    q=read(question)
    if q['question_type']!='room_size_estimation':raise ValueError('Selector JSON must use room_size_estimation and explicit room floor/ceiling mesh names')
    identity=dict(blend_sha256=sha(blend),video_sha256=sha(video),question_sha256=sha(question),selector_commit=COMMIT)
    if (out/'room-input.json').exists() and read(out/'room-input.json')!=identity:raise ValueError('Room inputs changed; use fresh output')
    save(out/'room-input.json',identity)
    selector=out/'geometry-selector'
    if not (selector/'render_manifest.json').exists():
        with (out/'selector.log').open('a') as log:
            subprocess.run([sys.executable,str(VENDOR/'scripts/select_views.py'),'--blend',str(blend),'--video',str(video),
                '--question',str(question),'--out',str(selector),'--blender',blender],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    if sha(blend)!=identity['blend_sha256']:raise RuntimeError('Original blend changed')
    views=next(r for r in read(selector/'render_manifest.json')['questions'] if r['qa_id']==str(q['qa_id']))
    if not views['qa_ready'] or not views['views']:raise RuntimeError('No accepted room view; diagnostic views cannot enter QA')
    spec=importlib.util.spec_from_file_location('bvb_v13_room_protocol',VENDOR/'fixed_protocol.py')
    protocol=importlib.util.module_from_spec(spec);spec.loader.exec_module(protocol)
    sys.path.insert(0,str(ROOT/'scripts'))
    from subscription_direct_client import request,image_item
    prompt=('Answer the public room-area question using only the supplied geometry-selected reconstructed view(s). '
        'These are views of one existing reconstructed scene, not original video frames. '
        'The roof is hidden only for display; furniture-covered floor remains part of the area. '
        'There is no calibrated metric scale and no mesh coordinates are provided. '
        'Do not treat the image boundary as a floor boundary. Follow the question scope.\n'
        +'Question: '+q['question']+'\nReasoning procedure:\n'+protocol.STEPS['room_size_estimation']
        +'\nGive a brief evidence summary (at most 120 words); describe scale assumptions and uncertainty. '
        'Refer only to views actually supplied. End with exactly one line: ANSWER: <number in square metres, one decimal>')
    from .progressive_qa import answer as progressive_answer,InsufficientEvidence
    prompt=prompt.replace('End with exactly one line: ANSWER: <number in square metres, one decimal>',
        'Return JSON {"answer":number,"unit":"m²","reason":"Chinese concise explanation","confidence":"low|medium|high","sufficient_evidence":boolean,"limitations":"Chinese"}')
    def validate(a,ids):
        value=a.get('answer')
        if type(value) not in (int,float) or not math.isfinite(value) or value<0 or a.get('unit')!='m²':raise ValueError('Invalid room area response')
    candidates=[dict(id=v['view_id'],path=v['file'],sha256=sha(v['file'])) for v in views['views']]
    state(out,'qa')
    try:a=progressive_answer(out,prompt,validate,views=candidates,task='room_size',question=q)
    except InsufficientEvidence as exc:
        state(out,'insufficient-evidence');return exc.result
    a.update(source_scene_reused=True,historical_scene_generation_cost_included=False)
    save(out/'qa/answer.json',a)
    state(out,'completed')
