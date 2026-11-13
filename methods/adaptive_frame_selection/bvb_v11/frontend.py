"""Preserve the latest 2fps per-question feature-cache selection recipe."""
import json,os,subprocess,sys
from pathlib import Path
from bvb_v10.pipeline import ROOT,METHOD,read,save,sha

def run(out,name,args):
    env=os.environ.copy();env['PYTHONPATH']=str(METHOD)+os.pathsep+env.get('PYTHONPATH','')
    env.setdefault('HF_ENDPOINT','https://hf-mirror.com')
    env['MPLCONFIGDIR']=str(out/'mpl-cache')
    if args[:2]==['-m','bvb_v3.adaptive_peaks']:
        # Existing peak plots name a macOS font; use DejaVu on Linux workers.
        boot="from PIL import ImageFont; from pathlib import Path; import sys,runpy; old=ImageFont.truetype; ImageFont.truetype=lambda name,*a,**k: old('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf' if str(name)=='/System/Library/Fonts/Supplemental/Arial.ttf' and not Path(name).exists() else name,*a,**k); sys.argv=['bvb_v3.adaptive_peaks']+sys.argv[1:]; runpy.run_module('bvb_v3.adaptive_peaks',run_name='__main__')"
        args=['-c',boot,*args[2:]]
    with (out/(name+'.log')).open('w') as f:
        subprocess.run([sys.executable,*map(str,args)],cwd=METHOD,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)

def sample(cache,poses,flow,questions,out,device,counting_mode=False):
    cache,poses,flow,questions,out=[p.resolve() for p in [cache,poses,flow,questions,out]]
    if out.exists():raise FileExistsError('Use a fresh sampling output')
    cfg=read(cache/'config.json');rr=read(cache/'candidates.json');sampling=read(poses/'sampling.json')
    if cfg['candidate_fps']!=2 or sampling['sampling_fps']!=2:raise ValueError('V10 default feature caches must both be 2fps')
    if cfg['video_sha256']!=sampling['video_sha256']:raise ValueError('Video hashes differ')
    for a,b in zip(rr,sampling['frames'],strict=True):
        if (a['source_frame_index'],a['frame_sha256'])!=(b['source_frame_index'],b['sha256']):raise ValueError('Feature frame mismatch')
    raw=read(questions);qq=raw['questions'] if isinstance(raw,dict) else raw
    public=[{k:q[k] for k in ['id','scene_name','question','question_type','options','roles','objects'] if k in q} for q in qq]
    if any(not q.get('objects') for q in public):raise ValueError('Explicit public object extraction required; global/temporal supplementation is not inferred here')
    if len({str(q['id']) for q in public})!=len(public):raise ValueError('Question IDs must be unique')
    from .counting_mode import peak_threshold
    threshold=peak_threshold(public,counting_mode)
    objects=sorted({o for q in public for o in q['objects']})
    if not set(objects)<=set(cfg['objects']):raise ValueError('Some question objects are missing from detector cache vocabulary')
    out.mkdir(parents=True);payload=dict(objects=objects,questions=public)
    save(out/'questions.json',payload)
    detections=[json.loads(s) for s in (cache/'detections.jsonl').read_text().splitlines()]
    for name,labels,ds in [('coarse',cfg['objects'],detections),('keyboard-pass',['keyboard'],
            [dict(d,detections=[b for b in d['detections'] if b['object']=='keyboard']) for d in detections])]:
        folder=out/name;folder.mkdir();save(folder/'config.json',dict(cfg,objects=labels));save(folder/'candidates.json',rr)
        save(folder/'detector-info.json',dict(source=str(cache),keyboard_semantics='pass-through; piano separate'))
        (folder/'detections.jsonl').write_text(''.join(json.dumps(d)+'\n' for d in ds));(folder/'frames').symlink_to(cache/'frames')
    run(out,'01-peaks',['-m','bvb_v3.adaptive_peaks','--coarse',out/'coarse','--local',cache,'--piano',out/'keyboard-pass',
        '--questions',out/'questions.json','--native-flow',flow,'--out',out/'candidates','--peak-threshold',str(threshold)])
    stub=out/'appearance-preparation';save(stub/'groups.json',[]);save(stub/'anchor-geometry.json',{})
    run(out,'02-rois',['-m','bvb_v5.dinov2_appearance','prepare','--result',stub,'--peaks',out/'candidates/report/peaks.json','--cache',cache,'--out',out/'appearance'])
    run(out,'03-appearance',['-m','bvb_v5.dinov2_appearance','extract','--out',out/'appearance',
        '--revision','f9e44c814b77203eaa57a6bdbbd535f21ede1415','--batch-size','8','--device',device])
    v3=out/'question-input';save(v3/'questions.json',payload);(v3/'peaks.json').symlink_to(out/'candidates/report/peaks.json')
    run(out,'04-select',['-m','bvb_v6','select','--v3',v3,'--cache',cache,'--poses',poses,'--appearance',out/'appearance','--out',out/'baseline','--frame-cap','8'])
    run(out,'05-clusters',[METHOD/'scripts/run_v6_object_clustering.py','--source',out/'baseline/selection.json','--minimum-frame-pairs','2','--out',out/'clusters'])
    save(out/'sampling-receipt.json',dict(version='bvb_v10',questions=len(public),candidate_fps=2,peak_threshold=threshold,
         frames_per_peak=4,frame_cap_per_group=8,minimum_frame_pairs=2,new_detection=False,semantic_review=False,
         input_sha256={str(p):sha(p) for p in [questions,flow,cache/'config.json',cache/'detections.jsonl',poses/'camera-poses.npz']}))

def audit(selection,out):
    selection=selection.resolve();out=out.resolve()
    if out.exists():raise FileExistsError('Use fresh audit directory; resume existing runs using the established audit runner')
    sys.path[:0]=[str(METHOD),str(ROOT/'scripts')]
    import review_revsi_first10_astra as ar
    from bvb_v6 import semantic_review as sr
    d=read(selection)
    for row in d['questions'].values():
        row['question']={k:row['question'][k] for k in ['id','scene_name','question','question_type','options','roles','objects'] if k in row['question']}
    save(out/'input/selection.json',d)
    sr.prepare(out/'input/selection.json',out/'input/review')
    ar.OUT=out;ar.REVIEW=out/'input/review';ar.main()
