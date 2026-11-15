"""Seven-task dispatch; model inputs remain public questions and visual evidence."""
from pathlib import Path
import shutil
import sys
from bvb_v10.pipeline import ROOT,read,save,sha
from . import pipeline,reliability,counting,object_size,route,abs_distance_context
from .common import state
from .view_selection import configure, ensure_selection

KINDS={
 'object_rel_direction':'rel_dir',
 'object_rel_distance_closest':'rel_dist','object_rel_distance_farthest':'rel_dist',
 'object_abs_distance':'abs_dist',
 'object_counting_single':'counting','object_counting_multiple':'counting',
 'object_size_estimation':'object_size','route_planning':'route',
 'room_size_estimation':'room_size',
}

def kind(question):
    if question.get('question_type','').startswith('object_rel_direction_'):return 'rel_dir'
    if question.get('question_type','').startswith('room_size_estimation'):return 'room_size'
    try:return KINDS[question['question_type']]
    except KeyError:raise ValueError('Unsupported question type: '+str(question.get('question_type')))

def prepare(source,out,poses,*,context=None,view_selection=None):
    view_selection=configure(out,view_selection) if (out/"experiment.json").exists() else (view_selection or "fixed")
    if view_selection not in {"fixed","astra"}:raise ValueError("Unknown view-selection mode")
    task=kind(read(source/'manifest.json')['question'])
    if task=='room_size':raise ValueError('Use room with an existing blend and explicit floor/ceiling mapping')
    if task=='abs_dist':
        if context is None:raise ValueError('Absolute distance requires a 2-second context plan and source/scale-reference.json')
        reference=read(source/'scale-reference.json')
        if reference.get('scale_reference_policy')=='stable_width_only':
            from .abs_distance_width import prepare as prepare_width
            prepare_width(source,out,poses,context)
        else:abs_distance_context.prepare(source,out,poses,context)
    elif task=='counting':counting.prepare(source,out,poses,counting_mode=True,allow_all_rejected=True)
    else:{'rel_dir':pipeline,'rel_dist':reliability,'object_size':object_size,'route':route}[task].prepare(source,out,poses)
    exp=read(out/'experiment.json');exp.update(release='13.0.0',task_adapter=task,reconstruction_output='compact-scene-v2',qa_policy='progressive-answer-only-half-resolution-v3',generated_image_scale=.5)
    from .scene_prompt import build as scene_prompt
    (out/'system-prompt.txt').write_text(scene_prompt(read(out/'input.json')))
    exp['prompt_sha256']=sha(out/'system-prompt.txt')
    save(out/'experiment.json',exp)
    configure(out,view_selection)

def _qa(out,view_selection=None):
    configure(out,view_selection)
    ensure_selection(out)
    sys.path.insert(0,str(ROOT/'scripts'))
    task=kind(read(out/'question-public.json'))
    if task=='abs_dist' and read(out/'experiment.json').get('scale_reference_policy')=='stable_width_only':
        from .abs_distance_width import qa as answer_width
        return answer_width(out)
    if task=='rel_dir':
        from bvb_v10.qa import qa as answer
    else:answer={'rel_dist':reliability,'abs_dist':abs_distance_context,'counting':counting,'object_size':object_size,'route':route}[task].qa
    return answer(out)

def qa(out,view_selection=None):
    from .progressive_qa import InsufficientEvidence
    try:return _qa(out,view_selection)
    except InsufficientEvidence as exc:
        state(out,'insufficient-evidence')
        return exc.result

def run(out,blender,view_selection=None):
    """Resume successful stages; at most three generated scripts, preserving receipts."""
    from .progressive_qa import completion_stage
    out=out.resolve();configure(out,view_selection);sys.path.insert(0,str(ROOT/'scripts'))
    task=kind(read(out/'question-public.json'))
    if task=='rel_dir':
        from bvb_v10.pipeline import execute
    else:execute=reliability.execute
    try:
        if (out/'qa/answer.json').exists():state(out,completion_stage(out));return
        attempts=[int(p.name.split('-')[1]) for p in out.glob('attempt-*') if p.is_dir() and p.name.split('-')[1].isdigit()]
        number=max(attempts,default=1)
        def infer_valid(number):
            from .compact_inference import SceneSpecError
            while True:
                try:
                    pipeline.infer(out,f'attempt-{number:02d}');return number
                except SceneSpecError:
                    if number>=3:raise
                    number+=1;state(out,'repairing-scene-spec',attempt=number)
        if not (out/'generated.py').exists():
            state(out,'reconstructing');number=infer_valid(number)
        while True:
            try:
                if not (out/'scope-validation.json').exists():state(out,'building');execute(out,blender)
                if not (out/'views/views.json').exists():state(out,'rendering');execute(out,blender,True)
                break
            except Exception:
                if number>=3:raise
                # A new script invalidates all derived scene/render artifacts.
                number+=1
                archive=out/f'before-repair-{number:02d}';archive.mkdir(exist_ok=True)
                for name in ['scope-validation.json','coverage-validation.json','validation.json','scene.blend','views']:
                    if (out/name).exists():shutil.move(str(out/name),str(archive/name))
                state(out,'repairing',attempt=number);number=infer_valid(number)
        qa(out);state(out,completion_stage(out))
    except Exception as exc:
        state(out,'failed',error=str(exc));raise
