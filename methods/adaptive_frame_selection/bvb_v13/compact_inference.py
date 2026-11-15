"""Ask only for scene parameters, then compile the fixed local Blender program."""
import json,sys
from pathlib import Path
from bvb_v10.pipeline import ROOT,read,save,sha
from .scene_spec import validate,compile_script,VERSION
from .scene_prompt import build as prompt_for,GEOMETRY_POLICY,task_for
from .scene_evidence import pack,dumps
from .temporal_frames import select as select_frames

class SceneSpecError(ValueError):pass

def infer(out,attempt='attempt-01'):
    from .pipeline import prepare_reference_images
    from .common import parse
    sys.path.insert(0,str(ROOT/'scripts'))
    from subscription_direct_client import request,image_item
    out=Path(out);d=read(out/'input.json');exp=read(out/'experiment.json');folder=out/attempt
    if sha(out/'input.json')!=exp['input_sha256']:raise ValueError('Scene evidence changed')
    if not d.get('room_mode') and sha(out/'spatial-reference.json')!=exp['reference_sha256']:raise ValueError('Soft reference changed')
    d,selection=select_frames(d)
    selection['source_input_sha256']=exp['input_sha256']
    selection_path=out/'reconstruction-frame-selection.json'
    if selection_path.exists() and read(selection_path)!=selection:raise ValueError('Reconstruction frame selection changed; use fresh output')
    save(selection_path,selection)
    save(folder/'reconstruction-image-inputs.json',dict(policy=selection['policy'],source_input_sha256=exp['input_sha256'],images=[dict(path=f['path'],sha256=f['sha256'],timestamp_s=f['timestamp_s'],source_packet_index=i) for i,f in zip(selection['kept_source_packet_indices'],d['frames'])]))
    prompt=prompt_for(d);folder.mkdir(parents=True,exist_ok=True)
    (folder/'scene-prompt.txt').write_text(prompt)
    ref=read(out/'spatial-reference.json') if not d.get('room_mode') else None
    evidence=pack(d,ref['anchors'] if ref is not None else None)
    save(folder/'evidence-transport.json',evidence)
    content=[dict(type='input_text',text=dumps(evidence['header']))]
    for i,f in enumerate(d['frames']):
        if sha(f['path'])!=f['sha256']:raise ValueError('Original frame changed')
        content.extend([dict(type='input_text',text=dumps(evidence['frames'][i])),image_item(f['path'])])
    if (out/'pose-reference-vlm.json').exists():
        content.append(dict(type='input_text',text=read(out/'pose-reference-vlm.json')['explanation'].replace('Camera row follows each image:', 'compact_camera:')))
    if not d.get('room_mode'):
        content.append(dict(type='input_text',text=dumps(evidence['anchors'])))
        for image in prepare_reference_images(out,exp,attempt):
            content.extend([dict(type='input_text',text=image['name']+' empty-space reference'),image_item(image['path'])])
    if (out/'scale-reference.json').exists():
        info=read(out/'scale-reference.json')
        content.append(dict(type='input_text',text='Reference object: '+json.dumps(dict(category=info['category'],role=info.get('reference_role'),dimension=info['choice']['dimension'],dimension_description=info['choice']['dimension_description']),ensure_ascii=False)))
    if attempt!='attempt-01' and (out/'repair-feedback.txt').exists():
        previous=read(out/'scene-spec.json') if (out/'scene-spec.json').exists() else None
        content.append(dict(type='input_text',text='Correct the reported scene-data/schema errors using the images. Return a complete replacement JSON. Do not invent missing evidence.\n'+(out/'repair-feedback.txt').read_text()[-5000:]+'\nPrevious spec: '+json.dumps(previous,separators=(',',':'))))
    raw=request(folder,content,instructions=prompt,model='gpt-6-astra',effort='high')
    try:
        spec=parse(raw['answer'])
        if not isinstance(spec,dict) or type(spec.get('v')) is not int or spec['v']!=2:
            raise ValueError('New inference requires compact scene parameters with v=2')
        normalized=validate(spec,d);code=compile_script(spec,d,out/'scene.blend')
    except (ValueError,TypeError,KeyError,IndexError) as exc:
        save(folder/'scene-validation.json',dict(valid=False,error=str(exc)))
        (out/'repair-feedback.txt').write_text('Invalid scene specification: '+str(exc))
        raise SceneSpecError(str(exc)) from exc
    save(folder/'scene-spec.json',spec);save(folder/'scene-validation.json',dict(valid=True,schema=VERSION,geometry_policy=GEOMETRY_POLICY,task=task_for(d),objects=len(normalized['objects']),primitives=sum(len(x['parts']) for x in normalized['objects']),single_primitive_objects=sum(len(x['parts'])==1 for x in normalized['objects'])))
    save(folder/'scene-normalized.json',normalized)
    save(out/'scene-spec.json',spec);(out/(attempt+'.py')).write_text(code);(out/'generated.py').write_text(code)
    save(folder/'compiler-receipt.json',dict(schema=VERSION,model_output='scene parameters only',compiler_sha256=sha(Path(__file__).with_name('scene_runtime.py')),spec_sha256=sha(folder/'scene-spec.json'),compiled_code_sha256=sha(out/'generated.py'),generated_code_is_model_output=False))
