import bpy,json,sys
from pathlib import Path
out=Path(sys.argv[sys.argv.index('--')+1]).resolve();sys.path.insert(0,str(out))
from coverage_check import validate
try:
    (out/'coverage-failure.json').unlink(missing_ok=True)
    records=[dict(name=o.name,role=o.get('role'),category=o.get('category'),instance_id=o.get('instance_id'),
                  dimensions=list(o.dimensions),color=list(o.color),hide_render=o.hide_render)
             for o in bpy.context.scene.objects if o.type=='MESH']
    result=validate(json.loads((out/'input.json').read_text()),records,bpy.context.scene.get('target_evidence_inventory'))
    # Re-open the saved artifact: validation must apply to what QA renders, not
    # just unsaved in-memory additions after the generated script's save call.
    bpy.ops.wm.open_mainfile(filepath=str(out/'scene.blend'))
    records=[dict(name=o.name,role=o.get('role'),category=o.get('category'),instance_id=o.get('instance_id'),
                  dimensions=list(o.dimensions),color=list(o.color),hide_render=o.hide_render)
             for o in bpy.context.scene.objects if o.type=='MESH']
    saved=validate(json.loads((out/'input.json').read_text()),records,bpy.context.scene.get('target_evidence_inventory'))
    if saved['instances_by_category']!=result['instances_by_category']:raise ValueError('Saved scene target instances differ from in-memory scene')
    result=saved
    result['saved_target_presence_checked']=True
    (out/'coverage-validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print('V10_2_COVERAGE_VALID')
except Exception as exc:
    data=json.loads((out/'input.json').read_text())
    present={r['category'] for r in records if r['role']=='target'}
    (out/'coverage-failure.json').write_text(json.dumps(dict(error=str(exc),
        missing_categories=sorted(set(data['question']['objects'])-present)),ensure_ascii=False,indent=2))
    (out/'scope-validation.json').unlink(missing_ok=True)
    raise
