"""V10_2: audited evidence coverage, category identity and three-view joint QA.
No answer keys are read here. The original V10 source/lock stays unchanged.
"""
import ast,json,sys,shutil,subprocess
from pathlib import Path
from bvb_v10.pipeline import ROOT,HERE,read,save,sha
from . import pipeline as base
REVISION='20261111-coverage-colors-threeview'
PALETTE=[
    (.85,.04,.025,1),  # red
    (.025,.50,.06,1),  # green
    (.025,.16,.90,1),  # blue
    (.95,.36,.01,1),   # orange
    (.55,.025,.80,1),  # violet
    (.01,.65,.72,1),   # cyan
    (.85,.10,.42,1),   # rose
    (.38,.22,.07,1),   # brown
    (.65,.85,.015,1),  # lime
    (.48,.44,.015,1),  # olive
    (.005,.27,.22,1),  # teal
    (.025,.035,.24,1), # navy
    (.32,.015,.06,1),  # burgundy
    (.95,.73,.02,1),   # golden yellow
    (.10,.10,.10,1),   # charcoal
    (.15,.46,.90,1),   # sky blue
]
EXTRA='''
V10_2 COVERAGE AND IDENTITY CONTRACT (priority over earlier missing-object allowances):
Use exactly the category colors in CATEGORY_COLOR_MAP below for EVERY target mesh's
object.color. Different categories have distinct colors; all instances of one category
share that category color. No text labels, arrows or legend geometry in the room.
Preserve every image-supported distinct instance that can participate in this question.
Do NOT restrict a category to one instance. Do NOT confuse mesh parts with instances.
Do NOT create one copy per frame or per evidence group: multiple groups can observe the
same physical instance, and one group can show several instances.
Do not silently omit an image-supported question category or substitute another object
of the same color. Evidence grouping is contextual, NOT a required one-to-one mapping:
there is NO requirement to assign every group to an instance, or every instance to a group.
Use any supplied input frame to identify additional relevant instances, including frames
selected for another category. Do not fabricate geometry to satisfy a detector proposal.
If a candidate appears to be a false positive, you may omit that candidate and optionally
record the uncertainty. scene['target_evidence_inventory'] and additional evidence notes
are optional diagnostics; incomplete or empty mappings do not block validation.
The hard coverage check only requires at least one real instance of every question category.
Keep all evidenced walls/openings/floors and direct target supports/contacts as white blocks.
No unrelated furnishings. Do not solve or use the answer to place geometry.
The installed Blender is 5.3: use BLENDER_WORKBENCH if assigning a render engine.
BLENDER_EEVEE_NEXT is not available here. Final overview rendering is controlled externally.
'''

def prepare(source,out,poses,replace_poses=False):
    ref=base.prepare(source,out,poses,replace_poses)
    d=read(out/'input.json');objects=d['question']['objects']
    if len(objects)>len(PALETTE):raise ValueError('Extend distinct palette before preparing >16 categories; never recycle colors')
    d['colors']={obj:list(PALETTE[i]) for i,obj in enumerate(objects)}
    if d['question']['question_type'] in {'object_rel_distance_closest','object_rel_distance_farthest','object_counting_single','object_counting_multiple'}:
        d['instance_limits']={obj:None for obj in objects}
    d['coverage_inventory_required']=False;d['category_coverage_required']=True;d['revision']=REVISION
    save(out/'input.json',d)
    from bvb_v10.prompt import build
    from .reference import plot
    source_manifest=read(source/'manifest.json')
    first=source/source_manifest['groups'][0]['folder']/'pose-reference-vlm.json'
    encoding=read(first)['explanation']
    prompt=build(d['question'],d['instance_limits'],d['colors'],out,encoding)+base.EXTRA+EXTRA+'\nCATEGORY_COLOR_MAP='+json.dumps(d['colors'])
    (out/'system-prompt.txt').write_text(prompt)
    plot(out,ref,d['colors'])
    save(out/'color-map.json',dict(colors=d['colors'],same_category_same_color=True,instance_ids_internal_only=True))
    exp=read(out/'experiment.json');exp.update(revision=REVISION,qa_view_count=3,qa_walls_hidden=True,unique_category_colors=True,
        remove_overhead_occluders=True,
        strict_evidence_coverage=False,category_coverage_required=True,multiple_instances=True,input_sha256=sha(out/'input.json'),
        prompt_sha256=sha(out/'system-prompt.txt'),reference_images={f:sha(out/f) for f in ['reference-top.png','reference-oblique.png']})
    save(out/'experiment.json',exp)
    return ref

def run(out):
    """Standalone relative-distance run; batch runner adds bounded repair retries."""
    sys.path.insert(0,str(ROOT/'scripts'))
    from bvb_v11.common import state,BLENDER
    try:
        if not (out/'generated.py').exists():state(out,'reconstructing');base.infer(out)
        if not (out/'coverage-validation.json').exists():state(out,'building');execute(out,BLENDER)
        if not (out/'views/views.json').exists():state(out,'rendering');execute(out,BLENDER,True)
        state(out,'selecting-views');qa(out);state(out,'completed')
    except Exception as exc:
        state(out,'failed',error=str(exc));raise

def execute(out,blender,render=False):
    """Same OS isolation as V10, with strict post-build evidence validation."""
    out=out.resolve()
    if sys.platform!='darwin' or not Path('/usr/bin/sandbox-exec').exists():raise RuntimeError('Requires isolated macOS Blender adapter')
    for name in ['home','tmp']:(out/name).mkdir(exist_ok=True)
    sandbox='(version 1)\n(allow default)\n(deny network*)\n(deny file-read* (subpath '+json.dumps(str(Path.home()))+'))\n(allow file-read* (subpath '+json.dumps(str(out))+'))\n(deny file-write*)\n(allow file-write* (subpath '+json.dumps(str(out))+') (subpath "/private/tmp") (subpath "/tmp") (literal "/dev/null"))\n'
    (out/'execution.sb').write_text(sandbox)
    env=dict(PATH='/usr/bin:/bin:/usr/sbin:/sbin',HOME=str(out/'home'),TMPDIR=str(out/'tmp'))
    cmd=['/usr/bin/sandbox-exec','-f',str(out/'execution.sb'),blender,'--background','--disable-autoexec','--python-exit-code','1']
    shutil.copy2(HERE/'contract.py',out/'v10_contract.py')
    if render:
        if not read(out/'coverage-validation.json').get('valid'):raise ValueError('Strict coverage must pass before rendering')
        # Add intermediate elevation to the original top/oblique candidate views.
        script=(HERE/'runtime/render.py').read_text().replace("[(f'oblique-{yaw:03d}',[20,-20,yaw]) for yaw in range(0,360,45)]", "[(f'oblique-{yaw:03d}',[20,-20,yaw]) for yaw in range(0,360,45)]+[(f'elevated-{yaw:03d}',[45,0,yaw]) for yaw in range(0,360,90)]")
        # The original scene and pose renders retain architecture. Only candidate
        # QA overview images hide wall/opening-boundary meshes. Restore flags
        # before writing the separate reference-camera blend.
        script=script.replace('points=points_for(meshes);views=[]',"points=points_for(meshes);views=[]\nwall_states={o.name:o.hide_render for o in meshes if o.get('role')=='structure' and o.get('category')=='wall'}")
        script=script.replace("path=out/'views'/(name+'.png');s.render.filepath=str(path);bpy.ops.render.render(write_still=True)","""if name=='top':
        s.render.filepath=str(out/'views'/'overview-with-walls.png');bpy.ops.render.render(write_still=True)
    for wall_name in wall_states:bpy.data.objects[wall_name].hide_render=True
    path=out/'views'/(name+'.png');s.render.filepath=str(path);bpy.ops.render.render(write_still=True)""")
        script=script.replace("dict(views=views)","dict(views=views,qa_walls_hidden=True,hidden_wall_mesh_count=len(wall_states),full_overview='overview-with-walls.png')")
        script=script.replace("for o in meshes:\n    assert max", "for wall_name,hidden in wall_states.items():bpy.data.objects[wall_name].hide_render=hidden\nfor o in meshes:\n    assert max")
        if not read(out/'experiment.json').get('qa_walls_hidden',True):
            # Route planning depends on walls and doorway connectivity.
            script=script.replace("for wall_name in wall_states:bpy.data.objects[wall_name].hide_render=True", "pass # Keep walls visible for route planning")
            script=script.replace("qa_walls_hidden=True,hidden_wall_mesh_count=len(wall_states)", "qa_walls_hidden=False,hidden_wall_mesh_count=0")
        # Shared by all V10_2 question types. Original/pose views and scene.blend
        # retain all geometry. Only actual higher occluders are hidden per view.
        if read(out/'experiment.json').get('remove_overhead_occluders', True):
            shutil.copy2(Path(__file__).with_name('runtime_occlusion.py'),out/'runtime_occlusion.py')
            script=script.replace('points=points_for(meshes);views=[]',
                'from runtime_occlusion import render_with_overhead_removal\nocclusion_reports=[]\npoints=points_for(meshes);views=[]')
            script=script.replace("path=out/'views'/(name+'.png');s.render.filepath=str(path);bpy.ops.render.render(write_still=True)",
                "path=out/'views'/(name+'.png')\n    occlusion_reports.append(render_with_overhead_removal(s,overview,meshes,path,out,name))")
            script=script.replace("(out/'views/views.json').write_text", "(out/'occlusion-report.json').write_text(json.dumps(dict(views=occlusion_reports,geometry_modified=False,visibility_restored=True),indent=2))\n(out/'views/views.json').write_text")
            script=script.replace('dict(views=views,qa_walls_hidden=', 'dict(views=views,overhead_occlusion_removed=True,occlusion_report="../occlusion-report.json",qa_walls_hidden=')
        (out/'render.py').write_text(script)
        cmd += [str(out/'scene.blend'),'--python',str(out/'render.py'),'--',str(out)]
    else:
        (out/'scope-validation.json').unlink(missing_ok=True)
        (out/'coverage-validation.json').unlink(missing_ok=True)
        shutil.copy2(HERE/'runtime/validate.py',out/'validate.py')
        shutil.copy2(Path(__file__).with_name('coverage_check.py'),out/'coverage_check.py')
        shutil.copy2(Path(__file__).with_name('runtime_coverage.py'),out/'coverage.py')
        executable=adapt_engine(out)
        cmd += ['--factory-startup','--python',str(executable),'--python',str(out/'validate.py')]
        if read(out/'input.json').get('counting_mode'):
            for filename in ['counting_mode.py','runtime_counting_colors.py']:
                shutil.copy2(Path(__file__).with_name(filename),out/filename)
            save(out/'counting-palette.json',PALETTE)
            cmd += ['--python',str(out/'runtime_counting_colors.py')]
        cmd += ['--python',str(out/'coverage.py'),'--',str(out)]
    name='render' if render else 'build'
    with (out/(name+'.log')).open('w') as log:
        r=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
    if r.returncode:
        if not render:(out/'scope-validation.json').unlink(missing_ok=True)
        tail=(out/(name+'.log')).read_text()[-10000:];(out/'repair-feedback.txt').write_text(tail)
        raise RuntimeError(name+' failed; inspect '+name+'.log')
    if not render and read(out/'input.json').get('counting_mode'):
        exp=read(out/'experiment.json');exp.update(input_sha256=sha(out/'input.json'),
            instance_color_map_sha256=sha(out/'instance-color-map.json'),instance_colors_applied=True)
        save(out/'experiment.json',exp)

def adapt_engine(out):
    """Only normalize an obsolete engine enum; preserve the original model code."""
    source=out/'generated.py';tree=ast.parse(source.read_text());changes=[]
    class Adapter(ast.NodeTransformer):
        def visit_Assign(self,node):
            if (isinstance(node.value,ast.Constant) and node.value.value=='BLENDER_EEVEE_NEXT'
                and any(isinstance(t,ast.Attribute) and t.attr=='engine' and isinstance(t.value,ast.Attribute) and t.value.attr=='render' for t in node.targets)):
                changes.append(dict(line=node.lineno,from_value='BLENDER_EEVEE_NEXT',to_value='BLENDER_WORKBENCH'))
                node.value=ast.copy_location(ast.Constant('BLENDER_WORKBENCH'),node.value)
            return node
    tree=Adapter().visit(tree)
    if changes:
        target=out/'executed.py';target.write_text(ast.unparse(ast.fix_missing_locations(tree))+'\n')
    else:target=source
    save(out/'execution-adaptations.json',dict(source_sha256=sha(source),executed_sha256=sha(target),changes=changes,geometry_modified=False))
    return target

def qa(out):
    sys.path.insert(0,str(ROOT/'scripts'))
    from subscription_direct_client import request,image_item
    from bvb_v11.common import parse,state
    d=read(out/'input.json');q=d['question'];inventory=read(out/'coverage-validation.json')
    assert inventory['valid']
    views=read(out/'views/views.json')['views'];byid={v['id']:v for v in views}
    legend='Category RGB colors (0..1): '+json.dumps(d['colors'])+'. White=floor and direct supports/contacts. Walls and doorway boundaries are hidden ONLY for these views to remove occlusion; objects remain in their reconstructed positions. Absence of walls in these images does not imply shared room membership or an open walking path. Same color can represent MULTIPLE distinct instances. Mesh parts are NOT additional instances.'
    legend+=' Higher non-target objects that occlude targets may be hidden separately in each view. These are cutaway displays: do not infer changed height, lost support/contact or free passage from hidden geometry. Targets are never removed by this visibility step.'
    identities=[dict(category=k,declared_instance_count=len(ids)) for k,ids in inventory['instances_by_category'].items()]
    context=json.dumps(q,ensure_ascii=False)+'\n'+legend+'\nInternal scene instance inventory (no metric coordinates): '+json.dumps(identities)
    if not (out/'view-selection/selection.json').exists():
        prompt='Select EXACTLY THREE DISTINCT complementary supplied views for this question. At least two must be non-top views with different azimuths. Prioritize covering ALL categories and instances across the three images, including obscured targets and height differences. Avoid redundant directions. Do NOT answer the question. A declared object is not necessarily visually identifiable: report occluded or unresolved categories honestly. Return JSON {"view_ids":[id,id,id],"reason":"Chinese","limitations":"Chinese","coverage":[{"category":str,"visible_in":[view_id,...],"uncertainty":str}]}. Include every question category in coverage, and only selected view IDs in visible_in.\n'+context
        content=[dict(type='input_text',text=prompt)]
        for v in views:
            p=out/'views'/v['image'];assert sha(p)==v['sha256']
            content.extend([dict(type='input_text',text='VIEW '+v['id']),image_item(p)])
        a=parse(request(out/'view-selection',content,model='gpt-6-astra',effort='high')['answer'])
        validate_views(a,byid,q['objects'])
        a['view_id']=a['view_ids'][0] # compatibility only; QA always consumes all three.
        save(out/'view-selection/selection.json',a)
    chosen=read(out/'view-selection/selection.json');validate_views(chosen,byid,q['objects'])
    state(out,'qa')
    if not (out/'qa/answer.json').exists():
        options=dict(o.split('. ',1) for o in q['options'])
        prompt='Answer using ONLY these THREE reconstructed views of the SAME scene, the question, color legend and declared instance counts. Integrate the views jointly, do not vote independently. Identify categories from their assigned colors, not guessed appearance. Same-category instances share color; track them spatially across views without double counting. Compare straight-line 3D distances between closest SURFACE points, not centers or walking paths. Consider ALL reference and candidate instances; do not silently discard a far or inconvenient instance. Follow the question wording for closest/farthest and explain the multi-instance interpretation if ambiguous. A declared instance is not proof it is visible or accurate. Disclose missing/occluded or unreliable evidence; still choose the best option. Do not assume calibrated metric scale. Return JSON {"option":letter,"answer":exact_option_text_without_letter,"reason":"Chinese","confidence":"low|medium|high","sufficient_evidence":boolean,"limitations":"Chinese","instance_consideration":[{"category":str,"considered_count":integer,"unresolved_count":integer,"assessment":str}],"cross_view_consistency":str}. For each category, considered_count plus unresolved_count must equal its declared instance count. Describe visible instances by their relative image locations and account for unresolved ones honestly; do not invent ID-to-image correspondences.\n'+context
        content=[dict(type='input_text',text=prompt)]
        for vid in chosen['view_ids']:
            v=byid[vid];p=out/'views'/v['image'];assert sha(p)==v['sha256']
            content.extend([dict(type='input_text',text='VIEW '+vid),image_item(p)])
        a=parse(request(out/'qa',content,model='gpt-6-astra',effort='high')['answer'])
        assert a['option'] in options and a['answer'].strip()==options[a['option']].strip()
        assert type(a['sufficient_evidence']) is bool
        validate_instances(a,inventory['instances_by_category'])
        a.update(view_ids=chosen['view_ids'],revision=REVISION)
        save(out/'qa/answer.json',a)

def validate_views(a,byid,categories):
    ids=a['view_ids']
    if len(ids)!=3 or len(set(ids))!=3 or not set(ids)<=set(byid):raise ValueError('Exactly 3 distinct supplied views required')
    angles={int(v.rsplit('-',1)[1]) for v in ids if v!='top'}
    if len(angles)<2:raise ValueError('Need at least two different oblique azimuths')
    rows=a['coverage']
    if len(rows)!=len(categories) or {r['category'] for r in rows}!=set(categories):raise ValueError('Incomplete visibility coverage report')
    if any(not set(r['visible_in'])<=set(ids) for r in rows):raise ValueError('Coverage references unselected view')

def validate_instances(a,expected):
    rows=a['instance_consideration']
    if len(rows)!=len(expected) or {r['category'] for r in rows}!=set(expected):raise ValueError('QA must consider all target categories')
    for r in rows:
        a,b=r['considered_count'],r['unresolved_count']
        if type(a) is not int or type(b) is not int or min(a,b)<0 or a+b!=len(expected[r['category']]):raise ValueError('QA silently omitted or invented instances: '+r['category'])
