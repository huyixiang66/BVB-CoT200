"""Blender-only deterministic meter ruler; no model-authored executable code."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import bpy
from mathutils import Euler, Vector

p=argparse.ArgumentParser()
p.add_argument('--source',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
manifest=json.loads((a.source/'manifest.json').read_text())
assert manifest.get('object_size_reference_mode') and len(manifest['groups'])==1
folder=a.source/manifest['groups'][0]['folder']/'assets'
audit=json.loads((folder/'question-audit.json').read_text())
spec=json.loads((folder/'scene-spec.json').read_text())
reference=audit.get('scale_reference')
scene=bpy.context.scene
out=a.out
record={'metric_scale_verified':False,'ground_truth_used':False,'scale_reference':reference,'ruler_present':False,
        'renderer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
if reference:
    obj=next(o for o in spec['objects'] if o['id']==reference['geometry_id'])
    center=Vector(obj['p']);size=Vector(obj['s'])
    rot=Euler([math.radians(v) for v in obj.get('r',[0,0,0])],'XYZ').to_matrix()
    axis=reference['axis'];length=reference['assumed_length_m'];unit=reference['meters_per_scene_unit']
    # Choose a face normal perpendicular to the measured axis. Show that face
    # directly, preserving the ruler's metric spacing without foreshortening.
    normal_axis=1 if axis!=1 else 0
    tick_axis=({0,1,2}-{axis,normal_axis}).pop()
    normal=rot.col[normal_axis].copy()
    tangent=rot.col[tick_axis].copy()
    direction=rot.col[axis].copy()
    offset=max(size)*.015
    start_local=Vector((0,0,0));start_local[axis]=-size[axis]/2
    start_local[normal_axis]=-size[normal_axis]/2-offset
    start_local[tick_axis]=size[tick_axis]/2
    start=center+rot@start_local
    radius=max(size)*.0035
    collection=bpy.data.collections.new('Estimated meter ruler');scene.collection.children.link(collection)
    def link_only(o):
        for c in list(o.users_collection):c.objects.unlink(o)
        collection.objects.link(o)
        o['scale_annotation']=True
    def line(name,pa,pb):
        delta=pb-pa
        bpy.ops.mesh.primitive_cylinder_add(vertices=12,radius=radius,depth=delta.length,location=(pa+pb)/2)
        o=bpy.context.object;o.name=name;o.rotation_euler=delta.to_track_quat('Z','Y').to_euler()
        o.color=(.03,.03,.03,1);link_only(o)
    line('Meter baseline',start,start+direction*size[axis])
    step=.02 if length<=.3 else (.05 if length<=.6 else (.1 if length<=1.2 else (.2 if length<=2.5 else .5)))
    values=[round(i*step,8) for i in range(math.floor(length/step)+1)]
    if length-values[-1]>step*.3:values.append(length)
    labels=[];ticks=[]
    textsize=max(size[axis]/max(len(values),2)*.29,max(size)*.02)
    for value in values:
        pos=start+direction*(value/unit)
        line('Tick '+str(value),pos-tangent*textsize*.4,pos+tangent*textsize*.8)
        curve=bpy.data.curves.new('Meter label','FONT');curve.body=f'{value:.2f}'.rstrip('0').rstrip('.')+' m'
        curve.size=textsize;curve.align_x='CENTER';curve.align_y='CENTER'
        label=bpy.data.objects.new('Meter '+str(value),curve);collection.objects.link(label)
        label.location=pos+tangent*textsize*1.8;label.color=(.02,.02,.02,1);label['scale_annotation']=True
        labels.append(label);ticks.append({'meters':value,'world_position':list(pos)})
    camera=scene.camera
    # Frame target and reference together; hide occluding structural slabs in
    # this evidence close-up only. Geometry remains intact in the saved scene.
    target_ids={x['id'] for x in audit['targets']}|{reference['geometry_id']}
    meshes=[o for o in scene.objects if o.type=='MESH' and
        any(c.name.rsplit('::',1)[-1] in target_ids for c in o.users_collection)]
    # The merger keeps asset_id on part meshes and prefixes their names.
    if not meshes:
        meshes=[o for o in scene.objects if o.type=='MESH' and any(t in o.name for t in target_ids)]
    bounds=[o.matrix_world@Vector(v) for o in meshes for v in o.bound_box]
    bounds += [start,start+direction*size[axis],start+tangent*textsize*3]
    camera.rotation_euler=(normal-tangent*.7).to_track_quat('-Z','Y').to_euler()
    basis=camera.rotation_euler.to_matrix()
    projected=[basis.transposed()@v for v in bounds]
    lo=Vector([min(v[i] for v in projected) for i in range(3)])
    hi=Vector([max(v[i] for v in projected) for i in range(3)])
    mid=(lo+hi)/2
    camera.location=basis@Vector((mid.x,mid.y,hi.z+max(size)*10+10))
    camera.data.ortho_scale=max(hi.x-lo.x,hi.y-lo.y)*1.35
    for label in labels:label.rotation_euler=camera.rotation_euler
    # Reference face is on negative local normal side; camera must face it.
    # to_track above looks along +normal, hence camera is on -normal.
    hidden=[]
    for o in scene.objects:
        if o.type=='MESH' and o not in meshes and not o.get('scale_annotation') and not o.hide_render:
            hidden.append(o);o.hide_render=True
    scene.display.shading.show_shadows=False
    scene.render.resolution_x=scene.render.resolution_y=1400
    scene.render.filepath=str(out/'reference-ruler.png')
    bpy.ops.render.render(write_still=True)
    overview_location=camera.location.copy();overview_scale=camera.data.ortho_scale
    reference_meshes=[o for o in meshes if any(c.name.rsplit('::',1)[-1]==reference['geometry_id'] for c in o.users_collection)]
    bpy.context.view_layer.update()
    detail_bounds=[o.matrix_world@Vector(v) for o in reference_meshes+labels for v in o.bound_box]
    projected=[basis.transposed()@v for v in detail_bounds]
    lo=Vector([min(v[i] for v in projected) for i in range(3)])
    hi=Vector([max(v[i] for v in projected) for i in range(3)])
    mid=(lo+hi)/2
    camera.location=basis@Vector((mid.x,mid.y,hi.z+max(size)*10+10))
    camera.data.ortho_scale=max(hi.x-lo.x,hi.y-lo.y)*1.2
    for o in meshes:
        if o not in reference_meshes:o.hide_render=True
    scene.render.filepath=str(out/'reference-detail.png')
    bpy.ops.render.render(write_still=True)
    for o in meshes:o.hide_render=False
    camera.location=overview_location;camera.data.ortho_scale=overview_scale
    scene.render.filepath=str(out/'reference-ruler.png')
    record.update(ruler_present=True,ticks=ticks,tick_step_m=step,reference_geometry=obj,
                  preview='reference-ruler.png',preview_structure_hidden=True,preview_context_hidden=True,
                  detail_preview='reference-detail.png',saved_context_geometry_retained=True,
                  saved_context_hidden_for_render=True,
                  note='Meter labels use an estimated reference dimension, not measured calibration.')
    bpy.ops.wm.save_as_mainfile(filepath=str(out/'scene-with-ruler.blend'))
else:
    record['reason']=audit.get('reference_unavailable_reason')
(out/'scale-reference.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
