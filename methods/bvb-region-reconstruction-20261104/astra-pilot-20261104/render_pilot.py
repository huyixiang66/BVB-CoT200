"""Compile Astra JSON fragments and render in their shared world coordinates."""
from pathlib import Path
import importlib
import math
import sys

import bpy
from mathutils import Vector

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import build_session_blender as session

COLORS={'path':(.98,.20,.16),'start':(.08,.72,.24),'end':(.12,.43,.95),'direction':(.94,.45,.04),'query':(.52,.16,.84)}


def color(obj, label):
    name='Trajectory '+label
    mat=bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.diffuse_color=(*COLORS[label],1)
    obj.data.materials.append(mat)
    return obj


def sphere(name,point,radius,label):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=16,ring_count=8,radius=radius,location=point)
    ob=bpy.context.object;ob.name=name
    color(ob,label)


def arrow(name,origin,direction,size,label):
    direction=direction.normalized();end=origin+size*direction
    center=(origin+end)/2
    rotation=direction.to_track_quat('Z','Y').to_euler()
    bpy.ops.mesh.primitive_cylinder_add(vertices=12,radius=size*.045,depth=size,location=center,rotation=rotation)
    ob=bpy.context.object;ob.name=name+' shaft';color(ob,label)
    bpy.ops.mesh.primitive_cone_add(vertices=12,radius1=size*.15,radius2=0,depth=size*.27,
                                    location=end,rotation=rotation)
    ob=bpy.context.object;ob.name=name+' tip';color(ob,label)


def show_trajectory(meta):
    rows=meta['rows']
    positions=[Vector([r['c2w_blender'][i][3] for i in range(3)]) for r in rows]
    span=max((max(p[i] for p in positions)-min(p[i] for p in positions)) for i in range(3))
    size=max(.12*span,.025)
    curve=bpy.data.curves.new('Predicted camera path','CURVE');curve.dimensions='3D';curve.bevel_depth=size*.045
    spline=curve.splines.new('POLY');spline.points.add(len(positions)-1)
    for v,point in zip(spline.points,positions,strict=True):
        v.co=(*point,1)
    ob=bpy.data.objects.new('Predicted camera path',curve);bpy.context.collection.objects.link(ob)
    color(ob,'path');bpy.context.view_layer.objects.active=ob;ob.select_set(True)
    bpy.ops.object.convert(target='MESH');ob=bpy.context.object;ob.select_set(False)
    sphere(f"START {rows[0]['timestamp_s']:.2f}s",positions[0],size*.19,'start')
    sphere(f"END {rows[-1]['timestamp_s']:.2f}s",positions[-1],size*.19,'end')
    indices=set(round(i*(len(rows)-1)/5) for i in range(6))
    query=meta.get('highlight_time_s')
    highlight=min(range(len(rows)),key=lambda i:abs(rows[i]['timestamp_s']-query)) if query is not None else None
    if highlight is not None:indices.add(highlight)
    for i in sorted(indices):
        r=rows[i];direction=Vector([-r['c2w_blender'][j][2] for j in range(3)])
        if i==highlight:sphere(f"QUERY {r['timestamp_s']:.2f}s",positions[i],size*.22,'query')
        arrow(f"Camera forward {r['timestamp_s']:.2f}s",positions[i],direction,
              size*(1.35 if i==highlight else .78),'query' if i==highlight else 'direction')


def main():
    catalog=session.read(HERE/'server-inputs/manifest.json')['questions']
    requested=set(sys.argv[sys.argv.index('--')+1:]) if '--' in sys.argv else set()
    output=HERE/'rendered';output.mkdir(exist_ok=True)
    render=session.render
    bpy.context.preferences.filepaths.save_version=0
    for question in catalog:
        case=question['case'];local=output/case;local.mkdir(parents=True,exist_ok=True)
        if requested and case not in requested:
            continue
        fragments=[]
        for group in question['groups']:
            if group['status']!='ready':continue
            gid=group['id'];source=HERE/'server-inputs'/case/gid;target=local/'groups'/gid
            target.mkdir(parents=True,exist_ok=True)
            spec=session.read(source/'scene-spec.json');session.save(target/'scene-spec.json',spec)
            bpy.ops.wm.read_factory_settings(use_empty=True)
            importlib.reload(session.preset_runtime)
            session.asset_runtime.compile_scene(spec,str(target/'scene.blend'))
            render(target);fragments.append((gid,target))
        traj=question['trajectory']
        if not fragments and not traj:
            print('MISSING_CASE',case,flush=True);continue
        if traj:
            def with_trajectory(path):
                show_trajectory(traj)
                render(path)
            session.render=with_trajectory
        else:
            session.render=render
        session.merge(fragments,local,traj['rows'] if traj else question['references'])
        bpy.context.scene['backend']='Codex child requested as gpt-6-astra; cached LingBot pose overlays'
        bpy.ops.wm.save_as_mainfile(filepath=str(local/'scene.blend'))
        build=session.read(local/'build.json')
        build.update(backend='Codex child requested as gpt-6-astra',trajectory_visible=bool(traj),
                     target_groups=len(fragments),full_scene_reconstruction=False)
        session.save(local/'build.json',build)
        print('CASE_DONE',case,len(fragments),'trajectory',bool(traj),flush=True)
    session.save(output/'manifest.json',dict(questions=catalog))


if __name__=='__main__':
    main()
