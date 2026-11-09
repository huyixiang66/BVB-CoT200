"""Blender: merge pose-conditioned fragments in their shared LingBot world frame."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Euler, Vector


def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def error(a,b):return max(abs(a[i][j]-b[i][j]) for i in range(4) for j in range(4))
def bbox(objects):
    pts=[o.matrix_world@Vector(v) for o in objects for v in o.bound_box]
    return {'min':[min(v[i] for v in pts) for i in range(3)],'max':[max(v[i] for v in pts) for i in range(3)]}


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args(sys.argv[sys.argv.index('--')+1:]);src=args.source.resolve();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    manifest=read(src/'manifest.json');shared=None;reference_rows=[];parts=[];sources={}
    bpy.ops.wm.read_factory_settings(use_empty=True);scene=bpy.context.scene;scene.unit_settings.system='METRIC';scene.unit_settings.scale_length=1
    scene['coordinate_system']='Shared approximate LingBot-to-Blender Sim3 world; not metric ground truth'
    for g in manifest['groups']:
        folder=src/g['folder'];assets=folder/'assets';source=assets/'scene.blend';sources[str(source)]=sha(source)
        reference=read(folder/'pose-reference.json')
        if shared is None:shared=reference['source_hashes']
        assert reference['source_hashes']==shared,'Different coordinate provenance'
        experiment=read(folder/'experiment.json');assert experiment['camera_pose_reference'] and experiment['pose_input_format']=='compact'
        spec=read(assets/'scene-spec.json');audit=read(assets/'question-audit.json');inspection=read(assets/'scene-inspection.json')
        expected={r['name']:r for r in inspection['objects']}
        ids=[o['id'] for o in spec['objects']];target_ids={r['id'] for r in audit['targets']}
        target_labels={r['id']:r['label'] for r in audit['targets']}
        chunk=bpy.data.collections.new(g['group_id']);scene.collection.children.link(chunk)
        frame=bpy.data.objects.new(g['group_id']+' | shared-world frame',None);chunk.objects.link(frame);frame.matrix_world=Matrix.Identity(4)
        frame['world_from_fragment']=sum([list(r) for r in Matrix.Identity(4)],[])
        frame['why_identity']='Fragment geometry was already generated in the same supplied global camera coordinate system'
        with bpy.data.libraries.load(str(source),link=False) as (available,loaded):
            assert set(ids)<=set(available.collections)
            loaded.collections=ids
        meshes=[];asset_rows=[];max_bbox_error=0.
        for col in loaded.collections:
            original_id=col.name;chunk.children.link(col);bpy.context.view_layer.update()
            for o in list(col.objects):
                original_name=o.name
                if o.type=='MESH':
                    original_bbox=bbox([o]);expected_bbox=expected[original_name]['bbox']
                    max_bbox_error=max(max_bbox_error,max(abs(original_bbox[k][i]-expected_bbox[k][i]) for k in ['min','max'] for i in range(3)))
                    meshes.append(o)
                    o.color=manifest['colors'][target_labels[original_id]] if original_id in target_ids else (o.data.materials[0].diffuse_color if o.data.materials else (.85,.85,.82,1))
                if o.parent is None:
                    old=o.matrix_world.copy();o.parent=frame;o.matrix_world=old
                    asset_rows.append({'id':original_id,'position':list(old.translation)})
                o['source_fragment']=g['group_id'];o['source_object_name']=original_name;o.name=g['group_id']+'::'+original_name
            col.name=g['group_id']+'::'+original_id
        bpy.context.view_layer.update();assert max_bbox_error<1e-4,max_bbox_error
        for item in spec['objects']:
            root=next(o for o in chunk.all_objects if o.get('asset_id')==item['id'])
            assert (root.matrix_world.translation-Vector(item['p'])).length<1e-4
        parts.append({'group_id':g['group_id'],'target_category':g['object'],'world_from_fragment':[list(r) for r in Matrix.Identity(4)],
                      'source_blend':str(source),'source_sha256':sources[str(source)],'mesh_parts':len(meshes),'asset_count':len(ids),
                      'targets':audit['targets'],'missing_keywords':audit['missing_keywords'],'bounds':bbox(meshes),
                      'max_world_bbox_difference':max_bbox_error,'assets':asset_rows})
        for row in reference['rows']:reference_rows.append({**row,'group_id':g['group_id']})
    # Reference cameras retain the full precision poses, not the rounded VLM transport.
    refs=bpy.data.collections.new(f'POSE REFERENCES | {len(reference_rows)} selected views');scene.collection.children.link(refs)
    camera_records=[]
    for row in reference_rows:
        name=f'{row["group_id"]} | {row["timestamp_s"]:.3f}s | frame{row["source_frame_index"]}'
        data=bpy.data.cameras.new(name);data.type='PERSP';data.sensor_fit='HORIZONTAL';data.sensor_width=36
        k=row['intrinsics_640x480'];data.lens=k[0][0]*36/640;data.shift_x=-(k[0][2]-320)/640;data.shift_y=(k[1][2]-240)/640
        data.display_size=.18;camera=bpy.data.objects.new(name,data);refs.objects.link(camera)
        camera.matrix_world=Matrix(row['c2w_blender']);camera.color=manifest['colors'][next(g['object'] for g in manifest['groups'] if g['group_id']==row['group_id'])]
        camera['source_frame_index']=row['source_frame_index'];camera['timestamp_s']=row['timestamp_s'];camera['intrinsics_640x480']=sum(k,[])
        camera['reference_resolution']=[640,480];camera['pixel_aspect_y_for_reference_render']=k[1][1]/k[0][0]
        camera['note']='Full matrix and intrinsics are authoritative. Overview render resolution is separate.'
        assert error(camera.matrix_world,Matrix(row['c2w_blender']))<1e-5
        camera_records.append({'name':name,'source_frame_index':row['source_frame_index'],'timestamp_s':row['timestamp_s'],'group_id':row['group_id'],'c2w_blender':row['c2w_blender'],'intrinsics_640x480':k})
    # Camera centers are also available as a separate, optional diagnostic collection.
    dots=bpy.data.collections.new('POSE OVERLAY | optional render markers');scene.collection.children.link(dots)
    for row in reference_rows:
        bpy.ops.mesh.primitive_uv_sphere_add(segments=12,ring_count=8,radius=.045,location=Matrix(row['c2w_blender']).translation)
        o=bpy.context.object;o.name=f'Pose dot | {row["group_id"]} | {row["timestamp_s"]:.3f}s'
        for col in list(o.users_collection):col.objects.unlink(o)
        dots.objects.link(o);o.color=manifest['colors'][next(g['object'] for g in manifest['groups'] if g['group_id']==row['group_id'])]
    dots.hide_render=True;dots.hide_viewport=True
    scene.render.engine='BLENDER_WORKBENCH';scene.render.resolution_x=1400;scene.render.resolution_y=1400;scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB';scene.view_settings.view_transform='Standard'
    scene.world=bpy.data.worlds.new('Merged world');scene.world.color=(.8,.8,.8)
    sh=scene.display.shading;sh.light='STUDIO';sh.color_type='OBJECT';sh.show_shadows=True;sh.show_cavity=True;sh.cavity_type='BOTH';sh.background_type='WORLD'
    data=bpy.data.cameras.new('Merged overview');data.type='ORTHO';data.clip_end=1000
    camera=bpy.data.objects.new('Merged overview',data);scene.collection.objects.link(camera);scene.camera=camera
    scene.frame_start=scene.frame_end=1
    geometry=[o for o in scene.objects if o.type=='MESH' and o.get('source_fragment')]
    points=[o.matrix_world@Vector(v) for o in geometry for v in o.bound_box];views={}
    for view,angles in [('top',(0,0,0)),('oblique',(20,-20,0))]:
        rotation=Euler([math.radians(v) for v in angles],'XYZ');basis=rotation.to_matrix();projected=[basis.transposed()@p for p in points]
        lo=Vector([min(p[i] for p in projected) for i in range(3)]);hi=Vector([max(p[i] for p in projected) for i in range(3)]);center=(lo+hi)/2
        data.ortho_scale=max(hi.x-lo.x,hi.y-lo.y)*1.15;camera.rotation_euler=rotation;camera.location=basis@Vector((center.x,center.y,hi.z+30))
        scene.render.filepath=str(out/f'merged-{view}.png');bpy.ops.render.render(write_still=True)
        views[view]={'rotation_degrees':angles,'location':list(camera.location),'ortho_scale':data.ortho_scale}
    # The saved blend opens with the same oblique overview and colored targets.
    for area in bpy.context.screen.areas if bpy.context.screen else []:
        if area.type=='VIEW_3D':area.spaces.active.shading.color_type='OBJECT'
    bpy.ops.wm.save_as_mainfile(filepath=str(out/'merged-scene.blend'))
    dots.hide_render=False;dots.hide_viewport=False;bpy.context.view_layer.update()
    scene.render.filepath=str(out/'merged-oblique-with-pose-centers.png');bpy.ops.render.render(write_still=True)
    for path,digest in sources.items():assert sha(Path(path))==digest
    save(out/'merge.json',{'source':str(src),'method':'Preserve shared-world coordinates from pose-conditioned reconstruction. Identity transform per fragment; no second alignment, ICP, wall snapping or topology repair.',
         'shared_pose_source_hashes':shared,'parts':parts,'reference_cameras':camera_records,'views':views,
         'source_blends_unchanged':True,'geometric_accuracy_verified':False,'walls_deduplicated':False,
         'missing_target_groups':[p['group_id'] for p in parts if p['missing_keywords']],
         'merged_blend_sha256':sha(out/'merged-scene.blend')})
    print('MERGED',len(parts),'fragments',len(camera_records),'reference cameras',len(geometry),'meshes')

if __name__=='__main__':main()
