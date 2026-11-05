"""Compile validated region packets using the pinned local asset runtime."""
from pathlib import Path
import shutil
import subprocess
import sys

from .reconstruction import read, write, sha, expand

ROOT = Path(__file__).parent


def run(args):
    source, out = args.source.resolve(), args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    manifest = read(source/'manifest.json')
    out.mkdir(parents=True)
    staged = out/'fragments';staged.mkdir()
    ready, reports = [], []
    def execute(name, command):
        with (out/(name+'.log')).open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError('Blender failed; see '+str(out/(name+'.log')))
    for group in manifest['groups']:
        folder = source/group['folder']
        from .semantic_review import reconstruction_gate
        reconstruction_gate(folder)
        reference = read(folder/'pose-reference.json')
        frames = read(folder/'input-frames.json')
        for f, p in zip(frames, reference['rows'], strict=True):
            assert f['source_frame_index']==p['source_frame_index'] and f['timestamp_s']==p['timestamp_s']
            assert sha(Path(f['path']))==f['sha256']
        assets = folder/'assets'
        transport = folder/'scene-transport.json'
        if transport.exists():
            spec, audit = expand(read(transport), read(folder/'question.json'), walls_only=True)
        elif (assets/'scene-spec.json').exists():
            spec, audit = read(assets/'scene-spec.json'), read(assets/'question-audit.json')
            # Revalidate the compiled schema; never execute model-authored Python.
            from schema import validate
            validate(spec)
        else:
            reports.append({'group_id':group['group_id'],'status':'missing_response'})
            continue
        if not spec['objects']:
            reports.append({'group_id':group['group_id'],'status':'empty','missing_keywords':audit['missing_keywords']})
            continue
        target = staged/group['folder'];dest=target/'assets';dest.mkdir(parents=True)
        write(dest/'scene-spec.json',spec);write(dest/'question-audit.json',audit)
        write(dest/'target-colors-input.json',manifest['colors'])
        shutil.copy2(folder/'pose-reference.json',target/'pose-reference.json')
        shutil.copy2(folder/'experiment.json',target/'experiment.json')
        command=[args.blender,'--background','--disable-autoexec','--python-exit-code','1']
        execute(group['group_id']+'-compile',command+['--python',str(ROOT/'runtime/assets/asset_runtime.py'),
                '--',str(dest/'scene-spec.json'),str(dest/'scene.blend')])
        execute(group['group_id']+'-preview',[args.blender,'--background','--disable-autoexec',str(dest/'scene.blend'),
                '--python-exit-code','1','--python',str(ROOT/'runtime/verify_question_scene.py'),
                '--python',str(ROOT/'runtime/render_fps_ablation.py'),
                '--python',str(ROOT/'runtime/render_question_highlight.py'),'--',str(dest)])
        ready.append(group)
        reports.append({'group_id':group['group_id'],'status':'compiled','missing_keywords':audit['missing_keywords'],
                        'target_count':len(audit['targets']),'scene_sha256':sha(dest/'scene.blend')})
    write(staged/'manifest.json',{**manifest,'groups':ready})
    if ready:
        execute('merge',[args.blender,'--background','--disable-autoexec','--python-exit-code','1',
                '--python',str(ROOT/'runtime/merge_cluster_pose_scenes.py'),'--','--source',str(staged),'--out',str(out/'merged')])
    write(out/'build.json',{'version':'bvb_v6','groups':reports,'compiled_groups':len(ready),
                          'semantic_missing_categories':manifest.get('semantic_missing_categories',[]),
                          'missing_responses':sum(r['status']=='missing_response' for r in reports),
                          'all_groups_compiled':len(ready)==len(manifest['groups']),
                          'shared_world_preserved':bool(ready),'geometric_accuracy_verified':False,
                          'cross_group_deduplication':False,'qa_run':False,'model_calls':0})
    print('Compiled',len(ready),'of',len(manifest['groups']),'groups:',out)
