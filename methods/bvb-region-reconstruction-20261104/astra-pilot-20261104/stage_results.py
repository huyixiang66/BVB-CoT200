"""Validate each Astra transport with the senior schema and stage server inputs."""
from collections import defaultdict
import json
from pathlib import Path
import sys
import tarfile

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from reconstruct_regions import expand, read, sha, write

PALETTE=([.85,.08,.055],[.06,.30,.90],[.95,.43,.04],[.05,.48,.15],[.52,.16,.84],[.95,.76,.08])


def main():
    manifests=sorted(HERE.glob('*/*/manifest.json'))
    output=HERE/'server-inputs'
    output.mkdir(exist_ok=True)
    questions=[]
    for path in manifests:
        manifest=read(path);case=path.parent.relative_to(HERE).as_posix()
        groups=[];references={};counts=defaultdict(int)
        for packet in manifest['groups']:
            source=path.parent/packet['folder']; target=output/case/packet['folder']
            pose=read(source/'pose-reference.json')
            frames=read(source/'input-frames.json')
            for frame,row in zip(frames,pose['rows'],strict=True):
                assert frame['source_frame_index']==row['source_frame_index'] and frame['timestamp_s']==row['timestamp_s']
                assert sha(Path(frame['path']))==frame['sha256']
                references[row['source_frame_index']]=row
            record=dict(id=packet['group_id'],frames=len(frames),objects=packet['objects'])
            transport=source/'scene-transport.json'
            if not transport.exists():
                record['status']='missing_response'
            elif read(transport)=={'structure':[],'targets':[],'context':[]}:
                record.update(status='empty',mesh_assets=0,target_labels=[],transport_sha256=sha(transport))
            else:
                try:
                    spec,audit=expand(read(transport),read(source/'question.json'),walls_only=True)
                except (ValueError,KeyError,TypeError,AssertionError) as exc:
                    record.update(status='invalid_response',error=str(exc))
                else:
                    target.mkdir(parents=True,exist_ok=True)
                    write(source/'scene-spec.json',spec)
                    write(source/'scene-audit.json',audit)
                    if case=='vsi/2607':
                        for ob in spec['objects']:
                            if ob['id'].startswith('target_'):
                                ob['color']=PALETTE[counts['door']%len(PALETTE)];counts['door']+=1
                    write(target/'scene-spec.json',spec)
                    record.update(status='ready' if spec['objects'] else 'empty',
                                  mesh_assets=len(spec['objects']),target_labels=[o['label'] for o in audit['targets']],
                                  transport_sha256=sha(transport))
            groups.append(record)
        trajectory=path.parent/'trajectory-reference.json'
        questions.append(dict(case=case,question=manifest['question'],groups=groups,
            references=sorted(references.values(),key=lambda r:r['timestamp_s']),
            trajectory=read(trajectory) if trajectory.exists() else None))
    write(output/'manifest.json',dict(questions=questions))
    with tarfile.open(HERE/'server-inputs.tgz','w:gz') as archive:
        archive.add(output,arcname='server-inputs')
        archive.add(HERE/'render_pilot.py',arcname='render_pilot.py')
    for q in questions:
        print(q['case'],[(g['id'],g['status'],g.get('target_labels',[])) for g in q['groups']])


if __name__=='__main__':
    main()
