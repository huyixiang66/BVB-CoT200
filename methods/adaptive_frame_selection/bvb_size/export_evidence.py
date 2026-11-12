"""Prepare one cached V10 candidate-group evidence file without model calls."""
import argparse
import json
import shutil
import sys
from pathlib import Path
from .handoff import digest, public_question, write


def export(delivery_path, poses, detector_root, upstream_root, output):
    delivery_path, poses, detector_root, output = map(Path, (delivery_path, poses, detector_root, output))
    sys.path.insert(0,str(Path(upstream_root).resolve()))
    from bvb_v6.reconstruction import shared_pose
    from bvb_v6.runtime.compact_pose_input import build_transport
    raw=json.loads(delivery_path.read_text(encoding='utf8'))
    samples=json.loads((poses/'sampling.json').read_text(encoding='utf8'))['frames']
    byid={s['source_frame_index']:s for s in samples}
    if len(byid)!=len(samples):raise ValueError('Duplicate source frame indices')
    matrices,intrinsics,sources=shared_pose(poses,None)
    if len(samples)!=len(matrices) or len(samples)!=len(intrinsics):raise ValueError('Pose/sample count mismatch')
    frames=[]
    for frame in raw['frames']:
        sample=byid[frame['source_frame_index']];idx=sample['sample_index']
        if sample['timestamp_s']!=frame['timestamp_s']:raise ValueError('Pose timestamp mismatch')
        relative=Path(frame['image'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe cache frame path')
        # Upstream caches use symlinks; content is bound by both recorded hashes.
        path=(detector_root/relative).resolve()
        if digest(path)!=frame['sha256'] or sample['sha256']!=frame['sha256']:
            raise ValueError('Frame/cache hash mismatch')
        row=dict(frame,c2w_blender=matrices[idx].tolist(),intrinsics_640x480=intrinsics[idx].tolist())
        row['compact_camera']=build_transport({'rows':[row]})['rows'][0]
        row['file']='frames/'+str(len(frames)).zfill(5)+path.suffix
        frames.append((path,row))
    if not frames:raise ValueError('No cached candidates')
    output.mkdir(parents=True,exist_ok=False);(output/'frames').mkdir()
    for path,row in frames:shutil.copy2(path,output/row['file'])
    result={'question':public_question(raw['question']),'frames':[row for _,row in frames],
            'shared_pose_sources':sources}
    write(output/'evidence.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('delivery','poses','detector-root','upstream-root','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();export(a.delivery,a.poses,a.detector_root,a.upstream_root,a.output)


if __name__=='__main__':main()
