import json,hashlib,sys,time
from pathlib import Path
from types import SimpleNamespace
INSTALL=None
def read(p):return json.loads(p.read_text())
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def poses(folder,limit=None):
    import numpy as np
    import torch
    sys.path.insert(0,str(INSTALL/'lingbot-map'))
    from demo import load_images,load_model,postprocess
    out=folder/'poses';out.mkdir(exist_ok=True)
    candidates=read(folder/'cache/candidates.json')
    if limit:candidates=candidates[:limit]
    frames=out/'frames';frames.mkdir(exist_ok=True)
    for row in candidates:
        dst=frames/f'{row["index"]:06d}.jpg'
        if not dst.exists():dst.symlink_to((folder/'cache'/row['frame']).resolve())
    samples=[dict(sample_index=r['index'],source_frame_index=r['source_frame_index'],timestamp_s=r['timestamp_s'],filename=f'{r["index"]:06d}.jpg',sha256=r['frame_sha256']) for r in candidates]
    save(out/'sampling.json',dict(frames=samples,video_sha256=read(folder/'cache/config.json')['video_sha256'],sampling_fps=read(folder/'cache/config.json')['candidate_fps']))
    images,paths,_=load_images(image_folder=str(frames),image_size=518,patch_size=14)
    assert len(paths)==len(samples)
    for path,sample in zip(paths,samples):assert Path(path).name==sample['filename'] and sha(Path(path))==sample['sha256']
    cfg=SimpleNamespace(mode='streaming',model_path=str(INSTALL/'models/lingbot-map.pt'),image_size=518,
                        patch_size=14,enable_3d_rope=True,max_frame_num=max(4096,len(samples)+1),kv_cache_sliding_window=24,
                        num_scale_frames=4,use_sdpa=True,camera_num_iterations=4)
    torch.set_num_threads(8);device=torch.device('cuda:0');start=time.time()
    model=load_model(cfg,device);model.aggregator=model.aggregator.to(dtype=torch.bfloat16)
    images=images.cpu()
    with torch.inference_mode(),torch.amp.autocast('cuda',dtype=torch.bfloat16):
        pred=model.inference_streaming(images,num_scale_frames=4,keyframe_interval=1,output_device=torch.device('cpu'))
    pred,rgb=postprocess(pred,pred['images'])
    ext=pred['extrinsic'].float().numpy();k=pred['intrinsic'].float().numpy();depth=pred['depth'].float().numpy()
    if depth.ndim==4:depth=depth[...,0]
    assert ext.shape==(len(samples),3,4) and depth.ndim==3
    w2c=np.broadcast_to(np.eye(4),(len(ext),4,4)).copy();w2c[:,:3]=ext;c2w=np.linalg.inv(w2c)
    assert np.isfinite(c2w).all() and np.isfinite(k).all()
    assert np.max(np.abs(np.linalg.det(c2w[:,:3,:3])-1))<.001
    assert np.mean(np.isfinite(depth)&(depth>0))>.99
    np.savez_compressed(out/'camera-poses.npz',c2w=c2w,intrinsics=k)
    h,w=depth.shape[1:];x,y=np.meshgrid(np.arange(8,w,16),np.arange(8,h,16));pixels=np.stack([x.ravel(),y.ravel(),np.ones(x.size)],1)
    points=[];ids=[]
    for i,(zmap,pose,intrinsics) in enumerate(zip(depth,c2w,k)):
        z=zmap[y,x].ravel();valid=np.isfinite(z)&(z>1e-5)
        xyz=(pixels[valid]@np.linalg.inv(intrinsics).T)*z[valid,None]
        points.append(xyz@pose[:3,:3].T+pose[:3,3]);ids.append(np.full(len(xyz),i,dtype=np.int32))
    np.savez_compressed(out/'sparse-geometry-corrected.npz',points=np.concatenate(points).astype('float32'),frame_ids=np.concatenate(ids))
    save(out/'geometry-info.json',dict(image_size_wh=[w,h],pose_type='OpenCV camera-to-world',scene_units='native arbitrary scale'))
    save(out/'validation.json',dict(frames=len(samples),sampling_fps=read(folder/'cache/config.json')['candidate_fps'],positive_depth=True,rotation_valid=True,
          video_sha256=read(folder/'cache/config.json')['video_sha256'],elapsed_s=time.time()-start,
          gpu_peak_bytes=torch.cuda.max_memory_allocated(),metric_scale=False))
    print('POSES_READY',len(samples),flush=True)
