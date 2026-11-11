"""Question-conditioned soft evidence; no GT or old reconstruction geometry is read."""
import json,sys
from pathlib import Path
import numpy as np
from bvb_v10.pipeline import METHOD,read,save,sha

def build(out,poses):
    sys.path.insert(0,str(METHOD))
    from bvb_v6.reconstruction import shared_pose
    from bvb_v6.object_clustering import roi_surface
    d=read(out/'input.json');raw=dict(np.load(poses/'camera-poses.npz',allow_pickle=False))
    cloud=dict(np.load(poses/'sparse-geometry-corrected.npz',allow_pickle=False))
    samples=read(poses/'sampling.json')['frames'];index={r['source_frame_index']:i for i,r in enumerate(samples)}
    cams,intr,source=shared_pose(poses,None)
    assert source==d['shared_pose_sources'],'Input must use the same shared LingBot coordinate policy'
    rotation=cams[0,:3,:3]@np.diag([1,-1,-1])@raw['c2w'][0,:3,:3].T
    translation=cams[0,:3,3]-rotation@raw['c2w'][0,:3,3]
    assert np.allclose(cams[:,:3,3],raw['c2w'][:,:3,3]@rotation.T+translation,atol=1e-6)
    records=[];seen=set();selected=[]
    for frame in d['frames']:
        i=index[frame['source_frame_index']]
        assert np.allclose(cams[i],frame['c2w_blender'],atol=1e-6)
        assert np.allclose(intr[i],frame['intrinsics_640x480'],atol=1e-5)
        assert sha(Path(frame['path']))==frame['sha256']
        selected.append(dict(sample_index=i,source_frame_index=frame['source_frame_index'],timestamp_s=frame['timestamp_s'],groups=frame['groups'],objects=sorted(set(o['object'] for o in frame['observations'])),position=cams[i,:3,3].tolist(),forward=(-cams[i,:3,2]).tolist(),up=cams[i,:3,1].tolist()))
        for obs in frame['observations']:
            box=obs.get('box_xyxy')
            if box is None:continue
            key=(frame['source_frame_index'],obs['object'],tuple(box))
            if key in seen:continue
            seen.add(key)
            surface=roi_surface(i,raw['c2w'][i],intr[i],cloud,[640,480],box)
            rec=dict(object=obs['object'],timestamp_s=frame['timestamp_s'],frame_index=frame['source_frame_index'],dino_score=obs.get('dino_score'),box=box,groups=frame['groups'],status='insufficient_surface_points')
            if surface:
                xyz=surface['samples']@rotation.T+translation
                rec.update(status='provisional_surface',center=np.median(xyz,axis=0).tolist(),surface_q10=np.quantile(xyz,.1,axis=0).tolist(),surface_q90=np.quantile(xyz,.9,axis=0).tolist(),points=surface['points'],quality_heuristic=surface['quality'],box_coverage=surface['box_coverage'],neighbor_pose_indices=surface['sources'])
            records.append(rec)
    anchors=[]
    for group in d['groups']:
        for obj in group['primary_objects']:
            observations=[r for r in records if r['object']==obj and group['group_id'] in r['groups']]
            valid=[r for r in observations if r['status']=='provisional_surface']
            anchor=dict(group_id=group['group_id'],object=obj,valid_observation_frames=len(valid),tested_box_frames=len(observations),confidence='unknown',center=None,note='Heuristic visible-surface evidence, not an object mask, true center, full extent, metric ruler, or hard constraint.')
            if valid:
                xyz=np.array([v['center'] for v in valid]);weights=np.array([max(v['quality_heuristic'],.01) for v in valid]);center=[]
                for j in range(3):
                    order=np.argsort(xyz[:,j]);center.append(float(xyz[order[np.searchsorted(np.cumsum(weights[order]),weights.sum()/2)],j]))
                spread=float(np.median(np.linalg.norm(xyz-np.array(center),axis=1)))
                q=float(np.median([v['quality_heuristic'] for v in valid]))
                confidence='medium' if len(valid)>=2 and q>=.3 else 'low'
                anchor.update(center=center,confidence=confidence,median_quality_heuristic=q,between_frame_spread=spread,surface_envelope_low=np.min([v['surface_q10'] for v in valid],axis=0).tolist(),surface_envelope_high=np.max([v['surface_q90'] for v in valid],axis=0).tolist(),timestamps_s=[v['timestamp_s'] for v in valid])
            anchors.append(anchor)
    reference=dict(version='bvb_v10_2',question_id=d['question']['id'],scene=d['question']['scene_name'],coordinate_policy=source['coordinate_policy'],units='native relative scale, not calibrated meters',policy='Soft hints only: Astra may reject or reposition suspected background/incorrect-instance anchors using original images. No post-hoc geometry registration or coordinate clamping.',trajectory=[dict(timestamp_s=r['timestamp_s'],position=cams[i,:3,3].tolist()) for i,r in enumerate(samples)],selected_frames=selected,anchors=anchors,sources={**source,'sparse-geometry-corrected.npz':sha(poses/'sparse-geometry-corrected.npz')})
    save(out/'spatial-reference.json',reference);save(out/'anchor-observations.json',records)
    save(out/'reference-validation.json',dict(exact_selected_pose_matches=len(selected),total_trajectory_samples=len(samples),source_pose_hash_matches=True,cloud_transform_matches_shared_camera_transform=True,ground_truth_used=False,original_bvb_geometry_used=False,position_constraints_enforced=False))
    plot(out,reference,d['colors'])
    return reference

def plot(out,ref,colors):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    trajectory=np.array([r['position'] for r in ref['trajectory']]);times=np.array([r['timestamp_s'] for r in ref['trajectory']]);scale=max(np.linalg.norm(np.ptp(trajectory,axis=0)),.1)
    # Pure geometry diagrams; no room or object shape is asserted by these figures.
    for name,is3d in [('reference-top',False),('reference-oblique',True)]:
        fig=plt.figure(figsize=(12,10));ax=fig.add_subplot(111,projection='3d' if is3d else None)
        ax.plot(*trajectory.T[:3 if is3d else 2],color='#9ca3af',alpha=.65,lw=1,label='Full camera path (not walls)')
        pts=ax.scatter(*trajectory.T[:3 if is3d else 2],c=times,s=5,cmap='viridis',alpha=.6)
        for obj,rgba in colors.items():
            if obj=='scene context':continue
            selected=[f for f in ref['selected_frames'] if obj in f['objects']]
            if selected:
                pos=np.array([f['position'] for f in selected]);ax.scatter(*pos.T[:3 if is3d else 2],color=rgba,s=28,label=obj+' camera frames')
                for f in selected:
                    p=np.array(f['position']);v=np.array(f['forward'])*scale*.025
                    if is3d:ax.quiver(*p,*v,color=rgba,arrow_length_ratio=.35)
                    else:ax.arrow(*p[:2],*v[:2],color=rgba,head_width=scale*.007,length_includes_head=True)
                # Label first and last selected camera of each category, avoiding 20+ overprints.
                for f in [selected[0],selected[-1]]:
                    p=f['position'];ax.text(*p[:3 if is3d else 2],f" {f['timestamp_s']:g}s",fontsize=8,color=rgba)
        for anchor in ref['anchors']:
            if anchor['center'] is None:continue
            c=anchor['center'];rgba=colors[anchor['object']];ax.scatter(*c[:3 if is3d else 2],color=rgba,marker='*',s=220,edgecolors='black')
            ax.text(*c[:3 if is3d else 2],f" {anchor['object']}\n {anchor['confidence']} confidence",fontsize=9)
            if not is3d:
                lo=anchor['surface_envelope_low'];hi=anchor['surface_envelope_high']
                ax.add_patch(Rectangle(lo[:2],hi[0]-lo[0],hi[1]-lo[1],facecolor=rgba,alpha=.12,edgecolor=rgba))
        ax.set_xlabel('X (relative units)');ax.set_ylabel('Y (relative units)')
        if is3d:
            ax.set_zlabel('Z (estimated up)');ax.view_init(elev=40,azim=-65);ax.set_box_aspect(np.maximum(np.ptp(trajectory,axis=0),scale*.25))
        else:ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2)
        fig.colorbar(pts,ax=ax,shrink=.5,pad=.1,label='Video time (seconds)')
        ax.legend(loc='upper left',fontsize=8);ax.set_title('V10_2 | Empty-space trajectory + selected cameras + SOFT surface anchors\nStars/envelopes are uncertain observations, NOT verified object geometry',fontsize=12)
        fig.savefig(out/(name+'.png'),dpi=150,bbox_inches='tight');plt.close(fig)
