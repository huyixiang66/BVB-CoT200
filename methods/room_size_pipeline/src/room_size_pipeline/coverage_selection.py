"""Cached sparse coverage, position groups, reference cues and transition witnesses.

Ported from the 2026-11-12 low-score pilot. No detector/model calls or case IDs.
Coverage is a sparse geometric proxy, not proof of complete rooms or metric scale.
"""
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.cluster.hierarchy import linkage, fcluster
from .selection import sha256


def geometry(poses, samples, M):
    poses=Path(poses)
    raw=np.load(poses/'camera-poses.npz'); cameras=raw['c2w']; K=raw['intrinsics']
    size=json.loads((poses/'geometry-info.json').read_text())['image_size_wh']; width,height=size
    cloud=np.load(poses/'sparse-geometry-corrected.npz')
    points=cloud['points'].astype(float); frame_ids=cloud['frame_ids']; N=len(samples)
    R=M[0,:3,:3]@np.diag([1,-1,-1])@cameras[0,:3,:3].T
    t=M[0,:3,3]-R@cameras[0,:3,3]
    finite=np.isfinite(points).all(axis=1)
    zpoints=points[finite]@R.T+t
    span=np.quantile(zpoints,.98,axis=0)-np.quantile(zpoints,.02,axis=0)
    voxel=max(float(span.max()/30),1e-4)
    keys,inv=np.unique(np.floor(zpoints/voxel).astype(int),axis=0,return_inverse=True)
    sums=np.zeros((len(keys),3));np.add.at(sums,inv,zpoints)
    centers=sums/np.bincount(inv)[:,None]
    support=np.zeros(len(keys),int)
    for i in range(N):support[np.unique(inv[frame_ids[finite]==i])]+=1
    centers=centers[support>=2]
    native=(centers-t)@R
    visible=np.zeros((N,len(centers)),bool)
    for i in range(N):
        cp=(points[frame_ids==i]-cameras[i,:3,3])@cameras[i,:3,:3]
        cp=cp[np.isfinite(cp).all(axis=1)&(cp[:,2]>1e-5)]
        if not len(cp):continue
        uv=cp@K[i].T;uv=uv[:,:2]/uv[:,2,None]
        projected=(native-cameras[i,:3,3])@cameras[i,:3,:3]
        xy=projected@K[i].T;xy=xy[:,:2]/np.maximum(xy[:,2,None],1e-8)
        dist,nearest=cKDTree(uv).query(xy)
        visible[i]=(projected[:,2]>0)&(xy[:,0]>=0)&(xy[:,0]<width)&(xy[:,1]>=0)&(xy[:,1]<height)&(dist<12)&(np.abs(projected[:,2]-cp[nearest,2])<np.maximum(2*voxel,.08*cp[nearest,2]))
    reliable=visible.sum(axis=0)>=2
    centers=centers[reliable];visible=visible[:,reliable]
    if len(centers)<20:raise ValueError('Too little repeat-observed geometry')
    _,nn=cKDTree(centers).query(centers,k=min(12,len(centers)))
    neighbors=centers[nn];delta=neighbors-neighbors.mean(axis=1,keepdims=True)
    eig,vec=np.linalg.eigh(np.einsum('nki,nkj->nij',delta,delta))
    normals=vec[:,:,0]; planar=eig[:,0]/np.maximum(eig.sum(axis=1),1e-12)<.08
    horizontal=planar&(np.abs(normals[:,2])>.8)
    floor=horizontal&(centers[:,2]<np.quantile(centers[:,2],.25))
    ceiling=horizontal&(centers[:,2]>np.quantile(centers[:,2],.8))
    wall=planar&(np.abs(normals[:,2])<.3)
    weights=np.ones(len(centers));weights[floor|wall]=2.;weights[ceiling]=.15
    pitches=np.degrees(np.arcsin(np.clip(-M[:,2,2],-1,1)))
    allowed=(pitches<=25)&(pitches>=-65)
    # Report coverage over ALL repeat-observed cells, including unseen filtered cells.
    reachable=visible[allowed].any(axis=0)
    total=float(weights.sum()); reachable_weight=float(weights[reachable].sum())
    if reachable_weight<=0:raise ValueError('No geometry after pitch filtering')
    return visible,weights,allowed,pitches,voxel


def choose_coverage_frames(config, samples, M):
    if not config.get('detections'):raise ValueError('coverage_dino requires cached detections (640x480 box coordinates)')
    N=len(samples)
    if not N:raise ValueError('Empty frame cache')
    M=np.asarray(M,dtype=float);frames=Path(config['frames_root'])
    if M.shape!=(N,4,4) or not np.isfinite(M).all():raise ValueError('Invalid camera binding')
    visible,weights,allowed,pitches,voxel=geometry(config['poses'],samples,M)
    positions=np.stack([np.median(M[max(0,i-2):min(N,i+3),:3,3],axis=0) for i in range(N)])
    span=float(np.linalg.norm(np.quantile(positions,.95,axis=0)-np.quantile(positions,.05,axis=0)))
    # Complete linkage limits the diameter of each position group. These are NOT room labels.
    groups=(fcluster(linkage(positions,method='complete'),max(span*.60,1e-6),criterion='distance')
            if N>1 else np.ones(1,dtype=int))
    group_ids=sorted(set(groups));valid_groups=[x for x in group_ids if np.count_nonzero((groups==x)&allowed)>=2]
    sharp=[]
    for s in samples:
     if Path(s['filename']).name!=s['filename']:raise ValueError('Sampling filename must be a basename')
     p=frames/s['filename']
     if sha256(p)!=s['sha256']:raise ValueError('Image/pose hash mismatch')
     im=cv2.imread(str(p))
     if im is None:raise ValueError('Unreadable cached image: '+str(p))
     gray=cv2.cvtColor(cv2.resize(im,(160,120)),cv2.COLOR_BGR2GRAY)
     sharp.append(float(cv2.Laplacian(gray,cv2.CV_64F).var()))
    sharp=np.minimum(np.array(sharp)/max(np.quantile(sharp,.9),1e-8),1.)
    byfid={s['source_frame_index']:i for i,s in enumerate(samples)}
    anchors=[{} for _ in samples];accepted=[];rejected=[]
    aliases={'washing machine':'washer','washbasin':'sink'}
    for line in Path(config['detections']).read_text(encoding='utf8').splitlines():
     row=json.loads(line);i=byfid.get(row['source_frame_index'])
     if i is None:continue
     for d in row.get('detections',[]):
      label=aliases.get(d.get('object'),d.get('object'))
      if label not in ['washer','toilet','sink','bathtub','bed','table','desk','door','sofa','refrigerator']:continue
      x1,y1,x2,y2=d['box_xyxy'];area=(x2-x1)*(y2-y1)/(640*480)
      edge=sum(v<.015 for v in [x1/640,y1/480,(640-x2)/640,(480-y2)/480])
      score=float(d['score'])
      reason=None
      if score<.45:reason='low_confidence'
      elif not .04<=area<=.65:reason='too_small_or_frame_filling'
      elif edge>=2:reason='multiple_clipped_edges'
      # A high score on a frame-filling wall is not reliable reference evidence.
      if reason:
       rejected.append({'time':samples[i]['timestamp_s'],'label':label,'reason':reason});continue
      value=score*(.65 if edge else 1.)*(.5+.5*sharp[i])
      key=(int(groups[i]),label)
      anchors[i][key]=max(value,anchors[i].get(key,0))
      accepted.append({'time':samples[i]['timestamp_s'],'label':label,'group':int(groups[i]),'value':value,'box':d['box_xyxy'],'clipped_edges':edge})
    selected=[];reasons={}
    def add(i,reason):
     if i not in selected:selected.append(i);reasons[i]=[]
     reasons[i].append(reason)
    def pick(candidates,values):
     best=max(values[i] for i in candidates)
     near=[i for i in candidates if values[i]>=.95*best]
     return max(near,key=lambda i:sharp[i])
    # Protect underrepresented position groups before global coverage accumulation.
    for group in valid_groups:
     ii=np.flatnonzero((groups==group)&allowed)
     local=visible[ii].any(axis=0);values=visible[:,local]@weights[local]
     add(pick(ii,values),'位置区域'+str(group)+'的覆盖代表')
    # At most one reference witness per position group/category, avoiding global
    # category deduplication that could erase a second room's same-category object.
    keys=set(k for i in np.flatnonzero(allowed) for k in anchors[i])
    for key in sorted(keys):
     ii=[i for i in np.flatnonzero(allowed) if key in anchors[i]]
     best=max(ii,key=lambda i:anchors[i][key]);bestv=anchors[best][key]
     if any(anchors[i].get(key,0)>=bestv*.9 for i in selected):continue
     add(best,'检测参照候选：'+key[1]+'（位置区域'+str(key[0])+'）')
    # Retain both sides of large chronological spatial transitions. This is a
    # transition cue, not a claim that a doorway was semantically detected.
    transitions=[]
    for i in range(1,N):
     if groups[i]==groups[i-1]:continue
     if i<3 or i+3>N or len(set(groups[i-3:i]))!=1 or len(set(groups[i:i+3]))!=1:continue
     left=np.flatnonzero((groups==groups[i-1])&allowed&(np.arange(N)<i)&(np.arange(N)>=i-4))
     right=np.flatnonzero((groups==groups[i])&allowed&(np.arange(N)>=i)&(np.arange(N)<i+4))
     if not len(left) or not len(right):continue
     # Only fill a transition side if no selected nearby view already represents it.
     for side in [left,right]:
      if any(j in selected for j in side):continue
      values=visible@weights
      j=pick(side,values);add(j,'位置区域转换的连接证据')
     transitions.append(float(samples[i]['timestamp_s']))
    covered=visible[selected].any(axis=0);total=float(weights.sum());reachable=visible[allowed].any(axis=0)
    while weights[covered].sum()/weights[reachable].sum()<.90:
     gains=visible[:,~covered]@weights[~covered];own=visible@weights
     good=allowed&(gains/np.maximum(own,1e-8)>=.15);good[selected]=False
     if not good.any() or gains[good].max()/weights[reachable].sum()<.025:break
     i=pick(np.flatnonzero(good),gains);add(i,'补充未覆盖三维结构');covered|=visible[i]
    # Remove geometric redundancies, but retain spatial/anchor evidence constraints.
    removed=[]
    for i in sorted(selected,key=lambda i:sharp[i]):
     others=[j for j in selected if j!=i]
     if not others:continue
     if groups[i] in valid_groups and not any(groups[j]==groups[i] for j in others):continue
     if any(anchors[i].get(k,0)>0 and not any(anchors[j].get(k,0)>=anchors[i][k]*.9 for j in others) for k in anchors[i]):continue
     if any('连接' in r for r in reasons[i]):continue
     loss=float(weights[visible[i]&~visible[others].any(axis=0)].sum()/total)
     if loss<.015:selected.remove(i);removed.append({'time':samples[i]['timestamp_s'],'lost_coverage':loss})
    selected=sorted(map(int,selected))
    if not selected:raise ValueError('No usable coverage/reference frames')
    report={'method':'coverage_dino','candidate_count':N,'voxel_count':len(weights),'voxel_size_native':voxel,
     'selected_times':[samples[i]['timestamp_s'] for i in selected],
     'position_group_count':len(group_ids),'protected_position_groups':list(map(int,valid_groups)),
     'group_assignment':list(map(int,groups)),'position_diameter_threshold':span*.60,
     'groups_are_not_room_labels':True,'accepted_detection_cues':accepted,'rejected_detection_cues':rejected,
     'selected_reasons':{str(samples[i]['timestamp_s']):reasons[i] for i in selected},
     'transitions_s':transitions,'pruned':removed,
     'weighted_coverage':float(weights[visible[selected].any(axis=0)].sum()/total),
     'pitch_filter_degrees':[-65,25],
     'filtered':[{'timestamp_s':samples[i]['timestamp_s'],'pitch_degrees':float(pitches[i])} for i in range(N) if not allowed[i]],
     'vertical_policy':'upstream camera-derived up, not measured gravity',
     'limitations':['Position groups are not semantic rooms.','DINO labels may be false positives.',
     'Sparse coverage is not true room completeness.'],'model_calls':0}
    return report['selected_times'],report
