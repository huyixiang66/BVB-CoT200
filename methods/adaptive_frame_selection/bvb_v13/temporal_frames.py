"""Greedy time-only deduplication of model-facing reconstruction frames."""
import copy
import math

POLICY='earliest-frame-gap-gt-0.5s-v1'
INTERVAL_S=0.5


def select(packet):
    frames=packet['frames']
    for f in frames:
        t=f.get('timestamp_s')
        if type(t) not in (int,float) or not math.isfinite(t) or t<0:
            raise ValueError('Temporal selection requires finite nonnegative timestamps')
    order=sorted(range(len(frames)),key=lambda i:(frames[i]['timestamp_s'],i))
    kept=[];removed=[]
    for i in order:
        if kept and frames[i]['timestamp_s']-frames[kept[-1]]['timestamp_s']<=INTERVAL_S+1e-9:
            removed.append(dict(source_packet_index=i,kept_source_packet_index=kept[-1],timestamp_s=frames[i]['timestamp_s']))
        else:kept.append(i)
    model=copy.deepcopy(packet)
    model['frames']=[model['frames'][i] for i in kept]
    groups=[]
    for g in model['groups']:
        gid=g['group_id']
        before=sum(gid in f.get('groups',[]) for f in frames)
        times=[f['timestamp_s'] for f in model['frames'] if gid in f.get('groups',[])]
        # Do not transfer boxes, identity evidence or group memberships to a nearby image.
        if 'timestamps_s' in g:g['timestamps_s']=times
        if before and not times:g['supplied_image_status']='no original image retained after temporal dedup; soft evidence only'
        groups.append(dict(group_id=gid,before=before,after=len(times)))
    receipt=dict(policy=POLICY,interval_s=INTERVAL_S,before=len(frames),after=len(kept),removed=len(removed),
        kept_source_packet_indices=kept,removed_frames=removed,groups=groups,
        rule='Sort by timestamp; keep earliest; drop frames <=0.5s after the LAST KEPT frame. Equal times retain input order. No transitive chaining or group exceptions.',
        scope='reconstruction original frames only; semantic review, reference diagrams and raw input.json unchanged')
    return model,receipt
