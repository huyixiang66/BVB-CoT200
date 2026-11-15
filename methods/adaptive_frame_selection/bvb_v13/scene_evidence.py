"""Compact model-facing evidence; original packets remain the audit authority."""
import copy
import json

VERSION = 'scene-evidence-tables-v1'


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def table(rows):
    # Keep unknown evidence fields as well as known fields. Missing cells are null.
    fields = list(dict.fromkeys(k for row in rows for k in row))
    return dict(fields=fields, rows=[[row.get(k) for k in fields] for row in rows])


def pack(packet, anchors=None):
    """Return header, one metadata row per image, anchors, and local provenance."""
    d = copy.deepcopy(packet)
    ids = [g['group_id'] for g in d['groups']]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate evidence group IDs')
    aliases = {key: 'g' + str(i) for i, key in enumerate(ids)}

    def group_id(value):
        if value not in aliases:
            raise ValueError('Unknown evidence group: ' + str(value))
        return aliases[value]

    groups = []
    for g in d['groups']:
        old = g['group_id']
        # Only omit timestamps if exactly reconstructible from frame membership.
        times = [f['timestamp_s'] for f in d['frames'] if old in f.get('groups', [])]
        if g.get('timestamps_s') == times:
            del g['timestamps_s']
        g['group_id'] = group_id(old)
        groups.append(g)
    observations = []
    for f in d['frames']:
        for o in f.get('observations', []):
            o.pop('peak_id', None)  # Detector provenance, not a geometric constraint.
            if o.get('box_xyxy') is not None:
                o['box_xyxy'] = [round(x, 1) for x in o['box_xyxy']]
            # Confidence, camera and world coordinates are not further rounded.
            observations.append(o)
    obs_fields = table(observations)['fields']
    rows = [[i, f.get('timestamp_s'), [group_id(g) for g in f.get('groups', [])],
             f.get('compact_camera'),
             [[o.get(k) for k in obs_fields] for o in f.get('observations', [])]]
            for i, f in enumerate(d['frames'])]
    public = {k: d['question'][k] for k in ['question', 'question_type', 'roles', 'options']
              if d['question'].get(k) is not None}
    header = dict(format=VERSION, question=public, categories=d['question'].get('objects', []),
                  instance_limits=d.get('instance_limits', {}), groups=table(groups),
                  frame_fields=['i', 'timestamp_s', 'groups', 'compact_camera', 'observations'],
                  observation_fields=obs_fields,
                  legend='Table rows follow fields. Each frame row precedes its image; i is its zero-based image index. '
                         'gN names one evidence group, NOT a physical instance. Missing group timestamps follow frame membership. '
                         'Observations follow observation_fields; boxes are image-pixel xyxy. Null means unavailable.')
    anchor_table = None
    if anchors is not None:
        # Match the existing transport's four-decimal anchor precision.
        anchors = json.loads(json.dumps(anchors), parse_float=lambda x: round(float(x), 4))
        notes = []
        for a in anchors:
            if 'group_id' in a:
                a['group_id'] = group_id(a['group_id'])
            if 'note' in a:
                note = a.pop('note')
                if note not in notes:
                    notes.append(note)
                a['note_index'] = notes.index(note)
        anchor_table = dict(soft_surface_anchors=table(anchors), notes=notes)
    return dict(header=header, frames=rows, anchors=anchor_table,
                provenance=dict(format=VERSION, group_aliases=aliases,
                                omitted=['question.id', 'question.scene_name', 'observation.peak_id'],
                                box_decimals=1, anchor_decimals=4))
