"""Validate the data boundary between inference and deterministic geometry."""
import math


def require(ok, message):
    if not ok:
        raise ValueError(message)


def vector(value, length, positive=False):
    require(isinstance(value, list) and len(value) == length, 'Invalid vector')
    require(all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
                and (not positive or x > 0) for x in value), 'Invalid numeric values')


def validate_regions(spec):
    require(isinstance(spec.get('regions'), list) and spec['regions'], 'No regions')
    ids = [r['id'] for r in spec['regions']]
    require(len(set(ids)) == len(ids), 'Duplicate region id')
    for r in spec['regions']:
        vector(r['size_m'], 2, True)
        require(r.get('space_id') and r.get('evidence_times_s'), 'Region needs space identity/evidence')
    refs = spec.get('references', [])
    require(refs and len({x['id'] for x in refs}) == len(refs), 'Missing/duplicate references')
    for ref in refs:
        require(ref['region_id'] in ids, 'Reference region unknown')
        vector(ref['size_m'], 3, True)
        require(ref.get('label') and ref.get('source') and ref.get('evidence_times_s'), 'Missing reference evidence')
        require(ref.get('qa_axes') and set(ref['qa_axes']) <= {0, 1, 2}, 'Invalid QA dimension axes')
        ranges = ref['size_ranges_m']; require(len(ranges) == 3, 'Missing reference ranges')
        for value, bounds in zip(ref['size_m'], ranges):
            vector(bounds, 2, True)
            require(bounds[0] <= value <= bounds[1], 'Adopted reference dimension outside estimate range')
    require(all(any(ref['region_id'] == r['id'] for ref in refs) or
                any(x['space_id'] == r['space_id'] and any(ref['region_id'] == x['id'] for ref in refs)
                    for x in spec['regions']) for r in spec['regions']),
            'Each physical space needs a retained scale reference')
    for edge in spec.get('connections', []):
        require(len(edge['regions']) == 2 and all(x in ids for x in edge['regions']), 'Unknown connection')
        require(len(set(edge['regions']))==2,'Self connection')
        require(edge['kind'] in ['open', 'door'], 'Unknown connection type')
        spaces={r['id']:r['space_id'] for r in spec['regions']}
        require(spaces[edge['regions'][0]]!=spaces[edge['regions'][1]] or edge['kind']=='open',
                'Accounting subdivisions must not invent a wall')
    # Frozen regional-estimation method records uncertainty but still gives a best estimate.
    # Invalid/missing numeric dimensions remain an error; nonempty evidence notes are not.
    return spec


def rectangles(spec, scene):
    layout = scene['layout']; require(set(layout) == {x['id'] for x in spec['regions']}, 'Layout region mismatch')
    result = {}
    for r in spec['regions']:
        item = layout[r['id']]; vector(item['origin'], 2)
        require(type(item['swap']) is bool, 'swap must be Boolean')
        x, y = item['origin']; w, h = r['size_m'][::-1] if item['swap'] else r['size_m']
        result[r['id']] = (x, x + w, y, y + h)
    return result


def edge(rect, side):
    x0, x1, y0, y1 = rect
    return {'north': ('h', y1, x0, x1), 'south': ('h', y0, x0, x1),
            'east': ('v', x1, y0, y1), 'west': ('v', x0, y0, y1)}[side]


def validate_reference_size(ref, sizes):
    vector(sizes,3,True)
    require(all(lo-1e-4<=v<=hi+1e-4 for v,(lo,hi) in zip(sizes,ref['size_ranges_m'])),
            'Actual reference size outside original-frame estimate range: '+ref['id'])


def validate_bounds(rect, points):
    x0,x1,y0,y1=rect
    spill=max(x0-min(p[0] for p in points),max(p[0] for p in points)-x1,
              y0-min(p[1] for p in points),max(p[1] for p in points)-y1,0)
    require(spill<=.025,'Furniture outside fixed region')


def validate_scene(spec, scene):
    validate_regions(spec); rects = rectangles(spec, scene); items = list(rects.items())
    for i, (a, p) in enumerate(items):
        for b, q in items[i + 1:]:
            require(max(0, min(p[1], q[1]) - max(p[0], q[0])) *
                    max(0, min(p[3], q[3]) - max(p[2], q[2])) < 1e-6, f'Overlapping regions {a}/{b}')
    openings = scene.get('openings', [])
    for o in openings:
        require(o['region'] in rects and o['side'] in ['north','south','east','west'], 'Invalid opening')
        vector([o['start'], o['end']], 2)
        e = edge(rects[o['region']], o['side'])
        require(0 <= o['start'] < o['end'] <= e[3] - e[2] + 1e-6, 'Opening outside wall')
    adjacency = {k: set() for k in rects}
    opposite = {'north':'south','south':'north','east':'west','west':'east'}
    for link in spec.get('connections', []):
        a, b = link['regions']; matched = False
        for oa in openings:
            if oa['region'] != a: continue
            ea = edge(rects[a], oa['side'])
            for ob in openings:
                if ob['region'] != b or ob['side'] != opposite[oa['side']]: continue
                eb = edge(rects[b], ob['side'])
                if ea[0] != eb[0] or abs(ea[1] - eb[1]) > 1e-5: continue
                lo, hi = max(ea[2], eb[2]), min(ea[3], eb[3])
                clear = min(ea[2]+oa['end'], eb[2]+ob['end']) - max(ea[2]+oa['start'], eb[2]+ob['start'])
                spaces={r['id']:r['space_id'] for r in spec['regions']}
                if link['kind'] == 'open' and spaces[a]==spaces[b]:
                    matched |= hi > lo and clear >= hi-lo-1e-5
                else: matched |= clear >= .7
        require(matched, f'Missing aligned {link["kind"]} connection {a}/{b}')
        adjacency[a].add(b); adjacency[b].add(a)
    seen = {next(iter(rects))}
    while True:
        expanded = seen | set().union(*(adjacency[x] for x in seen))
        if expanded == seen: break
        seen = expanded
    require(seen == set(rects), 'Disconnected regions')
    return rects
