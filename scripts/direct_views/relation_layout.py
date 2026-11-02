"""Geometry-only view groups and standing-neighbourhood camera positions.

Grouping uses only reconstructed boxes and question roles. It does not rank
the candidates by the answer to a distance question.
"""
import math


def _group_span(semantics, items):
    boxes = [item['box'] for item in items if item['semantic'] in semantics]
    if not boxes:
        raise ValueError('No mapped boxes for comparison group')
    low = [min(box['min'][i] for box in boxes) for i in range(3)]
    high = [max(box['max'][i] for box in boxes) for i in range(3)]
    return math.sqrt(sum((high[i]-low[i])**2 for i in range(3)))


def comparison_groups(question, items):
    """Return two candidate groups sharing the question's anchor.

    For four unique candidates, examine all three 2+2 partitions and minimize
    the summed union-AABB diagonal. Every mapped instance contributes a box.
    Ties keep question order. Missing roles produce no groups.
    """
    roles = question.get('roles', {})
    anchors = roles.get('anchor', [])
    if not anchors:
        return []
    anchor = anchors[0]
    candidates = list(dict.fromkeys(c for c in roles.get('candidates', []) if c != anchor))
    if not candidates:
        return []
    mapped = {item['semantic'] for item in items}
    missing = set([anchor]+candidates)-mapped
    if missing:
        raise ValueError('Unmapped comparison objects: '+', '.join(sorted(missing)))
    if len(candidates) != 4:
        return [[anchor]+candidates[i:i+2] for i in range(0, len(candidates), 2)]
    a, b, c, d = candidates
    partitions = (((a,b),(c,d)), ((a,c),(b,d)), ((a,d),(b,c)))
    best = min(partitions, key=lambda pairs: sum(
        _group_span([anchor]+list(pair), items) for pair in pairs))
    return [[anchor]+list(pair) for pair in best]


def direction_points(standing_box, facing_center, floor_box,
                     radii=(.3, .65), height=1.5):
    """Sample outside a standing object's XY box without moving into it.

    Each radial ray starts at the standing-box centre, reaches the XY-box
    boundary, then continues by a radius. Returned stand_distance is the
    actual nearest XY-AABB distance, which can be smaller than that radius.
    horizontal_heading points from the candidate camera toward facing_center;
    facing_alignment prioritizes the side nearest the specified facing object.
    Room AABB checks are conservative; root must still test actual free space.
    """
    middle = [(standing_box['min'][i]+standing_box['max'][i])/2 for i in range(2)]
    half = [(standing_box['max'][i]-standing_box['min'][i])/2 for i in range(2)]
    delta = [facing_center[i]-middle[i] for i in range(2)]
    length = math.hypot(*delta)
    reference = [v/length for v in delta] if length > 1e-8 else [0., 1.]
    angle = math.atan2(reference[1], reference[0])
    results, seen = [], set()
    for offset in (0., 45., -45., 90., -90., 135., -135., 180.):
        direction = [math.cos(angle+math.radians(offset)), math.sin(angle+math.radians(offset))]
        edge_distance = min(half[i]/abs(direction[i]) for i in range(2) if abs(direction[i]) > 1e-10)
        for radius in radii:
            if radius <= 0:
                raise ValueError('Standing offsets must be positive')
            xy = [middle[i]+direction[i]*(edge_distance+radius) for i in range(2)]
            if not all(floor_box['min'][i]+.05 <= xy[i] <= floor_box['max'][i]-.05 for i in range(2)):
                continue
            key = tuple(round(v, 8) for v in xy)
            if key in seen:
                continue
            seen.add(key)
            nearest = [max(standing_box['min'][i], min(standing_box['max'][i], xy[i])) for i in range(2)]
            stand_distance = math.hypot(*(xy[i]-nearest[i] for i in range(2)))
            toward = [facing_center[i]-xy[i] for i in range(2)]
            toward_length = math.hypot(*toward)
            heading = [v/toward_length for v in toward] if toward_length > 1e-8 else reference[:]
            results.append({'location': xy+[floor_box['max'][2]+height],
                'stand_distance': stand_distance, 'horizontal_heading': heading+[0.],
                'facing_alignment': sum(direction[i]*reference[i] for i in range(2))})
    return sorted(results, key=lambda p: (-p['facing_alignment'], p['stand_distance'], p['location']))
