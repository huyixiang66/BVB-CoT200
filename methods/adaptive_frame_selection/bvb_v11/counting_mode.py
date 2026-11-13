"""Opt-in counting policy and deterministic neighbour-aware instance colors."""
import math

TYPES = {'object_counting_single', 'object_counting_multiple'}


def enabled(question, switch=False):
    return bool(switch) and question.get('question_type') in TYPES


def peak_threshold(questions, switch=False, default=.5):
    if switch:
        if not questions or not all(enabled(q, True) for q in questions):
            raise ValueError('Counting-mode peak extraction requires a counting-only batch')
        return .25
    return default


def assign_colors(instances, palette):
    """DSATUR graph coloring: same-category nearest-3 neighbours cannot share color.

    Category palettes are disjoint. Palette colors may repeat for non-neighbours;
    color cardinality therefore never declares the physical instance count to QA.
    """
    categories=sorted({r['category'] for r in instances})
    if len(categories)>len(palette):raise ValueError('Insufficient category palette')
    maps={};legend={};edges=[]
    for ci,category in enumerate(categories):
        colors=[list(c) for c in palette[ci::len(categories)]]
        legend[category]=colors
        rows=sorted([r for r in instances if r['category']==category],key=lambda r:r['instance_id'])
        byid={r['instance_id']:r for r in rows}
        if len(byid)!=len(rows):raise ValueError('Duplicate physical instance ID')
        graph={k:set() for k in byid}
        def distance(a,b):return math.dist(byid[a]['center'],byid[b]['center'])
        for ident in graph:
            for other in sorted((k for k in graph if k!=ident),key=lambda k:(distance(ident,k),k))[:3]:
                graph[ident].add(other);graph[other].add(ident)
        assigned={}
        while len(assigned)<len(graph):
            ident=max((k for k in graph if k not in assigned),
                      key=lambda k:(len({assigned[n] for n in graph[k] if n in assigned}),len(graph[k]),k))
            forbidden={assigned[n] for n in graph[ident] if n in assigned}
            available=[i for i in range(len(colors)) if i not in forbidden]
            if not available:raise ValueError('Neighbour color graph exceeds palette; extend palette')
            def quality(i):
                separation=min((math.dist(colors[i][:3],colors[j][:3]) for j in forbidden),default=1.)
                return (separation,-sum(j==i for j in assigned.values()),-i)
            assigned[ident]=max(available,key=quality)
        for ident,i in assigned.items():maps[ident]=dict(category=category,rgba=colors[i])
        edges.extend([a,b] for a in graph for b in sorted(graph[a]) if a<b)
    assert all(maps[a]['rgba']!=maps[b]['rgba'] for a,b in edges)
    return dict(instances=maps,category_palette=legend,neighbour_edges=edges,
                rule='same-category symmetrized nearest-3 instance centers; distinct adjacent colors',
                same_category_same_color=False,geometry_modified=False)
