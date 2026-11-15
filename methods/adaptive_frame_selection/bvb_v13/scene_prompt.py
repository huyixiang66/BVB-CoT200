"""Task-aware minimal geometry; fixed properties and Blender code stay local."""
GEOMETRY_POLICY='task-essential-geometry-v1'
PROMPT='''Infer ONE coherent 3D scene from original images and soft pose/anchor evidence. Return ONLY compact JSON, no Python, prose or answer. Z up; use the poses' uncalibrated native world frame, not meters. Noisy poses/anchors are visibility/surface hints, not room boundaries or enforced object centers. Verify identities; merge views/clusters of the same instance, retain every distinct evidenced target within instance limits. Never fabricate a target or arrange geometry from a guessed answer.
Output {"v":2,"o":[...],"a":[...]} with optional c,g. Object rows: [category_index,world_xyz,geometry,optional_rotation]. Geometry is full dimensions [dx,dy,dz] for a centered box, or [shape,dx,dy,dz] for c=cylinder along Z, s=ellipsoid, t=torus normal Z. Default rotation is zero; emit only nonzero XYZ Euler degrees, or scalar Z yaw. Row order supplies instance IDs; category indices map to question.objects. Local code fills category names, colors, materials and all runtime metadata.
Start with ONE primitive per physical instance. Omit decoration, seams, spokes and internal construction. Merge parts only when doing so preserves task-relevant silhouette, surfaces, gaps, orientation and supports. Never merge distinct instances, fill a relevant hollow space, enlarge a nearest surface, or trim a measured dimension to save output. If one primitive fails, use geometry {"p":[parts]}: each part [shape,local_xyz,full_dimensions,optional_rotation], shape b,c,s,t. Use the fewest necessary parts; no arbitrary cap that discards relevant geometry. Object rotation applies to all local parts.
Architecture a rows: [kind,world_xyz,full_dimensions,evidence_frame_indices,optional_rotation], f=floor,w=wall. Inspect all images; preserve evidenced floors, partitions, connections and irregular footprints with cuboids. Openings are gaps, never solid boxes. No roofs, invented rooms, giant floors or unsupported enclosing rectangles. Do not remove walls for display.
Optional c rows: [relation,target_object_indices,world_xyz,full_dimensions,optional_rotation], s=direct support,t=direct contact. Only necessary directly touching/supporting context; no recursive chains, unrelated furniture or duplicate targets. Optional g rows: [architecture_wall_indices,evidence_frame_indices], at least two real boundary walls; declarations add no geometry. Empty c,g and zero rotations should be omitted. Architecture/context are white locally.
All indices zero-based; frame indices address supplied originals, not video frame numbers. Dimensions finite positive; keep full 3D height/depth. Structure-only frames authorize architecture, not furniture. A scale reference is the same endpoint instance or one separate object: preserve its specified dimension and recognizable shape without forcing native scale to its uncertain size prior. Never drop required evidence for a shorter output. Local code handles Blender, lighting, cameras, labels and rendering.
'''

TASK_RULES={
 'rel_dir':'Preserve relative positions, height, facing cues and recognizable silhouettes. Omit small parts that do not affect identity, facing or occlusion.',
 'rel_dist':'Preserve closest surfaces of ALL candidate/reference instances, height, depth and relevant gaps. Do not replace an open or concave object by a solid bounding box when that would change distances.',
 'abs_dist':'Preserve endpoint closest surfaces, height/depth, the scale reference dimension and relevant empty gaps. Simplify away from these measurement surfaces only; supports matter when they establish height.',
 'counting':'Preserve separate physical instances and enough silhouette to recognize each category. Merge redundant parts within one instance; never merge adjacent instances or count parts as objects.',
 'object_size':'Preserve the complete outer extent in all three dimensions, including protrusions that set the longest dimension. Simplify internal details only when identity and measured extent remain clear.',
 'route':'Preserve start/facing/waypoint identity and placement, walls, doorway gaps and obstacles relevant to the specified route. Simplify furniture away from route boundaries; never open a blocked passage.',
 'room_size':'Preserve all evidenced floor boundaries, irregular footprints and room connections, including floor below furniture. Keep a small set of recognizable evidenced furniture useful for scale; simplify its parts without changing scale-bearing extents. No area calculation or numeric answer.',
}


def task_for(packet):
    if packet.get('room_mode'):return 'room_size'
    q=packet.get('question',{}).get('question_type','')
    if q.startswith('object_rel_direction'):return 'rel_dir'
    if q.startswith('object_rel_distance'):return 'rel_dist'
    if q.startswith('object_counting'):return 'counting'
    if q.startswith('room_size_estimation'):return 'room_size'
    return {'object_abs_distance':'abs_dist','object_size_estimation':'object_size','route_planning':'route'}.get(q)


def build(packet):
    task=task_for(packet)
    text=PROMPT+('\nTask geometry: '+TASK_RULES[task] if task else '')
    if packet.get('room_mode'):
        text+='\nROOM MODE: category uses short STRINGS instead of indices. Reconstruct evidenced rooms as one coherent space; revisits are not new rooms. Floor includes furniture-covered areas, excludes unseen exterior. Native pose scale remains uncalibrated.'
    zero=packet.get('counting_zero_evidence_categories',[]) if packet.get('counting_allow_all_rejected') else []
    if zero:text+='\nZero-evidence categories may remain absent; never fabricate them: '+repr(zero)
    return text
