"""Question targets, direct supports/contacts and all evidenced architecture."""
from pathlib import Path
import json

def build(question,limits,colors,out,encoding):
    original=(Path(__file__).parent/'original-bvb-prompt.txt').read_text()
    return original+'''

BVB_V10 EXPERIMENT OVERRIDES: These explicitly replace conflicting original harness rules.
Frames have ALREADY been extracted, clustered, reviewed and supplied. Inspect ALL groups
together to reconstruct ONE coherent scene, not independent rooms later appended.
Return exactly one complete executable Python bpy script in a python code fence. No bash
or frames tools, no prose, and do not answer the benchmark question. No GT is supplied.
Use only bpy, math, mathutils, json. No network, external models, image textures or arbitrary
mesh construction. Use standard primitive assemblies for the question-relevant objects.

STRICT SCENE SCOPE:
1. TARGETS: ONLY visible objects in the full question object list (including option objects).
   Reconstruct recognizable primitive geometry. Additional visible question targets may be
   included without a DINO box. Boxes and cluster labels are provisional, not proof.
2. CONTEXT: ONLY separate objects DIRECTLY physically supporting OR DIRECTLY TOUCHING
   a visible TARGET are allowed. Represent each as a plain WHITE oriented cuboid.
   Require observed physical contact/support evidence. Contact may be beside, above or
   below the target; an object resting on a target also qualifies as direct contact.
   Proximity, containment without touching, being a landmark or a route obstacle do NOT qualify.
   No recursive support chains or contact chains. An allowed context already in the question
   list is a TARGET, not a duplicate white box. No decorative/detail subparts for context.
   Keep the observed extent of the actual contacting/supporting object, not unrelated units.
3. ARCHITECTURE IS MANDATORY AND INDEPENDENT OF TARGET RELEVANCE: Reconstruct ALL walls,
   partitions, wall returns, door openings and floor surfaces supported by ANY supplied frame,
   including those not near a target. Do NOT reduce architecture to a kitchen backsplash or
   local target enclosure. Inspect every frame for structure before placing objects. Preserve
   the full evidenced room outline, partitions separating areas, passages and visible floor
   connections. Do not delete a wall merely because it occludes a target in a render.
   Use simple WHITE cuboids. Door openings are EMPTY SPACE bounded by wall segments/header,
   never a solid box filling the opening. Door leaves are not architecture and qualify only
   as targets or direct physical contacts. Omit roofs/ceilings. Do not invent unseen rooms or
   complete an unsupported closed rectangle; flag uncertain structure in the inventory.
   Architecture is exempt from the support/contact filtering rule. A counter's contact with
   its cabinet does not justify retaining that cabinet unless it also directly touches a TARGET.
4. OMIT all remaining NONARCHITECTURAL objects. Never apply this omission rule to evidenced
   walls, partitions, door openings or floors.

ONE SHARED POSE FRAME: Infer all target and structural geometry jointly using the supplied
LingBot cameras. These use Z up, first-video-camera origin and native approximate units,
not calibrated meters. Do not assume z=0 is floor. Do not independently recenter/rotate/
rescale clusters. Use cross-frame visibility and known poses to infer positions/depths.
Deduplicate repeated views; group labels are not separate rooms or physical instances.
Do not infer a new camera trajectory. The local runner attaches exact reference cameras
and checks their numerical projection. That does not guarantee geometry correctness.

BLENDER OUTPUT CONTRACT:
All meshes are static and must have obj['role'] = 'target', 'context', or 'structure'.
Every target part: obj['category']=exact question label, obj['instance_id']=stable physical
instance ID (shared across that instance's parts). Use different IDs for distinct objects.
Respect final instance caps below; null means multiple allowed. Never invent unseen targets
to satisfy coverage. Missing targets should remain absent and will be reported.
Every context cube: obj['relation']='support' with obj['supports']=json.dumps([instance_id,...]),
OR obj['relation']='contact' with obj['contacts']=json.dumps([instance_id,...]). References must
be directly supported/touched TARGET instances, never another context ID or category label.
Every structure cube: obj['category']='wall' or 'floor'. Door openings use separated wall
cubes with a clear gap; do not create a mesh with category door unless it is a target.
Non-target meshes must be cubes with 8 vertices, 6 faces, no modifiers, no extra detailing.
Use white RGBA [1,1,1,1] for context and structure. Set both material.diffuse_color and
object.color; materials use a numeric Principled BSDF. Use diagnostic target colors below.
Build meaningful named collections. Only write the exact scene.blend path below. No other
filesystem access. Include one temporary overview camera and light. Do not render.
Save with bpy.ops.wm.save_as_mainfile(filepath=OUTPUT_BLEND_PATH).
Before saving, set scene['architecture_inventory']=json.dumps([...]). Enumerate EACH evidenced
wall/partition, floor and door opening with {'id':unique_string,'kind':'wall'|'floor'|'opening',
'label':description,'evidence_times_s':[supplied_frame_timestamp,...],'mesh_names':[actual_names,...]}.
Every structure mesh must be listed. Opening entries reference at least two wall boundary meshes,
never a solid opening mesh. One wall item may reference several segments around an opening.
Include a 'note' for uncertain extent; references must point to actual objects, not invented names.
This inventory records evidence and exposes omissions; it is not a claim of verified geometry.
'''+ '\nOUTPUT_BLEND_PATH = '+str(out/'scene.blend')+'\nPUBLIC QUESTION = '+json.dumps(question,ensure_ascii=False)+\
      '\nFINAL INSTANCE LIMITS = '+json.dumps(limits)+'\nTARGET COLORS = '+json.dumps(colors)+\
      '\nCAMERA ENCODING = '+encoding
