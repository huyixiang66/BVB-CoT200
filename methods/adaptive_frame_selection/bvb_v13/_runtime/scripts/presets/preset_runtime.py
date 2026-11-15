"""Scene-independent Blender primitive/runtime library. No scene coordinates or assets."""
import bpy
import math
from mathutils import Vector
from math import pi, sin, cos, atan2, sqrt

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
for _c in list(bpy.data.collections):
    bpy.data.collections.remove(_c)
scene = bpy.context.scene
scene.unit_settings.system = 'METRIC'
scene.unit_settings.scale_length = 1.0
_groups = {}
_active_group = 'Architecture'
_camera = None
_duration = 90.0


def material(name, color, roughness=.5, metallic=0, transmission=0, emission=0):
    m = bpy.data.materials.new(name)
    m.diffuse_color = (*color[:3], 1)
    m.use_nodes = True
    p = m.node_tree.nodes.get('Principled BSDF')
    for key, value in [('Base Color', (*color[:3], 1)), ('Roughness', roughness),
                       ('Metallic', metallic), ('Transmission Weight', transmission),
                       ('Emission Color', (*color[:3], 1)), ('Emission Strength', emission)]:
        if key in p.inputs: p.inputs[key].default_value = value
    return m


M = {}
for _name, _color, _rough, _metal, _trans in [
    ('white',(.85,.85,.82),.5,0,0),('plaster',(.8,.79,.75),.85,0,0),
    ('tile',(.72,.71,.66),.3,0,0),('grout',(.35,.35,.33),.9,0,0),
    ('wood',(.42,.23,.11),.5,0,0),('lightwood',(.65,.43,.20),.5,0,0),
    ('darkwood',(.20,.11,.06),.5,0,0),('black',(.018,.022,.025),.45,0,0),
    ('gray',(.32,.34,.35),.6,0,0),('lightgray',(.65,.66,.65),.6,0,0),
    ('metal',(.65,.68,.70),.22,.85,0),('glass',(.55,.72,.75),.18,.05,.45),
    ('mirror',(.8,.84,.85),.08,1,0),('fabric',(.46,.44,.40),1,0,0),
    ('leather',(.045,.05,.055),.4,0,0),('cream',(.77,.73,.62),.85,0,0),
    ('red',(.62,.05,.04),.5,0,0),('green',(.23,.42,.09),.5,0,0),
    ('blue',(.07,.20,.37),.5,0,0),('yellow',(.85,.57,.10),.5,0,0),
    ('orange',(.8,.30,.05),.5,0,0),('brown',(.28,.17,.10),.6,0,0)]:
    M[_name] = material(_name, _color, _rough, _metal, _trans)


def group(name):
    global _active_group
    _active_group = name
    if name not in _groups:
        c = bpy.data.collections.new(name)
        scene.collection.children.link(c)
        _groups[name] = c
    return _groups[name]


def _finish(o, name, mat, collection, smooth=False, bevel=0):
    o.name = name
    target = collection or _active_group
    if target not in _groups:
        c = bpy.data.collections.new(target); scene.collection.children.link(c); _groups[target] = c
    for c in list(o.users_collection): c.objects.unlink(o)
    _groups[target].objects.link(o)
    if mat is not None: o.data.materials.append(M[mat] if isinstance(mat, str) else mat)
    if smooth:
        for p in o.data.polygons: p.use_smooth = True
    if bevel:
        mod = o.modifiers.new('Preset bevel', 'BEVEL'); mod.width = bevel; mod.segments = 3
        o.modifiers.new('Preset normals', 'WEIGHTED_NORMAL')
    return o


def box(name, pos, size, mat='white', rot=(0,0,0), bevel=0, collection=None):
    bpy.ops.mesh.primitive_cube_add(size=1, location=pos, rotation=rot)
    o = bpy.context.object; o.dimensions = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return _finish(o, name, mat, collection, bevel=bevel)


def sphere(name, pos, size, mat='white', rot=(0,0,0), collection=None):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=1, location=pos, rotation=rot)
    o = bpy.context.object; o.scale = [v/2 for v in size]
    return _finish(o, name, mat, collection, smooth=True)


def cylinder(name, pos, radius, depth, mat='metal', rot=(0,0,0), collection=None):
    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=radius, depth=depth, location=pos, rotation=rot)
    return _finish(bpy.context.object, name, mat, collection, smooth=True)


def cone(name, pos, radius1, radius2, depth, mat='white', rot=(0,0,0), collection=None):
    bpy.ops.mesh.primitive_cone_add(vertices=32, radius1=radius1, radius2=radius2, depth=depth, location=pos, rotation=rot)
    return _finish(bpy.context.object, name, mat, collection, smooth=True)


def torus(name, pos, radius, tube, mat='metal', rot=(0,0,0), scale=(1,1,1), collection=None):
    bpy.ops.mesh.primitive_torus_add(major_segments=40, minor_segments=10, major_radius=radius, minor_radius=tube, location=pos, rotation=rot)
    o = bpy.context.object; o.scale = scale
    return _finish(o, name, mat, collection, smooth=True)


def rod(name, a, b, radius=.015, mat='metal', collection=None):
    a, b = Vector(a), Vector(b); direction = b-a
    if direction.length < 1e-8: raise ValueError('Rod endpoints must differ')
    o = cylinder(name, (a+b)/2, radius, direction.length, mat, collection=collection)
    o.rotation_euler = direction.to_track_quat('Z','Y').to_euler()
    return o


def polyline(name, points, radius=.015, mat='metal', collection=None):
    return [rod(name+str(i), a, b, radius, mat, collection) for i,(a,b) in enumerate(zip(points,points[1:]))]


def area_light(name, pos, target, power=300, size=3, color=(1,1,1)):
    d = bpy.data.lights.new(name, 'AREA'); d.energy=power; d.shape='DISK'; d.size=size; d.color=color
    o = bpy.data.objects.new(name,d); scene.collection.objects.link(o); o.location=pos
    o.rotation_euler=(Vector(target)-Vector(pos)).to_track_quat('-Z','Y').to_euler()
    return o


def camera_path(shots, lens=24, duration=90, fps=24):
    """shots: [(seconds, position_xyz, look_at_xyz), ...], in meters."""
    global _camera, _duration
    if len(shots)<2: raise ValueError('At least two camera samples required')
    times=[s[0] for s in shots]
    if times!=sorted(set(times)): raise ValueError('Camera times must be unique and increasing')
    d=bpy.data.cameras.new('Walkthrough camera');d.lens=lens;d.sensor_width=36;d.clip_start=.03;d.clip_end=200
    o=bpy.data.objects.new(d.name,d);scene.collection.objects.link(o);scene.camera=o
    previous=None
    for t,pos,target in shots:
        direction=Vector(target)-Vector(pos)
        if direction.length<1e-8: raise ValueError('Camera target equals camera position')
        o.location=pos;e=direction.to_track_quat('-Z','Y').to_euler('XYZ')
        if previous is not None:
            for i in range(3):
                while e[i]-previous[i]>pi:e[i]-=2*pi
                while e[i]-previous[i]<-pi:e[i]+=2*pi
        o.rotation_euler=e;previous=e.copy();frame=1+round(t*fps)
        o.keyframe_insert('location',frame=frame);o.keyframe_insert('rotation_euler',frame=frame)
    action=o.animation_data.action
    if hasattr(action,'fcurves'): curves=action.fcurves
    else: curves=[fc for layer in action.layers for strip in layer.strips for bag in strip.channelbags for fc in bag.fcurves]
    for fc in curves:
        for k in fc.keyframe_points:k.interpolation='LINEAR'
    scene.render.fps=fps;scene.frame_start=1;scene.frame_end=1+round(duration*fps)
    scene.frame_set(1);_camera=o;_duration=duration
    return o


def save(path='/workspace/output/result.blend', require_animation=True):
    if not scene.camera or (require_animation and not scene.camera.animation_data):
        raise ValueError('An animated camera is required' if require_animation else 'A static camera is required')
    meshes=[o for o in scene.objects if o.type=='MESH']
    if not meshes: raise ValueError('No scene geometry')
    if not any(o.type=='LIGHT' for o in scene.objects):
        bpy.context.view_layer.update()
        points=[o.matrix_world@Vector(p) for o in meshes for p in o.bound_box]
        lo=Vector([min(p[i] for p in points) for i in range(3)]);hi=Vector([max(p[i] for p in points) for i in range(3)])
        center=(lo+hi)/2
        area_light('Preset overhead',(center.x,center.y,hi.z+3),center,1800,max(3,(hi-lo).length))
    scene.world=scene.world or bpy.data.worlds.new('Preset world');scene.world.use_nodes=True
    scene.world.node_tree.nodes['Background'].inputs[0].default_value=(.32,.37,.43,1)
    scene.world.node_tree.nodes['Background'].inputs[1].default_value=.20
    scene.render.engine='CYCLES';scene.cycles.samples=24;scene.cycles.use_denoising=True
    scene.render.resolution_x=640;scene.render.resolution_y=480;scene.render.resolution_percentage=100
    scene.view_settings.view_transform='AgX';scene.frame_set(1)
    bpy.ops.wm.save_as_mainfile(filepath=path)
    print('PRESET_SAVED',len(meshes),'mesh parts',len(bpy.data.collections),'collections',scene.frame_end,'frames')
