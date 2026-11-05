"""Generic semantic assets. No reconstruction-specific layout or object inventory.

Each asset is built in local coordinates, fitted to its requested full bounding
box, then rotated and translated. Its visible front is local -Y; Z is up.
"""
import json
import math
from pathlib import Path
import sys
import bpy
from mathutils import Matrix, Vector, Euler
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'presets'))
import preset_runtime as P
from schema import validate


def asset(o):
    name=o['id'];kind=o['kind'];style=o.get('style','standard')
    color=o.get('color','white')
    mat=P.material(name+' color',color) if isinstance(color,list) else color
    col=P.group(name)
    def b(label,p,s,m=None):return P.box(label,p,s,mat if m is None else m)
    def c(label,p,r,d,m=None):return P.cylinder(label,p,r,d,mat if m is None else m)
    def e(label,p,s,m=None):return P.sphere(label,p,s,mat if m is None else m)
    def rod(label,a,z,r=.025,m=None):return P.rod(label,a,z,r,mat if m is None else m)
    def legs(height=.7,spread=.38):
        for x in [-spread,spread]:
            for y in [-spread,spread]:b('leg',(x,y,height/2),(.055,.055,height),'darkwood')
    def frame():
        for x in [-.46,.46]:b('jamb',(x,0,.5),(.08,.18,1))
        b('lintel',(0,0,.96),(1,.18,.08))
    if kind=='wall':
        # Partition around the union of openings. No duplicate infill or hidden
        # solid wall behind an open doorway; all inputs are local metric units.
        w,d,h=o['s'];holes=o.get('openings',[])
        xs=sorted(set([-w/2,w/2]+[v for x,z,ww,hh in holes for v in [x-ww/2,x+ww/2]]))
        zs=sorted(set([-h/2,h/2]+[v for x,z,ww,hh in holes for v in [z-hh/2,z+hh/2]]))
        for a,z in zip(xs,xs[1:]):
            for lo,hi in zip(zs,zs[1:]):
                x=(a+z)/2;zz=(lo+hi)/2
                if any(abs(x-hx)<hw/2 and abs(zz-hz)<hh/2 for hx,hz,hw,hh in holes):continue
                b('wall panel',(x,0,zz),(z-a,d,hi-lo))
    elif kind in ['floor','ceiling','roof','rug','towel','box']:
        b(kind,(0,0,.5),(1,1,1))
    elif kind in ['door','doorway','window','mirror','picture']:
        frame()
        if kind not in ['doorway'] and not (kind=='door' and style=='open'):
            panel='glass' if kind=='window' else 'mirror' if kind=='mirror' else mat
            b('panel',(0,.02,.49),(.84,.055,.90),panel)
        if kind=='door':
            if style=='open':
                leaf=b('open door leaf',(-.43,-.4,.47),(.055,.8,.9))
            rod('handle',(.29,-.12,.48),(.39,-.12,.48),.018,'metal')
        if kind=='window':
            b('sill',(0,-.04,.025),(1,.3,.05));b('mullion',(0,-.02,.5),(.025,.1,.92))
        if kind=='picture':b('art inset',(0,-.04,.5),(.78,.02,.75),'blue')
    elif kind in ['table','stool']:
        legs(.90)
        if style=='round':c('top',(0,0,.95),.5,.1)
        else:b('top',(0,0,.95),(1,1,.1))
    elif kind=='chair':
        legs(.45)
        b('seat',(0,0,.48),(1,.92,.10));b('back',(0,.42,.78),(1,.12,.44))
        if style=='office':
            c('pedestal',(0,0,.25),.045,.5,'metal')
            for a in range(5):
                t=a*2*math.pi/5;rod('base spoke',(0,0,.07),(.45*math.cos(t),.45*math.sin(t),.07),.025,'black')
    elif kind in ['sofa','armchair']:
        b('base',(0,0,.24),(1,.9,.32));b('back',(0,.39,.66),(1,.2,.68))
        for x in [-.44,.44]:b('arm',(x,0,.55),(.12,1,.48))
        n=3 if kind=='sofa' else 1
        for i in range(n):b('seat cushion',(-.36+(i+.5)*.72/n,-.06,.48),(.70/n,.72,.16))
        for x in [-.40,.40]:
            for y in [-.36,.36]:b('foot',(x,y,.06),(.07,.07,.12),'darkwood')
    elif kind=='bed':
        b('base',(0,0,.22),(1,1,.34),'wood');b('mattress',(0,-.025,.5),(.96,.95,.22),'cream')
        b('headboard',(0,.46,.56),(1,.08,.88));b('blanket',(0,-.16,.63),(.94,.62,.04))
        for x in [-.24,.24]:e('pillow',(x,.27,.66),(.4,.23,.10),'white')
    elif kind in ['cabinet','shelf','fridge','dishwasher']:
        for x in [-.48,.48]:b('side',(x,0,.5),(.04,1,1))
        for z in [.03,.97]:b('cap',(0,0,z),(1,1,.06))
        b('back',(0,.48,.5),(1,.04,1))
        if kind=='shelf' or style=='open':
            for z in [.25,.50,.75]:b('shelf',(0,0,z),(.96,1,.035))
        else:
            n=2 if style=='double' else 1
            for i in range(n):
                x=-.5+(i+.5)/n;b('door',(x,-.48,.5),(.97/n,.05,.94))
                rod('handle',(x+.3/n,-.53,.45),(x+.3/n,-.53,.65),.014,'metal')
    elif kind=='washing_machine':
        b('body',(0,0,.5),(1,1,1));b('controls',(0,-.51,.88),(.9,.025,.15))
        port=c('door rim',(0,-.53,.47),.32,.055,'metal');port.rotation_euler=(math.pi/2,0,0)
        port=c('door glass',(0,-.565,.47),.265,.025,'black');port.rotation_euler=(math.pi/2,0,0)
        e('control dial',(.30,-.54,.88),(.10,.055,.10),'gray')
    elif kind in ['microwave','oven']:
        b('body',(0,0,.5),(1,1,1));b('glass front',(-.08,-.51,.5),(.73,.025,.7),'black')
        rod('handle',(.24,-.55,.25),(.24,-.55,.73),.025,'metal')
        for z in [.3,.7]:e('dial',(.40,-.54,z),(.1,.055,.1),'metal')
        if kind=='oven':
            for x in [-.25,.25]:
                for y in [-.25,.25]:c('burner',(x,y,1.01),.17,.025,'black')
    elif kind in ['sink','basin','bathtub']:
        b('bottom',(0,0,.10),(.90,.85,.20))
        for x in [-.46,.46]:b('rim',(x,0,.53),(.08,1,.94))
        for y in [-.46,.46]:b('rim',(0,y,.53),(.84,.08,.94))
        rod('tap',(0,.38,.7),(0,.38,1.15),.025,'metal')
        rod('spout',(0,.38,1.15),(0,.15,1.15),.025,'metal')
    elif kind=='toilet':
        e('foot',(0,0,.24),(.55,.65,.48));e('bowl',(0,-.1,.48),(.9,.85,.35))
        e('seat',(0,-.1,.66),(.90,.85,.07),'cream');b('tank',(0,.34,.74),(.80,.25,.52))
    elif kind=='shower':
        b('tray',(0,0,.025),(1,1,.05))
        for x in [-.49,.49]:b('glass side',(x,0,.5),(.025,1,1),'glass')
        b('glass back',(0,.49,.5),(1,.025,1),'glass')
        rod('riser',(.3,.43,.2),(.3,.43,.9),.012,'metal')
        e('shower head',(.3,.30,.9),(.23,.25,.03),'metal')
    elif kind in ['curtain','blind','radiator']:
        n=12
        for i in range(n):
            if kind=='blind':b('slat',(0,0,(i+.5)/n),(1,.10,.8/n))
            else:b('fold or fin',(-.5+(i+.5)/n,0,.5),(.8/n,.15,1))
        if kind=='radiator':
            for z in [.07,.93]:b('header',(0,0,z),(1,.18,.05))
    elif kind in ['tv','monitor']:
        b('display',(0,0,.59),(1,.12,.82),'black');b('screen',(0,-.066,.59),(.94,.012,.74))
        b('stand',(0,.02,.13),(.08,.10,.26),'black');b('base',(0,0,.025),(.45,.40,.05),'black')
    elif kind=='computer':
        b('case',(0,0,.5),(1,1,1));b('front',(0,-.51,.5),(.9,.02,.9),'black')
    elif kind=='lamp':
        c('foot',(0,0,.03),.30,.06,'metal');c('stem',(0,0,.43),.025,.8,'metal')
        P.cone('shade',(0,0,.84),.5,.3,.32,mat)
    elif kind=='plant':
        P.cone('pot',(0,0,.15),.20,.28,.3,'brown')
        for i in range(7):
            a=i*2*math.pi/7;x=.26*math.cos(a);y=.26*math.sin(a);z=.58+(i%3)*.12
            rod('stem',(0,0,.25),(x,y,z),.012,'green');e('foliage',(x,y,z),(.34,.28,.45),mat)
    elif kind in ['pot','cup','bin']:
        P.cone('vessel',(0,0,.5),.36,.5,1,mat)
        c('dark opening',(0,0,1.005),.43,.01,'black')
        if kind=='cup':P.torus('handle',(.48,0,.5),.24,.035,mat,rot=(math.pi/2,0,0))
    elif kind=='pan':
        c('pan',(0,0,.08),.35,.16);c('inside',(0,0,.17),.31,.02,'black')
        rod('handle',(.30,0,.10),(.85,0,.10),.04,'black')
    elif kind=='bottle':
        c('body',(0,0,.36),.26,.72);P.cone('shoulder',(0,0,.78),.26,.11,.12,mat)
        c('neck',(0,0,.92),.11,.16);c('cap',(0,0,1.01),.12,.06,'black')
    elif kind in ['pillow','sphere']:e(kind,(0,0,.5),(1,1,1))
    elif kind=='cylinder':c(kind,(0,0,.5),.5,1)
    else:raise ValueError(kind)
    bpy.context.view_layer.update()
    parts=list(col.objects)
    pts=[ob.matrix_world@Vector(v) for ob in parts for v in ob.bound_box]
    lo=Vector([min(p[i] for p in pts) for i in range(3)])
    hi=Vector([max(p[i] for p in pts) for i in range(3)])
    # Wall panels already have metric coordinates; retain openings exactly.
    if kind=='wall':fit=Matrix.Identity(4)
    else:
        scale=Vector([o['s'][i]/(hi[i]-lo[i]) for i in range(3)])
        fit=Matrix.Diagonal((*scale,1))@Matrix.Translation(-(lo+hi)/2)
    rot=Euler([math.radians(a) for a in o.get('r',[0,0,0])],'XYZ').to_matrix().to_4x4()
    root=bpy.data.objects.new(name+' [asset]',None);col.objects.link(root)
    root.location=o['p'];root.rotation_euler=rot.to_euler()
    root['asset_kind']=kind;root['asset_id']=name;root['requested_size']=o['s']
    col['asset_kind']=kind;col['asset_id']=name
    for ob in parts:
        local=fit@ob.matrix_world;ob.parent=root;ob.matrix_parent_inverse=Matrix.Identity(4);ob.matrix_basis=local
    return root


def compile_scene(data,path):
    validate(data)
    for o in data['objects']:asset(o)
    if 'camera' in data:
        P.camera_path(data['camera'])
    else:
        # Local visualization preset only: no video pose estimation or keyframes.
        bpy.context.view_layer.update()
        points=[o.matrix_world@Vector(v) for o in bpy.context.scene.objects if o.type=='MESH' for v in o.bound_box]
        rotation=Euler((math.radians(20),math.radians(-20),0),'XYZ')
        basis=rotation.to_matrix();projected=[basis.transposed()@p for p in points]
        lo=Vector([min(p[i] for p in projected) for i in range(3)])
        hi=Vector([max(p[i] for p in projected) for i in range(3)]);center=(lo+hi)/2
        cam_data=bpy.data.cameras.new('Static asset overview');cam_data.type='ORTHO'
        cam_data.ortho_scale=max(hi.x-lo.x,hi.y-lo.y)*1.2;cam_data.clip_end=1000
        cam=bpy.data.objects.new(cam_data.name,cam_data);bpy.context.scene.collection.objects.link(cam)
        cam.rotation_euler=rotation;cam.location=basis@Vector((center.x,center.y,hi.z+30))
        bpy.context.scene.camera=cam;bpy.context.scene.frame_start=1;bpy.context.scene.frame_end=1
    # Interior lighting is generated locally from each horizontal floor slab.
    floors=[o for o in data['objects'] if o['kind']=='floor']
    for o in floors:
        x,y,z=o['p'];w,d,h=o['s']
        P.area_light(o['id']+' fill',(x,y,z+h/2+2.3),(x,y,z),300*max(1,w*d/12),max(1,min(w,d)*.7))
    P.save(path,require_animation='camera' in data)
    print('ASSETS_COMPILED',len(data['objects']),'semantic objects')


if __name__=='__main__':
    args=sys.argv[sys.argv.index('--')+1:]
    compile_scene(json.loads(Path(args[0]).read_text()),args[1])
