"""Data-only scene contract, usable without Blender. Coordinates are meters."""
import math

KINDS = ('wall floor ceiling roof door doorway window table chair stool sofa armchair bed '
         'cabinet shelf fridge microwave oven dishwasher sink basin toilet bathtub shower '
         'curtain blind rug radiator tv monitor computer lamp plant pot pan cup bottle '
         'picture mirror pillow towel bin washing_machine box sphere cylinder').split()
COLORS = ('white plaster tile grout wood lightwood darkwood black gray lightgray metal '
          'glass mirror fabric leather cream red green blue yellow orange brown').split()


def vector(value, n, label, positive=False):
    if not isinstance(value, list) or len(value) != n:
        raise ValueError(f'{label}: expected {n} numbers')
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value):
        raise ValueError(f'{label}: finite numbers required')
    if positive and any(x <= 0 for x in value):
        raise ValueError(f'{label}: values must be positive')


def validate(data):
    if not isinstance(data, dict) or set(data) not in ({'objects'},{'objects', 'camera'}):
        raise ValueError('Top-level keys must be objects, with optional camera')
    if not isinstance(data['objects'], list) or not 1 <= len(data['objects']) <= 500:
        raise ValueError('objects must contain 1..500 semantic objects')
    # Optional compact transport. The compiler always sees the same named fields.
    normalized=[]
    fields=['id','kind','p','s','r','color','style','openings']
    for o in data['objects']:
        if isinstance(o,list):
            if not 4<=len(o)<=8:raise ValueError('Compact object requires 4..8 fields')
            o={k:v for k,v in zip(fields,o) if v is not None}
        normalized.append(o)
    data=dict(data,objects=normalized)
    ids = set()
    for o in data['objects']:
        if not isinstance(o, dict) or set(o) - {'id','kind','p','s','r','color','style','openings'}:
            raise ValueError('Unknown object fields')
        if not {'id','kind','p','s'} <= set(o):
            raise ValueError('Each object requires id,kind,p,s')
        if not isinstance(o['id'], str) or not o['id'] or o['id'] in ids:
            raise ValueError('Object ids must be unique nonempty strings')
        ids.add(o['id'])
        if o['kind'] not in KINDS: raise ValueError(f'Unsupported kind: {o["kind"]}')
        vector(o['p'],3,o['id']+'.p');vector(o['s'],3,o['id']+'.s',True)
        vector(o.get('r',[0,0,0]),3,o['id']+'.r')
        color=o.get('color','white')
        if isinstance(color,list):
            vector(color,3,'color')
            if any(x<0 or x>1 for x in color):raise ValueError('RGB must be in [0,1]')
        elif color not in COLORS: raise ValueError('Use palette color or RGB')
        if o.get('style','standard') not in ['standard','round','open','office','double']:
            raise ValueError('Unsupported style')
        holes=o.get('openings',[])
        if not isinstance(holes,list) or len(holes)>30:raise ValueError('Invalid openings')
        if holes and o['kind']!='wall':raise ValueError('Only walls accept openings')
        for h in holes:
            vector(h,4,'opening [x,z,width,height]')
            x,z,w,hgt=h
            if w<=0 or hgt<=0 or abs(x)+w/2>o['s'][0]/2+1e-6 or abs(z)+hgt/2>o['s'][2]/2+1e-6:
                raise ValueError('Opening must fit wall local width/height')
    if 'camera' not in data:return data
    shots=data['camera']
    if not isinstance(shots,list) or len(shots)<2:raise ValueError('Need at least 2 camera samples')
    prev=-1
    for shot in shots:
        if not isinstance(shot,list) or len(shot)!=3:raise ValueError('camera item: [seconds,position,target]')
        t,p,target=shot;vector([t],1,'time');vector(p,3,'camera position');vector(target,3,'camera target')
        if not prev<t<=90 or p==target:raise ValueError('Camera times must increase in [0,90]; distinct targets')
        prev=t
    if shots[0][0]!=0:raise ValueError('Camera must start at t=0')
    return data
