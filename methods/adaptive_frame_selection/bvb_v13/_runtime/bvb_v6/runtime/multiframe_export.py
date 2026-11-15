"""Read target geometry from an existing merged scene, without modifying it."""
import json
from pathlib import Path
import sys
import bpy
import numpy as np

source=Path(sys.argv[sys.argv.index('--')+1]);out=Path(sys.argv[sys.argv.index('--')+2])
merge=json.loads((source/'merge.json').read_text(encoding='utf8'))
arrays={};targets=[];skipped=[]
for part in merge['parts']:
    gid=part['group_id']; category=part['target_category'];items=part.get('targets',[])
    if len(items)!=1 or category in part.get('missing_keywords',[]):
        reason='同组有多个目标，无法可靠指定检测框对应哪一个' if len(items)>1 else '未生成该组目标'
        skipped.append({'group_id':gid,'category':category,'status':'skipped','reason':reason});continue
    target=items[0];prefix=gid+'::'+target['id']
    cols=[c for c in bpy.data.collections if c.name==prefix or (c.name.startswith(prefix+'.') and c.name[len(prefix)+1:].isdigit())]
    meshes=[o for c in cols for o in c.all_objects if o.type=='MESH']
    points=np.asarray([tuple(o.matrix_world@v.co) for o in meshes for v in o.data.vertices],dtype=np.float64)
    if not len(points):
        skipped.append({'group_id':gid,'category':category,'status':'skipped','reason':'缺少目标网格'});continue
    arrays[gid]=points
    targets.append({'group_id':gid,'category':category,'target':target,'vertices':len(points)})
np.savez_compressed(out/'target-vertices.npz',**arrays)
(out/'targets.json').write_text(json.dumps(targets,ensure_ascii=False,indent=2),encoding='utf8')
(out/'skipped.json').write_text(json.dumps(skipped,ensure_ascii=False,indent=2),encoding='utf8')
print('EXPORT',len(targets),'SKIP',len(skipped))
