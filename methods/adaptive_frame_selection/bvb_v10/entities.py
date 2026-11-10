"""Public-only entity preparation, including option nouns. Existing roles are preserved."""
import json,sys
from pathlib import Path
from .pipeline import ROOT,read,save
KEYS=('id','scene_name','question','question_type','options','objects','roles')
def public_questions(path):
 raw=read(path);rows=raw['questions'] if isinstance(raw,dict) else raw
 clean=[{k:q[k] for k in KEYS if k in q} for q in rows]
 if not clean or len({str(q['id']) for q in clean})!=len(clean):raise ValueError('Need nonempty questions with unique IDs')
 if len({q['scene_name'] for q in clean})!=1:raise ValueError('Run one scene/video at a time')
 for q in clean:
  if not str(q['id']).replace('-','').replace('_','').isalnum():raise ValueError('Unsafe question ID')
  if not q.get('question'):raise ValueError('Missing question')
 return clean

def prepare(source,out):
 qs=public_questions(source);out=out.resolve();out.mkdir(parents=True,exist_ok=True)
 for q in qs:
  if q.get('objects') and q.get('roles'):continue
  sys.path.insert(0,str(ROOT/'scripts'))
  from subscription_direct_client import request
  content=[dict(type='input_text',text='Extract question-relevant physical object categories from the public question AND ALL options. Disambiguate keyboard/piano etc using wording, not assumed answers. Return JSON {"objects":["English detector phrase"],"roles":{"standing":[],"facing":[],"query":[]},"regions":[],"requires_temporal_evidence":false}. Do not answer the question. Include regions separately; never pretend a room is a detectable object.\n'+json.dumps(q,ensure_ascii=False))]
  text=request(out/'requests'/str(q['id']),content,effort='high')['answer'].strip()
  if text.startswith('```'):text=text.split('\n',1)[1].rsplit('```',1)[0]
  parsed=json.loads(text)
  if parsed.get('regions') or parsed.get('requires_temporal_evidence'):raise ValueError('V10 static-object workflow needs explicit supplemental region/temporal evidence; inspect extraction receipt')
  objs=parsed['objects'];roles=parsed['roles']
  if not isinstance(objs,list) or not objs or any(not isinstance(x,str) or not x.strip() for x in objs):raise ValueError('Invalid extracted categories')
  if not isinstance(roles,dict) or any(not isinstance(v,list) or any(x not in objs for x in v) for v in roles.values()):raise ValueError('Invalid role references')
  q.update(objects=list(dict.fromkeys(objs)),roles=roles)
 save(out/'questions.json',dict(questions=qs,objects=sorted({o for q in qs for o in q['objects']})))
 return out/'questions.json'
