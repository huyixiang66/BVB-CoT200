import json,sys,time
from .pipeline import ROOT,read,save,sha
sys.path.insert(0,str(ROOT/"scripts"))
from subscription_direct_client import request,image_item
def state(out,stage):save(out/"status.json",dict(stage=stage,updated=time.time()))

def parse(text):
 text=text.strip()
 if text.startswith('```'):text=text.split('\n',1)[1].rsplit('```',1)[0]
 return json.loads(text)

def qa(out):
 q=read(out/'question-public.json');legend='Target label colors (RGBA): '+json.dumps(read(out/'input.json')['colors'])+'. White/gray geometry represents evidenced walls, floor, openings, and direct supports or contacts. Colors are labels, not independent verification.'
 views=read(out/'views/views.json')['views']
 if not (out/'view-selection/selection.json').exists():
  content=[dict(type='input_text',text='Select ONE supplied view best showing relevant objects and their horizontal relationship. Do not answer the question. Return JSON {"view_id":"supplied id","reason":"Chinese reason","limitations":"Chinese limitations"}.\n'+json.dumps(q,ensure_ascii=False)+'\n'+legend)]
  for v in views:
   p=out/'views'/v['image'];assert sha(p)==v['sha256'];content += [dict(type='input_text',text='VIEW '+v['id']),image_item(p)]
  a=parse(request(out/'view-selection',content,model='gpt-6-astra',effort='high')['answer']);assert a['view_id'] in {v['id'] for v in views};save(out/'view-selection/selection.json',a)
 state(out,'qa')
 if not (out/'qa/answer.json').exists():
  a=read(out/'view-selection/selection.json');v=next(v for v in views if v['id']==a['view_id'])
  text='Answer using ONLY this reconstructed scene image, question, and color legend. Use imagined observer direction, not screen-left/right. State uncertainty if a target is missing or ambiguous but give the best option. Return JSON {"option":"option letter","answer":"exact option text without letter","reason":"Chinese explanation","confidence":"low|medium|high","limitations":"Chinese limitations"}.\n'+json.dumps(q,ensure_ascii=False)+'\n'+legend
  a=parse(request(out/'qa',[dict(type='input_text',text=text),image_item(out/'views'/v['image'])],model='gpt-6-astra',effort='high')['answer']);opts=dict(x.split('. ',1) for x in q['options']);assert a['option'] in opts and a['answer'].strip()==opts[a['option']].strip();save(out/'qa/answer.json',a)
