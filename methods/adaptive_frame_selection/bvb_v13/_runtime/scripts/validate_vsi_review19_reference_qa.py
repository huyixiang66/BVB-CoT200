"""Verify QA receipts, source-image identity, scores, and local report links."""
from html.parser import HTMLParser
import hashlib
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/vsi-review19-v8-reference-20261109'
QA=BASE/'qa'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

questions=read(BASE/'questions.public.json');results=read(QA/'results.json');summary=read(QA/'summary.json')
assert len(results)==summary['completed']==19
assert {q['id'] for q in questions}=={r['id'] for r in results}
images=0;original_frames=0
for r in results:
    qid=str(r['id']);packet=read(QA/'inputs'/f'{qid}.json')
    assert r['input_manifest_sha256']==sha(QA/'inputs'/f'{qid}.json')
    assert not packet['ground_truth_sent'] and not packet['prior_answers_sent']
    assert set(packet['question'])=={'id','question','scene_name','question_type','objects'}
    assert len(packet['images'])==r['original_frame_count']+2
    manifest=read(BASE/'questions'/qid/'prepared/manifest.json');assert len(manifest['groups'])==1
    group=manifest['groups'][0];assert group['group_id']==packet['selected_group_id']
    source=read(BASE/'questions'/qid/'prepared'/group['folder']/'input-frames.json')
    actual=[f for f in packet['images'] if f['role']=='original_v8_frame']
    assert [(f['sha256'],f['source_frame_index']) for f in actual]==[(f['sha256'],f['source_frame_index']) for f in source]
    for f in packet['images']:assert sha(Path(f['path']))==f['sha256']
    images+=len(packet['images']);original_frames+=len(actual)
    receipt=read(QA/'requests'/qid/'request.json');assert receipt['status']=='completed'
    assert receipt['response_id']==r['response_id']
    text=receipt['answer'].strip()
    if text.startswith('```'):text=text.split('\n',1)[1].rsplit('```',1)[0]
    model=json.loads(text)
    for k in ('answer_cm','reason','confidence','sufficient_evidence','limitations'):assert model[k]==r[k]
    assert abs(r['mra']-float((abs(r['answer_cm']-r['ground_truth_cm'])/r['ground_truth_cm']<=1-np.linspace(.5,.95,11)).mean()))<1e-12
    input_text=read(QA/'requests'/qid/'input-text.json')
    assert input_text['image_count']==len(packet['images'])
    assert not any(key in '\n'.join(input_text['text_blocks']) for key in ('"ground_truth_cm"','"prior_answer_cm"','"mra"'))
assert abs(summary['mean_mra']-sum(r['mra'] for r in results)/19)<1e-12
class Links(HTMLParser):
    def __init__(self):super().__init__();self.urls=[];self.ids=[]
    def handle_starttag(self,tag,attrs):
        for k,v in attrs:
            if k in ('src','href') and v and not v.startswith(('#','https:','http:')):self.urls.append(v)
            if tag=='section' and k=='id':self.ids.append(v)
parser=Links();page=(BASE/'index.html').read_text();parser.feed(page)
assert len(parser.ids)==len(set(parser.ids))==19
assert page.count('<h3>本次新重构 QA</h3>')==19 and '本轮未重新进行 QA' not in page
for url in parser.urls:assert (BASE/unquote(urlsplit(url).path)).resolve().exists(),url
record={'status':'passed','questions':19,'completed_independent_requests':19,
    'total_input_images':images,'original_v8_frame_slots':original_frames,'reconstruction_images':38,
    'source_hashes_verified':True,'responses_match_report':True,'scores_recomputed':True,
    'local_report_links_verified':len(parser.urls),'ground_truth_and_prior_answer_fields_excluded':True,
    'browser_interaction_verified':False,'browser_note':'Browser host unavailable in this session; static links, JavaScript syntax and data validated.'}
(QA/'validation.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(record,ensure_ascii=False,indent=2))
