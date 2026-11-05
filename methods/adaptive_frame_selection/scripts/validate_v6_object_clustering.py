"""Validate the offline clustering comparison and portable media provenance."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import argparse,json,hashlib
from collections import Counter

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--delivery',type=Path,required=True);a=ap.parse_args()
    s=read(a.source/'selection.json');e=read(a.source/'evidence.json');d=a.delivery
    assert s['questions']
    assert not s['new_model_calls'] and not s['reconstruction_run']
    for p,h in s['provenance'].items():assert sha(Path(p))==h,p
    total={};bounds=Counter()
    for qid,q in s['questions'].items():
        expected={p for g in q['arms']['baseline'] for p in g['peak_ids']}
        frames={r['source_frame_index'] for g in q['arms']['baseline'] for r in g['candidate_frames']}
        for mode,groups in q['arms'].items():
            ids=[p for g in groups for p in g['peak_ids']]
            assert len(ids)==len(set(ids)) and set(ids)==expected
            assert {r['source_frame_index'] for g in groups for r in g['candidate_frames']}==frames
            assert q['summary'][mode]['image_slots']==sum(len(g['selected']) for g in groups)
            for g in groups:
                selected=g['selected'];assert 4<=len(selected)<=8
                fid=[r['source_frame_index'] for r in selected];assert fid==sorted(set(fid))
                assert set(fid)<={r['source_frame_index'] for r in g['candidate_frames']}
                for r in g['candidate_frames']:assert sha(d/'frames'/f'{r["source_frame_index"]:07d}.jpg')==r['frame_sha256']
                if mode!='baseline':
                    assert not any(p['cannot_link'] and set(p['peaks'])<=set(g['peak_ids']) for p in e[g['object']]['pairs'])
                if mode=='roi_robust':bounds[len(selected)]+=1
        for mode,counts in q['summary'].items():
            total.setdefault(mode,Counter()).update(counts)
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[];self.ids=set();self.panels=Counter()
        def handle_starttag(self,t,attrs):
            v=dict(attrs)
            for k in ['href','src']:
                if v.get(k):self.links.append(v[k])
            if v.get('id'):self.ids.add(v['id'])
            if v.get('data-panel'):self.panels[v['data-panel']]+=1
    p=Links();p.feed((d/'index.html').read_text())
    for mode in ['baseline','roi_complete','roi_robust']:assert p.panels[mode]==len(s['questions'])
    for link in p.links:
        u=urlsplit(link)
        if u.scheme:continue
        if u.path and u.path!='validation.json':assert (d/unquote(u.path)).is_file(),link
        if not u.path and u.fragment:assert u.fragment in p.ids
    assert sha(d/'original-video.mp4')==read(d/'media-manifest.json')['video_sha256']
    result=dict(status='passed',questions=len(s['questions']),totals=total,selected_frames_per_new_group=dict(bounds),
                frame_hashes_verified=True,peak_membership_preserved=True,cannot_links_respected=True,
                source_code_hashes_verified=True,html_links_checked=len(p.links),comparison_panels=dict(p.panels),
                llm_calls=0,reconstruction_run=False,clustering_accuracy_verified=False)
    for path in [a.source/'validation.json',d/'validation.json']:path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
