"""Validate the delivered run against frozen images, scene and meter geometry."""
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/vsi-review19-v8-reference-20261109'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    questions=read(BASE/'questions.public.json');assert len(questions)==19
    rows=[]
    for q in questions:
        qid=str(q['id']);folder=BASE/'questions'/qid;prepared=folder/'prepared'
        result=read(folder/'result.json');assert result['status']=='complete'
        manifest=read(prepared/'manifest.json');assert manifest['object_size_reference_mode']
        assert len(manifest['groups'])==1
        decision=read(folder/'cluster-comparison/decision.json');g=manifest['groups'][0]
        assert decision['selected_group_id']==g['group_id']
        assert manifest['cluster_comparison']['sha256']==sha(folder/'cluster-comparison/decision.json')
        sf=prepared/g['folder'];frames=read(sf/'input-frames.json')
        assert 1<=len(frames)<=8
        for f in frames:assert sha(Path(f['path']))==f['sha256']
        public=read(sf/'question.json')
        assert public['question']==q['question'] and public['objects']==q['objects']
        assert not any(k in public for k in ('ground_truth','ground_truth_cm','model_answer_cm','answer','mra'))
        audit=read(sf/'assets/question-audit.json');assert len(audit['targets'])==1
        assert not audit['ground_truth_used']
        instance=read(prepared/'instance-resolution/result.json')
        assert instance['status']=='validated' and len(instance['decision']['instances'])==1
        build=read(folder/'build/build.json')
        assert build['final_scene_complete'] and build['compiled_groups']==1
        ruler=read(folder/'build/merged/scale-reference.json')
        assert not ruler['metric_scale_verified'] and not ruler['ground_truth_used']
        if ruler['ruler_present']:
            ref=ruler['scale_reference'];assert ref['scale_basis']=='estimated_visual_prior'
            assert all(any(abs(t-f['timestamp_s'])<.011 for f in frames) for t in ref['source_timestamps_s'])
            ticks=ruler['ticks'];assert len(ticks)>=2 and ticks[0]['meters']==0
            for a,b in zip(ticks,ticks[1:]):
                distance=math.sqrt(sum((x-y)**2 for x,y in zip(a['world_position'],b['world_position'])))
                assert math.isclose(distance*ref['meters_per_scene_unit'],b['meters']-a['meters'],rel_tol=1e-4,abs_tol=1e-5)
            assert (folder/'build/merged/scene-with-ruler.blend').stat().st_size>1000
            from PIL import Image
            with Image.open(folder/'build/merged/reference-ruler.png') as im:
                assert im.size==(1400,1400)
        else:assert ruler['reason']
        rows.append({'id':q['id'],'scene':q['scene_name'],'cluster':g['group_id'],'frames':len(frames),
            'ruler_present':ruler['ruler_present'],'reference':ruler['scale_reference']['object'][1] if ruler['ruler_present'] else None})
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.urls=[]
        def handle_starttag(self,tag,attrs):
            for k,v in attrs:
                if k in ('src','href') and v and not v.startswith(('#','http:','https:')):self.urls.append(v)
    parser=Links();parser.feed((BASE/'index.html').read_text())
    for url in parser.urls:assert (BASE/unquote(urlsplit(url).path)).resolve().exists(),url
    report={'questions':19,'reconstructions_validated':len(rows),'rulers':sum(r['ruler_present'] for r in rows),
        'image_hashes_verified':True,'one_cluster_per_question':True,'one_target_per_question':True,
        'meter_tick_spacing_verified':True,'ground_truth_excluded_from_model_question_inputs':True,
        'report_local_links_checked':len(parser.urls),'rows':rows}
    (BASE/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
