"""Reuse the user's VSI-Bench extraction function without importing its Gemini client."""
import ast
import contextlib
import io
import json
import re
from pathlib import Path

VENDOR=Path(__file__).parent/'vendor'
source=(VENDOR/'vsi_question_parser_original.py').read_text()
# The original templates indent continuation lines, unlike dataset questions.
source=source.replace("question = doc['question']", "question = ' '.join(doc['question'].split())")
source=source.replace('re.escape(template)', "re.escape(' '.join(template.split()))")
source=source.replace('re.escape(OBJ_REL_DISTANCE_TEMPLATE)', "re.escape(' '.join(OBJ_REL_DISTANCE_TEMPLATE.split()))")
scope={};exec(compile(source,str(VENDOR/'vsi_question_parser_original.py'),'exec'),scope)

def extract_question(q):
    public={k:q[k] for k in ('id','scene_name','question_type','question','options') if k in q}
    parser_input=dict(public)
    if public['question_type']=='route_planning':
        parser_input['question']=re.sub(r'beginning by the ', 'beginning at the ',public['question'],flags=re.I)
        parser_input['question']=re.sub(r'Go forward until past the ', 'Go forward until the ',parser_input['question'],flags=re.I)
    with contextlib.redirect_stdout(io.StringIO()):
        try:roles=scope['extract_categories_of_interest'](parser_input)
        except UnboundLocalError as exc:raise ValueError(f'Unsupported question type: {public["question_type"]}') from exc
    if not isinstance(roles,dict):raise ValueError(f'Question did not match parser: {public.get("id")}')
    if public['question_type']=='route_planning' and 'start_position' in roles:
        match=re.fullmatch(r'(?:the )?(.+?) and facing (?:the )?(.+)',roles['start_position'],re.I)
        if match:roles.update(start_position=match[1],facing_object=match[2])
    objects=[]
    for value in roles.values():
        for obj in (value if isinstance(value,list) else [value]):
            obj=re.sub(r'^the\s+','',str(obj).lower().strip()).strip(' .?')
            if obj and obj not in objects:objects.append(obj)
    if not objects and public['question_type']!='room_size_estimation':
        raise ValueError(f'No objects parsed for question {public.get("id")}')
    return {**public,'roles':roles,'objects':objects,'context_only':not objects}

def prepare(path,scene=None):
    rows=[json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    qs=[extract_question(q) for q in rows if scene is None or q.get('scene_name')==scene]
    if not qs:raise ValueError('No matching questions')
    return {'questions':qs,'objects':sorted({o for q in qs for o in q['objects']}),
            'parser_provenance':json.loads((VENDOR/'provenance.json').read_text()),
            'ground_truth_used':False,'parser_uses_llm':False}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('questions');p.add_argument('output');p.add_argument('--scene');a=p.parse_args()
    plan=prepare(a.questions,a.scene);Path(a.output).write_text(json.dumps(plan,indent=2)+'\n')
    print(f'{len(plan["questions"])} questions; objects: {plan["objects"]}')
