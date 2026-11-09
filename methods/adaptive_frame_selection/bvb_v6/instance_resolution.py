"""Question-wide instance adjudication, with immutable image/spec provenance.

Local regions may reconstruct all question categories. Only this final stage
limits instances; supporting observations and original region files are retained.
"""
import argparse
import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import time

from .question_policy import question_policy


def read(p):
    return json.loads(Path(p).read_text())


def save(p, value):
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def instance_policy(question):
    policy = question_policy(question)
    objects = list(dict.fromkeys(question['objects']))
    limits = {o: 1 if policy['category']=='single' else None for o in objects}
    # An explicit plural/repeated role must not be collapsed merely by category.
    text = str(question.get('question', '')).lower()
    plural = re.search(r'\b(two|three|four|both|multiple|all|each)\s+\w+(?:\s+\w+)?s\b', text)
    roles = [v.lower().strip() for v in question.get('roles', {}).values() if isinstance(v,str)]
    ambiguous = bool(plural or len(roles)!=len(set(roles)))
    if ambiguous:
        limits = dict.fromkeys(objects, None)
    overrides = question.get('instance_limits', {})
    if set(overrides)-set(objects):
        raise ValueError('instance_limits references unknown category')
    for obj, value in overrides.items():
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError('Instance limit must be a positive integer or null')
        limits[obj] = value
    return {'scope':'final_question_scene', 'limits':limits,
            'question_policy':policy, 'ambiguous_referents':ambiguous,
            'minimum_when_candidates_exist':1, 'limit_observations':False}


def local_instruction(question, primary):
    policy = instance_policy(question)
    return ('\nQUESTION-WIDE TARGET SCOPE:\n'+json.dumps({
        'all_question_objects':question['objects'], 'primary_objects':primary,
        'final_instance_limits':policy['limits']}, ensure_ascii=False)+
        '\nPrioritize the primary objects, but actively inspect these same clean images for ALL '
        'question-relevant categories, including option objects. Reconstruct other visible relevant '
        'objects as labeled targets, not white context boxes. This does not require a detector peak '
        'or a detector hint for the additional object. Never invent an unseen object to fill the list. '
        'Repeated views of one physical object are one local target. For a category with final limit 1, '
        'prefer the question referent if distinguishable; if several distinct candidates remain '
        'ambiguous, retain those observed alternatives for the final question-wide comparison. '
        'The final limit applies across ALL regions, not one new object per region. '
        'Do not merge visibly distinct objects simply to satisfy a limit. Semantic review of the '
        'primary object does not certify newly observed categories.\n')


def make_plan(source):
    source = Path(source)
    manifest = read(source/'manifest.json')
    question = manifest['question']
    policy = manifest.get('final_instance_policy') or instance_policy(question)
    candidates, frames, inputs, missing_groups, cameras = [], {}, {}, [], {}
    for group in manifest['groups']:
        folder = source/group['folder']
        spec_path, audit_path = folder/'assets/scene-spec.json', folder/'assets/question-audit.json'
        if not spec_path.exists() or not audit_path.exists():
            missing_groups.append(group['group_id'])
            continue
        # Transport overrides are deliberately disallowed: the plan hashes exactly what build uses.
        if (folder/'scene-transport.json').exists():
            raise ValueError('Resolve scene-transport into scene-spec/audit before instance adjudication')
        spec, audit = read(spec_path), read(audit_path)
        if manifest.get('context_scope')=='direct_physical_support_only':
            target_ids={t['id'] for t in audit['targets']}
            relations=audit.get('context_relations',[])
            if {r['id'] for r in relations} != set(audit.get('background',[])):
                raise ValueError('Every white box requires a direct support audit')
            if any(r.get('relation')!='support' or not r.get('targets') or
                   not set(r['targets']) <= target_ids for r in relations):
                raise ValueError('Only direct physical support context may enter final integration')
        objects = {o['id']:o for o in spec['objects']}
        rows = read(folder/'input-frames.json')
        cameras[group['group_id']] = read(folder/'pose-reference-vlm.json')
        inputs[group['group_id']] = {name:file_sha(folder/name) for name in
            ['assets/scene-spec.json','assets/question-audit.json','input-frames.json','pose-reference.json','pose-reference-vlm.json','question.json','system-prompt.txt']}
        for f in rows:
            if file_sha(f['path']) != f['sha256']:
                raise ValueError('Input image changed')
            key = f['sha256']
            frames.setdefault(key, {'path':f['path'],'sha256':key,'timestamp_s':f['timestamp_s'],
                                    'source_frame_index':f['source_frame_index'],'groups':[]})
            if group['group_id'] not in frames[key]['groups']:
                frames[key]['groups'].append(group['group_id'])
        for t in audit['targets']:
            if t['label'] not in question['objects'] or t['id'] not in objects:
                raise ValueError('Target audit is inconsistent with question/spec')
            candidates.append({'candidate_id':group['group_id']+'::'+t['id'],
                'group_id':group['group_id'],'target_id':t['id'],'label':t['label'],
                'geometry':objects[t['id']], 'frames':[r['sha256'] for r in rows]})
    if missing_groups:
        raise ValueError('Finish local reconstructions before adjudication: '+', '.join(missing_groups))
    payload = {'version':1,'question':{k:question[k] for k in
        ('id','question','question_type','options','roles','objects') if k in question},
        'policy':policy,'candidates':candidates,'frames':list(frames.values()),
        'inputs':inputs,'cameras':cameras,'manifest_sha256':file_sha(source/'manifest.json'),
        'adjudication_prompt_sha256':digest(PROMPT)}
    payload['request_hash'] = digest(payload)
    return payload


PROMPT = '''Adjudicate question-relevant physical instances across reconstructed regions.
Use the supplied CLEAN original video frames, public question and approximate shared-world geometry.
Do not answer the benchmark question. Do not use detector score alone as identity evidence.
Several candidate IDs may be observations/reconstructions of the SAME object from different views.
Return JSON only: {"instances":[{"label":"...","representative":"candidate ID",
"members":["candidate ID"],"status":"confirmed|provisional","reason":"visual evidence"}],
"excluded":[{"candidate_id":"...","reason":"wrong referent/false positive evidence"}]}.
Account for EVERY candidate exactly once, as an instance member or excluded. A representative
must be one of that instance's members and supplies its existing geometry; never invent geometry.
Merge only when images support the same physical instance. Keep distinct instances for counting
or unconstrained categories. For a final limit of 1, choose exactly one question referent, not
one per region. Keep its supporting views as members. Exclude other referents with explanations.
If a category has any candidates, keep at least one; if evidence is ambiguous keep the best as
provisional and explain uncertainty. If no candidates exist, do not invent a missing target.
Absence of a detector box is not evidence that an object is absent. Geometric overlap alone
does not establish identity. Compare appearance, scene context, timestamps and camera evidence.
'''


def validate_decision(plan, decision):
    if not isinstance(decision, dict) or set(decision) != {'instances','excluded'}:
        raise ValueError('Expected instances and excluded')
    available = {c['candidate_id']:c for c in plan['candidates']}
    if len(available)!=len(plan['candidates']):
        raise ValueError('Duplicate candidate IDs')
    seen, counts, kept = set(), {}, set()
    for inst in decision['instances']:
        members = inst.get('members')
        if not isinstance(members,list) or not members or inst.get('representative') not in members:
            raise ValueError('Invalid instance members/representative')
        if inst.get('status') not in ('confirmed','provisional') or not str(inst.get('reason','')).strip():
            raise ValueError('Instance requires evidence/status')
        for member in members:
            if member not in available or member in seen or available[member]['label']!=inst['label']:
                raise ValueError('Unknown, duplicate or cross-category member')
            seen.add(member)
        counts[inst['label']] = counts.get(inst['label'],0)+1
        kept.add(inst['representative'])
    for row in decision['excluded']:
        member = row.get('candidate_id')
        if member not in available or member in seen or not str(row.get('reason','')).strip():
            raise ValueError('Invalid excluded candidate')
        seen.add(member)
    if seen != set(available):
        raise ValueError('Decision must account for every candidate')
    for label in {c['label'] for c in available.values()}:
        if counts.get(label,0)<1:
            raise ValueError('Category floor violated: '+label)
        cap = plan['policy']['limits'][label]
        if cap is not None and counts[label]>cap:
            raise ValueError('Final instance limit violated: '+label)
    return kept


def resolve_for_build(source):
    plan = make_plan(source)
    path = Path(source)/'instance-resolution/result.json'
    if not path.exists():
        raise ValueError('Run resolve-instances before final build; no silent winner selection')
    record = read(path)
    if record.get('status')!='validated' or record.get('request_hash')!=plan['request_hash']:
        raise ValueError('Instance resolution missing, failed or stale')
    kept = validate_decision(plan, record['decision'])
    return plan, record, kept


def filter_region(spec, audit, gid, kept):
    """Retain one representative geometry; keep source files and all frame evidence untouched."""
    spec, audit = copy.deepcopy(spec), copy.deepcopy(audit)
    old_targets = {t['id'] for t in audit['targets']}
    retained = {t for t in old_targets if gid+'::'+t in kept}
    removed = old_targets-retained
    relations=[]
    for rel in audit.get('context_relations',[]):
        rel['targets'] = [t for t in rel['targets'] if t in retained]
        if rel['targets']:
            relations.append(rel)
        else:
            removed.add(rel['id'])
    spec['objects'] = [o for o in spec['objects'] if o['id'] not in removed]
    audit['targets'] = [t for t in audit['targets'] if t['id'] in retained]
    audit['context_relations'] = relations
    audit['background'] = [i for i in audit.get('background',[]) if i not in removed]
    audit['missing_keywords'] = sorted(set(audit['keywords'])-{t['label'] for t in audit['targets']})
    audit['instance_filter'] = {'removed_geometry_ids':sorted(removed),'source_evidence_preserved':True}
    return spec, audit


def run(source, execute=False, env_file=None, model='gpt-6-astra', provider=None):
    source = Path(source).resolve()
    plan = make_plan(source)
    out = source/'instance-resolution';out.mkdir(exist_ok=True)
    if (out/'result.json').exists():
        resolve_for_build(source)
        return read(out/'result.json')
    save(out/'input.json',plan)
    (out/'prompt.txt').write_text(PROMPT)
    if not execute:
        return {'status':'planned','candidates':len(plan['candidates']),'images':len(plan['frames'])}
    from openai import OpenAI
    from .api_provider import resolve
    connection=resolve(provider,env_file)
    ledger = {'status':'started','request_hash':plan['request_hash'],'model':model,
              'reasoning_effort':'high','max_completion_tokens':None,'started_unix':time.time(),**connection.metadata()}
    # Exclusive ledger prevents duplicate billing on accidental retries.
    with (out/'request.json').open('x') as f:
        json.dump(ledger,f)
    content=[{'type':'text','text':PROMPT+'\n'+json.dumps({k:v for k,v in plan.items() if k not in ('frames','inputs')},ensure_ascii=False)}]
    for frame in plan['frames']:
        content += [{'type':'text','text':json.dumps({k:v for k,v in frame.items() if k!='path'})},
                    {'type':'image_url','image_url':{'detail':'high','url':'data:image/jpeg;base64,'+base64.b64encode(Path(frame['path']).read_bytes()).decode()}}]
    try:
        from .api_provider import create_client
        client=create_client(connection,timeout=600)
        response=client.chat.completions.create(model=model,reasoning_effort='high',
            messages=[{'role':'user','content':content}])
        reply=response.choices[0].message.content or ''
        (out/'response.txt').write_text(reply)
        ledger.update(status='received',usage=response.usage.model_dump() if response.usage else None,
                      response_id=response.id,finish_reason=response.choices[0].finish_reason)
        if reply.strip().startswith('```'):
            reply=reply.strip().split('\n',1)[1].rsplit('```',1)[0]
        decision=json.loads(reply)
        validate_decision(plan,decision)
        record={'status':'validated','request_hash':plan['request_hash'],'decision':decision,
                'missing_categories':sorted(set(plan['question']['objects'])-{c['label'] for c in plan['candidates']})}
        save(out/'result.json',record)
        ledger['status']='validated'
        return record
    except Exception as exc:
        ledger.update(status='failed',error_type=type(exc).__name__)
        raise
    finally:
        save(out/'request.json',ledger)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--env-file',type=Path)
    p.add_argument('--model',default='gpt-6-astra')
    p.add_argument('--provider',choices=['openai','boyue','subscription','subscription_direct'])
    a=p.parse_args()
    print(json.dumps(run(a.source,a.execute,a.env_file,a.model,a.provider),ensure_ascii=False))


if __name__=='__main__':
    main()
