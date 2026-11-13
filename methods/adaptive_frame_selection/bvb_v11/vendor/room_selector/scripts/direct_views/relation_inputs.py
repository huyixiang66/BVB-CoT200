"""Role labels and image-specific prompts; never calls a model or reads GT."""
import argparse
import json
from pathlib import Path


def role_palette(question):
    roles = question['roles']
    if question['question_type'].startswith('object_rel_direction'):
        return {roles['standing'][0]: (.9, .08, .06),
                roles['facing'][0]: (.06, .3, .9),
                roles['query'][0]: (.08, .72, .18)}
    if question['question_type'] == 'object_abs_distance':
        a, b = roles['endpoints']
        return {a: (.9, .08, .06), b: (.06, .3, .9)}
    return {}


def prepare_input(question, rendered, *, allow_overview_only=False):
    """Overview fallback is explicit and does not change geometric qa_ready."""
    views = rendered['views']
    roles = question['roles']
    if question['question_type'].startswith('object_rel_direction'):
        standing = [v for v in views if v.get('role') == 'standing_reference']
        overview = [v for v in views if v.get('role') == 'direction_layout']
        if len(overview) != 1 or len(standing) > 1:
            raise ValueError('Expected one accepted layout and at most one standing view')
        if not standing and not allow_overview_only:
            raise ValueError('Standing view missing; overview-only requires explicit approval')
        if standing and not rendered['qa_ready']:
            raise ValueError('Complete direction geometry did not pass')
        s, f, t = [roles[k][0] for k in ('standing', 'facing', 'query')]
        selected = standing + overview
        intro = (f'Image 1 is a view from beside the {s}, looking toward the {f}; '
                 f'it shows both the {f} and the {t}. Image 2 is an overview showing all three objects.'
                 if standing else 'The single image is an overview of the three objects. '
                 'No first-person standing view is provided.')
        prompt = intro + f' Red marks the {s} (standing object); blue marks the {f} '
        prompt += f'(facing object); green marks the {t} (queried object). '
        prompt += ('Judge direction relative to the specified standing position and heading, '
                   'not screen left/right in the overview. Reply with only the option letter.')
        mode = 'standing_and_overview' if standing else 'overview_only_explicit_fallback'
    elif question['question_type'] == 'object_abs_distance':
        if not rendered['qa_ready'] or not views:
            raise ValueError('Distance geometry did not pass')
        selected = views
        a, b = roles['endpoints']
        prompt = (f'Red marks the {a}; blue marks the {b}. The images show both endpoints '
                  'in the same scene. Estimate the distance requested in the question, '
                  'using its stated definition and units. Do not treat projected pixel '
                  'distance as metric distance. Reply with a numerical answer.')
        mode = 'joint_endpoints'
    else:
        raise ValueError('Only absolute distance and relative direction are supported')
    if any(not v.get('geometry_accepted', False) for v in selected):
        raise ValueError('Diagnostic views cannot enter model inputs')
    prompt += '\nColors are temporary identity labels.\nQuestion: ' + question['question']
    if question.get('options'):
        prompt += '\nOptions:\n' + '\n'.join(question['options'])
    return {'qa_id': str(question['qa_id']), 'mode': mode,
            'geometry_qa_ready': rendered['qa_ready'],
            'images': [{'file': v['file'], 'sha256': v['sha256'], 'role': v.get('role')}
                       for v in selected], 'prompt': prompt, 'qa_run': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--question', type=Path, required=True)
    parser.add_argument('--renders', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--allow-overview-only', action='store_true')
    args = parser.parse_args()
    question = json.loads(args.question.read_text(encoding='utf-8'))
    manifest = json.loads(args.renders.read_text(encoding='utf-8'))
    if not manifest.get('role_colors'):
        raise ValueError('Render with --role-colors before using role-color prompts')
    records = manifest['questions']
    rendered = next(r for r in records if str(r['qa_id']) == str(question['qa_id']))
    result = prepare_input(question, rendered, allow_overview_only=args.allow_overview_only)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
