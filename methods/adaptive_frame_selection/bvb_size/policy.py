"""Pure rules for the size prototype; measurements are in scene units."""
import json
import math
from pathlib import Path


def load_priors(path=None):
    path = Path(path) if path else Path(__file__).with_name('priors.json')
    return json.loads(path.read_text(encoding='utf-8'))


def reference_requirement(category, priors):
    """Decide target policy independently of candidate availability."""
    category = category.strip().lower()
    rules = json.loads(Path(__file__).with_name('target_reference_policy.json').read_text(encoding='utf-8'))['categories']
    if category in rules:
        return rules[category]['use_reference'], 'paired20'
    if category in priors['reliable']:
        return False, 'target_prior_reliable'
    if category in priors['unreliable']:
        return True, 'target_prior_unreliable'
    return False, 'unknown_target_category'


def choose_reference(target, candidates, priors):
    """Return (candidate, reason); unknown or unverified inputs fail closed.

    Each object has id, category, extent (three scene-unit dimensions), and
    verified (confirmed complete and belonging to the scene). Instances must
    already share a reliable scale; this routine does not establish one.
    """
    category = target['category'].strip().lower()
    required, reason = reference_requirement(category, priors)
    if not required:
        return None, 'paired20_target_only' if reason == 'paired20' else reason
    target_size = max(target['extent'])
    if target_size <= 0:
        return None, 'invalid_target_extent'
    ranked = []
    for item in candidates:
        label = item['category'].strip().lower()
        if (not item.get('verified') or item['id'] == target['id'] or
                label not in priors['reliable'] or label in priors['poor_reference']):
            continue
        size = max(item['extent'])
        ratio = size / target_size
        if size > 0 and 0.25 <= ratio <= 4.0:
            ranked.append((abs(math.log(ratio)), str(item['id']), item))
    if not ranked:
        return None, 'no_eligible_reference'
    ranked.sort(key=lambda x: (x[0], x[1]))
    return ranked[0][2], 'reference_selected'


def needs_detail(max_pixel_extent, axis_retention, min_pixels=200, min_axis=0.8):
    """Heuristic only: geometry cannot guarantee semantic recognizability."""
    return max_pixel_extent < min_pixels or axis_retention < min_axis


def reference_prompt(question, target, reference):
    base = 'Answer the question using any provided images. Give only the numeric answer in centimeters. Question: ' + question
    if reference:
        base += f' The object on the left is the {target}; the object on the right is the {reference}, which is the reference object.'
    return base
