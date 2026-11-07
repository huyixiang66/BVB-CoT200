"""Pure rules for the size prototype; measurements are in scene units."""
import json
import math
from pathlib import Path


def load_priors(path=None):
    path = Path(path) if path else Path(__file__).with_name('priors.json')
    return json.loads(path.read_text(encoding='utf-8'))


def choose_reference(target, candidates, priors):
    """Return (candidate, reason); unknown or unverified inputs fail closed.

    Each object has id, category, extent (three scene-unit dimensions), and
    verified (confirmed complete and belonging to the scene). Instances must
    already share a reliable scale; this routine does not establish one.
    """
    category = target['category'].strip().lower()
    if category in priors['reliable']:
        return None, 'target_prior_reliable'
    if category not in priors['unreliable']:
        return None, 'unknown_target_category'
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
