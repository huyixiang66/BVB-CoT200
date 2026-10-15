"""Question-only adaptation of the release protocols to Luna scene assets."""

import re

from fixed_protocol import build_prompt, resolve_roles
from name_resolver import COMPOUND_BLOCK, SYN, head_term


def legacy_roles(qa):
    question = qa["question"]
    kind = qa["question_type"]
    if kind == "object_counting":
        match = re.search(r"How many (.+?)\(s\)", question)
        terms = [(match[1], "target category", "RED")] if match else []
    elif kind.startswith("object_rel_direction"):
        match = re.search(r"standing by the (.+?) and facing the (.+?), is (?:the )?(.+?) to (?:my|the)",
                          question)
        terms = list(zip(match.groups(), ("standing position", "facing target", "query object"),
                         ("GREEN", "BLUE", "RED"))) if match else []
    elif kind == "obj_appearance_order":
        match = re.search(r"video: (.+?)\?", question)
        names = [name.strip() for name in match[1].split(",")] if match else []
        terms = list(zip(names, ["queried category"] * 4, ("RED", "BLUE", "GREEN", "ORANGE")))
        if len(names) != 4:
            raise ValueError("Expected four appearance categories: " + qa["qa_id"])
    else:
        raise ValueError("Unsupported legacy question type")
    if not terms:
        raise ValueError("Cannot parse question roles: " + qa["qa_id"])
    return [{"term": term, "role": role, "color": color, "anchor": False}
            for term, role, color in terms]


def luna_recipe(qa, astra):
    if "spec" not in astra:
        return {"protocol": astra["protocol"], "scene": qa["scene_name"],
                "roles": astra["roles"], "views": ["original_camera"] * 16,
                "jpeg_quality": astra["jpeg_quality"], "timeline_source": "blend"}
    roles = legacy_roles(qa)
    kind = qa["question_type"]
    views = ({"trajectory": {"n": 24}} if kind == "obj_appearance_order" else
             {"plan": 1, "focus": 8, "plan_end": 1})
    if kind == "object_counting":
        views["oblique"] = 4
    hide = [word for word in ("ceiling", "roof")
            if not any(word in role["term"].lower() for role in roles)]
    spec = {"qa_id": qa["qa_id"], "scene": qa["scene_name"], "roles": roles,
            "question_type": kind, "hide": hide,
            "grid": {"enable": kind != "obj_appearance_order"}, "views": views,
            "scene_model": "luna"}
    return {"protocol": "legacy-v2-luna-name-match-single-answer",
            "scene": qa["scene_name"], "spec": spec, "jpeg_quality": 85}


def resolve_legacy_spec(spec, names):
    resolved = resolve_roles(spec["roles"], names, SYN, head_term, COMPOUND_BLOCK)
    views = dict(spec["views"])
    if spec["question_type"].startswith("object_rel_direction"):
        standing = next(r for r in resolved if "standing position" in r["roles"])
        facing = next(r for r in resolved if "facing target" in r["roles"])
        if standing["objects"] and facing["objects"]:
            views["startcam"] = {"at": standing["objects"], "look": facing["objects"], "eye": 1.25}
    return dict(spec, resolved_roles=resolved, views=views, exact_names=True,
                highlights=[{"color": r["color"].lower(), "match": r["objects"]}
                            for r in resolved if r["objects"]])


def legacy_views(kind, roles, timeline):
    highlighted = any(r["objects"] for r in roles)
    if kind == "obj_appearance_order":
        start, end = timeline
        if end < start:
            raise ValueError("Invalid scene timeline")
        n = len({round(start + (end - start) * i / 23) for i in range(24)})
        return [f"traj_{i:02d}" for i in range(n)]
    views = []
    if kind.startswith("object_rel_direction"):
        found = {label for r in roles if r["objects"] for label in r["roles"]}
        if {"standing position", "facing target"} <= found:
            views.append("startcam")
    views.append("plan")
    if kind == "object_counting":
        views += [f"oblique_{i}" for i in range(4)]
    if highlighted:
        views += [f"focus_{i}" for i in range(8)]
    return views + ["plan_end"]


def render_prompt(qa, roles, views):
    if qa["question_type"] not in {"object_counting", "obj_appearance_order"} and not \
            qa["question_type"].startswith("object_rel_direction"):
        return build_prompt(qa, "C", roles)
    descriptions = {
        "startcam": "standing-position view facing the named facing target",
        "plan": "top-down plan with a 1-metre grid", "plan_end": "the plan again for verification",
        "oblique": "oblique overview", "focus": "close-up of name-matched candidates",
        "traj": "original BVB camera, in chronological order",
    }
    layout = "\n".join(f"Frame {i}: {descriptions[view.split('_')[0] if view not in descriptions else view]}"
                       for i, view in enumerate(views, 1))
    legend = "\n".join(f"{r['color']}: {r['term']} ({', '.join(r['roles'])})"
                       for r in roles if r["objects"]) or "No reliable colour guide is available."
    parts = [
        "Answer a spatial question about a reconstructed indoor scene using the supplied frames.",
        f"The {len(views)} frames are ordered:\n" + layout,
        "Colour guide:\n" + legend,
        "Highlights are name-matched candidate parts, not verified instances. Uncoloured objects are "
        "not necessarily absent. Confirm object identity from appearance and context; do not count "
        "separate parts as separate instances. Highlight colours are not original object colours. "
        "Ground conclusions in visible evidence; do not invent missing views or measurements.",
        "Question: " + qa["question"],
    ]
    if qa.get("options"):
        parts.append("Options:\n" + "\n".join(qa["options"]))
    kind = qa["question_type"]
    if kind == "object_counting":
        steps = (
            "1. Divide the plan into regions anchored to furniture.\n"
            "2. List visible instances of the queried category by region and landmark, using overviews "
            "and any supplied close-ups. Inspect unhighlighted objects too.\n"
            "3. Merge repeat sightings of the same physical instance across frames; do not count parts twice.\n"
            "4. Verify each instance belongs to the requested category, not a look-alike.\n"
            "5. Count the merged list, recount, and resolve discrepancies using the frames.")
    elif kind.startswith("object_rel_direction"):
        steps = (
            "1. Establish forward from the named standing position toward the named facing target. "
            "Cross-check the plan with the standing-position view only if supplied.\n"
            "2. Compare the vector from the standing position to the query object against forward. "
            "Use the question's quadrant or direction definitions in the observer's coordinate frame, "
            "not screen-left or the walkthrough camera's direction.\n"
            "3. If facing in the opposite direction, flip forward by 180 degrees.\n"
            "4. Cross-check near-boundary bearings using the available views. Flag unresolved identity "
            "or facing instead of inventing a missing view.\n"
            "5. Map the resulting direction to the actual option text.")
    else:
        steps = (
            f"1. Scan all {len(views)} frames in order. Record each queried category's first visible "
            "frame, including partial visibility. Inspect unhighlighted objects too.\n"
            "2. Sort categories by first visibility; for ties prefer clearer visibility. An unseen "
            "category remains unresolved, not automatically last.\n"
            "3. Match the full category sequence to the option texts, not a fixed letter order.")
    parts += ["Steps:\n" + steps, "Give a brief evidence summary citing frame numbers.",
              "End with exactly one line: ANSWER: <answer>"]
    if qa.get("options"):
        parts.append("Map your conclusion to the actual option letter; letters are shuffled.")
    return "\n\n".join(parts) + "\n"
