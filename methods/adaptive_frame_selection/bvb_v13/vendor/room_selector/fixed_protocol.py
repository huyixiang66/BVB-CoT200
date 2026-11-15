"""Frozen fixed-camera C protocol; extracted without sampling/aggregation code."""
import re

TYPES = (
    "object_abs_distance",
    "room_size_estimation",
    "route_planning",
    "object_rel_distance",
    "object_size_estimation",
)

ARMS = ("A", "B", "C", "D")

VERSION = "fixed-camera-v1"

COLORS = {
    "RED": (0.85, 0.025, 0.025, 1.0),
    "BLUE": (0.025, 0.15, 0.9, 1.0),
    "GREEN": (0.025, 0.7, 0.07, 1.0),
    "ORANGE": (1.0, 0.3, 0.015, 1.0),
    "PURPLE": (0.55, 0.025, 0.85, 1.0),
    "YELLOW": (0.95, 0.85, 0.02, 1.0),
    "CYAN": (0.01, 0.7, 0.8, 1.0),
}

STEPS = {
    "room_size_estimation": """1. Establish the target floor area from the rooms covered by the walkthrough and the question. An overhead overview, if supplied, may include additional rooms or outdoor ground: those are not automatically part of the answer. Do not use the image border as the room boundary.
2. Use a clearly identifiable table as the primary horizontal ruler when one is visible. Identify its subtype (dining table, desk, coffee table, etc.), and state a plausible approximate tabletop LENGTH and WIDTH. These are uncertain real-world size assumptions, not provided measurements. Do not assign one universal size to every table. If no usable table is visible, explicitly select another visible fixture as the ruler.
3. Estimate room length in table-length units and room width in table-width units, taking their orientations into account. In perspective views, do not directly equate tabletop pixels to floor pixels: they can differ in depth and height. Prefer approximately comparable depth, cross-check other original views, and use the overhead view for horizontal proportions if available.
4. Convert these two ratios to metres and multiply. Split an L-shaped or multi-room target into non-overlapping simple regions and add their areas. Include the floor under furniture; do not subtract furniture footprints or add an unseen room merely because it is connected.
5. Check the scale using a second clearly visible reference, such as a door or bed. If estimates disagree, revisit the assumed reference size or the included floor boundary instead of averaging incompatible estimates. State the approximate length and width supporting the final area.""",
    "object_abs_distance": """1. Identify the two complete physical objects named by the question across the original views. Distinguish repeated instances using the question's spatial qualifiers. A highlighted part alone does not define the whole object.
2. Locate the nearest surfaces of the two objects, not their centres. Prefer an original view containing both objects and a useful size reference; otherwise combine consistent views into one spatial layout. Image-space separation alone is not metric distance.
3. Use a visible reference such as a table, door or counter to estimate the surface gap in reference-length units. State the reference's approximate physical size as an assumption, then convert the gap to metres. Account for perspective and do not compare raw pixel lengths at different depths.
4. Check the vertical relationship in the original views. A top-down gap, if available, only constrains horizontal separation. If nearest surfaces have a height offset, include it in the three-dimensional distance estimate; do not automatically set it to zero just because both objects touch the floor.
5. Cross-check the same instance pair in another original view. Report the shortest straight surface-to-surface distance, not a centre distance or a walking distance around obstacles.""",
    "object_rel_distance": """1. Read whether the question asks for the closest or farthest candidate. Identify the reference and each candidate by name and spatial qualifiers; preserve physical instance identity across views.
2. Establish a single consistent spatial layout from the original views and the overhead overview if supplied. Compare nearest surface-to-surface gaps, including a clear height offset when relevant. Do not compare raw pixel gaps measured in different frames.
3. Rank candidates qualitatively first: clearly nearer, unresolved, clearly farther. Avoid inventing precise metre values when only ordering is needed. A candidate that is not visible or not highlighted is unresolved, not automatically far away or absent.
4. Compare the two contenders most relevant to the requested extreme, using a common view or their positions in the same layout. State the evidence for the ordering and flag a near tie rather than claiming unsupported precision.
5. Select the requested closest or farthest candidate, then match its NAME to the actual option letter. Never infer the letter from a fixed colour order.""",
    "object_size_estimation": """1. Identify one complete instance of the requested object. Separate it from neighbouring objects, trim, wall sections and other same-category instances. Multiple disconnected regions of one highlight colour must not be merged into one giant bounding box.
2. Estimate WIDTH, HEIGHT and DEPTH separately using the most informative original views. An overhead overview, if supplied, can help with horizontal dimensions but is not evidence of height. Do not treat the longest projected image segment as the longest physical dimension.
3. Calibrate using a different clearly visible object or fixture at a comparable depth, such as a table or door. State its approximate size as an uncertain assumption. Correct for perspective by comparing several views; if a dimension is hidden, label it estimated rather than measured.
4. Take the largest of the three estimated physical dimensions and convert metres to centimetres. Do not assume height is always the maximum.
5. Check the target boundary again. Typical target sizes are only a sanity check, never a replacement for the observed ratio and the complete-instance boundary.""",
    "route_planning": """1. Read the complete route and identify the starting location, the initial facing direction, every ordered landmark instruction and every blank. The walkthrough CAMERA is an observer, NOT the hypothetical walker in this question; its motion and screen-left do not define the walker's route or left.
2. Build a consistent layout of the landmarks and traversable space from the original views, using the overhead overview if available. Follow the QUESTION'S landmark order, not their order of appearance in the video. Preserve qualifiers such as 'past', 'before', 'edge' and 'corner'; a stopping position need not be an object's centre.
3. At each blank, track: current stopping position; current heading from the previous instructed leg (or the stated initial facing); direction of the next instructed leg. Determine the required turn in this walker's coordinate frame, then update the heading before processing the next blank.
4. Check the proposed movement against visible walls and openings. Do not connect object centres through walls. Do not silently select the nearest same-category instance or replace an unclear initial facing with the direction to the first landmark.
5. If identity or facing is ambiguous, state that uncertainty briefly and choose the option best supported by the full route. Starting by X and facing X is not automatically impossible: the walker can stand beside the object. Compare the COMPLETE ordered turn sequence with the option texts, then return its actual letter.""",
}

def bvb_sample_frames(start: int, end: int, video_count: int, n: int = 16) -> list[int]:
    """Reproduce official 64-frame rounding, then judge's truncating subsample."""
    span = end - start + 1
    if span <= 0 or video_count <= 0:
        raise ValueError("Invalid timeline or video frame count")
    if span <= 64:
        rendered = list(range(start, end + 1))
    else:
        rendered = [start + round(i * (span - 1) / 63) for i in range(64)]
    if len(rendered) != video_count:
        raise ValueError(f"Official sample count {len(rendered)} != video count {video_count}")
    return [rendered[int(i * (video_count - 1) / max(n - 1, 1))] for i in range(n)]

def word_match(name: str, term: str) -> bool:
    return bool(re.search(r"(?<![a-z0-9])" + re.escape(term.lower()) + r"(?:s|es)?(?![a-z])",
                          name.lower()))

def clean_term(value: str) -> str:
    return re.sub(r"^(?:the|a|an)\s+", "", value.strip(), flags=re.I).strip(" .")

def resolve_roles(roles: list[dict], names: list[str], synonyms: dict,
                  head_term, compound_block: dict) -> list[dict]:
    """Conservative name candidates, without GT, nearest-instance selection or bbox unions."""
    result = []
    by_term: dict[str, dict] = {}
    for role in roles:
        term = role["term"]
        normalized = clean_term(term).lower()
        if normalized in by_term:
            by_term[normalized]["roles"].append(role["role"])
            continue
        head = head_term(normalized).lower() or normalized
        # Unlike the legacy resolver, never fall back from table to counter/desk/island.
        keywords = [normalized, head] + list(synonyms.get(head, [[]])[0])
        if head == "table":
            keywords = ["table"]
        if head in {"room", "space", "area", "hallway", "corridor"}:
            keywords = [normalized] if len(normalized.split()) > 1 else []
        matches = [name for name in names if any(word_match(name, k) for k in keywords if k)]
        if not any(word_match(normalized, w) for w in ("wall", "floor", "ceiling", "roof")):
            matches = [name for name in matches
                       if not any(word_match(name, w) for w in ("wall", "floor", "ceiling", "roof"))]
        if head in compound_block:
            matches = [name for name in matches
                       if not any(word_match(name, w) for w in compound_block[head])]
        entry = dict(role, roles=[role["role"]], objects=sorted(matches),
                     status="name_candidates" if matches else "unresolved", conflicts=[])
        by_term[normalized] = entry
        result.append(entry)
    owners: dict[str, list[dict]] = {}
    for entry in result:
        for name in entry["objects"]:
            owners.setdefault(name, []).append(entry)
    for name, entries in owners.items():
        if len(entries) > 1:
            # A generic scale anchor must never steal a queried object's material.
            non_anchors = [entry for entry in entries if not entry["anchor"]]
            excluded = ([entry for entry in entries if entry["anchor"]]
                        if len(non_anchors) == 1 else entries)
            for entry in excluded:
                entry["objects"].remove(name)
                entry["conflicts"].append(name)
    for entry in result:
        if not entry["objects"]:
            entry["status"] = "unresolved"
    return result

def build_prompt(qa: dict, arm: str, resolved: list[dict] | None = None) -> str:
    if arm not in ARMS or qa["question_type"] not in TYPES:
        raise ValueError("Unsupported arm or question type")
    layout = ("Frames 1-16 are in chronological order from the BVB scene's ORIGINAL camera "
              "and fixed sampling times. They are observations, not the question's hypothetical route.")
    if arm == "D":
        layout += (" Frame 17 is ONE additional orthographic overhead overview. It is an auxiliary "
                   "map, NOT the next video moment. Its roof may be hidden; the first 16 frames retain "
                   "their original occlusion. There is no known metric scale printed on this map.")
    else:
        layout += " No overhead, elevation, close-up or arrival-view panel is supplied."
    parts = ["Answer a spatial question about the reconstructed indoor scene using the supplied frames.",
             layout]
    if arm in {"C", "D"}:
        legend = []
        for entry in resolved or []:
            if entry.get("objects"):
                legend.append(f"{entry['color']}: {entry['term']} ({', '.join(entry['roles'])})")
        parts.append("Colour guide:\n" + ("\n".join(legend) if legend else "No reliable colour guide is available."))
        parts.append("Highlights are name-matched candidate parts, not verified instance IDs or answers. "
                     "One colour can occur on several physical instances. Confirm the question's target "
                     "from appearance and context. Uncoloured objects are not necessarily absent. "
                     "Highlight colours replace original appearance colours; do not treat them as "
                     "the original colour adjectives in the question.")
    options = qa.get("options") or []
    parts.append("Question: " + qa["question"])
    if options:
        parts.append("Options:\n" + "\n".join(map(str, options)))
    if arm != "A":
        parts.append("Reasoning procedure:\n" + STEPS[qa["question_type"]])
        parts.append("Give a brief evidence summary (at most 120 words), citing useful frame numbers. "
                     "Distinguish what is visible from uncertain size assumptions. Do not invent "
                     "measurements, unseen instances or missing views.")
    else:
        parts.append("Give only the answer, without an explanation.")
    answer_format = {
        "object_abs_distance": "<number in metres, one decimal>",
        "room_size_estimation": "<number in square metres, one decimal>",
        "object_size_estimation": "<integer in centimetres>",
        "object_rel_distance": "<option letter>",
        "route_planning": "<option letter>",
    }[qa["question_type"]]
    parts.append("End with exactly one line: ANSWER: " + answer_format)
    if options:
        parts.append("Map the conclusion to the actual option text; letters are not in a fixed order.")
    return "\n\n".join(parts) + "\n"

def extract_answer(text: str) -> str:
    matches = re.findall(r"(?im)^\s*ANSWER:\s*(.+)$", text or "")
    return matches[-1].strip() if matches else ""
