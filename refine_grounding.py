"""Validate source-anchored correspondences without trusting model source coordinates."""

import refine_measurement as m

PROTOCOL = "geometry_anchored_correspondence_v1"
POLICY = {"max_target_points": 6, "max_background_points": 12,
          "min_point_spacing": 0.03, "max_background_distance": 0.35}


def observation(response, proposal):
    """Only original coordinates are inferred; source identity/coordinates are frozen."""
    result = {"same_physical_target": response.get("same_physical_target") is True,
              "features": [], "landmarks": [], "reason": response.get("reason", "")}
    available = {p["id"]: (role, p) for role in ("features", "landmarks") for p in proposal[role]}
    if len(available) != sum(len(proposal[role]) for role in ("features", "landmarks")):
        raise ValueError("Duplicate source anchor IDs")
    matches = response.get("matches")
    if not isinstance(matches, list):
        raise ValueError("Missing matches")
    seen = set()
    for match in matches:
        if not isinstance(match, dict) or match.get("id") not in available or match["id"] in seen:
            raise ValueError("Unknown or duplicate anchor ID")
        seen.add(match["id"])
        if match.get("same_physical_point") is not True or match.get("original") is None:
            continue
        if not m.coordinates(match["original"], 2):
            raise ValueError("Invalid original coordinate")
        role, anchor = available[match["id"]]
        result[role].append({"id": anchor["id"], "baseline": anchor["xy"],
                             "original": match["original"], "same_physical_point": True})
    return result


def prompt(label, proposal):
    import json
    catalog = {role: [{"id": p["id"], "mesh_name": p["object_name"],
                      "baseline_xy": p["xy"]} for p in proposal[role]]
               for role in ("features", "landmarks")}
    return (
        f"Match visible local surface points of the {label} between ORIGINAL and BASELINE. "
        "The attached catalog fixes actual visible BASELINE surface points and their source mesh names. "
        "Mesh names are hints, not proof of identity. Inspect the exact listed baseline coordinate; "
        "do NOT replace it with a nearby corner or a different surface. Find the SAME physical detail "
        "in ORIGINAL. If that exact point is not distinctive, ambiguous, hidden, or absent, omit it. "
        "Partial visibility is allowed; never infer hidden geometry or force matches. "
        "Background points must be stationary and not part of the moving target or accessories. "
        "Do not match a different same-category instance. Only return ORIGINAL normalized [0,1] "
        "coordinates, TOP LEFT origin. Do not output baseline coordinates or change catalog IDs. "
        "Do not estimate distance, metric size, camera correction or a desired modification. "
        'Return JSON: {"same_physical_target":false,"matches":[{"id":"catalog ID",'
        '"original":[0.2,0.3],"same_physical_point":true}],"reason":"uncertainty"}. '
        "Return fewer matches or none when necessary. Catalog: " + json.dumps(catalog)
    )


def background_observation(response, proposal):
    """A background-only task must account for every frozen anchor, including rejection."""
    available = {p["id"]: p for p in proposal["landmarks"]}
    if len(available) != len(proposal["landmarks"]):
        raise ValueError("Duplicate source background ID")
    matches = response.get("matches")
    if not isinstance(matches, list):
        raise ValueError("Missing background decisions")
    seen, result = set(), []
    for match in matches:
        if not isinstance(match, dict) or not isinstance(match.get("id"), str):
            raise ValueError("Invalid background decision")
        identity = match["id"]
        if identity not in available or identity in seen:
            raise ValueError("Unknown or duplicate background ID")
        seen.add(identity)
        status = match.get("status")
        if (status not in ("matched", "unmatchable", "target_or_accessory")
                or not isinstance(match.get("reason"), str) or not match["reason"].strip()):
            raise ValueError("Missing explicit background decision")
        if status == "matched":
            if not m.coordinates(match.get("original"), 2):
                raise ValueError("Invalid background coordinate")
            result.append({"id": identity, "baseline": available[identity]["xy"],
                           "original": match["original"]})
    if seen != set(available):
        raise ValueError("Incomplete background decisions")
    return result


def background_prompt(label, proposal):
    import json
    catalog = [{"id": p["id"], "mesh_name": p["object_name"], "baseline_xy": p["xy"]}
               for p in proposal["landmarks"]]
    for entry, point in zip(catalog, proposal["landmarks"]):
        if "point_description" in point:
            entry["geometry_hint"] = point["point_description"]
    return (
        "Your ONLY task is stationary BACKGROUND point correspondence between ORIGINAL and BASELINE. "
        f"The moving target is {label}; exclude it and anything attached to or resting on it. "
        "Do not measure or locate the target. The catalog fixes visible BASELINE coordinates. "
        "Mesh names are hints, not verified identity. Inspect the exact coordinate, not a nearby corner. "
        "For every catalog ID return exactly one decision: matched (same identifiable physical point), "
        "unmatchable (uncertain, absent, occluded, or indistinct), or target_or_accessory. "
        "For matched points give ONLY ORIGINAL normalized [0,1] coordinates, TOP LEFT origin. "
        "Never invent a correspondence, infer hidden geometry, or substitute a similar instance. "
        "Every decision needs a short reason describing the detail or why it cannot be matched. "
        "No metric distances, QA answer, camera correction, or recommended motion. "
        'Return JSON {"matches":[{"id":"catalog ID","status":"unmatchable",'
        '"original":null,"reason":"brief"}]}. Empty matches is invalid when the catalog is nonempty; '
        "reject individual points explicitly instead. Catalog: " + json.dumps(catalog)
    )
