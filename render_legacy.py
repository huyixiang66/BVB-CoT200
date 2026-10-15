# -*- coding: utf-8 -*-
"""Render per-question QA frames from a BVB blend (v2).

Usage:
  blender -b scene.blend --python render_qa_frames.py -- --spec spec.json

Spec JSON:
{
  "qa_id": "44", "scene": "42898486",
  "out_dir": "tmp/bvb/pilot/renders/42898486/44",
  "res": 512, "frame": 1,
  "hide": ["ceiling", "roof"], "hide_exempt": ["diffuser", "fixture"],
  "grid": {"enable": true, "bbox": null, "z": null},
  "highlights": [{"color": "red", "match": ["Upholstered base"]}],
  "views": {
    "startcam": {"at": [...], "look": [...], "eye": 1.2, "standoff": 0.0, "lens": 32},
    "plan": 1, "oblique": 4, "focus": 8, "plan_end": 1,
    "elev": {"front": 1, "side": 1, "iso": true},
    "arrive": [{"at": [...], "face_dir": [1, 0], "standoff": 0.8, "eye": 1.2}],
    "trajectory": {"n": 24}
  }
}
Shot order: startcam -> plan -> oblique -> elev -> focus -> arrive -> trajectory -> plan_end
"""
import bpy, sys, json, math, time
from pathlib import Path
from mathutils import Vector

def arg(name, default=None):
    argv = sys.argv
    if "--" not in argv:
        return default
    args = argv[argv.index("--") + 1:]
    for i, a in enumerate(args):
        if a == name and i + 1 < len(args):
            return args[i + 1]
    return default

spec_path = arg("--spec")
spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
out_dir = Path(spec["out_dir"]).resolve()
out_dir.mkdir(parents=True, exist_ok=True)
scene = bpy.context.scene
scene.frame_set(spec.get("frame", 1))
orig_cam = scene.camera

# ---------- engine ----------
engine_set = None
for eng in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
    try:
        scene.render.engine = eng
        engine_set = eng
        break
    except Exception:
        pass
print("[rqa] engine=%s" % engine_set)

# ---------- render settings ----------
scene.render.resolution_x = spec.get("res", 512)
scene.render.resolution_y = spec.get("res", 512)
scene.render.image_settings.file_format = "PNG"
scene.render.film_transparent = False
try:
    scene.eevee.taa_render_samples = 16
except Exception:
    pass

# ---------- safety light / world ----------
if not any(o.type == "LIGHT" for o in bpy.data.objects):
    light_data = bpy.data.lights.new("rqa_sun", type="SUN")
    light_data.energy = 3.0
    sun = bpy.data.objects.new("rqa_sun", light_data)
    sun.rotation_euler = (math.radians(50), 0, math.radians(30))
    scene.collection.objects.link(sun)
    print("[rqa] added safety sun")
if scene.world is None:
    w = bpy.data.worlds.new("rqa_world")
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[0].default_value = (0.6, 0.6, 0.6, 1)
    scene.world = w
    print("[rqa] added safety world")

# ---------- hygiene hides ----------
hide_terms = [t.lower() for t in spec.get("hide", ["ceiling", "roof"])]
keep_terms = [t.lower() for t in spec.get("hide_exempt", [])]
hid = 0
for o in bpy.data.objects:
    if o.type == "MESH":
        low = o.name.lower()
        if any(t in low for t in hide_terms) and not any(k in low for k in keep_terms):
            o.hide_render = True
            hid += 1
print("[rqa] hidden=%d by %s exempt=%s" % (hid, hide_terms, keep_terms))

meshes = [o for o in bpy.data.objects if o.type == "MESH" and not o.hide_render]

def bbox_of(objs):
    xs = []; ys = []; zs = []
    for o in objs:
        mw = o.matrix_world
        for c in o.bound_box:
            v = mw @ Vector(c)
            xs.append(v.x); ys.append(v.y); zs.append(v.z)
    return (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))

# ---------- highlight ----------
COLORS = {
    "red": (1.0, 0.05, 0.05), "blue": (0.05, 0.25, 1.0), "green": (0.05, 0.9, 0.1),
    "orange": (1.0, 0.45, 0.0), "purple": (0.6, 0.1, 1.0), "yellow": (1.0, 0.9, 0.0),
}
def make_mat(name, rgb, strength=1.6):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (rgb[0], rgb[1], rgb[2], 1.0)
    em.inputs["Strength"].default_value = strength
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    return m

def match_objs(names):
    out = []
    for o in meshes:
        for p in names:
            if o.name.lower().startswith(p.lower()):
                out.append(o)
                break
    return out

hl_objs = []
hl_set = set()
for grp in spec.get("highlights", []):
    color = grp["color"]
    mat = make_mat("HL_" + color, COLORS[color])
    matched = match_objs(grp["match"])
    for o in matched:
        if o.data.users > 1:
            o.data = o.data.copy()
        o.data.materials.clear()
        o.data.materials.append(mat)
        hl_objs.append(o)
        hl_set.add(o)
    names = [o.name for o in matched]
    print("[rqa] HL %s matched %d: %s%s" % (color, len(names), names[:6], " ..." if len(names) > 6 else ""))

if not hl_objs:
    print("[rqa] WARNING: no highlight objects matched")

grid_mat = make_mat("RQAGrid", (0.85, 0.85, 0.85))
ann_mat = make_mat("RQAAnn", (0.02, 0.02, 0.02))
fine_mat = make_mat("RQAFine", (0.55, 0.55, 0.55))
ann_objs = []       # annotation objects shown only in the plan shots (dims, heading arrows)
gap_objs = []       # fine grid + gap line, shown only in the tight-crop measuring panel
grid_objs = []      # the 1 m floor grid and the elevation backdrops (hidden in the tight measuring panel)

# ---------- grid ----------
grid_info = None
if spec.get("grid", {}).get("enable"):
    gb = spec["grid"].get("bbox")
    gz = spec["grid"].get("z")
    if gb is None or gz is None:
        floors = [o for o in meshes if any(k in o.name.lower() for k in ("floor", "foundation", "ground"))]
        # One anchor point per highlight group; the grid should cover the SMALLEST floor
        # that still contains them, i.e. the room, not the whole storey.
        anchors = []
        for grp in spec.get("highlights", []):
            g_objs = match_objs(grp["match"])
            if g_objs:
                bb = bbox_of(g_objs)
                anchors.append(((bb[0] + bb[1]) / 2, (bb[2] + bb[3]) / 2))

        def contains(b, p, slack=0.5):
            return b[0] - slack <= p[0] <= b[1] + slack and b[2] - slack <= p[1] <= b[3] + slack

        fits = []
        for o in floors:
            b = bbox_of([o])
            area = (b[1] - b[0]) * (b[3] - b[2])
            if not (3.0 <= area <= 400.0):
                continue
            n_in = sum(1 for p in anchors if contains(b, p))
            fits.append((n_in, area, b, o.name))
        best = None
        if fits and anchors:
            full = [f for f in fits if f[0] == len(anchors)]
            pool = full if full else [f for f in fits if f[0] > 0]
            if pool:
                pool.sort(key=lambda f: f[1])
                n_in, area, b, name = pool[0]
                best = (b, area)
                print("[rqa] grid floor=%r anchors_in=%d/%d area=%.1f" % (name, n_in, len(anchors), area))
        if best:
            gb = [best[0][0], best[0][1], best[0][2], best[0][3]]
            gz = best[0][5] + 0.006
        else:
            b = bbox_of(hl_objs) if hl_objs else bbox_of(meshes)
            gb = [b[0] - 0.5, b[1] + 0.5, b[2] - 0.5, b[3] + 0.5]
            gz = 0.01
    x0, x1, y0, y1 = gb
    nlines = 0
    def bar(cx, cy, sx, sy):
        global nlines
        me = bpy.data.meshes.new("gridbar")
        dv = [(-sx/2, -sy/2, 0), (sx/2, -sy/2, 0), (sx/2, sy/2, 0), (-sx/2, sy/2, 0),
              (-sx/2, -sy/2, -0.006), (sx/2, -sy/2, -0.006), (sx/2, sy/2, -0.006), (-sx/2, sy/2, -0.006)]
        me.from_pydata(dv, [], [(0,1,2,3),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)])
        me.update()
        ob = bpy.data.objects.new("gridbar", me)
        ob.location = (cx, cy, gz)
        ob.data.materials.append(grid_mat)
        try:
            ob.visible_shadow = False
        except Exception:
            pass
        scene.collection.objects.link(ob)
        grid_objs.append(ob)
        nlines += 1
    xs = list(range(math.ceil(x0 - 1e-6), math.floor(x1 + 1e-6) + 1))
    ys = list(range(math.ceil(y0 - 1e-6), math.floor(y1 + 1e-6) + 1))
    for gx in xs:
        bar(gx, (y0 + y1) / 2, 0.02, y1 - y0)
    for gy in ys:
        bar((x0 + x1) / 2, gy, x1 - x0, 0.02)
    grid_info = {"bbox": gb, "z": gz, "n_lines": nlines}
    print("[rqa] grid lines=%d bbox=[%.2f,%.2f,%.2f,%.2f] z=%.3f" % (nlines, x0, x1, y0, y1, gz))

# ---------- cameras ----------
cam_data = bpy.data.cameras.new("RQACam")
cam = bpy.data.objects.new("RQACam", cam_data)
scene.collection.objects.link(cam)

def look_at(loc, target):
    d = Vector(target) - Vector(loc)
    return d.to_track_quat("-Z", "Y").to_euler()

def inside_solid(p):
    """True when the point sits inside solid geometry (all axis rays start on a back face)."""
    deps = bpy.context.evaluated_depsgraph_get()
    hits = back = 0
    for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        ok, loc, nrm, _i, _ob, _mx = scene.ray_cast(deps, Vector(p), Vector(d), distance=80.0)
        if not ok:
            continue
        hits += 1
        if Vector(d).dot(nrm) > 0:
            back += 1
    return hits >= 3 and back >= hits - 1

def view_blocked(p, direction, min_dist=1.2):
    """True when the line of sight from p is blocked closer than min_dist (wall, cabinet, ...)."""
    deps = bpy.context.evaluated_depsgraph_get()
    ok, _loc, _nrm, _i, _ob, _mx = scene.ray_cast(deps, Vector(p), Vector(direction), distance=min_dist)
    return bool(ok)

def choose_camera_loc(acx, acy, d, eye, standoff):
    """Pick the first-person camera: least displacement from the object that still sees the room.
    A blocked start view (object against a wall, camera inside the object) is what makes these
    frames come back as a single flat colour, which is useless for the question."""
    perp = (-d.y, d.x)
    cands = []
    for t in (standoff, standoff + 0.5, standoff + 1.0):
        cands.append((acx - d.x * t, acy - d.y * t, eye))
    for s in (0.9, 1.4):
        for sgn in (-1.0, 1.0):
            cands.append((acx + perp[0] * s * sgn - d.x * standoff, acy + perp[1] * s * sgn - d.y * standoff, eye))
    cands.append((acx - d.x * (standoff + 0.5), acy - d.y * (standoff + 0.5), eye + 0.7))
    deps = bpy.context.evaluated_depsgraph_get()
    base = math.atan2(d.y, d.x)
    best, best_score = cands[0], -1
    for p in cands:
        if inside_solid(p):
            continue
        if scene.ray_cast(deps, Vector(p), Vector(d), distance=1.2)[0]:
            continue
        hits = 0
        for k in range(9):
            a = base - math.radians(50) + math.radians(100) * k / 8.0
            if scene.ray_cast(deps, Vector(p), Vector((math.cos(a), math.sin(a), 0.0)), distance=30.0)[0]:
                hits += 1
        if hits > best_score:
            best, best_score = p, hits
    return best, best_score

def box_obj(name, center, size, mat):
    x, y, z = center
    sx, sy, sz = size
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    me = bpy.data.meshes.new(name)
    verts = [(-hx,-hy,-hz),(hx,-hy,-hz),(hx,hy,-hz),(-hx,hy,-hz),
             (-hx,-hy,hz),(hx,-hy,hz),(hx,hy,hz),(-hx,hy,hz)]
    faces = [(0,1,2,3),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]
    me.from_pydata(verts, [], faces)
    me.update()
    ob = bpy.data.objects.new(name, me)
    ob.location = (x, y, z)
    ob.data.materials.append(mat)
    try:
        ob.visible_shadow = False
    except Exception:
        pass
    scene.collection.objects.link(ob)
    return ob

if grid_info:
    gx0, gx1, gy0, gy1 = grid_info["bbox"]
    room = (gx0, gx1, gy0, gy1)
else:
    b = bbox_of(hl_objs) if hl_objs else bbox_of(meshes)
    room = (b[0], b[1], b[2], b[3])

cx = (room[0] + room[1]) / 2
cy = (room[2] + room[3]) / 2
dx = room[1] - room[0]
dy = room[3] - room[2]
zmax = max((bbox_of([o])[5] for o in meshes), default=2.5)

shots = []

def add_label(txt, loc, size=0.30, rot_z=0.0):
    cu = bpy.data.curves.new("rqatext", type="FONT")
    cu.body = txt
    cu.size = size
    cu.align_x = "CENTER"
    cu.align_y = "CENTER"
    ob = bpy.data.objects.new("rqatext", cu)
    ob.location = loc
    ob.rotation_euler = (0, 0, rot_z)
    ob.data.materials.append(ann_mat)
    scene.collection.objects.link(ob)
    return ob

def add_dims():
    """Dimension lines with metre values drawn INSIDE the grid extent (bottom = width, left = depth),
    so the plan keeps exactly the framing it has without annotations."""
    if not grid_info:
        return
    gx0, gx1, gy0, gy1 = grid_info["bbox"]
    z = grid_info["z"] + 0.010
    w, d = gx1 - gx0, gy1 - gy0
    off = 0.35
    sz = max(0.20, min(0.34, min(w, d) * 0.05))
    ann_objs.append(box_obj("dimline", ((gx0 + gx1) / 2, gy0 + off, z), (w, 0.035, 0.008), ann_mat))
    ann_objs.append(box_obj("dimline", (gx0 + off, (gy0 + gy1) / 2, z), (0.035, d, 0.008), ann_mat))
    ann_objs.append(add_label("%.1f m" % w, ((gx0 + gx1) / 2, gy0 + off + sz * 0.85, z + 0.010), size=sz))
    ann_objs.append(add_label("%.1f m" % d, (gx0 + off + sz * 0.85, (gy0 + gy1) / 2, z + 0.010),
                              size=sz, rot_z=math.radians(90)))
    print("[rqa] dims width=%.2f depth=%.2f m" % (w, d))

def add_arrows(items):
    """Heading arrows on the plan: dark = direction of travel on arrival, coloured = toward next target."""
    if not grid_info:
        return
    z = grid_info["z"] + 0.014
    for it in items:
        ax, ay = it["at"]
        dxr, dyr = it["dir"]
        n = math.hypot(dxr, dyr)
        if n < 1e-6:
            continue
        dxr, dyr = dxr / n, dyr / n
        L = float(it.get("len", 1.5))
        mat = ann_mat if it.get("color", "yellow") == "dark" else make_mat("RQAArow_%s" % it.get("color", "yellow"),
                                                                          COLORS.get(it.get("color", "yellow")))
        ang = math.atan2(dyr, dxr)
        sh = box_obj("arrowshaft", (ax + dxr * L * 0.34, ay + dyr * L * 0.34, z), (L * 0.68, 0.085, 0.010), mat)
        sh.rotation_euler = (0, 0, ang)
        hd = box_obj("arrowhead", (ax + dxr * L * 0.76, ay + dyr * L * 0.76, z), (0.30, 0.30, 0.012), mat)
        hd.rotation_euler = (0, 0, ang - math.radians(45))
        ann_objs.extend([sh, hd])
    print("[rqa] arrows=%d" % len(items))

def add_plan(tag):
    shots.append((tag, False, {"type": "ORTHO", "scale": max(dx, dy) * 1.06,
                               "loc": (cx, cy, zmax + 2.0), "rot": (0, 0, 0), "extra": ann_objs}))

def add_oblique(n=4):
    R = max(dx, dy) * 0.85 + 2.0
    for i, ang in enumerate([45, 135, 225, 315][:n]):
        a = math.radians(ang)
        elev = math.radians(38)
        loc = (cx + R * math.cos(a) * math.cos(elev), cy + R * math.sin(a) * math.cos(elev), 0.4 + R * math.sin(elev))
        shots.append(("oblique_%d" % i, False, {"type": "PERSP", "lens": 32, "loc": loc, "rot": look_at(loc, (cx, cy, 0.4))}))

def add_focus(n=8):
    if not hl_objs:
        return
    b = bbox_of(hl_objs)
    hcx, hcy = (b[0] + b[1]) / 2, (b[2] + b[3]) / 2
    hcz = min(max((b[4] + b[5]) / 2, 0.3), 1.4)
    diag = math.hypot(b[1] - b[0], b[3] - b[2])
    R = max(1.0, diag * 0.75 + 0.8)
    idx = 0
    for elev_deg in (20, 45):
        for ang in (45, 135, 225, 315):
            if idx >= n:
                break
            a = math.radians(ang + (15 if elev_deg == 45 else 0))
            e = math.radians(elev_deg)
            loc = (hcx + R * math.cos(a) * math.cos(e), hcy + R * math.sin(a) * math.cos(e), hcz + R * math.sin(e))
            shots.append(("focus_%d" % idx, False, {"type": "PERSP", "lens": 40, "loc": loc, "rot": look_at(loc, (hcx, hcy, hcz))}))
            idx += 1
        if idx >= n:
            break

def add_elev(cfg):
    if not hl_objs:
        print("[rqa] WARNING: elev skipped, no highlights")
        return
    b = bbox_of(hl_objs)
    bcx, bcy = (b[0] + b[1]) / 2, (b[2] + b[3]) / 2
    z_lo = 0.0 if b[4] < 0.10 else b[4] - 0.10
    D = 4.0
    if cfg.get("front"):
        span = float(max(1, math.ceil(max(b[1] - b[0], b[5] - b[4], 0.8) * 1.25 - 1e-6)))
        zc = z_lo + span / 2
        loc = (bcx, b[2] - D, zc)
        shots.append(("elev_front", False, {"type": "ORTHO", "scale": span, "loc": loc,
                                            "rot": look_at(loc, (bcx, bcy, zc)),
                                            "iso": cfg.get("iso", False),
                                            "meta": {"frame_m": span, "axis": "front"}}))
        yb = b[3] + 0.15
        x0 = math.floor(bcx - span / 2); x1 = math.ceil(bcx + span / 2)
        for xi in range(x0, x1 + 1):
            grid_objs.append(box_obj("elevbar", (xi, yb, zc), (0.012, 0.012, span), grid_mat))
        for zi in range(0, int(math.ceil(zc + span / 2)) + 1):
            grid_objs.append(box_obj("elevbar", (bcx, yb, zi), (span, 0.012, 0.012), grid_mat))
    if cfg.get("side"):
        span = float(max(1, math.ceil(max(b[3] - b[2], b[5] - b[4], 0.8) * 1.25 - 1e-6)))
        zc = z_lo + span / 2
        loc = (b[0] - D, bcy, zc)
        shots.append(("elev_side", False, {"type": "ORTHO", "scale": span, "loc": loc,
                                           "rot": look_at(loc, (bcx, bcy, zc)),
                                           "iso": cfg.get("iso", False),
                                           "meta": {"frame_m": span, "axis": "side"}}))
        xb = b[1] + 0.15
        y0 = math.floor(bcy - span / 2); y1 = math.ceil(bcy + span / 2)
        for yi in range(y0, y1 + 1):
            grid_objs.append(box_obj("elevbar", (xb, yi, zc), (0.012, 0.012, span), grid_mat))
        for zi in range(0, int(math.ceil(zc + span / 2)) + 1):
            grid_objs.append(box_obj("elevbar", (xb, bcy, zi), (0.012, span, 0.012), grid_mat))

def add_pairs(cfg):
    """One tight top-down crop per candidate: reference + that candidate only, over the 1 m grid."""
    ref = match_objs(cfg.get("ref") or [])
    if not ref:
        print("[rqa] WARNING: pairs skipped, no ref match %s" % cfg.get("ref"))
        return
    zcam = bbox_of(meshes)[5] + 2.0
    pad = float(cfg.get("pad", 0.45))
    for k, cand in enumerate(cfg.get("cands") or []):
        co = match_objs(cand)
        if not co:
            print("[rqa] WARNING: pair %d skipped, no match %s" % (k, cand))
            continue
        keep = set(ref) | set(co)
        bb = bbox_of(list(keep))
        cx2, cy2 = (bb[0] + bb[1]) / 2, (bb[2] + bb[3]) / 2
        span = max(1.2, max(bb[1] - bb[0], bb[3] - bb[2]) + 2 * pad)
        loc = (cx2, cy2, zcam)
        shots.append(("pair_%d" % k, False, {"type": "ORTHO", "scale": span, "loc": loc,
                                             "rot": (0, 0, 0), "keep": keep,
                                             "meta": {"frame_m": round(span, 2), "cand": k}}))

def add_viewpoint(cfg):
    at = match_objs(cfg["at"])
    if not at:
        print("[rqa] WARNING: viewpoint skipped, no 'at' match %s" % cfg["at"])
        return
    ab = bbox_of(at)
    acx, acy = (ab[0] + ab[1]) / 2, (ab[2] + ab[3]) / 2
    eye = cfg.get("eye", 1.2)
    standoff = cfg.get("standoff", 0.5)      # never sit inside the object we stand at
    lens = cfg.get("lens", 32)
    if cfg.get("away"):
        # "with your back to X" -> look away from X
        t = match_objs(cfg["away"])
        tb = bbox_of(t)
        tcx, tcy = (tb[0] + tb[1]) / 2, (tb[2] + tb[3]) / 2
        d = Vector((acx - tcx, acy - tcy, 0.0))
    elif cfg.get("look"):
        t = match_objs(cfg["look"])
        tb = bbox_of(t)
        tcx, tcy = (tb[0] + tb[1]) / 2, (tb[2] + tb[3]) / 2
        d = Vector((tcx - acx, tcy - acy, 0.0))
    else:
        fd = cfg["face_dir"]
        d = Vector((fd[0], fd[1], 0.0))
    if d.length < 1e-6:
        d = Vector((1.0, 0.0, 0.0))
    d.normalize()
    loc, seen = choose_camera_loc(acx, acy, d, eye, standoff)
    plain = (acx - d.x * standoff, acy - d.y * standoff, eye)
    if max(abs(loc[0] - plain[0]), abs(loc[1] - plain[1]), abs(loc[2] - plain[2])) > 1e-6:
        print("[rqa] %s: camera moved off the object, obstacles in view=%d" % (cfg.get("tag", "view"), seen))
    target = (loc[0] + d.x, loc[1] + d.y, eye)
    shots.append((cfg.get("tag", "view"), False, {"type": "PERSP", "lens": lens, "loc": loc, "rot": look_at(loc, target)}))

def add_gap_panel(cfg):
    """Tight top-down measuring panel: reference + target only, over a fine grid, with the
    closest-point gap drawn and its value printed in metres."""
    ref = match_objs(cfg.get("ref") or [])
    tgt = match_objs(cfg.get("target") or [])
    if not ref or not tgt:
        print("[rqa] WARNING: gap panel skipped, ref=%d target=%d" % (len(ref), len(tgt)))
        return
    rb = bbox_of(ref)
    tb = bbox_of(tgt)
    cell = float(cfg.get("cell", 0.5))
    pad = float(cfg.get("pad", 0.7))
    gx0, gx1 = min(rb[0], tb[0]), max(rb[1], tb[1])
    gy0, gy1 = min(rb[2], tb[2]), max(rb[3], tb[3])
    span = max(1.6, max(gx1 - gx0, gy1 - gy0) + 2 * pad)
    pcx, pcy = (gx0 + gx1) / 2, (gy0 + gy1) / 2
    gx0, gx1 = pcx - span / 2, pcx + span / 2
    gy0, gy1 = pcy - span / 2, pcy + span / 2
    z = min(rb[4], tb[4]) + 0.014
    for k in range(int(math.floor(gx0 / cell)), int(math.ceil(gx1 / cell)) + 1):
        gap_objs.append(box_obj("finebar", (k * cell, pcy, z), (0.012, gy1 - gy0, 0.006), fine_mat))
    for m in range(int(math.floor(gy0 / cell)), int(math.ceil(gy1 / cell)) + 1):
        gap_objs.append(box_obj("finebar", (pcx, m * cell, z), (gx1 - gx0, 0.012, 0.006), fine_mat))

    def closest(p0, p1, q0, q1):
        if p1 < q0:
            return q0 - p1, p1, q0
        if q1 < p0:
            return p0 - q1, p0, q1
        mid = (max(p0, q0) + min(p1, q1)) / 2
        return 0.0, mid, mid

    dxr, px, qx = closest(rb[0], rb[1], tb[0], tb[1])
    dyr, py, qy = closest(rb[2], rb[3], tb[2], tb[3])
    dzr = closest(rb[4], rb[5], tb[4], tb[5])[0]
    dist = math.sqrt(dxr ** 2 + dyr ** 2 + dzr ** 2)
    lx, ly = qx - px, qy - py
    L = math.hypot(lx, ly)
    sz = max(0.17, span * 0.05)
    if L > 1e-6:
        ang = math.atan2(ly, lx)
        ln = box_obj("gapline", ((px + qx) / 2, (py + qy) / 2, z + 0.004), (L, 0.05, 0.010), ann_mat)
        ln.rotation_euler = (0, 0, ang)
        gap_objs.append(ln)
    for (ex, ey) in ((px, py), (qx, qy)):
        gap_objs.append(box_obj("gapdot", (ex, ey, z + 0.006), (0.10, 0.10, 0.014), ann_mat))
    if cfg.get("show_value", True):
        nx, ny = -ly / L if L > 1e-6 else 0.0, lx / L if L > 1e-6 else 1.0
        gap_objs.append(add_label("%.2f m" % dist,
                                  ((px + qx) / 2 + nx * sz * 1.3, (py + qy) / 2 + ny * sz * 1.3, z + 0.012),
                                  size=sz))
    shots.append(("gap_panel", False, {"type": "ORTHO", "scale": span, "loc": (pcx, pcy, zmax + 2.0),
                                       "rot": (0, 0, 0), "keep": set(ref) | set(tgt),
                                       "no_grid": True, "extra": gap_objs,
                                       "meta": {"frame_m": round(span, 2), "cell_m": cell,
                                                "gap_m": round(dist, 2)}}))
    print("[rqa] gap panel: span=%.2f cell=%.2f gap=%.3f m" % (span, cell, dist))

def add_trajectory(cfg):
    if orig_cam is None:
        print("[rqa] WARNING: trajectory skipped, no scene camera")
        return
    n = cfg.get("n", 24)
    f0, f1 = scene.frame_start, scene.frame_end
    if n < 2 or f1 <= f0:
        frames = [f0]
    else:
        frames = sorted({int(round(f0 + (f1 - f0) * k / (n - 1))) for k in range(n)})
    for k, f in enumerate(frames):
        shots.append(("traj_%02d" % k, True, {"frame": f}))

vw = spec.get("views", {})
if spec.get("dims"):
    add_dims()
if vw.get("arrows"):
    add_arrows(vw["arrows"])
if vw.get("startcam"):
    item = dict(vw["startcam"])
    item["tag"] = "startcam"
    add_viewpoint(item)
if vw.get("plan", 0):
    add_plan("plan")
if vw.get("oblique", 0):
    add_oblique()
if vw.get("elev"):
    add_elev(vw["elev"])
if vw.get("pairs"):
    add_pairs(vw["pairs"])
if vw.get("focus", 0):
    add_focus(int(vw["focus"]))
if vw.get("gap"):
    add_gap_panel(vw["gap"])
for i, item in enumerate(vw.get("arrive") or []):
    it = dict(item)
    it["tag"] = "arrive_%d" % i
    add_viewpoint(it)
if vw.get("trajectory"):
    add_trajectory(vw["trajectory"])
if vw.get("plan_end", 0):
    add_plan("plan_end")

print("[rqa] shots=%d types=%s" % (len(shots), sorted(set(t for t, _, _ in shots))))

for o in ann_objs + gap_objs:
    o.hide_render = True

# ---------- render ----------
manifest = {"qa_id": spec.get("qa_id"), "scene": spec.get("scene"), "spec": spec,
            "engine": engine_set, "frames": [], "grid": grid_info}
t0 = time.time()
for i, (tag, use_scene_cam, params) in enumerate(shots, 1):
    if use_scene_cam:
        scene.camera = orig_cam
        scene.frame_set(params["frame"])
    else:
        scene.camera = cam
        if params["type"] == "ORTHO":
            cam_data.type = "ORTHO"
            cam_data.ortho_scale = params["scale"]
        else:
            cam_data.type = "PERSP"
            cam_data.lens = params.get("lens", 32)
        cam.location = params["loc"]
        cam.rotation_euler = params["rot"]
    keep = params.get("keep")
    iso_hidden = []
    if keep is not None or params.get("iso"):
        for o in meshes:
            if o.name.startswith("gridbar") or o.name.startswith("elevbar"):
                continue
            if keep is not None:
                if o in keep:
                    continue
            elif o in hl_set:
                continue
            if not o.hide_render:
                o.hide_render = True
                iso_hidden.append(o)
    grid_hidden = []
    if params.get("no_grid"):
        for o in grid_objs:
            if not o.hide_render:
                o.hide_render = True
                grid_hidden.append(o)
    extra_on = params.get("extra") or []
    for o in extra_on:
        o.hide_render = False
    fn = "frame_%02d_%s.png" % (i, tag)
    scene.render.filepath = str(out_dir / fn)
    bpy.ops.render.render(write_still=True)
    for o in extra_on:
        o.hide_render = True
    for o in grid_hidden:
        o.hide_render = False
    for o in iso_hidden:
        o.hide_render = False
    frame_row = {"file": fn, "view": tag}
    frame_row.update(params.get("meta") or {})
    manifest["frames"].append(frame_row)
    print("[rqa] rendered %s" % fn, flush=True)

manifest["seconds"] = round(time.time() - t0, 1)
(out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
print("[rqa] RENDER_OK frames=%d dir=%s seconds=%.1f" % (len(shots), out_dir, manifest["seconds"]))
