"""Render color-only counting evidence from a Blender scene and instance mapping."""

import argparse
import colorsys
import json
from pathlib import Path
import sys

import bpy
from mathutils import Vector


def corners(obj):
    return [obj.matrix_world @ Vector(point) for point in obj.bound_box]


def bounds(points):
    return tuple(Vector([fn(point[i] for point in points) for i in range(3)])
                 for fn in (min, max))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--global-view", choices=("top", "oblique"), default="top")
    parser.add_argument("--local-views", type=int, choices=range(5), default=4)
    parser.add_argument("--axis", choices=("x", "y"), default="x")
    parser.add_argument("--split", type=float, nargs="+", help="Explicit local region boundaries")
    parser.add_argument("--hide-above", type=float, help="Hide non-target meshes entirely above this Z")
    parser.add_argument("--resolution", type=int, default=1024)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    if args.split and (args.split != sorted(set(args.split)) or len(args.split) >= args.local_views):
        parser.error("--split requires sorted, unique boundaries within the local view budget")
    if args.resolution < 1:
        parser.error("--resolution must be positive")

    spec = json.loads(args.groups.read_text(encoding="utf-8"))
    scene = bpy.context.scene
    instances = spec["groups"][args.target]
    if not instances or not all(instances):
        parser.error("target requires nonempty instance groups")
    names = [name for group in instances for name in group]
    if len(names) != len(set(names)) or any(
        scene.objects.get(name) is None or scene.objects[name].type != "MESH" for name in names
    ):
        parser.error("each target mesh must exist and belong to exactly one instance")
    args.out.mkdir(parents=True, exist_ok=False)
    items = []
    for i, group in enumerate(instances, 1):
        rgb = colorsys.hsv_to_rgb((i * .61803398875) % 1, .66, .80 + .18 * (i % 2))
        key = f"{args.target}#{i}"
        material = bpy.data.materials.new(f"count:{key}")
        material.diffuse_color = (*rgb, 1)
        points = []
        for name in group:
            obj = scene.objects[name]
            points.extend(corners(obj))
            obj.data = obj.data.copy()
            obj.data.materials.clear()
            obj.data.materials.append(material)
            for face in obj.data.polygons:
                face.material_index = 0
        low, high = bounds(points)
        items.append(dict(id=key, meshes=group, rgb=rgb, points=points, center=(low + high) / 2))

    room = spec.get("room", {})
    ceilings = set(room.get("ceiling", []))
    target_names = set(names)
    hidden = []
    for obj in scene.objects:
        if obj.type != "MESH" or obj.name in target_names:
            continue
        if obj.name in ceilings or (args.hide_above is not None and min(p.z for p in corners(obj)) >= args.hide_above):
            obj.hide_render = True
            obj.hide_set(True)
            hidden.append(obj.name)

    camera = bpy.data.objects.new("count_camera", bpy.data.cameras.new("count_camera"))
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.data.type = "ORTHO"
    scene.render.engine = "BLENDER_WORKBENCH"
    shading = scene.display.shading
    shading.color_type = "MATERIAL"
    shading.light = "STUDIO"
    shading.show_shadows = False
    shading.show_cavity = True
    shading.show_object_outline = True
    scene.render.resolution_x = scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1
    scene.render.image_settings.file_format = "PNG"
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 1

    def render(name, group, direction, frame_points=None):
        points = frame_points or [point for item in group for point in item["points"]]
        low, high = bounds(points)
        target = (low + high) / 2
        camera.location = target + max(10, 3 * (high - low).length) * Vector(direction).normalized()
        camera.rotation_euler = ((target - camera.location).to_track_quat("-Z", "Y").to_euler()
                                 if direction != (0, 0, 1) else (0, 0, 0))
        bpy.context.view_layer.update()
        inverse = camera.matrix_world.inverted()
        low, high = bounds([inverse @ point for point in points])
        mid = Vector(((low.x + high.x) / 2, (low.y + high.y) / 2, 0))
        camera.location += camera.rotation_euler.to_matrix() @ mid
        camera.data.ortho_scale = max(high.x - low.x, high.y - low.y, .15) * 1.16
        camera.data.clip_end = max(100, camera.location.length + 100)
        filename = name + ".png"
        scene.render.filepath = str(args.out.resolve() / filename)
        bpy.ops.render.render(write_still=True)
        return dict(file=filename, instance_ids=[item["id"] for item in group],
                    position=list(camera.location), ortho_scale=camera.data.ortho_scale)

    floor = [p for name in room.get("floor", []) if scene.objects.get(name)
             for p in corners(scene.objects[name])]
    top = args.global_view == "top"
    views = [render("global", items, (0, 0, 1) if top else (.85, -1.1, 1.2), floor if top else None)]
    if args.local_views and len(items) > 1:
        axis = 0 if args.axis == "x" else 1
        ordered = sorted(items, key=lambda item: item["center"][axis])
        bins = {}
        for i, item in enumerate(ordered):
            region = (sum(item["center"][axis] >= split for split in args.split)
                      if args.split else i * min(args.local_views, len(items)) // len(items))
            bins.setdefault(region, []).append(item)
        room_center = sum((item["center"] for item in items), Vector()) / len(items)
        for i, group in enumerate(bins.values(), 1):
            center = sum((item["center"] for item in group), Vector()) / len(group)
            side = 1 if center[axis] <= room_center[axis] else -1
            direction = (side, -.25, 2.5) if axis == 0 else (-.25, side, 2.5)
            views.append(render(f"local-{i:02d}", group, direction))

    manifest = dict(target=args.target, labels_rendered=False, hidden_meshes=hidden, views=views,
                    vlm_images=[view["file"] for view in views],
                    instances=[{k: item[k] for k in ("id", "meshes", "rgb")} for item in items])
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
