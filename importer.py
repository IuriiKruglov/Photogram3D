# SPDX-License-Identifier: GPL-3.0-or-later
# Photogram3D - photos or video to a textured mesh with COLMAP + OpenMVS
# Copyright (C) 2026 Kruglov Iurii
# Authors: Kruglov Iurii, Anthropic Claude AI
# License: GNU General Public License (GPL) v3 or later, https://www.blender.org/about/license/
"""Bring a reconstruction into Blender: textured mesh, cameras, root transform."""

import json
import math
import os
import statistics

import bpy
from bpy.app.handlers import persistent
from mathutils import Matrix, Vector

from . import colmap_io
from .common import TRANSFORM_FILE, current

# COLMAP world after model_orientation_aligner: +Y points down (gravity).
# Blender: +Z up. Rotation of -90 degrees about X maps y -> -z, z -> y.
COLMAP_TO_BLENDER = Matrix(((1.0, 0.0, 0.0, 0.0),
                            (0.0, 0.0, 1.0, 0.0),
                            (0.0, -1.0, 0.0, 0.0),
                            (0.0, 0.0, 0.0, 1.0)))

# COLMAP camera looks down +Z with +Y down; a Blender camera looks down -Z with +Y up.
CAMERA_FLIP = Matrix(((1.0, 0.0, 0.0, 0.0),
                      (0.0, -1.0, 0.0, 0.0),
                      (0.0, 0.0, -1.0, 0.0),
                      (0.0, 0.0, 0.0, 1.0)))

ROOT_PROP = "photogram3d_project"


def _ui_override():
    wm = bpy.context.window_manager
    for win in getattr(wm, "windows", []):
        screen = win.screen
        if screen is None:
            continue
        for area in screen.areas:
            if area.type == "VIEW_3D":
                region = next((r for r in area.regions if r.type == "WINDOW"), None)
                return {"window": win, "area": area, "region": region}
    if getattr(wm, "windows", None):
        return {"window": wm.windows[0]}
    return {}


def _import_mesh_file(path):
    before = set(bpy.data.objects)
    ext = os.path.splitext(path)[1].lower()
    with bpy.context.temp_override(**_ui_override()):
        if ext == ".obj":
            # forward=Y, up=Z is Blender's own convention -> no axis conversion
            bpy.ops.wm.obj_import(filepath=path, forward_axis="Y", up_axis="Z",
                                  use_split_objects=False, use_split_groups=False,
                                  validate_meshes=True)
        elif ext == ".ply":
            bpy.ops.wm.ply_import(filepath=path, forward_axis="Y", up_axis="Z")
        else:
            raise ValueError(f"unsupported mesh format: {ext}")
    return [o for o in bpy.data.objects if o not in before]


def _use_fresh_images(objs, images_before):
    """Give a new import its own images, read from disk now.

    Blender's OBJ importer reuses an image that is already loaded from the same path
    and keeps its old pixels. Every run rewrites scene_textured_material_*.jpg, so a
    second import into the same .blend showed the previous run's atlas on the new
    UVs. Objects imported earlier keep their images.
    """
    fresh = {}
    for obj in objs:
        for slot in obj.material_slots:
            mat = slot.material
            if mat is None or not mat.use_nodes:
                continue
            for node in mat.node_tree.nodes:
                img = node.image if node.type == "TEX_IMAGE" else None
                if img is None or img not in images_before:
                    continue
                if img not in fresh:
                    path = bpy.path.abspath(img.filepath)
                    fresh[img] = (bpy.data.images.load(path, check_existing=False)
                                  if os.path.isfile(path) else img)
                    fresh[img].colorspace_settings.name = img.colorspace_settings.name
                node.image = fresh[img]
    return len(fresh)


def _matte_materials(obj):
    """Scan textures already contain lighting: make the surface rough and non-specular."""
    for slot in obj.material_slots:
        mat = slot.material
        if mat is None or not mat.use_nodes:
            continue
        for node in mat.node_tree.nodes:
            if node.type == "BSDF_PRINCIPLED":
                for name, value in (("Roughness", 1.0), ("Specular IOR Level", 0.0)):
                    sock = node.inputs.get(name)
                    if sock is not None:
                        sock.default_value = value


def _move_to_collection(obj, coll):
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    coll.objects.link(obj)


def _camera_scale(images):
    centers = [Vector(colmap_io.camera_center(im)) for im in images.values()]
    if len(centers) < 2:
        return 1.0
    mid = sum(centers, Vector()) / len(centers)
    dists = [(c - mid).length for c in centers]
    return max(1e-3, statistics.median(dists))


RES_PROP = "photogram3d_resolution"          # on camera data: (width, height) of its frame
INTRINSICS_PROP = "photogram3d_intrinsics"   # on camera data: fx, fy, cx, cy, w, h as solved (0.3.9)
TURN_PROP = "photogram3d_turn"               # on a photo camera: quarter turns that made it upright


def _set_lens(data, fx, fy, cx, cy, w, h):
    """Lens, sensor fit, shift and frame size of a Blender camera for a pinhole camera of
    w x h pixels."""
    data.sensor_width = 36.0
    data.sensor_height = 36.0
    if w >= h:
        data.sensor_fit = "HORIZONTAL"
        data.lens = fx * 36.0 / w
    else:
        data.sensor_fit = "VERTICAL"
        data.lens = fy * 36.0 / h
    maxdim = float(max(w, h))
    data.shift_x = (w * 0.5 - cx) / maxdim
    data.shift_y = (cy - h * 0.5) / maxdim
    data[RES_PROP] = (int(round(w)), int(round(h)))


def _camera_data(name, cam, scale, turn=0):
    """Blender camera data matching a COLMAP pinhole camera (undistorted model), for its
    photo turned upright by ``turn`` counter-clockwise quarter turns."""
    fx, fy, cx, cy = colmap_io.pinhole_intrinsics(cam)
    w, h = int(cam["width"]), int(cam["height"])
    data = bpy.data.cameras.new(name)
    data[INTRINSICS_PROP] = (float(fx), float(fy), float(cx), float(cy), float(w), float(h))
    _set_lens(data, *colmap_io.turned_intrinsics(fx, fy, cx, cy, w, h, turn))
    data.display_size = 0.08 * scale
    data.clip_start = 0.01 * scale
    data.clip_end = 100.0 * scale
    return data


def _roll(turn):
    """The camera's own turn about its viewing axis that shows its photo upright."""
    return Matrix.Rotation(-(turn % 4) * math.pi / 2.0, 4, "Z")


def _turn_backgrounds(data, turn):
    """Show the photos (as stored) turned like the camera. Blender fits a background
    picture to the frame first and turns it afterwards (overlay_camera.hh: translate *
    rotate * scale), so a quarter turn uses Fit, scaled by the photo's long side over its
    short side, and the turned picture then fills the turned frame exactly."""
    k = turn % 4
    w, h = (data.get(INTRINSICS_PROP) or (0, 0, 0, 0, 1, 1))[4:6]
    angle = (0.0, -math.pi / 2.0, math.pi, math.pi / 2.0)[k]
    for bg in data.background_images:
        if bg.source != "IMAGE":
            continue
        bg.rotation = angle
        bg.offset = (0.0, 0.0)
        bg.use_flip_x = bg.use_flip_y = False
        if k % 2:
            bg.frame_method = "FIT"
            bg.scale = max(w, h) / max(1.0, min(w, h))
        else:
            bg.frame_method = "STRETCH"
            bg.scale = 1.0


PHOTO_STAMP_PROP = "photogram3d_file_stamp"   # on a camera's photo: the file it was read from


def _file_stamp(path):
    st = os.stat(path)
    return f"{st.st_mtime_ns}:{st.st_size}"


def _photo_image(path):
    """The image of a photo file: the one already loaded from that file when there is one
    (0.3.12; before, every import loaded the photos again, as IMG_1.jpg.001, .002 ...), read
    again from disk when the file changed since (a run solved again rewrites its undistorted
    photos under the same names)."""
    stamp = _file_stamp(path)
    known = {img.name for img in bpy.data.images}
    img = bpy.data.images.load(path, check_existing=True)
    if img.name in known and img.get(PHOTO_STAMP_PROP) != stamp:
        img.reload()
    img[PHOTO_STAMP_PROP] = stamp
    return img


def _add_photo(data, path, turn=0):
    """Show the undistorted photo behind the camera view (exact fit: same intrinsics)."""
    if not os.path.isfile(path):
        return False
    try:
        img = _photo_image(path)            # pixels load lazily
    except (RuntimeError, OSError):
        return False
    data.show_background_images = True
    bg = data.background_images.new()
    bg.image = img
    bg.alpha = 1.0
    bg.display_depth = "BACK"
    bg.frame_method = "STRETCH"
    _turn_backgrounds(data, turn)
    return True


def _make_camera(im, cam, scale, parent, sub, image_dir, photos, turn=0):
    stem = os.path.splitext(im["name"])[0]
    data = _camera_data(f"P3D_{stem}", cam, scale, turn)
    r = colmap_io.qvec_to_rotmat(im["qvec"])
    c = colmap_io.camera_center(im)
    # world_from_cam = [R^T | C]
    m = Matrix(((r[0][0], r[1][0], r[2][0], c[0]),
                (r[0][1], r[1][1], r[2][1], c[1]),
                (r[0][2], r[1][2], r[2][2], c[2]),
                (0.0, 0.0, 0.0, 1.0)))
    obj = bpy.data.objects.new(stem, data)
    sub.objects.link(obj)
    obj.parent = parent
    obj.matrix_parent_inverse = Matrix.Identity(4)
    obj.matrix_basis = m @ CAMERA_FLIP @ _roll(turn)
    obj["photogram3d_image"] = im["name"]
    obj[TURN_PROP] = int(turn) % 4
    if image_dir:
        path = os.path.join(image_dir, im["name"])
        obj["photogram3d_image_path"] = path
        if photos:
            _add_photo(data, path, turn)
    return obj


def photo_turns(result):
    """{photo name: counter-clockwise quarter turns} that show each photo upright: from the
    result (0.3.9), else from COLMAP's database (the gravity it took from the EXIF
    orientation), else from the EXIF orientation of the photos COLMAP read."""
    if isinstance(result.get("upright"), dict):
        return {k: int(v) % 4 for k, v in result["upright"].items() if int(v) % 4}
    project = result.get("project") or ""
    turns = colmap_io.upright_turns(os.path.join(project, "colmap", "database.db")) if project else {}
    if turns or not project:
        return turns
    from . import formats
    for folder in (os.path.join(project, "photos"), result.get("images_dir") or ""):
        if not folder or not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            k = colmap_io.EXIF_TURNS.get(formats.exif_orientation(os.path.join(folder, name)), 0)
            if k:
                turns.setdefault(name, k)
    return turns


def _set_render_size(scene, w, h):
    r = scene.render
    if (r.resolution_x, r.resolution_y) != (int(w), int(h)):
        r.resolution_x, r.resolution_y = int(w), int(h)
    r.resolution_percentage = 100
    r.pixel_aspect_x = r.pixel_aspect_y = 1.0


def match_render_size(scene, cameras):
    """The scene's frame for these cameras: the scene camera's own size when it is one of
    them, else the size most of them have."""
    sizes = [tuple(o.data[RES_PROP]) for o in cameras if o.type == "CAMERA" and RES_PROP in o.data.keys()]
    if not sizes:
        return None
    cam = scene.camera
    if cam is not None and cam in cameras and RES_PROP in cam.data.keys():
        size = tuple(cam.data[RES_PROP])
    else:
        size = max(sorted(set(sizes)), key=sizes.count)
    _set_render_size(scene, *size)
    return size


def import_cameras(model_dir, parent, coll, match_render=True, image_dir=None, photos=True, turns=None):
    """One camera object (and camera data, which owns the background photo) per image,
    turned so that its photo is upright (``turns``: photo_turns())."""
    cams, images = colmap_io.read_model(model_dir)
    scale = _camera_scale(images)
    sub = bpy.data.collections.new(coll.name + " Cameras")
    coll.children.link(sub)
    objects = []
    turns = turns or {}
    for iid in sorted(images, key=lambda k: images[k]["name"]):
        im = images[iid]
        objects.append(_make_camera(im, cams[im["camera_id"]], scale, parent, sub, image_dir, photos,
                                    turns.get(im["name"], 0)))
    if match_render and objects:
        match_render_size(bpy.context.scene, objects)
    return objects


def _intrinsics_of(data):
    """(fx, fy, cx, cy, w, h) as solved: stored by 0.3.9, or read back from a camera an
    earlier version made (never turned: its frame is the stored photo's)."""
    stored = data.get(INTRINSICS_PROP)
    if stored is not None and len(stored) == 6:
        return tuple(float(v) for v in stored)
    w, h = (int(v) for v in data[RES_PROP])
    maxdim = float(max(w, h))
    f = data.lens * (w if w >= h else h) / 36.0
    cx = w * 0.5 - data.shift_x * maxdim
    cy = h * 0.5 + data.shift_y * maxdim
    return f, f, cx, cy, float(w), float(h)


def turn_camera(obj, turn):
    """Turn a photo camera (and its background photo) so that it shows its photo turned by
    ``turn`` quarter turns from the way the file stores it. Nothing moves in the world: the
    camera only turns about its own viewing axis. Returns True when something changed."""
    data = obj.data
    if obj.type != "CAMERA" or RES_PROP not in data.keys():
        return False
    now = int(obj.get(TURN_PROP, 0)) % 4
    turn %= 4
    if now == turn and INTRINSICS_PROP in data.keys():
        return False
    intr = _intrinsics_of(data)
    data[INTRINSICS_PROP] = intr
    _set_lens(data, *colmap_io.turned_intrinsics(*intr, turn))
    obj.matrix_basis = obj.matrix_basis @ _roll(now).inverted() @ _roll(turn)
    obj[TURN_PROP] = turn
    _turn_backgrounds(data, turn)
    return now != turn


def photo_cameras(root):
    return [o for o in root.children_recursive if o.type == "CAMERA" and "photogram3d_image" in o.keys()]


def scene_camera_poses(root):
    """{photo name: (world-to-camera rotation, centre)} of a scan's photo cameras as they stand
    now, in the solve's frame (the scan root's own transform left out) and in COLMAP's camera
    axes for the photo as stored: what Re-solve From Cameras checks the photo pairs against.
    The importer's turn that shows a photo upright is taken off again."""
    poses = {}
    for obj in photo_cameras(root):
        turn = int(obj.get(TURN_PROP, 0)) % 4
        m = obj.matrix_basis @ _roll(turn).inverted() @ CAMERA_FLIP     # world_from_cam = [R^T | C]
        r = m.to_3x3().normalized().to_quaternion().normalized().to_matrix()   # rotation only
        R = [[r[0][0], r[1][0], r[2][0]], [r[0][1], r[1][1], r[2][1]], [r[0][2], r[1][2], r[2][2]]]
        poses[obj["photogram3d_image"]] = (R, tuple(m.translation))
    return poses


def sideways_cameras(root, turns=None):
    """The root's photo cameras that do not show their photo upright."""
    if turns is None:
        turns = photo_turns(_result_for_root(root))
    return [o for o in photo_cameras(root)
            if int(o.get(TURN_PROP, 0)) % 4 != turns.get(o.get("photogram3d_image"), 0)]


def _result_for_root(root):
    project = root.get(ROOT_PROP, "")
    try:
        return load_result_json(project) if project else {}
    except (OSError, ValueError):
        return {"project": project}


_TURNS_CACHE = {}


def root_turns(root):
    """photo_turns() for a scan's root, read again only when its database changes (the
    panel asks on every redraw)."""
    project = root.get(ROOT_PROP, "")
    try:
        stamp = os.stat(os.path.join(project, "colmap", "database.db")).st_mtime_ns
    except OSError:
        stamp = None
    key = (_norm(project), stamp)
    if key not in _TURNS_CACHE:
        if len(_TURNS_CACHE) > 8:
            _TURNS_CACHE.clear()
        _TURNS_CACHE[key] = photo_turns(_result_for_root(root))
    return _TURNS_CACHE[key]


def upright_cameras(root, turns=None):
    """Turn every photo camera of a scan to show its photo upright. Returns how many turned."""
    if turns is None:
        turns = photo_turns(_result_for_root(root))
    cams = photo_cameras(root)
    done = sum(1 for o in cams if turn_camera(o, turns.get(o.get("photogram3d_image"), 0)))
    scene = bpy.context.scene
    if done and scene.camera in cams:
        _set_render_size(scene, *scene.camera.data[RES_PROP])
    return done


# The scene's frame follows the scene camera: a photo camera turned upright has a portrait
# frame while its neighbours may be landscape, and Blender keeps one frame size per scene.
_SCENE_CAMERA = {}


def sync_render_size(scene):
    """When the scene camera changes to a Photogram3D camera, give the scene its frame."""
    cam = scene.camera
    name = cam.name if cam is not None else ""
    if _SCENE_CAMERA.get(scene.name) == name:
        return False
    _SCENE_CAMERA[scene.name] = name
    if cam is None or cam.type != "CAMERA" or RES_PROP not in cam.data.keys():
        return False
    _set_render_size(scene, *cam.data[RES_PROP])
    return True


@persistent
def _on_depsgraph(scene, depsgraph=None):
    try:
        sync_render_size(scene)
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        pass


@persistent
def _on_load(*_args):
    """A file just opened keeps the frame it was saved with."""
    _SCENE_CAMERA.clear()
    try:
        scenes = list(bpy.data.scenes)
    except AttributeError:            # add-on registration: bpy.data is not open yet
        return
    for scene in scenes:
        _SCENE_CAMERA[scene.name] = scene.camera.name if scene.camera is not None else ""


def register():
    if _on_depsgraph not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph)
    if _on_load not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_load)
    _on_load()


def unregister():
    for handlers, fn in ((bpy.app.handlers.depsgraph_update_post, _on_depsgraph),
                         (bpy.app.handlers.load_post, _on_load)):
        if fn in handlers:
            handlers.remove(fn)


def import_more_cameras(result, root, cams, photos=True):
    """Video: static cameras also for the solved frames the depth maps did not use (long
    videos use photos a few degrees apart, see pipeline.DENSE_SUBSET_MIN). Their
    background is the extracted frame itself, with the lens distortion still in it."""
    model = result.get("model_distorted")
    if not cams or not model or not result.get("frames_json") or not os.path.isdir(model):
        return []
    frames_dir = os.path.dirname(result["frames_json"])
    have = {o.get("photogram3d_image") for o in cams}
    models, images = colmap_io.read_model(model)
    missing = sorted((iid for iid, im in images.items() if im["name"] not in have),
                     key=lambda k: images[k]["name"])
    if not missing:
        return []
    sub = cams[0].users_collection[0]
    scale = _camera_scale(images)
    return [_make_camera(images[iid], models[images[iid]["camera_id"]], scale, root, sub,
                         frames_dir, photos) for iid in missing]


def scan_videos(result):
    """The videos a scan's tracked cameras follow: a joined scan's (the result's, or for a
    scan joined before 0.3.15 those of its join file's runs, pipeline.joined_videos), else the
    video run's own; [] for photos."""
    if result.get("videos"):
        return list(result["videos"])
    from . import pipeline
    desc = pipeline.joined_description(result["project"]) if result.get("project") else None
    if desc is not None:
        return pipeline.joined_videos(desc)
    return []


def _ensure_tracked_camera(result, root, coll, enabled=True, errors=None):
    """Video input: one camera that follows the video (see tracking.py). A joined scan
    (0.3.15): one for each video it was joined from, moved into its coordinates. ``errors``:
    a list that gets why a camera was not made."""
    if not enabled:
        return None
    from . import tracking
    errors = [] if errors is None else errors
    videos = scan_videos(result)
    if videos:
        prefix = run_names(result)[2]
        stems = [os.path.splitext(os.path.basename(v.get("video") or ""))[0] or str(k + 1)
                 for k, v in enumerate(videos)]
        made = []
        for k, (v, stem) in enumerate(zip(videos, stems)):
            if stems.count(stem) > 1:
                stem = f"{stem}_{k + 1}"
            run = {"project": v.get("project", ""), "video": v.get("video", ""),
                   "frames_json": v.get("frames_json", ""), "model_distorted": v.get("model_distorted", "")}
            try:
                obj = tracking.ensure_tracked_camera(
                    run, root, coll, source=_norm(run["project"]) or f"video {k + 1}",
                    transform=v.get("matrix"), name=f"{prefix}_video_camera_{stem}", set_scene=not made,
                    solve=v.get("solve"))
                made.append(obj)
            except Exception as exc:  # noqa: BLE001 - an extra: never let it stop the import
                errors.append(f"{stem}: {exc}")
                print(f"Photogram3D: tracked camera of {stem} not created: {exc}")
        return made[0] if made else None
    if not result.get("frames_json"):
        return None
    try:
        return tracking.ensure_tracked_camera(result, root, coll)
    except Exception as exc:  # noqa: BLE001 - an extra: never let it stop the import
        errors.append(str(exc))
        print(f"Photogram3D: tracked camera not created: {exc}")
        return None


def _video_result(root):
    """What _ensure_tracked_camera needs for a scan in the scene: its run's result, else (no
    result file yet: the crop stage) the run's video and the scan's name as imported."""
    from . import pipeline
    result = _result_for_root(root)
    project = result.get("project") or root.get(ROOT_PROP, "")
    if not result.get("frames_json") and not scan_videos(result) and project:
        vids = pipeline.run_videos(pipeline.project_paths(project))
        if vids:
            result = dict(result, **{k: vids[0][k] for k in ("video", "frames_json", "model_distorted")})
    if not result.get("scan"):
        desc = pipeline.joined_description(project) if project else None
        coll = _root_collection(root).name
        scan = (desc or {}).get("scan_name") or (coll.split(" ", 1)[1] if coll.startswith("Photogram3D_")
                                                 and " " in coll else "")
        result = dict(result, attempt=int(root.get("photogram3d_attempt") or 1), **({"scan": scan} if scan else {}))
    return result


def add_video_cameras(root):
    """Make a scan's missing tracked cameras (imported without them, or joined before 0.3.15,
    when the joined scan got none). Returns (cameras made, why others were not)."""
    from . import tracking
    before = set(tracking.tracked_cameras(root))
    errors = []
    _ensure_tracked_camera(_video_result(root), root, _root_collection(root), errors=errors)
    bpy.context.view_layer.update()
    return [o for o in tracking.tracked_cameras(root) if o not in before], errors


_VIDEOS_CACHE = {}


def scan_video_count(root):
    """How many videos a scan has tracked cameras for, or could have: len(scan_videos()), or 1
    for a video run. Read again only when the run's result or join file changes (the panel
    asks on every redraw)."""
    from . import pipeline
    project = root.get(ROOT_PROP, "")
    if not project:
        return 0
    stamps = []
    for path in (current(os.path.join(project, "photogram3d_result.json")),
                 os.path.join(project, pipeline.JOIN_FILE),
                 os.path.join(project, "frames", "frames.json")):
        try:
            stamps.append(os.stat(path).st_mtime_ns)
        except OSError:
            stamps.append(None)
    key = (_norm(project), tuple(stamps))
    if key not in _VIDEOS_CACHE:
        if len(_VIDEOS_CACHE) > 8:
            _VIDEOS_CACHE.clear()
        try:
            result = _video_result(root)
            n = len(scan_videos(result)) or (1 if result.get("frames_json") else 0)
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            n = 0
        _VIDEOS_CACHE[key] = n
    return _VIDEOS_CACHE[key]


MESH_FALLBACKS = ("scene_textured.obj", "scene_dense_mesh_refine.ply", "scene_dense_mesh.ply")


def resolve_mesh(result):
    """(path, note): the result's mesh, or the best mesh still in its mvs folder.

    A result file written before Photogram3D 0.2.1 can outlive its mesh (a later run
    that failed while texturing had already cleared the mvs folder).
    """
    mesh_path = result.get("mesh")
    if mesh_path and os.path.isfile(mesh_path):
        return mesh_path, ""
    folders = []
    if mesh_path:
        folders.append(os.path.dirname(mesh_path))
    if result.get("project"):
        folders.append(os.path.join(result["project"], "mvs"))
    for folder in folders:
        for name in MESH_FALLBACKS:
            path = os.path.join(folder, name)
            if os.path.isfile(path):
                return path, f"{os.path.basename(mesh_path or '?')} is missing; imported {name}"
    raise FileNotFoundError(f"reconstructed mesh not found: {mesh_path}")


SOLVE_PROP = "photogram3d_solve"   # on a root: the id of the camera solve it shows (0.3.5)
# On a crop box: True while it is parented to its root (0.3.3 and earlier, and from Build
# Mesh on), else the root's ROOT_ID_PROP: until Build Mesh the box is a free object.
CROP_BOX_PROP = "photogram3d_crop_box"
CROP_FIT_PROP = "photogram3d_crop_fit"     # the box as Photogram3D last fitted it (untouched test)
ROOT_ID_PROP = "photogram3d_id"
CLOUD_PROP = "photogram3d_cloud"
MARKER_PROP = "photogram3d_marker"
MESH_PROP = "photogram3d_mesh"
ROI_FIT_PROP = "photogram3d_roi_fit"
MARKER_NAMES = {"G1": "Ground 1", "G2": "Ground 2", "G3": "Ground 3", "A": "Anchor A", "B": "Anchor B"}


def _norm(path):
    return os.path.normcase(os.path.normpath(os.path.abspath(path))) if path else ""


def load_user_transform(project, solve=None):
    """Alignment and scale saved for a project's camera solve ``solve`` (identity if none:
    an alignment made for another solve of the run does not fit this one)."""
    try:
        with open(current(os.path.join(project, TRANSFORM_FILE)), "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("solve") != solve:
            return Matrix.Identity(4)
        return Matrix(data["matrix"])
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return Matrix.Identity(4)


def user_transform_saved(project, solve):
    """A place is saved in ``project`` for its camera solve ``solve``."""
    try:
        with open(current(os.path.join(project, TRANSFORM_FILE)), "r", encoding="utf-8") as f:
            return json.load(f).get("solve") == solve
    except (OSError, ValueError, AttributeError):
        return False


def hide_scans(projects):
    """Hide the scans of these run folders (their collection's eye in the Outliner, else the
    objects): Complete Joining shows the joined scan in their place. The names hidden."""
    want = {_norm(p) for p in projects if p}
    view_layer = bpy.context.view_layer
    hidden = []
    for root in [o for o in bpy.context.scene.objects if ROOT_PROP in o.keys()]:
        if _norm(root.get(ROOT_PROP, "")) not in want:
            continue
        coll = _root_collection(root)
        layer = _layer_collection(view_layer.layer_collection, coll) \
            if coll is not bpy.context.scene.collection else None
        if layer is not None:
            layer.hide_viewport = True
        else:
            for obj in [root] + list(root.children_recursive) + crop_boxes(root):
                try:
                    obj.hide_set(True)
                except RuntimeError:      # not in the view layer
                    pass
        hidden.append(root.name)
    return hidden


def save_user_transform(root):
    """Remember where the user put the scan: root = user transform @ COLMAP_TO_BLENDER."""
    project = root.get(ROOT_PROP, "")
    if not project or not os.path.isdir(project):
        return False
    return write_user_transform(project, root.matrix_world, root.get(SOLVE_PROP))


def write_user_transform(project, root_matrix, solve):
    """Save the place of a scan whose root has ``root_matrix``, for the camera solve ``solve``
    of the run in ``project`` (a joined run is placed where the scan it joins onto is)."""
    t = Matrix(root_matrix) @ COLMAP_TO_BLENDER.inverted()
    path = current(os.path.join(project, TRANSFORM_FILE))
    try:
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump({"matrix": [list(row) for row in t], "solve": solve}, f, indent=2)
        os.replace(path + ".tmp", path)
    except OSError:
        return False
    return True


def _children(root, prop):
    if prop == CROP_BOX_PROP:
        return crop_boxes(root)
    return [o for o in root.children_recursive if prop in o.keys()]


def root_id(root):
    if ROOT_ID_PROP not in root.keys():
        import uuid
        root[ROOT_ID_PROP] = uuid.uuid4().hex
    return root[ROOT_ID_PROP]


def crop_boxes(root):
    """The root's crop boxes: its children, and the free box of the crop stage."""
    out = [o for o in root.children_recursive if CROP_BOX_PROP in o.keys()]
    rid = root.get(ROOT_ID_PROP)
    if rid:
        scene = bpy.context.scene
        out += [o for o in (scene.objects if scene else ()) if o.get(CROP_BOX_PROP) == rid
                and o.parent is None and o not in out]
    return out


def box_root(box):
    """The root a crop box belongs to: its parent, or for a free box the root it names."""
    if box.parent is not None:
        return find_root(box.parent)
    rid = box.get(CROP_BOX_PROP)
    if isinstance(rid, str):
        for obj in bpy.data.objects:
            if obj.get(ROOT_ID_PROP) == rid and ROOT_PROP in obj.keys():
                return obj
    return None


def find_project_root(project, prop=CROP_BOX_PROP, solve=None):
    """The newest root in this scene for ``project`` that holds an object with ``prop`` and
    shows the camera solve ``solve`` (a root of an earlier solve of the run has other
    coordinates: its cameras, cloud and box do not fit the new result)."""
    want = _norm(project)
    best = None
    for obj in bpy.context.scene.objects:
        if (ROOT_PROP in obj.keys() and _norm(obj[ROOT_PROP]) == want and obj.get(SOLVE_PROP) == solve
                and _children(obj, prop)):
            if best is None or obj.get("photogram3d_created", 0) > best.get("photogram3d_created", 0):
                best = obj
    return best


EARLIER_SOLVE = " (earlier solve)"


def _remove_object(obj):
    data = obj.data
    bpy.data.objects.remove(obj, do_unlink=True)
    if data is not None and data.users == 0:
        if isinstance(data, bpy.types.Camera):
            bpy.data.cameras.remove(data)
        elif isinstance(data, bpy.types.Mesh):
            bpy.data.meshes.remove(data)


def _remove_empty_collections(coll):
    """``coll`` and its sub-collections, those left without objects."""
    for child in list(coll.children):
        _remove_empty_collections(child)
    if not coll.objects and not coll.children:
        bpy.data.collections.remove(coll)


def retire_earlier_solves(project, solve):
    """Scans in this scene of the run ``project`` that show another camera solve than
    ``solve``. The run was solved again (Reconstruct with Resume after a fold was found or a
    setting changed), its folder now holds the new solve, and the earlier scan's cloud,
    cameras, crop box and markers fit nothing any more (0.3.12; before, they stayed, and the
    scene held two clouds and two sets of cameras with copies of every photo). They are
    removed; a mesh of the earlier solve stays with its root, in its collection renamed
    '... (earlier solve)' and hidden. Scans without a solve id (made before 0.3.5) are left
    alone. Returns [(root name, "removed" or "kept the mesh")]."""
    from . import tracking
    if not solve:
        return []
    want = _norm(project)
    done = []
    for root in [o for o in bpy.context.scene.objects
                 if ROOT_PROP in o.keys() and _norm(o[ROOT_PROP]) == want
                 and o.get(SOLVE_PROP) and o.get(SOLVE_PROP) != solve]:
        name = root.name
        coll = _root_collection(root)
        stale = set(crop_boxes(root))
        for prop in (CLOUD_PROP, MARKER_PROP, SPLATS_PROP, tracking.TRACK_PROP):
            stale.update(_children(root, prop))
        stale.update(photo_cameras(root))
        for obj in stale:
            _remove_object(obj)
        if root.children_recursive:           # a mesh, or anything else put under the root
            if coll is not bpy.context.scene.collection and not coll.name.endswith(EARLIER_SOLVE):
                coll.name = coll.name + EARLIER_SOLVE
                for layer in bpy.context.scene.view_layers:
                    lc = _layer_collection(layer.layer_collection, coll)
                    if lc is not None:
                        lc.exclude = True
            done.append((name, "kept the mesh"))
        else:
            bpy.data.objects.remove(root, do_unlink=True)
            if coll is not bpy.context.scene.collection:
                _remove_empty_collections(coll)
            done.append((name, "removed"))
    return done


def _layer_collection(lc, coll):
    if lc.collection == coll:
        return lc
    for child in lc.children:
        found = _layer_collection(child, coll)
        if found is not None:
            return found
    return None


def run_names(result, name=None):
    """(collection, root, object prefix) for a run: Photogram3D_2 Gas_mask, P3D_Gas_mask_2 and
    Gas_mask_2 for the second attempt on the Gas_mask photos."""
    project = result.get("project") or ""
    k = int(result.get("attempt") or 1)
    scan = result.get("scan") or name or os.path.basename(os.path.normpath(project or "scan")) or "scan"
    return f"Photogram3D_{k} {scan}", f"P3D_{scan}_{k}", f"{scan}_{k}"


def _new_root(result, name=None):
    import time
    project = result.get("project") or ""
    coll_name, root_name, _ = run_names(result, name)
    coll = bpy.data.collections.new(coll_name)
    bpy.context.scene.collection.children.link(coll)
    root = bpy.data.objects.new(root_name, None)
    root.empty_display_type = "ARROWS"
    coll.objects.link(root)
    root.matrix_world = load_user_transform(project, result.get("solve_id")) @ COLMAP_TO_BLENDER
    root[ROOT_PROP] = project
    if result.get("solve_id"):
        root[SOLVE_PROP] = result["solve_id"]
    root["photogram3d_created"] = time.time()
    root["photogram3d_attempt"] = int(result.get("attempt") or 1)
    if result.get("base"):
        root["photogram3d_base"] = result["base"]
    root_id(root)
    return root, coll


def _root_collection(root):
    return root.users_collection[0] if root.users_collection else bpy.context.scene.collection


def _has_cameras(root):
    return any(o.type == "CAMERA" for o in root.children)


def import_result(result, name=None, import_cameras_flag=True, match_render=True, frame_view=True,
                  camera_photos=True, tracked_camera=True):
    """Import a finished reconstruction. ``result`` is photogram3d_result.json content.

    After a crop stage in this scene the mesh joins that root (with its cameras,
    alignment and scale); the cloud and the crop box are hidden.
    """
    mesh_path, note = resolve_mesh(result)
    if note:
        print(f"Photogram3D: {note}")
    project = result.get("project", "")
    _, _, prefix = run_names(result, name)
    retired = retire_earlier_solves(project, result.get("solve_id")) if project else []
    root = find_project_root(project, solve=result.get("solve_id")) if project else None
    if root is not None:
        coll = _root_collection(root)
        for old in _children(root, MESH_PROP):      # a mesh from an earlier Build Mesh
            old.hide_set(True)
            old.hide_render = True
    else:
        root, coll = _new_root(result, name)

    images_before = set(bpy.data.images)
    new_objs = _import_mesh_file(mesh_path)
    _use_fresh_images(new_objs, images_before)
    meshes = [o for o in new_objs if o.type == "MESH"]
    for obj in new_objs:
        _move_to_collection(obj, coll)
        obj.parent = root
        obj.matrix_parent_inverse = Matrix.Identity(4)
    for obj in meshes:
        obj.name = f"{prefix}_scan"
        obj[MESH_PROP] = True
        _matte_materials(obj)

    cams = [o for o in root.children if o.type == "CAMERA"]
    model_dir = result.get("model")
    if import_cameras_flag and not cams and model_dir and os.path.isdir(model_dir):
        try:
            cams = import_cameras(model_dir, root, coll, match_render, result.get("images"),
                                  photos=camera_photos, turns=photo_turns(result))
            cams += import_more_cameras(result, root, cams, camera_photos)
        except (OSError, ValueError) as exc:
            print(f"Photogram3D: cameras not imported: {exc}")
    tracked = _ensure_tracked_camera(result, root, coll, tracked_camera)
    try:
        splat_obj = import_splats(result, root, coll)
    except Exception as exc:  # noqa: BLE001 - the mesh import must not fail over the splats
        print(f"Photogram3D: splats not imported: {exc}")
        splat_obj = None
    for obj in _children(root, CLOUD_PROP) + _children(root, CROP_BOX_PROP):
        obj.hide_set(True)

    # Parenting was done through matrix_basis; refresh matrix_world for callers
    # (and for background mode, where no depsgraph update happens on its own).
    bpy.context.view_layer.update()
    root.empty_display_size = max(0.1, _bounds_size(meshes) * 0.1) if meshes else 1.0
    if frame_view:
        _frame(meshes or [root])
    return {"root": root, "meshes": meshes, "cameras": cams, "collection": coll, "note": note,
            "tracked_camera": tracked, "splats": splat_obj, "retired": retired}


# ---------------------------------------------------------------------------
# Gaussian splats (0.3.5)
# ---------------------------------------------------------------------------
SPLATS_PROP = "photogram3d_splats"
SPLAT_PREVIEW_MAX = 2_000_000          # points in the Blender 5.2 preview


def _splat_material():
    mat = bpy.data.materials.get("Photogram3D Splats")
    if mat is not None:
        return mat
    mat = bpy.data.materials.new("Photogram3D Splats")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emit = nt.nodes.new("ShaderNodeEmission")
    clear = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    col = nt.nodes.new("ShaderNodeAttribute")
    col.attribute_name = "Col"
    alpha = nt.nodes.new("ShaderNodeAttribute")
    alpha.attribute_name = "opacity"
    nt.links.new(col.outputs["Color"], emit.inputs["Color"])
    nt.links.new(alpha.outputs["Fac"], mix.inputs["Fac"])
    nt.links.new(clear.outputs["BSDF"], mix.inputs[1])
    nt.links.new(emit.outputs["Emission"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    mat.surface_render_method = "DITHERED"
    mat.diffuse_color = (0.8, 0.8, 0.8, 1.0)
    return mat


def _splat_node_group():
    """Geometry nodes: each splat as a point with its own radius (attribute 'radius')."""
    ng = bpy.data.node_groups.get("Photogram3D Splats")
    if ng is not None:
        return ng
    ng = bpy.data.node_groups.new("Photogram3D Splats", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    gin = ng.nodes.new("NodeGroupInput")
    gout = ng.nodes.new("NodeGroupOutput")
    radius = ng.nodes.new("GeometryNodeInputNamedAttribute")
    radius.data_type = "FLOAT"
    radius.inputs["Name"].default_value = "radius"
    to_pts = ng.nodes.new("GeometryNodeMeshToPoints")
    setmat = ng.nodes.new("GeometryNodeSetMaterial")
    setmat.inputs["Material"].default_value = _splat_material()
    ng.links.new(gin.outputs[0], to_pts.inputs["Mesh"])
    ng.links.new(radius.outputs["Attribute"], to_pts.inputs["Radius"])
    ng.links.new(to_pts.outputs["Points"], setmat.inputs["Geometry"])
    ng.links.new(setmat.outputs["Geometry"], gout.inputs[0])
    return ng


def _splat_preview_object(name, path, root, coll):
    """Blender 5.2: the splats as round points with their colour, opacity and size."""
    import numpy as np
    from . import splats
    _, data = splats.read_ply(path)
    if len(data) > SPLAT_PREVIEW_MAX:          # the most opaque ones
        keep = np.argsort(-data["opacity"])[:SPLAT_PREVIEW_MAX]
        data = data[np.sort(keep)]
    xyz, linear, opacity, scale, _ = splats.splat_fields(data)
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(xyz))
    me.vertices.foreach_set("co", xyz.ravel())
    col = me.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
    rgba = np.ones((len(xyz), 4), np.float32)
    rgba[:, :3] = linear
    col.data.foreach_set("color", rgba.ravel())
    me.color_attributes.active_color = col
    a = me.attributes.new("opacity", "FLOAT", "POINT")
    a.data.foreach_set("value", opacity)
    # a splat's footprint: its two larger axes (one sigma), as one radius
    r = me.attributes.new("radius", "FLOAT", "POINT")
    r.data.foreach_set("value", np.sqrt(np.sort(scale, 1)[:, 1:].prod(1)).astype(np.float32))
    me.update()
    obj = bpy.data.objects.new(name, me)
    coll.objects.link(obj)
    obj.parent = root
    obj.matrix_parent_inverse = Matrix.Identity(4)
    mod = obj.modifiers.new("Photogram3D Splats", "NODES")
    mod.node_group = _splat_node_group()
    obj.data.materials.append(_splat_material())
    return obj


def import_splats(result, root, coll):
    """The trained splats (cropped to the box) under the root: Blender's own splat object
    from 5.3 on, a point preview in 5.2. Returns the object or None."""
    rel = result.get("splats")
    if not rel:
        return None
    path = rel if os.path.isabs(rel) else os.path.join(result.get("project", ""), rel)
    if not os.path.isfile(path):
        return None
    for old in _children(root, SPLATS_PROP):
        data, kind = old.data, old.type
        bpy.data.objects.remove(old)
        if data is not None and data.users == 0:
            getattr(bpy.data, "pointclouds" if kind == "POINTCLOUD" else "meshes").remove(data)
    _, _, prefix = run_names(result, None)
    obj = None
    if bpy.app.version >= (5, 3, 0):
        before = set(bpy.data.objects)
        try:
            bpy.ops.wm.ply_import(filepath=path, forward_axis="Y", up_axis="Z")
        except (RuntimeError, TypeError) as exc:
            print(f"Photogram3D: Blender's splat import failed ({exc}); using the point preview")
        new = [o for o in bpy.data.objects if o not in before]
        if len(new) == 1 and new[0].type == "POINTCLOUD":
            obj = new[0]
            for c in list(obj.users_collection):
                c.objects.unlink(obj)
            coll.objects.link(obj)
            obj.parent = root
            obj.matrix_parent_inverse = Matrix.Identity(4)
            obj.matrix_basis = Matrix.Identity(4)
            obj.name = f"{prefix}_splats"
        else:
            for o in new:
                bpy.data.objects.remove(o)
    if obj is None:
        obj = _splat_preview_object(f"{prefix}_splats", path, root, coll)
    obj[SPLATS_PROP] = True
    return obj


# ---------------------------------------------------------------------------
# Crop stage: dense point cloud, cameras and the crop box
# ---------------------------------------------------------------------------

def _points_material():
    mat = bpy.data.materials.get("Photogram3D Points")
    if mat is not None:
        return mat
    mat = bpy.data.materials.new("Photogram3D Points")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emit = nt.nodes.new("ShaderNodeEmission")
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "Col"
    nt.links.new(attr.outputs["Color"], emit.inputs["Color"])
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    mat.diffuse_color = (0.75, 0.75, 0.75, 1.0)
    return mat


def _points_node_group():
    """Geometry nodes: show the cloud's vertices as small points in the viewport."""
    ng = bpy.data.node_groups.get("Photogram3D Points")
    if ng is not None:
        return ng
    ng = bpy.data.node_groups.new("Photogram3D Points", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    radius = ng.interface.new_socket("Radius", in_out="INPUT", socket_type="NodeSocketFloat")
    radius.default_value = 0.005
    radius.min_value = 0.0
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    gin = ng.nodes.new("NodeGroupInput")
    gout = ng.nodes.new("NodeGroupOutput")
    to_pts = ng.nodes.new("GeometryNodeMeshToPoints")
    setmat = ng.nodes.new("GeometryNodeSetMaterial")
    setmat.inputs["Material"].default_value = _points_material()
    ng.links.new(gin.outputs[0], to_pts.inputs["Mesh"])
    ng.links.new(gin.outputs[1], to_pts.inputs["Radius"])
    ng.links.new(to_pts.outputs["Points"], setmat.inputs["Geometry"])
    ng.links.new(setmat.outputs["Geometry"], gout.inputs[0])
    return ng


def make_cloud_object(name, xyz, rgb, root, coll):
    """Vertex-only mesh with the cloud colours, drawn as points (geometry nodes)."""
    import numpy as np
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(xyz))
    me.vertices.foreach_set("co", np.ascontiguousarray(xyz, dtype=np.float32).ravel())
    col = me.color_attributes.new("Col", "BYTE_COLOR", "POINT")
    rgba = np.ones((len(xyz), 4), dtype=np.float32)
    rgba[:, :3] = np.asarray(rgb, dtype=np.float32) / 255.0
    col.data.foreach_set("color_srgb", rgba.ravel())
    me.color_attributes.active_color = col
    me.update()
    obj = bpy.data.objects.new(name, me)
    coll.objects.link(obj)
    obj.parent = root
    obj.matrix_parent_inverse = Matrix.Identity(4)
    obj[CLOUD_PROP] = True
    obj.hide_render = True            # a placing aid: hiding it later must also keep it out of F12
    mod = obj.modifiers.new("Photogram3D Points", "NODES")
    mod.node_group = _points_node_group()
    # point radius: a fraction of the typical spacing, from the cloud's extent
    lo, hi = np.percentile(xyz, 2, axis=0), np.percentile(xyz, 98, axis=0)
    diag = float(np.linalg.norm(hi - lo)) or 1.0
    socket = next(s for s in mod.node_group.interface.items_tree
                  if getattr(s, "in_out", "") == "INPUT" and s.name == "Radius")
    _set_modifier_input(mod, socket.identifier, diag * 0.0008)
    obj.data.materials.append(_points_material())
    return obj


def _set_modifier_input(mod, identifier, value):
    """Blender 5.2 keeps geometry-nodes inputs in mod.properties.inputs; older
    versions used ID properties on the modifier."""
    inputs = getattr(getattr(mod, "properties", None), "inputs", None)
    if inputs is not None and hasattr(inputs, identifier):
        getattr(inputs, identifier).value = value
    else:
        mod[identifier] = value


_CUBE = ((-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
         (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1))


def _cube_mesh(name):
    me = bpy.data.meshes.new(name)
    f = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    me.from_pydata(_CUBE, [], f)
    me.update()
    return me


def _is_box_mesh(box):
    return box.type == "MESH" and box.data is not None and len(box.data.vertices) == 8


def set_box_from_obb(box, obb):
    if _is_box_mesh(box):              # undo an applied transform (Ctrl+A) on the cube
        box.data.vertices.foreach_set("co", [c for v in _CUBE for c in v])
        box.data.update()
    box.matrix_basis = Matrix([list(row) for row in obb.matrix()])


def crop_box_obb(box):
    """The box in reconstruction (COLMAP / OpenMVS) coordinates.

    Read from the cube's corners rather than the object matrix, so the box stays right
    after Apply Location / Rotation / Scale (Ctrl+A), which moves the shape into the mesh."""
    from . import roi
    root = box_root(box)
    local = (root.matrix_world.inverted() @ box.matrix_world) if root else box.matrix_world.copy()
    if _is_box_mesh(box):
        v = [local @ box.data.vertices[i].co for i in range(8)]
        # corners 1, 3 and 4 are one edge away from corner 0 along the cube's x, y and z
        a, b, c = ((v[k] - v[0]) / 2.0 for k in (1, 3, 4))
        p = sum(v[1:], v[0].copy()) / 8.0
        local = Matrix(((a.x, b.x, c.x, p.x),
                        (a.y, b.y, c.y, p.y),
                        (a.z, b.z, c.z, p.z),
                        (0.0, 0.0, 0.0, 1.0)))
    return roi.OBB.from_matrix([list(row) for row in local])


def import_crop_stage(result, name=None, import_cameras_flag=True, camera_photos=True,
                      frame_view=True, tracked_camera=True):
    """The dense cloud and the cameras under one root, and a free crop box around the
    cloud (parented to the root at Build Mesh)."""
    from . import roi
    project = result.get("project", "")
    _, _, prefix = run_names(result, name)
    retired = retire_earlier_solves(project, result.get("solve_id")) if project else []
    root = find_project_root(project, solve=result.get("solve_id"))
    if root is None:
        root, coll = _new_root(result, name)
    else:
        coll = _root_collection(root)
    for old in _children(root, CLOUD_PROP):
        me = old.data
        bpy.data.objects.remove(old)
        if me is not None and me.users == 0:
            bpy.data.meshes.remove(me)
    xyz, rgb = roi.load_preview(result["cloud_preview"])
    cloud = make_cloud_object(f"{prefix}_cloud", xyz, rgb, root, coll)
    root[ROI_FIT_PROP] = json.dumps(roi.OBB.from_dict(result["roi_fit"]).as_dict())

    cams = [o for o in root.children if o.type == "CAMERA"]
    model_dir = result.get("model")
    if import_cameras_flag and not cams and model_dir and os.path.isdir(model_dir):
        try:
            cams = import_cameras(model_dir, root, coll, True, result.get("images"),
                                  photos=camera_photos, turns=photo_turns(result))
            cams += import_more_cameras(result, root, cams, camera_photos)
        except (OSError, ValueError) as exc:
            print(f"Photogram3D: cameras not imported: {exc}")
    _ensure_tracked_camera(result, root, coll, tracked_camera)
    bpy.context.view_layer.update()

    boxes = _children(root, CROP_BOX_PROP)
    if boxes:
        box = boxes[0]                       # keep the box the user already placed
        box.hide_set(False)
    else:
        box = bpy.data.objects.new(f"{prefix}_crop_box", _cube_mesh(f"{prefix}_crop_box"))
        coll.objects.link(box)
        box[CROP_BOX_PROP] = root_id(root)
        _style_crop_box(box)
        saved = result.get("roi_user")
        if saved:                            # the box of an earlier Build Mesh, as it was
            set_box_from_obb(box, roi.OBB.from_dict(saved))
            box.matrix_world = root.matrix_world @ box.matrix_basis
        else:
            fit_crop_box(root, box)
    bpy.context.view_layer.update()
    root.empty_display_size = max(0.1, _bounds_size([box]) * 0.1)
    if frame_view:
        _frame([box])
    return {"root": root, "cloud": cloud, "box": box, "cameras": cams, "collection": coll,
            "retired": retired}


def _crop_box_material():
    """See-through blue: the points inside the box show through it (Solid and Material
    Preview), so it is easy to see what the box cuts."""
    mat = bpy.data.materials.get("Photogram3D Crop Box")
    if mat is not None:
        return mat
    mat = bpy.data.materials.new("Photogram3D Crop Box")
    rgba = (0.25, 0.55, 1.0, 0.18)
    mat.diffuse_color = rgba                     # Solid mode: Material colour with alpha
    mat.use_backface_culling = False
    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = "BLENDED"    # EEVEE (Blender 4.2+)
    if hasattr(mat, "use_transparency_overlap"):
        mat.use_transparency_overlap = False
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfTransparent")
    emit = nt.nodes.new("ShaderNodeEmission")
    mix = nt.nodes.new("ShaderNodeMixShader")
    emit.inputs["Color"].default_value = rgba
    emit.inputs["Strength"].default_value = 1.0
    mix.inputs["Fac"].default_value = rgba[3]
    nt.links.new(bsdf.outputs[0], mix.inputs[1])
    nt.links.new(emit.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    return mat


def _style_crop_box(box):
    box.display_type = "TEXTURED"
    box.show_in_front = False
    box.show_wire = False
    box.hide_render = True
    box.visible_shadow = False
    if box.data is not None and not box.data.materials:
        box.data.materials.append(_crop_box_material())


def _cloud_world_points(root):
    import numpy as np
    pts = []
    for cloud in _children(root, CLOUD_PROP):
        n = len(cloud.data.vertices)
        if not n:
            continue
        co = np.empty(n * 3, dtype=np.float64)
        cloud.data.vertices.foreach_get("co", co)
        m = np.array(cloud.matrix_world)
        pts.append(co.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3])
    return np.concatenate(pts) if pts else None


def _box_state(box):
    return [round(v, 6) for row in box.matrix_world for v in row] + \
        [round(v, 6) for vert in box.data.vertices for v in vert.co]


def fit_crop_box(root, box):
    """Put a free box around the bulk of the cloud, along the world axes, with no rotation
    and unit scale: the size is in the mesh, the object sits at the cloud's centre."""
    import numpy as np
    pts = _cloud_world_points(root)
    if pts is None or len(pts) < 4:
        lo, hi = np.array([-1.0] * 3), np.array([1.0] * 3)
    else:
        lo, hi = np.percentile(pts, 0.5, axis=0), np.percentile(pts, 99.5, axis=0)
    centre = (lo + hi) / 2.0
    half = np.maximum((hi - lo) / 2.0 * 1.03, 1e-4)
    if box.parent is not None:
        box.parent = None
    box[CROP_BOX_PROP] = root_id(root)
    box.data.vertices.foreach_set("co", [c * half[i] for v in _CUBE for i, c in enumerate(v)])
    box.data.update()
    box.matrix_world = Matrix.Translation(Vector(centre.tolist()))
    bpy.context.view_layer.update()
    box[CROP_FIT_PROP] = json.dumps(_box_state(box))
    return box


def crop_box_untouched(box):
    try:
        return json.loads(box.get(CROP_FIT_PROP, "null")) == _box_state(box)
    except (TypeError, ValueError):
        return False


def attach_crop_box(root, box):
    """Build Mesh: the box joins the root (it then moves with the scan), staying in place."""
    if box.parent is root:
        return
    world = box.matrix_world.copy()
    box.parent = root
    box.matrix_parent_inverse = Matrix.Identity(4)
    box.matrix_world = world
    box[CROP_BOX_PROP] = True
    bpy.context.view_layer.update()


def reset_crop_box(root):
    boxes = _children(root, CROP_BOX_PROP)
    if not boxes:
        return None
    box = boxes[0]
    if box.parent is not None:
        box.parent = None
        box[CROP_BOX_PROP] = root_id(root)
    if not _is_box_mesh(box):
        box.data = _cube_mesh(box.name)
    _style_crop_box(box)
    fit_crop_box(root, box)
    return box


# ---------------------------------------------------------------------------
# Ground alignment and real-world scale (markers placed at the 3D cursor)
# ---------------------------------------------------------------------------

def scan_size(root):
    objs = [o for o in root.children_recursive if o.type == "MESH" and not o.hide_get()]
    return _bounds_size(objs) if objs else 1.0


def place_marker(root, slot, location):
    """Create or move the marker ``slot`` (G1, G2, G3, A, B) to a world location."""
    marker = next((o for o in _children(root, MARKER_PROP) if o[MARKER_PROP] == slot), None)
    if marker is None:
        marker = bpy.data.objects.new(f"PM {MARKER_NAMES[slot]}", None)
        _root_collection(root).objects.link(marker)
        marker.parent = root
        marker.matrix_parent_inverse = Matrix.Identity(4)
        marker[MARKER_PROP] = slot
        marker.empty_display_type = "SPHERE"
        marker.show_name = True
        marker.show_in_front = True
    marker.hide_set(False)
    bpy.context.view_layer.update()
    marker.matrix_world = Matrix.Translation(Vector(location))   # world size 1: size below is world
    marker.empty_display_size = 0.01 * scan_size(root)
    return marker


def marker_locations(root, slots):
    found = {o[MARKER_PROP]: o.matrix_world.translation.copy() for o in _children(root, MARKER_PROP)}
    missing = [MARKER_NAMES[s] for s in slots if s not in found]
    if missing:
        raise ValueError("place " + ", ".join(missing) + " first")
    return [found[s] for s in slots]


GROUND_AXES = {"POS_X": (1.0, 0.0, 0.0), "NEG_X": (-1.0, 0.0, 0.0),
               "POS_Y": (0.0, 1.0, 0.0), "NEG_Y": (0.0, -1.0, 0.0)}
RIGHT_ANGLE_TOLERANCE_DEG = 10.0


def ground_origin(p1, p2, p3):
    """Where Align Ground puts the origin. Three corners of a rectangle (three wheels'
    contact points) make a right triangle: the rectangle's centre, the middle of the
    longest side. Otherwise the centre of the three points."""
    pts = (Vector(p1), Vector(p2), Vector(p3))
    limit = math.sin(math.radians(RIGHT_ANGLE_TOLERANCE_DEG))
    for k in range(3):
        a, b = pts[(k + 1) % 3] - pts[k], pts[(k + 2) % 3] - pts[k]
        if a.length > 0 and b.length > 0 and abs(a.dot(b)) / (a.length * b.length) < limit:
            return (pts[(k + 1) % 3] + pts[(k + 2) % 3]) / 2.0, True
    return (pts[0] + pts[1] + pts[2]) / 3.0, False


def ground_transform(points, cameras=(), axis=None):
    """World transform that puts the plane through three points on Z = 0, normal +Z, with
    ground_origin() at the origin. The normal faces the cameras (or keeps +Z up). With
    ``axis`` (a GROUND_AXES key) the direction from the first point to the second, along
    the ground, turns to that axis; without it the scan keeps its turn about Z."""
    p1, p2, p3 = (Vector(p) for p in points)
    n = (p2 - p1).cross(p3 - p1)
    if n.length < 1e-9 * max(1e-9, (p2 - p1).length * (p3 - p1).length):
        raise ValueError("the three ground points are on one line")
    n.normalize()
    c, _ = ground_origin(p1, p2, p3)
    centroid = (p1 + p2 + p3) / 3.0
    cams = [Vector(x) for x in cameras]
    if cams:
        if sum(((x - centroid).dot(n) for x in cams), 0.0) < 0:
            n = -n
    elif n.z < 0:
        n = -n
    if axis in GROUND_AXES:
        d = (p2 - p1) - n * (p2 - p1).dot(n)       # Ground 1 -> 2, along the ground plane
        d.normalize()
        src = Matrix((d.cross(n), d, n)).transposed()      # columns: x, y, z of the scan
        y = Vector(GROUND_AXES[axis])
        z = Vector((0.0, 0.0, 1.0))
        dst = Matrix((y.cross(z), y, z)).transposed()      # where they go in the world
        rot = (dst @ src.transposed()).to_4x4()
    else:
        rot = n.rotation_difference(Vector((0.0, 0.0, 1.0))).to_matrix().to_4x4()
    return Matrix.Translation(-(rot @ c)) @ rot


def transform_root(root, delta):
    """Move the whole scan by the world transform ``delta`` and remember it. A free crop box
    (crop stage) moves along; one the user never touched is fitted to the cloud again,
    so after Align Ground it lies along the new ground."""
    free = [b for b in crop_boxes(root) if b.parent is None]
    untouched = [b for b in free if crop_box_untouched(b)]
    root.matrix_world = delta @ root.matrix_world
    for box in free:
        box.matrix_world = delta @ box.matrix_world
    bpy.context.view_layer.update()
    for box in untouched:
        fit_crop_box(root, box)
    save_user_transform(root)


def align_ground(root, axis=None):
    pts = marker_locations(root, ("G1", "G2", "G3"))
    cams = [o.matrix_world.translation.copy() for o in root.children if o.type == "CAMERA"]
    transform_root(root, ground_transform(pts, cams, axis))
    return ground_origin(*pts)[1]


def apply_real_scale(root, real_distance):
    """Scale the scan about the world origin so that Anchor A to B is real_distance."""
    a, b = marker_locations(root, ("A", "B"))
    current = (a - b).length
    if current <= 1e-12:
        raise ValueError("Anchor A and Anchor B are at the same place")
    factor = real_distance / current
    transform_root(root, Matrix.Scale(factor, 4))
    return factor


def reset_alignment(root):
    """Back to the scan as it was reconstructed (only up made +Z): undoes Align Ground and
    every scale."""
    transform_root(root, COLMAP_TO_BLENDER @ root.matrix_world.inverted())
    for prop in (JOINED_TO_PROP, JOIN_RESULT_PROP):
        if prop in root.keys():
            del root[prop]


# ---------------------------------------------------------------------------
# Joining two scans by control points (0.3.13, join.py)
# ---------------------------------------------------------------------------

JOIN_SLOTS = tuple(f"J{k}" for k in range(1, 9))
MARKER_NAMES.update({slot: f"Point {slot[1:]}" for slot in JOIN_SLOTS})
JOINED_TO_PROP = "photogram3d_joined_to"      # on the moved scan: the name of the scan it was joined onto
JOIN_RESULT_PROP = "photogram3d_join"         # on the moved scan: the fit, as JSON
JOIN_TREE_POINTS = 400_000                    # points of the fixed scan searched by the refinement


def join_pairs(moving, fixed):
    """(slots, the moving scan's points, the fixed scan's points): the control points
    placed on both scans, in world coordinates."""
    m = {o[MARKER_PROP]: tuple(o.matrix_world.translation) for o in _children(moving, MARKER_PROP)}
    f = {o[MARKER_PROP]: tuple(o.matrix_world.translation) for o in _children(fixed, MARKER_PROP)}
    slots = [slot for slot in JOIN_SLOTS if slot in m and slot in f]
    return slots, [m[slot] for slot in slots], [f[slot] for slot in slots]


def _join_record(moving, fixed, slots, fit, size, refined=None):
    from . import join
    rec = {"onto": fixed.name, "points": slots, "size": size, "scale": fit["scale"],
           "residuals": [float(r) for r in fit["residuals"]], "rms": fit["rms"],
           "suspect": [slots[i] for i in fit["suspect"]],
           "narrow": bool(fit.get("spread", 1.0) < join.SPREAD_MIN)}
    if fit.get("loo") is not None:
        rec["loo"] = [float(r) for r in fit["loo"]]
    if refined:
        rec["refined"] = refined
    moving[JOINED_TO_PROP] = fixed.name
    moving[JOIN_RESULT_PROP] = json.dumps(rec)
    return rec


def join_record(root):
    try:
        return json.loads(root.get(JOIN_RESULT_PROP) or "")
    except (TypeError, ValueError):
        return None


def join_scans(moving, fixed):
    """Move the scan ``moving`` onto ``fixed`` by the control points placed on both (at
    least three): scale, rotation and translation by least squares. Returns the record
    stored on the moved scan (join_record): each point's distance after the fit, the
    points that look misplaced."""
    from . import join
    if moving == fixed:
        raise ValueError("choose another scan to join onto")
    slots, P, Q = join_pairs(moving, fixed)
    fit = join.fit_points(P, Q)
    transform_root(moving, Matrix(fit["matrix"].tolist()))
    return _join_record(moving, fixed, slots, fit, scan_extent(fixed))


def scan_points(root, limit=None, seed=0):
    """World positions of a scan's dense cloud (its meshes' vertices when it has none), at
    most ``limit`` of them (an even random sample)."""
    import numpy as np
    objs = _children(root, CLOUD_PROP) or [o for o in root.children_recursive
                                           if o.type == "MESH" and MESH_PROP in o.keys()]
    parts = []
    for obj in objs:
        n = len(obj.data.vertices)
        if not n:
            continue
        co = np.empty(3 * n, np.float32)
        obj.data.vertices.foreach_get("co", co)
        M = np.array(obj.matrix_world)
        parts.append(co.reshape(-1, 3).astype(float) @ M[:3, :3].T + M[:3, 3])
    if not parts:
        return np.zeros((0, 3))
    pts = np.vstack(parts)
    if limit and len(pts) > limit:
        pts = pts[np.random.default_rng(seed).choice(len(pts), limit, replace=False)]
    return pts


def scan_extent(root):
    """The size of a scan: the diagonal of the box holding 98% of its points."""
    import numpy as np
    pts = scan_points(root, limit=200_000)
    if not len(pts):
        return scan_size(root)
    lo, hi = np.percentile(pts, 1, axis=0), np.percentile(pts, 99, axis=0)
    return float(np.linalg.norm(hi - lo)) or scan_size(root)


def refine_join(moving, fixed):
    """Refine the join of ``moving`` onto ``fixed`` where their clouds overlap (join.icp,
    with scale; the control points stay in the fit). The scan is moved only when the clouds
    come clearly closer (join.icp's "improved"); otherwise the points' join stays, and the
    record says so (refined["kept"] False). Returns the updated record."""
    from . import join
    import numpy as np
    rec = join_record(moving)
    if not rec or rec.get("onto") != fixed.name:
        raise ValueError("join the scans by their points first")
    size = scan_extent(fixed)
    target = scan_points(fixed, limit=JOIN_TREE_POINTS)
    source = scan_points(moving, limit=4 * join.ICP_SAMPLES, seed=1)
    if len(target) < 100 or len(source) < 100:
        raise ValueError("both scans need a point cloud or a mesh")
    slots, P, Q = join_pairs(moving, fixed)
    nearest = join.kdtree_nearest(target)
    start = max(3.0 * float(rec.get("rms") or 0.0), 0.01 * size)
    out = join.icp(source, nearest, np.eye(4), size, controls=(P, Q) if len(slots) >= 3 else None,
                   start_distance=start)
    info = {"median_before": out["median_before"], "median_after": out["median_after"],
            "iterations": out["iterations"]}
    if not out["improved"]:
        rec["refined"] = dict(info, kept=False)
        moving[JOIN_RESULT_PROP] = json.dumps(rec)
        return rec
    transform_root(moving, Matrix(out["matrix"].tolist()))
    slots, P, Q = join_pairs(moving, fixed)
    res = np.linalg.norm(np.array(P) - np.array(Q), axis=1) if slots else np.zeros(0)
    rec.update(points=slots, residuals=[float(r) for r in res],
               rms=float(np.sqrt((res ** 2).mean())) if len(res) else rec.get("rms"),
               refined=dict(info, kept=True, rms_before=out["rms_before"], rms=out["rms"], overlap=out["overlap"],
                            scale=float(np.cbrt(abs(np.linalg.det(out["matrix"][:3, :3]))))))
    moving[JOIN_RESULT_PROP] = json.dumps(rec)
    return rec


def join_transform(moving, fixed):
    """4 x 4 (numpy): the reconstruction coordinates of the scan ``moving`` into those of
    ``fixed``, from where both are in the scene: W_fixed^-1 W_moving. A joined run puts the
    second scan's cameras and points through it (joinrun.py), so it must be a similarity:
    a scan stretched unevenly or mirrored by hand is refused."""
    import numpy as np
    T = np.linalg.inv(np.array(fixed.matrix_world, float)) @ np.array(moving.matrix_world, float)
    A = T[:3, :3]
    det = float(np.linalg.det(A))
    if det <= 0:
        raise ValueError(f"{moving.name} or {fixed.name} is mirrored: undo the negative scale")
    s = det ** (1.0 / 3.0)
    if np.abs(A.T @ A / s ** 2 - np.eye(3)).max() > 1e-3:
        raise ValueError(f"{moving.name} or {fixed.name} is scaled unevenly: give it the same scale on "
                         "all three axes")
    return T


def _main_crop_box(root):
    """The scan's crop box: a visible one first."""
    boxes = crop_boxes(root)

    def hidden(box):
        try:
            return box.hide_get()
        except RuntimeError:   # not in the view layer
            return True
    boxes.sort(key=hidden)
    return boxes[0] if boxes else None


def scan_box(root):
    """A scan's crop box in its reconstruction coordinates: its box object, else, for a scan
    imported finished without its crop stage (no cloud), the box its run was meshed in (Build
    Mesh's, else the one a 0.3.13 joined run was meshed in); None otherwise (a crop stage's box
    deleted counts for nothing)."""
    from . import roi
    box = _main_crop_box(root)
    if box is not None:
        return crop_box_obb(box)
    if _children(root, CLOUD_PROP):
        return None
    project = root.get(ROOT_PROP, "")
    for path in ((os.path.join(project, "mvs", "crop_roi.txt"), os.path.join(project, "crop_roi.txt"))
                 if project else ()):
        if os.path.isfile(path):
            try:
                return roi.OBB.load(path)
            except (OSError, ValueError):
                pass
    return None


def joined_box(fixed, moving, T):
    """The crop box of the run joining ``moving`` onto ``fixed``, in the reconstruction
    coordinates of ``fixed``: with the axes of its crop box (upright when that scan was
    aligned to the ground), large enough to hold both scans' boxes. Each video's cloud often
    covers one side of the object only, so either box alone could cut the other side off.
    None when neither scan has a box."""
    obb_a, obb_b = scan_box(fixed), scan_box(moving)
    if obb_b is not None:
        obb_b = obb_b.moved(T)
    if obb_a is None or obb_b is None:
        return obb_a or obb_b
    return obb_a.enclosing(obb_b.corners())


def _bounds_size(objs):
    pts = []
    for o in objs:
        pts += [o.matrix_world @ Vector(c) for c in o.bound_box]
    if not pts:
        return 1.0
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return (hi - lo).length


def _frame(objs):
    if bpy.app.background:
        return
    try:
        view_layer = bpy.context.view_layer
        for o in view_layer.objects:
            o.select_set(False)
        for o in objs:
            o.select_set(True)
        view_layer.objects.active = objs[0]
        override = _ui_override()
        if "area" in override:
            with bpy.context.temp_override(**override):
                bpy.ops.view3d.view_selected()
    except RuntimeError:
        pass


def load_result_json(project_dir):
    path = current(os.path.join(project_dir, "photogram3d_result.json"))
    with open(path, "r", encoding="utf-8") as f:
        return rebase_result(json.load(f), project_dir)


def rebase_result(result, project_dir):
    """Point the paths of a result file into ``project_dir``, where it was found.

    The file holds absolute paths, so after the project folder was moved or copied they
    still name the old place (the mesh "not found", the alignment and scale not applied)."""
    old = result.get("project") or ""
    if not old or not project_dir or _norm(old) == _norm(project_dir):
        return result
    old_n = _norm(old)
    out = dict(result)
    for key, value in result.items():
        if not isinstance(value, str) or not value:
            continue
        v = _norm(value)
        if v == old_n or v.startswith(old_n.rstrip("\\/") + os.sep):
            out[key] = os.path.normpath(os.path.join(project_dir, os.path.relpath(value, old)))
    return out


def find_root(obj):
    while obj is not None:
        if ROOT_PROP in obj.keys():
            return obj
        obj = obj.parent
    return None
