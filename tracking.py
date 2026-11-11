# SPDX-License-Identifier: GPL-3.0-or-later
# Photogram3D - photos or video to a textured mesh with COLMAP + OpenMVS
# Copyright (C) 2026 Kruglov Iurii
# Authors: Kruglov Iurii, Anthropic Claude AI
# License: GNU General Public License (GPL) v3 or later, https://www.blender.org/about/license/
"""Video input: one camera that follows the video (0.3.4; every frame since 0.3.9).

The camera is keyed at the scene frame that shows each frame of the movie clip: Blender reads a
movie at FFmpeg's guess of its frame rate and counts its frames from the video stream's first
timestamp, so a frame at time t (from the container's start, as FFmpeg's -ss counts) is clip
frame 1 + (t - stream start) * rate, and the scene runs at that rate with the clip starting at
frame 1. With a camera track (camera_track.py, 0.3.9) the camera has a key on every frame (or
every second frame) and follows the footage; without one it has keys on the solved frames only
and is interpolated in between, where a hand-held camera's real path differs from the curve.
The camera's background is the video itself, undistorted with the lens COLMAP solved (Render
Undistorted).

The lens goes on the movie clip with Blender's conventions, measured by undistorting a
test image in the compositor (Movie Distortion node) against COLMAP's models: the
principal point counts y from the bottom (H - cy), and Blender's Brown model takes
COLMAP/OpenCV's tangential terms as brown_p1 = p2, brown_p2 = -p1 (0.1-0.6 px apart).
"""

import os

import bpy
from mathutils import Matrix, Quaternion

from . import colmap_io, video

TRACK_PROP = "photogram3d_tracked_camera"
KEYS_PROP = "photogram3d_track_keys"          # on the tracked camera: what its keys come from
# A joined scan has a tracked camera for each video it was joined from (0.3.15): the video's
# run folder, the matrix taking that run's solve into the joined scan's coordinates, and the
# id of that solve (a camera track of a later solve would be elsewhere).
SOURCE_PROP = "photogram3d_track_source"
MATRIX_PROP = "photogram3d_track_matrix"
SOLVE_PROP = "photogram3d_track_solve"


def scene_fps(fps):
    """(fps, fps_base) for Blender's integer frame rate: 29.97 -> (30, 1.001)."""
    fps = float(fps)
    if fps <= 0:
        return 24, 1.0
    n = round(fps)
    if n >= 1 and abs(n - fps) < 0.01:
        return int(n), 1.0
    n = round(fps * 1.001)
    if n >= 1 and abs(n / 1.001 - fps) < 0.01:
        return int(n), 1.001
    n = max(1, round(fps))
    return int(n), n / fps


def set_clip_lens(clip, cam):
    """Put COLMAP's camera (with its distortion) on the movie clip's tracking camera."""
    t = clip.tracking.camera
    w, h = int(cam["width"]), int(cam["height"])
    sx = clip.size[0] / w if clip.size[0] else 1.0
    sy = clip.size[1] / h if clip.size[1] else 1.0
    fx, fy, cx, cy = colmap_io.pinhole_intrinsics(cam)
    p = cam["params"]
    t.units = "PIXELS"
    t.pixel_aspect = 1.0
    t.focal_length_pixels = fx * sx
    t.principal_point_pixels = (cx * sx, (h - cy) * sy)
    model = cam["model"]
    if model in ("OPENCV", "FULL_OPENCV"):
        t.distortion_model = "BROWN"
        t.brown_k1, t.brown_k2 = p[4], p[5]
        t.brown_k3 = p[8] if model == "FULL_OPENCV" else 0.0
        t.brown_k4 = 0.0
        t.brown_p1, t.brown_p2 = p[7], -p[6]
    else:
        t.distortion_model = "POLYNOMIAL"
        t.k1 = p[3] if model in ("SIMPLE_RADIAL", "RADIAL") else 0.0
        t.k2 = p[4] if model == "RADIAL" else 0.0
        t.k3 = 0.0


def _load_clip(path):
    for clip in bpy.data.movieclips:
        if clip.filepath and os.path.normcase(bpy.path.abspath(clip.filepath)) == os.path.normcase(path):
            return clip
    return bpy.data.movieclips.load(path)


def _world_from_camera(qvec, tvec):
    from .importer import CAMERA_FLIP
    r = colmap_io.qvec_to_rotmat(qvec)
    c = colmap_io.camera_center({"qvec": qvec, "tvec": tvec})
    m = Matrix(((r[0][0], r[1][0], r[2][0], c[0]),
                (r[0][1], r[1][1], r[2][1], c[1]),
                (r[0][2], r[1][2], r[2][2], c[2]),
                (0.0, 0.0, 0.0, 1.0)))
    return m @ CAMERA_FLIP


def clip_frame(meta, f):
    """The scene frame (with the clip starting at frame 1) that shows video frame ``f`` of
    frames.json or track.json: from its time when the file has the stream's start and the
    clip's rate (0.3.9), else from its number."""
    rate = meta.get("clip_fps")
    if rate and "stream_start" in meta and "t" in f:
        return 1.0 + (float(f["t"]) - float(meta["stream_start"])) * float(rate)
    return float(int(f["n"]) + 1)


def _moved(keys, transform):
    """Keys moved by a similarity (4 x 4 rows, or None): the camera's place by all of it,
    its turn by its rotation (key_camera keeps no scale)."""
    if transform is None:
        return keys
    T = Matrix([list(map(float, row)) for row in transform])
    return [(f, T @ m) for f, m in keys]


def _keyframes(model_dir, frames, track=None):
    """[(scene frame, local camera matrix)] in time order: every tracked frame of ``track``
    (track.json), else the solved frames of ``model_dir`` found in frames.json."""
    keys = []
    if track and track.get("frames"):
        for f in track["frames"]:
            keys.append((clip_frame(track, f), _world_from_camera(f["qvec"], f["tvec"])))
    else:
        _, images = colmap_io.read_model(model_dir)
        by_file = {f["file"]: f for f in frames.get("frames", [])}
        for im in images.values():
            f = by_file.get(im["name"])
            if f is not None:
                keys.append((clip_frame(frames, f), _world_from_camera(im["qvec"], im["tvec"])))
    keys.sort(key=lambda k: k[0])
    return keys


def _track_for(result):
    from . import camera_track
    project = result.get("project") or ""
    return camera_track.load(project) if project else None


def _key_source(track, keys):
    return f"track:{len(keys)}" if track and track.get("frames") else f"solved:{len(keys)}"


def _channelbag_fcurves(obj):
    """The F-curves of an object's action (Blender 4.4+ layered actions, or the old list)."""
    ad = obj.animation_data
    if ad is None or ad.action is None:
        return []
    try:
        from bpy_extras import anim_utils
        bag = anim_utils.action_get_channelbag_for_slot(ad.action, ad.action_slot)
        if bag is not None:
            return list(bag.fcurves)
    except (ImportError, AttributeError):
        pass
    return list(getattr(ad.action, "fcurves", []) or [])


def key_camera(obj, keys):
    """Location and rotation keys on ``obj`` at the (possibly fractional) scene frames of
    ``keys``, the shortest way round between neighbouring rotations. Thousands of keys are
    written into the F-curves at once."""
    import numpy as np
    if obj.animation_data is not None:
        obj.animation_data_clear()
    obj.rotation_mode = "QUATERNION"
    frames = np.empty(len(keys))
    locs = np.empty((len(keys), 3))
    rots = np.empty((len(keys), 4))
    prev = None
    for i, (frame, m) in enumerate(keys):
        loc, rot, _ = m.decompose()
        if prev is not None and rot.dot(prev) < 0.0:      # the short way round
            rot.negate()
        prev = rot
        frames[i], locs[i], rots[i] = frame, loc, rot
    # one key per channel makes the action, its slot and the F-curves
    obj.location = locs[0].tolist()
    obj.rotation_quaternion = Quaternion(rots[0].tolist())
    obj.keyframe_insert("location", frame=float(frames[0]))
    obj.keyframe_insert("rotation_quaternion", frame=float(frames[0]))
    curves = {(fc.data_path, fc.array_index): fc for fc in _channelbag_fcurves(obj)}
    if len(curves) != 7:                                    # unknown layout: one by one
        for frame, loc, rot in zip(frames[1:], locs[1:], rots[1:]):
            obj.location = loc.tolist()
            obj.rotation_quaternion = Quaternion(rot.tolist())
            obj.keyframe_insert("location", frame=float(frame))
            obj.keyframe_insert("rotation_quaternion", frame=float(frame))
        return len(keys)
    for (path, idx), fc in curves.items():
        values = (locs if path == "location" else rots)[:, idx]
        fc.keyframe_points.clear()
        fc.keyframe_points.add(len(keys))
        co = np.empty(2 * len(keys), dtype=np.float32)
        co[0::2], co[1::2] = frames, values
        fc.keyframe_points.foreach_set("co", co)       # new points: Bezier, auto-clamped handles
        fc.update()
    return len(keys)


def _frames_meta(result):
    frames = video.load_frames_json(os.path.dirname(result["frames_json"]))
    if not frames:
        raise ValueError("frames.json is missing: extract the frames with Photogram3D 0.3.4")
    return frames


def _set_scene_timing(scene, meta, keys):
    scene.render.fps, scene.render.fps_base = scene_fps(meta.get("clip_fps") or meta.get("fps") or 30.0)
    scene.frame_start = int(keys[0][0])
    scene.frame_end = int(round(keys[-1][0]))


def _widen_range(scene, keys):
    scene.frame_start = min(scene.frame_start, int(keys[0][0]))
    scene.frame_end = max(scene.frame_end, int(round(keys[-1][0])))


def _source_of(obj):
    return obj.get(SOURCE_PROP, "")


def ensure_tracked_camera(result, root, coll, source="", transform=None, name="", set_scene=True,
                          solve=None):
    """The root's tracked camera, created from the video run's result when missing; an
    existing one gets the camera track's keys when the track is newer than its keys.

    A joined scan (0.3.15) has one for each video it was joined from: ``source`` names the
    video's run (its folder), ``transform`` takes that run's solve (id ``solve``) into the
    scan's coordinates, ``name`` the camera. ``set_scene``: the scene's frame range, frame
    rate, render size and camera follow this one; otherwise the frame range only widens to
    hold it too."""
    from .importer import _camera_data, _camera_scale
    frames = _frames_meta(result)
    project = result.get("project") or ""
    model_dir = result.get("model_distorted") or os.path.join(project, "colmap", "model")
    track = _track_for(result)
    for obj in root.children:
        if TRACK_PROP in obj.keys() and _source_of(obj) == source:
            if track and track.get("frames") and obj.get(KEYS_PROP, "") != _key_source(track, track["frames"]):
                retrack(obj, result, track)
            return obj
    cams, images = colmap_io.read_model(model_dir)
    keys = _moved(_keyframes(model_dir, frames, track), transform)
    if len(keys) < 2:
        raise ValueError("fewer than two video frames were solved")
    cam = next(iter(cams.values()))
    if not name:
        from .importer import run_names
        name = f"{run_names(result)[2]}_video_camera"
    scale = _camera_scale(images)
    if transform is not None:                      # the joined scan's units
        scale *= Matrix([list(map(float, row)) for row in transform]).to_scale()[0]
    data = _camera_data(name, cam, scale)
    obj = bpy.data.objects.new(name, data)
    coll.objects.link(obj)
    obj.parent = root
    obj.matrix_parent_inverse = Matrix.Identity(4)
    obj[TRACK_PROP] = True
    if source:
        obj[SOURCE_PROP] = source
        if transform is not None:
            obj[MATRIX_PROP] = [float(v) for row in transform for v in row]
        if solve:
            obj[SOLVE_PROP] = str(solve)
    data.display_size *= 2.0
    key_camera(obj, keys)
    obj[KEYS_PROP] = _key_source(track, keys)

    scene = bpy.context.scene
    if set_scene:
        _set_scene_timing(scene, track if track and track.get("frames") else frames, keys)
        scene.frame_set(int(round(keys[0][0])))
        scene.render.resolution_x, scene.render.resolution_y = int(cam["width"]), int(cam["height"])
        scene.render.resolution_percentage = 100
        scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1.0
        scene.camera = obj
    else:                                          # another video of the scan: the range holds both
        _widen_range(scene, keys)

    path = result.get("video") or frames.get("video") or ""
    if path and os.path.isfile(path):
        clip = _load_clip(path)
        set_clip_lens(clip, cam)
        clip.frame_start = 1          # scene frame n + 1 shows video frame n
        clip.frame_offset = 0
        data.show_background_images = True
        bg = data.background_images.new()
        bg.source = "MOVIE_CLIP"
        bg.clip = clip
        bg.use_camera_clip = False
        bg.clip_user.use_render_undistorted = True
        bg.alpha = 1.0
        bg.display_depth = "BACK"
        bg.frame_method = "STRETCH"
    return obj


def _stored_transform(obj):
    flat = obj.get(MATRIX_PROP)
    if flat is None or len(flat) != 16:
        return None
    flat = [float(v) for v in flat]
    return [flat[0:4], flat[4:8], flat[8:12], flat[12:16]]


def retrack(obj, result, track=None):
    """Key an existing tracked camera from the project's camera track (every frame); a joined
    scan's camera (0.3.15) through the matrix it keeps."""
    track = track or _track_for(result)
    if not track or not track.get("frames"):
        return 0
    keys = _moved(_keyframes(None, None, track), _stored_transform(obj))
    n = key_camera(obj, keys)
    obj[KEYS_PROP] = _key_source(track, keys)
    scene = bpy.context.scene
    others = [o for o in tracked_cameras(obj.parent) if o is not obj] if obj.parent is not None else []
    if not others:
        _set_scene_timing(scene, track, keys)
    else:              # a joined scan's: the frame range keeps the other videos (and the rate theirs)
        if scene.camera is obj:
            scene.render.fps, scene.render.fps_base = scene_fps(track.get("clip_fps") or track.get("fps") or 30.0)
        _widen_range(scene, keys)
    return n


def source_solved_again(obj):
    """The run a joined scan's tracked camera follows was solved again since the join: a
    camera track made now would not fit the scan. False for any other camera."""
    solve = obj.get(SOLVE_PROP)
    if not solve or not obj.get(SOURCE_PROP):
        return False
    from . import pipeline
    now = pipeline.solve_id(pipeline.project_paths(obj[SOURCE_PROP]))
    return now is not None and now != solve          # None: the run is not there (moved, deleted)


def tracked_cameras(root):
    return [obj for obj in root.children if TRACK_PROP in obj.keys()]


def tracked_camera(root, prefer=None):
    """The root's tracked camera: ``prefer`` when it is one of them (a joined scan has one
    for each video), else the first."""
    cams = tracked_cameras(root)
    if prefer is not None and prefer in cams:
        return prefer
    return cams[0] if cams else None


def track_project(obj, root):
    """The run folder whose video a tracked camera follows (its camera track goes there)."""
    from .importer import ROOT_PROP
    return _source_of(obj) or root.get(ROOT_PROP, "")


def track_state(root, prefer=None):
    """("every", n) when the root's tracked camera is keyed on a camera track, ("solved", n)
    when on the solved frames only, None without a tracked camera."""
    obj = tracked_camera(root, prefer) if root is not None else None
    if obj is None:
        return None
    kind, _, n = str(obj.get(KEYS_PROP, "solved:0")).partition(":")
    if not obj.get(KEYS_PROP):
        fcs = _channelbag_fcurves(obj)
        n = str(len(fcs[0].keyframe_points)) if fcs else "0"
    return ("every" if kind == "track" else "solved"), int(n or 0)
