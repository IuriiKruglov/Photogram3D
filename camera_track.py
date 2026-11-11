# SPDX-License-Identifier: GPL-3.0-or-later
# Photogram3D - photos or video to a textured mesh with COLMAP + OpenMVS
# Copyright (C) 2026 Kruglov Iurii
# Authors: Kruglov Iurii, Anthropic Claude AI
# License: GNU General Public License (GPL) v3 or later, https://www.blender.org/about/license/
"""Video: the camera on every frame (0.3.9).

The reconstruction solves a few frames of the video: Auto keeps one per 12% of the picture's
movement, which on a walk round a vehicle is one every 1-2 seconds, 60-120 frames of a 60 fps
video apart. Enough for the mesh, but a camera keyed only on those frames is interpolated
across the frames between them, and a hand-held camera does not move along a smooth curve:
the scan slides against the footage between two keyframes and meets it again at the next.

The track step places the camera on every frame (or every second frame), keeping the solved
frames exactly where the reconstruction put them:

1. FFmpeg writes every frame of the part of the video the reconstruction used, at the size of
   the solved frames (photo_max_size), so the solved frames come out pixel for pixel again.
2. COLMAP finds SIFT features on them with the lens fixed to the solved one, and matches each
   frame with frames STEPS_AHEAD after it and with the two solved frames around it; each
   solved frame also with the next KEY_LINKS solved frames.
3. The solved frames go in as a model with fixed poses: point_triangulator makes 3D points
   from their matches (poses and lens fixed), image_registrator places every other frame
   against those points (P3P RANSAC, then the pose refined on its inliers). When frames are
   left out, a second round makes points from the placed frames within NEAR_MISSING frames
   of them (and the solved ones) and places them on those; the frames placed in the first
   round keep their places. (Until 0.3.9 the second round triangulated with every placed
   frame and adjusted the points over all of them: 11 of the 26 minutes of 3,600 frames.)
4. track.json: each placed frame's time and pose, in the reconstruction's coordinates.

SIFT whatever the reconstruction used: it runs on any GPU build of COLMAP and on the CPU, and
the tracking database is its own (the reconstruction's database is not touched).
"""

import json
import math
import os

from . import colmap_io

TRACK_DIR = "track"
TRACK_JSON = "track.json"
STEPS_AHEAD = (1, 2, 3, 5, 8, 13)
KEY_LINKS = 3
SIFT_MAX_SIDE = 1600
SIFT_FEATURES = 2048      # enough for a pose; 4096 would make matching four times slower
JPEG_QUALITY = 3          # FFmpeg -q:v (2 is the reconstruction's; the track only needs features)
NEAR_MISSING = 15         # second round: points from the placed frames this close to a missing one


def paths(project):
    root = os.path.join(project, TRACK_DIR)
    return {"root": root, "frames": os.path.join(root, "frames"), "db": os.path.join(root, "track.db"),
            "pairs": os.path.join(root, "pairs.txt"), "seed": os.path.join(root, "seed"),
            "tri": os.path.join(root, "tri"), "reg": os.path.join(root, "reg"),
            "json": os.path.join(root, TRACK_JSON)}


def load(project):
    """track.json of a project, or None."""
    try:
        with open(os.path.join(project, TRACK_DIR, TRACK_JSON), encoding="utf-8") as f:
            data = json.load(f)
        return data if data.get("frames") else None
    except (OSError, ValueError):
        return None


def frame_number(t, stream_start, fps):
    """The frame's number counted from the video stream's first frame, as Blender counts the
    frames of a movie clip (time = stream start + number / fps)."""
    return int(math.floor((t - stream_start) * fps + 0.5))


def make_pairs(names, keys):
    """Image pairs to match: ``names`` all frames in time order, ``keys`` the indices of the
    solved ones among them (sorted)."""
    out = set()
    n = len(names)
    for i in range(n):
        for d in STEPS_AHEAD:
            if i + d < n:
                out.add((i, i + d))
    if keys:
        import bisect
        for i in range(n):
            j = bisect.bisect_right(keys, i)
            for k in (keys[j - 1] if j > 0 else None, keys[j] if j < len(keys) else None):
                if k is not None and k != i:
                    out.add((min(i, k), max(i, k)))
        for a in range(len(keys)):
            for b in range(a + 1, min(len(keys), a + 1 + KEY_LINKS)):
                out.add((keys[a], keys[b]))
    return [(names[a], names[b]) for a, b in sorted(out)]


def write_pairs(path, pairs):
    with open(path, "w", encoding="utf-8") as f:
        for a, b in pairs:
            f.write(f"{a} {b}\n")
    return len(pairs)


def db_cameras_and_images(db):
    """({image name: (image_id, camera_id)}, {camera_id: (model, w, h)}) of a COLMAP database."""
    import sqlite3
    con = sqlite3.connect(db, timeout=60)
    try:
        images = {name: (iid, cid)
                  for iid, name, cid in con.execute("SELECT image_id, name, camera_id FROM images")}
        cams = {cid: (model, w, h)
                for cid, model, w, h in con.execute("SELECT camera_id, model, width, height FROM cameras")}
        return images, cams
    finally:
        con.close()


def camera_line(cam_id, cam, width, height):
    """cameras.txt line for COLMAP's camera ``cam`` (as read by colmap_io) at another size."""
    sx, sy = width / float(cam["width"]), height / float(cam["height"])
    p = list(cam["params"])
    model = cam["model"]
    # focal lengths and principal point scale with the picture, distortion does not
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE",
                 "SIMPLE_DIVISION", "SIMPLE_FISHEYE"):
        p[0] *= sx
        p[1] *= sx
        p[2] *= sy
    else:
        p[0] *= sx
        p[1] *= sy
        p[2] *= sx
        p[3] *= sy
    return f"{cam_id} {model} {width} {height} " + " ".join(f"{v:.17g}" for v in p), p


def write_seed(seed_dir, model_dir, keyframes, db, size):
    """A COLMAP text model of the solved frames (``keyframes``: {track frame name: solved
    image name}) with their poses, the tracking database's camera with the solved lens, and
    no points. Returns the camera parameters written."""
    cams, images = colmap_io.read_model(model_dir)
    by_name = {im["name"]: im for im in images.values()}
    db_images, db_cams = db_cameras_and_images(db)
    os.makedirs(seed_dir, exist_ok=True)
    first = next(iter(cams.values()))
    cam_id = next(iter(db_cams))
    line, params = camera_line(cam_id, first, *size)
    with open(os.path.join(seed_dir, "cameras.txt"), "w", encoding="utf-8") as f:
        f.write(line + "\n")
    with open(os.path.join(seed_dir, "images.txt"), "w", encoding="utf-8") as f:
        for track_name, solved in sorted(keyframes.items()):
            im = by_name[solved]
            iid, cid = db_images[track_name]
            q, t = im["qvec"], im["tvec"]
            f.write(f"{iid} {q[0]:.17g} {q[1]:.17g} {q[2]:.17g} {q[3]:.17g} {t[0]:.17g} {t[1]:.17g} "
                    f"{t[2]:.17g} {cid} {track_name}\n\n")
    with open(os.path.join(seed_dir, "points3D.txt"), "w", encoding="utf-8") as f:
        f.write("")
    return params


def write_posed(dest, seed_dir, poses, db):
    """A COLMAP text model of the frames in ``poses`` ({name: (qvec, tvec)}) with the seed's
    camera and no points (the second round's input)."""
    db_images, _ = db_cameras_and_images(db)
    os.makedirs(dest, exist_ok=True)
    with open(os.path.join(seed_dir, "cameras.txt"), "r", encoding="utf-8") as f:
        cameras = f.read()
    with open(os.path.join(dest, "cameras.txt"), "w", encoding="utf-8") as f:
        f.write(cameras)
    with open(os.path.join(dest, "images.txt"), "w", encoding="utf-8") as f:
        for name in sorted(poses):
            iid, cid = db_images[name]
            q, t = poses[name]
            f.write(f"{iid} {q[0]:.17g} {q[1]:.17g} {q[2]:.17g} {q[3]:.17g} {t[0]:.17g} {t[1]:.17g} "
                    f"{t[2]:.17g} {cid} {name}\n\n")
    with open(os.path.join(dest, "points3D.txt"), "w", encoding="utf-8") as f:
        f.write("")


def near_missing(names, placed, keyframes, reach=NEAR_MISSING):
    """The placed frames within ``reach`` frames (in ``names`` order) of one that is not
    placed, and the solved frames: the second round's points come from them."""
    out = {n for n in keyframes if n in placed}
    for i, name in enumerate(names):
        if name not in placed:
            out.update(m for m in names[max(0, i - reach):i + reach + 1] if m in placed)
    return out


def read_poses(model_dir):
    """{image name: (qvec, tvec)} of the placed images of a model."""
    _, images = colmap_io.read_model(model_dir)
    return {im["name"]: (list(im["qvec"]), list(im["tvec"])) for im in images.values()}


def save(path, meta, frames, poses, keyframes, every):
    """track.json: the placed frames in time order with their time and pose."""
    out = []
    for name, n, t in frames:
        if name in poses:
            q, tv = poses[name]
            out.append({"file": name, "n": n, "t": round(t, 6), "qvec": [round(v, 12) for v in q],
                        "tvec": [round(v, 12) for v in tv], "key": name in keyframes})
    data = {"version": 1, "video": meta.get("video"), "fps": meta.get("fps"), "clip_fps": meta.get("clip_fps"),
            "stream_start": meta.get("stream_start", 0.0), "every": every, "total": len(frames),
            "placed": len(out), "keyframes": len(keyframes), "frames": out}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)
    return data
