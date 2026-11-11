# SPDX-License-Identifier: GPL-3.0-or-later
# Photogram3D - photos or video to a textured mesh with COLMAP + OpenMVS
# Copyright (C) 2026 Kruglov Iurii
# Authors: Kruglov Iurii, Anthropic Claude AI
# License: GNU General Public License (GPL) v3 or later, https://www.blender.org/about/license/
"""Minimal readers for COLMAP sparse models (binary and text formats)."""

import os
import struct

# COLMAP 4.2 camera model ids -> (name, number of parameters)
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
    11: ("RAD_TAN_THIN_PRISM_FISHEYE", 16),
    12: ("SIMPLE_DIVISION", 4),
    13: ("DIVISION", 5),
    14: ("SIMPLE_FISHEYE", 3),
    15: ("FISHEYE", 4),
    16: ("EUCM", 6),
}
MODEL_IDS = {v[0]: k for k, v in CAMERA_MODELS.items()}


def _read(f, fmt):
    size = struct.calcsize(fmt)
    data = f.read(size)
    if len(data) != size:
        raise ValueError("unexpected end of file")
    return struct.unpack(fmt, data)


def read_cameras_binary(path):
    cams = {}
    with open(path, "rb") as f:
        (num,) = _read(f, "<Q")
        for _ in range(num):
            cam_id, model_id, width, height = _read(f, "<IiQQ")
            if model_id not in CAMERA_MODELS:
                raise ValueError(f"unknown camera model id {model_id}")
            name, nparams = CAMERA_MODELS[model_id]
            params = _read(f, "<" + "d" * nparams)
            cams[cam_id] = {"model": name, "width": width, "height": height, "params": list(params)}
    return cams


def read_images_binary(path, read_points=False):
    images = {}
    with open(path, "rb") as f:
        (num,) = _read(f, "<Q")
        for _ in range(num):
            image_id, qw, qx, qy, qz, tx, ty, tz, cam_id = _read(f, "<I7dI")
            name = bytearray()
            while True:
                ch = f.read(1)
                if not ch:
                    raise ValueError("unexpected end of file in image name")
                if ch == b"\x00":
                    break
                name += ch
            (npts,) = _read(f, "<Q")
            if read_points:
                pts = [_read(f, "<ddq") for _ in range(npts)]
            else:
                f.seek(npts * 24, os.SEEK_CUR)
                pts = None
            images[image_id] = {
                "name": name.decode("utf-8", "replace"),
                "qvec": (qw, qx, qy, qz), "tvec": (tx, ty, tz),
                "camera_id": cam_id, "num_points2D": npts, "points2D": pts,
            }
    return images


def read_points3d_count_binary(path):
    with open(path, "rb") as f:
        (num,) = _read(f, "<Q")
    return num


def read_observations_binary(path):
    """The observations of a model's points (the sum of their track lengths): how many
    feature positions in the photos the solve explains."""
    total = 0
    with open(path, "rb") as f:
        (num,) = _read(f, "<Q")
        for _ in range(num):
            head = f.read(51)                          # id, xyz, rgb, error, track length
            if len(head) != 51:
                raise ValueError("unexpected end of file")
            (length,) = struct.unpack_from("<Q", head, 43)
            total += length
            f.seek(8 * length, os.SEEK_CUR)
    return total


def read_cameras_text(path):
    cams = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            el = line.split()
            cams[int(el[0])] = {"model": el[1], "width": int(el[2]), "height": int(el[3]),
                                "params": [float(x) for x in el[4:]]}
    return cams


def read_images_text(path):
    images = {}
    with open(path, "r", encoding="utf-8") as f:
        lines = [l for l in f if not l.startswith("#")]
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        el = line.split()
        image_id = int(el[0])
        q = tuple(float(x) for x in el[1:5])
        t = tuple(float(x) for x in el[5:8])
        images[image_id] = {"name": " ".join(el[9:]), "qvec": q, "tvec": t,
                            "camera_id": int(el[8]), "num_points2D": None, "points2D": None}
        i += 2  # skip the 2D points line
    return images


def image_observations(folder):
    """{image name: number of its keypoints that have a triangulated 3D point}."""
    import numpy as np
    out = {}
    ib = os.path.join(folder, "images.bin")
    if os.path.isfile(ib):
        with open(ib, "rb") as f:
            (num,) = _read(f, "<Q")
            for _ in range(num):
                _read(f, "<I7dI")
                name = bytearray()
                while (ch := f.read(1)) not in (b"\x00", b""):
                    name += ch
                (npts,) = _read(f, "<Q")
                ids = np.frombuffer(f.read(24 * npts), dtype="<i8")[2::3]  # (x, y, point3D_id)
                out[name.decode("utf-8", "replace")] = int((ids != -1).sum())
        return out
    with open(os.path.join(folder, "images.txt"), "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if not ln.startswith("#")]
    for i in range(0, len(lines) - 1, 2):
        el = lines[i].split()
        if len(el) >= 10:
            out[" ".join(el[9:])] = sum(1 for p in lines[i + 1].split()[2::3] if p != "-1")
    return out


def image_points(folder):
    """{image name: (xy float64 N x 2, point3D ids int64 N)} of the keypoints that have a
    triangulated 3D point, from a binary model (images.bin)."""
    import numpy as np
    out = {}
    with open(os.path.join(folder, "images.bin"), "rb") as f:
        (num,) = _read(f, "<Q")
        for _ in range(num):
            _read(f, "<I7dI")
            name = bytearray()
            while (ch := f.read(1)) not in (b"\x00", b""):
                name += ch
            (npts,) = _read(f, "<Q")
            raw = np.frombuffer(f.read(24 * npts), dtype=np.dtype([("x", "<f8"), ("y", "<f8"), ("id", "<i8")]))
            raw = raw[raw["id"] != -1]
            out[name.decode("utf-8", "replace")] = (np.stack([raw["x"], raw["y"]], 1), raw["id"].copy())
    return out


def read_model(folder):
    """Read cameras + images from ``folder`` (binary preferred)."""
    cb = os.path.join(folder, "cameras.bin")
    ib = os.path.join(folder, "images.bin")
    if os.path.isfile(cb) and os.path.isfile(ib):
        return read_cameras_binary(cb), read_images_binary(ib)
    ct = os.path.join(folder, "cameras.txt")
    it = os.path.join(folder, "images.txt")
    if os.path.isfile(ct) and os.path.isfile(it):
        return read_cameras_text(ct), read_images_text(it)
    raise FileNotFoundError(f"no COLMAP model in {folder}")


def count_registered(folder):
    """Number of registered images in a model folder (0 if unreadable)."""
    ib = os.path.join(folder, "images.bin")
    try:
        if os.path.isfile(ib):
            with open(ib, "rb") as f:
                return _read(f, "<Q")[0]
        it = os.path.join(folder, "images.txt")
        if os.path.isfile(it):
            return len(read_images_text(it))
    except (OSError, ValueError):
        return 0
    return 0


def find_models(sparse_dir):
    """Sub-folders of ``sparse_dir`` that hold a model, largest first."""
    found = []
    if os.path.isfile(os.path.join(sparse_dir, "images.bin")):
        found.append((count_registered(sparse_dir), sparse_dir))
    if os.path.isdir(sparse_dir):
        for entry in os.scandir(sparse_dir):
            if entry.is_dir():
                n = count_registered(entry.path)
                if n > 0:
                    found.append((n, entry.path))
    found.sort(key=lambda x: -x[0])
    return found


def qvec_to_rotmat(q):
    w, x, y, z = q
    return [
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ]


def rotmat_to_qvec(R):
    """Unit quaternion (w, x, y, z), w >= 0, of a rotation matrix (rows): the inverse of
    qvec_to_rotmat, from the largest of the four terms (sound near half a turn too)."""
    import math
    (m00, m01, m02), (m10, m11, m12), (m20, m21, m22) = [[float(v) for v in row] for row in R]
    tr = m00 + m11 + m22
    if tr > 0:
        s = 2.0 * math.sqrt(tr + 1.0)
        q = (0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s)
    elif m00 > m11 and m00 > m22:
        s = 2.0 * math.sqrt(1.0 + m00 - m11 - m22)
        q = ((m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s)
    elif m11 > m22:
        s = 2.0 * math.sqrt(1.0 + m11 - m00 - m22)
        q = ((m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s)
    else:
        s = 2.0 * math.sqrt(1.0 + m22 - m00 - m11)
        q = ((m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s)
    if q[0] < 0:
        q = tuple(-v for v in q)
    n = math.sqrt(sum(v * v for v in q)) or 1.0
    return tuple(v / n for v in q)


def write_text_model(folder, cameras, poses):
    """A COLMAP text model without points: ``cameras`` as read_model returns them,
    ``poses`` [(image id, name, camera id, world-to-camera rotation, centre)]."""
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "cameras.txt"), "w", encoding="utf-8") as f:
        for cid, c in sorted(cameras.items()):
            f.write(f"{cid} {c['model']} {c['width']} {c['height']} "
                    + " ".join(repr(float(v)) for v in c["params"]) + "\n")
    with open(os.path.join(folder, "images.txt"), "w", encoding="utf-8") as f:
        for iid, name, cid, R, C in poses:
            q = rotmat_to_qvec(R)
            t = [-sum(float(R[r][k]) * float(C[k]) for k in range(3)) for r in range(3)]
            f.write(f"{iid} " + " ".join(repr(v) for v in q) + " " + " ".join(repr(v) for v in t)
                    + f" {cid} {name}\n\n")
    with open(os.path.join(folder, "points3D.txt"), "w", encoding="utf-8"):
        pass
    return folder


def camera_center(image):
    """World position of the camera: C = -R^T t."""
    r = qvec_to_rotmat(image["qvec"])
    t = image["tvec"]
    return [-(r[0][i] * t[0] + r[1][i] * t[1] + r[2][i] * t[2]) for i in range(3)]


def pinhole_intrinsics(cam):
    """(fx, fy, cx, cy) for pinhole-like models (distortion ignored)."""
    p = cam["params"]
    model = cam["model"]
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE",
                 "RADIAL_FISHEYE", "SIMPLE_DIVISION", "SIMPLE_FISHEYE"):
        return p[0], p[0], p[1], p[2]
    return p[0], p[1], p[2], p[3]


# ---------------------------------------------------------------------------
# Upright photos (0.3.9). A phone stores a photo taken holding it upright (portrait) with
# the sensor's pixels as they are, landscape, and an EXIF orientation that tells viewers to
# turn it. COLMAP solves the photo as stored (a camera lying on its side, which is right
# for those pixels) and keeps the turn as a gravity direction per photo in its database.
# ---------------------------------------------------------------------------

def rot90_from_gravity(gravity):
    """Counter-clockwise quarter turns (0-3) that make a photo upright, from the gravity
    direction COLMAP keeps for it (x, y in the stored picture; y down is upright): COLMAP's
    ComputeRot90FromGravity. The same as ai_features.rot90_from_gravity."""
    import math
    if gravity is None:
        return 0
    angle = math.atan2(gravity[1], gravity[0])
    x = (angle - math.pi / 2) / (math.pi / 2)
    k = math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5)      # std::round
    k = int(math.fmod(k, 4))
    return k + 4 if k < 0 else k


EXIF_TURNS = {1: 0, 3: 2, 6: 3, 8: 1}     # EXIF orientation -> quarter turns (mirrored ones: none)


def gravity_priors(db_path):
    """{photo name: (gx, gy, gz)}: the gravity COLMAP keeps for a photo (pose_priors,
    COLMAP 4.2), in the stored picture's camera frame (y down is upright)."""
    import sqlite3
    if not db_path or not os.path.isfile(db_path):
        return {}
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
    except sqlite3.Error:
        return {}
    try:
        names = dict(con.execute("SELECT image_id, name FROM images"))
        out = {}
        for data_id, blob in con.execute("SELECT corr_data_id, gravity FROM pose_priors"):
            if data_id not in names or not blob or len(blob) < 24:
                continue
            g = struct.unpack("<3d", blob[:24])
            if all(v == v and abs(v) != float("inf") for v in g) and any(g):     # finite, set
                out[names[data_id]] = g
        return out
    except sqlite3.Error:
        return {}
    finally:
        con.close()


def upright_turns(db_path):
    """{photo name: quarter turns} for the photos COLMAP did not take upright, from the
    gravity in its database. Empty when there is none."""
    out = {}
    for name, g in gravity_priors(db_path).items():
        k = rot90_from_gravity((g[0], g[1]))
        if k:
            out[name] = k
    return out


def consensus_axis(axes, max_distance=0.05):
    """COLMAP's FindBestConsensusAxis: the mean of the largest group of unit axes within
    ``max_distance`` (1 - cosine) of one of them."""
    best, best_sum = [], float("inf")
    for i, ref in enumerate(axes):
        group, total = [], 0.0
        for j, a in enumerate(axes):
            d = 0.0 if i == j else 1.0 - sum(x * y for x, y in zip(ref, a))
            if d <= max_distance:
                group.append(j)
                total += d
        if len(group) > len(best) or (len(group) == len(best) and total < best_sum):
            best, best_sum = group, total
    if not best:
        return None
    return [sum(axes[j][c] for j in best) / len(best) for c in range(3)]


def downward_axes(model_dir, gravity=None):
    """Each solved photo's downward direction in the model's world: the stored picture's y
    axis (what COLMAP's IMAGE-ORIENTATION alignment assumes), or, for a photo with a
    gravity in ``gravity`` ({name: (gx, gy, gz)}), that direction (a portrait photo of a
    phone, stored on its side, looks down along its x axis)."""
    import math
    _, images = read_model(model_dir)
    out = []
    for im in images.values():
        r = qvec_to_rotmat(im["qvec"])                # rows: the camera's axes in the world
        gx, gy, _ = (gravity or {}).get(im["name"], (0.0, 1.0, 0.0))
        n = math.hypot(gx, gy) or 1.0
        a = [(gx * r[0][c] + gy * r[1][c]) / n for c in range(3)]
        out.append(a)
    return out


def rotation_between(a, b):
    """Unit quaternion (w, x, y, z) of the smallest rotation that turns direction a to b."""
    import math
    na = math.sqrt(sum(v * v for v in a)) or 1.0
    nb = math.sqrt(sum(v * v for v in b)) or 1.0
    a = [v / na for v in a]
    b = [v / nb for v in b]
    c = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
    d = sum(x * y for x, y in zip(a, b))
    if d < -1.0 + 1e-9:                     # opposite: half a turn about any perpendicular axis
        p = [0.0, -a[2], a[1]] if abs(a[0]) < 0.9 else [-a[1], a[0], 0.0]
        n = math.sqrt(sum(v * v for v in p))
        return (0.0, p[0] / n, p[1] / n, p[2] / n)
    w = 1.0 + d
    n = math.sqrt(w * w + sum(v * v for v in c))
    return (w / n, c[0] / n, c[1] / n, c[2] / n)


def turned_intrinsics(fx, fy, cx, cy, w, h, k):
    """The camera of the photo turned by k counter-clockwise quarter turns: (fx, fy, cx, cy,
    w, h) of the upright picture. Pixel (u, v) of the stored photo is, upright:
    k=1 (v, w - u), k=2 (w - u, h - v), k=3 (h - v, u)."""
    k %= 4
    if k == 1:
        return fy, fx, cy, w - cx, h, w
    if k == 2:
        return fx, fy, w - cx, h - cy, w, h
    if k == 3:
        return fy, fx, h - cy, cx, h, w
    return fx, fy, cx, cy, w, h
