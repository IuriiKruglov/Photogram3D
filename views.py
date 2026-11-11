# SPDX-License-Identifier: GPL-3.0-or-later
# Photogram3D - photos or video to a textured mesh with COLMAP + OpenMVS
# Copyright (C) 2026 Kruglov Iurii
# Authors: Kruglov Iurii, Anthropic Claude AI
# License: GNU General Public License (GPL) v3 or later, https://www.blender.org/about/license/
"""Photos far enough apart to be worth their cost (0.3.4).

A video gives hundreds of frames a fraction of a degree apart. Texturing gains nothing
from two views of the same spot from the same direction, and TextureMesh loads every
photo it is given. select_views() keeps the photos with the most solved points first and
drops a photo when a kept one looks at the same place, from within ``min_angle`` degrees
and from a similar distance; then it adds photos back until nearly every 3D point is
still seen by two kept photos, so no part of the scan loses its views.
"""

import math
import os
import struct

import numpy as np

from . import colmap_io

SAME_PLACE = 0.25        # targets closer than this share of the viewing distance: same place
SAME_SCALE = 1.4         # distance ratio below which two views see the surface at one scale


def read_points3d(path):
    """(xyz float64[N, 3], tracks: list of image-id arrays, point ids int64[N])."""
    with open(path, "rb") as f:
        data = f.read()
    (num,) = struct.unpack_from("<Q", data, 0)
    off = 8
    ids = np.empty(num, np.int64)
    xyz = np.empty((num, 3), np.float64)
    tracks = []
    for k in range(num):
        pid, x, y, z = struct.unpack_from("<Q3d", data, off)
        off += 8 + 24 + 3 + 8                      # id, xyz, rgb, error
        (n,) = struct.unpack_from("<Q", data, off)
        off += 8
        track = np.frombuffer(data, "<u4", 2 * n, off)[0::2]
        off += 8 * n
        ids[k] = pid
        xyz[k] = (x, y, z)
        tracks.append(track)
    return xyz, tracks, ids


def view_info(model_dir):
    """Per image: name, centre, optical axis, median depth, target point, point indices."""
    _, images = colmap_io.read_model(model_dir)
    xyz, tracks, _ = read_points3d(os.path.join(model_dir, "points3D.bin"))
    seen = {iid: [] for iid in images}
    for k, track in enumerate(tracks):
        for iid in set(int(i) for i in track):
            if iid in seen:
                seen[iid].append(k)
    out = []
    for iid, im in images.items():
        r = np.array(colmap_io.qvec_to_rotmat(im["qvec"]))
        c = np.array(colmap_io.camera_center(im))
        axis = r[2]                                   # camera +Z in the world
        idx = np.array(seen[iid], dtype=np.int64)
        if len(idx):
            depth = float(np.median((xyz[idx] - c) @ axis))
        else:
            depth = 0.0
        out.append({"id": iid, "name": im["name"], "centre": c, "axis": axis,
                    "depth": depth, "target": c + axis * max(depth, 0.0), "points": idx})
    return out, len(xyz), tracks


def select_views(model_dir, min_angle=20.0, coverage=0.99, min_views=8):
    """Names of the photos to keep (sorted); all of them when there is no reason to drop any."""
    info, n_points, tracks = view_info(model_dir)
    usable = [v for v in info if v["depth"] > 0 and len(v["points"])]
    if len(usable) <= min_views:
        return sorted(v["name"] for v in info)
    usable.sort(key=lambda v: (-len(v["points"]), v["name"]))
    cos_min = math.cos(math.radians(min_angle))
    kept = []
    targets, centres, depths = [], [], []
    for v in usable:
        if kept:
            t = np.array(targets)
            c = np.array(centres)
            d = np.array(depths)
            near = np.linalg.norm(t - v["target"], axis=1) < SAME_PLACE * np.minimum(d, v["depth"])
            mid = (t + v["target"]) / 2.0
            a = c - mid                               # from the shared target to each camera
            b = v["centre"] - mid
            cos = np.einsum("ij,ij->i", a, b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
                                                  + 1e-12)
            scale = np.maximum(d, v["depth"]) / np.minimum(d, v["depth"])
            if np.any(near & (cos > cos_min) & (scale < SAME_SCALE)):
                continue
        kept.append(v)
        targets.append(v["target"])
        centres.append(v["centre"])
        depths.append(v["depth"])

    # every 3D point seen by two photos should still be seen by two kept photos
    count = np.zeros(n_points, np.int32)
    multi = np.array([len(set(t.tolist())) >= 2 for t in tracks], dtype=bool)
    for v in kept:
        count[v["points"]] += 1
    kept_ids = {v["id"] for v in kept}
    rest = [v for v in usable if v["id"] not in kept_ids]
    total = max(1, int(multi.sum()))
    while rest:
        short = multi & (count < 2)
        if short.sum() <= (1.0 - coverage) * total:
            break
        gains = [int(short[v["points"]].sum()) for v in rest]
        best = int(np.argmax(gains))
        if gains[best] == 0:
            break
        v = rest.pop(best)
        kept.append(v)
        count[v["points"]] += 1
    if len(kept) < min_views:
        kept += rest[:min_views - len(kept)]
    return sorted(v["name"] for v in kept)
