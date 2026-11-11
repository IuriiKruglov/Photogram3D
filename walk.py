# SPDX-License-Identifier: GPL-3.0-or-later
# Photogram3D - photos or video to a textured mesh with COLMAP + OpenMVS
# Copyright (C) 2026 Kruglov Iurii
# Authors: Kruglov Iurii, Anthropic Claude AI
# License: GNU General Public License (GPL) v3 or later, https://www.blender.org/about/license/
"""Solving along the shooting order (0.3.5): the repair for folded camera paths.

Photos of a walk around an object are taken in order, and each one looks at nearly the
same side as the photos taken just before and after it. A symmetric object - a box printed
alike on opposite sides, the two sides of a car - also gives strong matches between photos
from opposite sides. The solvers then place part of the walk on top of the rest (a fold):
the cloud and the mesh hold two copies of the object facing each other.

The repair has three parts:

1. Orient the cameras with only the photo pairs taken a few shots apart (the band: about
   BAND_TURN_DEG of turning), by COLMAP's rotation averaging. Photos from opposite sides
   are never in the band, so these orientations cannot fold. Weak pairs (few matches)
   other than consecutive photos are left out of the band: their relative rotation is
   often wrong (one such pair, 24 matches, 76 degrees where its neighbours said 43, cut the
   gas mask walk in two).
2. Check every matched pair against those orientations. A pair's own two-view geometry
   gives the rotation between its two photos; when it differs from the rotation between
   the two oriented cameras by more than ROTATION_LIMIT, the pair links look-alike views
   of different sides, and its matches are left out. Only rotations are needed: they are
   much more robust than positions on weakly matched photos, and a symmetric object's
   wrong pairs are off by half a turn.
3. Solve again with every pair except those (pipeline.step_mapping does that with the
   usual solvers and checks).

Measured (0.3.5 tests, test_mapping_walk.py): on a rendered box with alike opposite sides
(40 photos, both solvers folded half the walk onto the other half) the repair leaves out 208
pairs and places all 40 cameras within 3 mm of the truth; on the gas-mask photos
(reduced to 480 x 639; 0.3.4 folded them) it leaves out 19 pairs, mostly front views matched
to back views, and the solve places all 41 photos without a fold. On the WRX photos (no fold
in 0.3.4) it judged 313 of 315 pairs as a check against the correct solve did.

Nothing here imports bpy.
"""

import math
import os
import sqlite3

from . import colmap_io

MAX_IMAGES = 2147483647           # COLMAP's pair id: MAX_IMAGES * smaller id + larger id
BAND_TURN_DEG = 100.0             # photos at most this much turning apart form the band
BAND_MIN, BAND_MAX, BAND_DEFAULT = 3, 15, 6
BAND_MIN_INLIERS = 30             # band pairs other than consecutive photos need this many
BAND_INLIER_SHARE = 0.5           # ... and half the matches consecutive photos typically have
ROTATION_JUMP_DEG = 60.0          # consecutive photos turned more than this (and 3 steps): wrong
ROTATION_LIMIT = 1.0              # radians (57 degrees); the wrong pairs measured 1.4 - 3.1
CHEIRALITY_SAMPLES = 80           # matches used to pick the essential matrix's rotation


def pair_key(a, b):
    return (a, b) if a < b else (b, a)


def _np():
    import numpy as np
    return np


# ---------------------------------------------------------------------------
# reading the match database
# ---------------------------------------------------------------------------

def image_names(db_path):
    con = sqlite3.connect(db_path, timeout=30)
    try:
        return {i: n for i, n in con.execute("SELECT image_id, name FROM images")}
    finally:
        con.close()


def _matrix(blob):
    np = _np()
    if blob is None or len(blob) != 72:
        return None
    m = np.frombuffer(blob, np.float64).reshape(3, 3)      # COLMAP stores them row-major
    return m if np.any(m) else None


def geometries(db_path):
    """Yield (a, b, matches N x 2, config, F, H) for every verified pair (a < b)."""
    np = _np()
    con = sqlite3.connect(db_path, timeout=30)
    try:
        for pid, rows, cols, data, config, F, H in con.execute(
                "SELECT pair_id, rows, cols, data, config, F, H FROM two_view_geometries "
                "WHERE rows > 0"):
            if not data or cols != 2:
                continue
            yield (pid // MAX_IMAGES, pid % MAX_IMAGES, np.frombuffer(data, np.uint32).reshape(rows, 2),
                   config, _matrix(F), _matrix(H))
    finally:
        con.close()


def verified_pairs(db_path):
    return list(pair_inliers(db_path))


def pair_inliers(db_path):
    """{(a, b): number of verified matches} for every verified pair (a < b)."""
    con = sqlite3.connect(db_path, timeout=30)
    try:
        return {(pid // MAX_IMAGES, pid % MAX_IMAGES): rows for pid, rows in
                con.execute("SELECT pair_id, rows FROM two_view_geometries WHERE rows > 0")}
    finally:
        con.close()


def keypoints(db_path, ids):
    """{image id: N x 2 float32 pixel positions} for ``ids``."""
    np = _np()
    want = set(ids)
    out = {}
    con = sqlite3.connect(db_path, timeout=30)
    try:
        for iid, rows, cols, data in con.execute("SELECT image_id, rows, cols, data FROM keypoints"):
            if iid in want and rows and data:
                out[iid] = np.frombuffer(data, np.float32).reshape(rows, cols)[:, :2].copy()
    finally:
        con.close()
    return out


def copy_database(src, dst, keep=None, cameras=None):
    """A copy of the match database with only the verified pairs in ``keep`` (all when
    None) and, when given, the solved ``cameras`` (colmap_io.read_model) as the cameras'
    starting intrinsics, marked as known (the global solver needs a sound focal length)."""
    np = _np()
    for suffix in ("", "-wal", "-shm"):
        try:
            os.remove(dst + suffix)
        except OSError:
            pass
    s = sqlite3.connect(src, timeout=30)
    d = sqlite3.connect(dst)
    try:
        s.backup(d)
    finally:
        s.close()
    try:
        if keep is not None:
            keep = {pair_key(a, b) for a, b in keep}
            gone = [(a * MAX_IMAGES + b,) for a, b in verified_pairs(dst) if (a, b) not in keep]
            d.executemany("DELETE FROM two_view_geometries WHERE pair_id = ?", gone)
            d.executemany("DELETE FROM matches WHERE pair_id = ?", gone)
        for cid, cam in (cameras or {}).items():
            params = np.asarray(cam["params"], np.float64)
            d.execute("UPDATE cameras SET params = ?, prior_focal_length = 1 WHERE camera_id = ?",
                      (params.tobytes(), cid))
        d.commit()
    finally:
        d.close()
    return dst


# ---------------------------------------------------------------------------
# the band
# ---------------------------------------------------------------------------

def _angle(R):
    np = _np()
    return float(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))


def step_turn(model_dir, order):
    """Median rotation (degrees) between photos taken one after the other, from a solved
    model (a folded one too: most consecutive photos are placed right relative to each
    other); None when too few consecutive photos are solved."""
    np = _np()
    _, images = colmap_io.read_model(model_dir)
    seq = sorted((order[im["name"]], np.array(colmap_io.qvec_to_rotmat(im["qvec"]), float))
                 for im in images.values() if im["name"] in order)
    turns = [_angle(r1 @ r0.T) for (p0, r0), (p1, r1) in zip(seq, seq[1:]) if p1 - p0 == 1]
    if len(turns) < 5:
        return None
    return math.degrees(float(np.median(turns)))


def band_width(turn_deg):
    """How many shots before and after each photo belong to its band."""
    if not turn_deg or turn_deg <= 0:
        return BAND_DEFAULT
    return int(min(BAND_MAX, max(BAND_MIN, round(BAND_TURN_DEG / turn_deg))))


def shot_distance(pos, a, b):
    return abs(pos[a] - pos[b])


def band_pairs(inliers, pos, width):
    """The pairs of photos at most ``width`` shots apart: all consecutive ones, the others
    when well matched (BAND_MIN_INLIERS, BAND_INLIER_SHARE)."""
    np = _np()
    near = {k: n for k, n in inliers.items()
            if k[0] in pos and k[1] in pos and shot_distance(pos, *k) <= width}
    consecutive = [n for k, n in near.items() if shot_distance(pos, *k) == 1]
    floor = BAND_MIN_INLIERS
    if consecutive:
        floor = max(floor, BAND_INLIER_SHARE * float(np.median(consecutive)))
    return {pair_key(*k) for k, n in near.items() if shot_distance(pos, *k) == 1 or n >= floor}


def rotation_jumps(model_dir, order):
    """Photos taken one after the other whose solved orientations differ by more than
    ROTATION_JUMP_DEG and three typical steps: [(position a, position b, degrees)]."""
    np = _np()
    _, images = colmap_io.read_model(model_dir)
    seq = sorted((order[im["name"]], np.array(colmap_io.qvec_to_rotmat(im["qvec"]), float))
                 for im in images.values() if im["name"] in order)
    steps = [(p0, p1, math.degrees(_angle(r1 @ r0.T))) for (p0, r0), (p1, r1) in zip(seq, seq[1:])]
    if not steps:
        return []
    typical = float(np.median([d / max(1, p1 - p0) for p0, p1, d in steps]))
    return [(p0, p1, d) for p0, p1, d in steps
            if d > max(ROTATION_JUMP_DEG, 3.0 * typical * max(1, p1 - p0))]


# ---------------------------------------------------------------------------
# checking pairs against solved cameras
# ---------------------------------------------------------------------------

def _intrinsics(cam):
    """(K, k1, k2) of a COLMAP camera; distortion beyond two radial terms is ignored."""
    np = _np()
    model, p = cam["model"], [float(x) for x in cam["params"]]
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE",
                 "RADIAL_FISHEYE"):
        f, cx, cy = p[0], p[1], p[2]
        K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
        k1 = p[3] if len(p) > 3 and model in ("SIMPLE_RADIAL", "RADIAL") else 0.0
        k2 = p[4] if model == "RADIAL" and len(p) > 4 else 0.0
    else:                                              # PINHOLE, OPENCV, FULL_OPENCV, ...
        K = np.array([[p[0], 0.0, p[2]], [0.0, p[1], p[3]], [0.0, 0.0, 1.0]])
        k1 = p[4] if len(p) > 4 and model != "PINHOLE" else 0.0
        k2 = p[5] if len(p) > 5 and model != "PINHOLE" else 0.0
    return K, k1, k2


def _rays(xy, intr):
    """Pixel positions -> undistorted normalized image coordinates (N x 3, z = 1)."""
    np = _np()
    K, k1, k2 = intr
    u = (xy[:, 0].astype(np.float64) - K[0, 2]) / K[0, 0]
    v = (xy[:, 1].astype(np.float64) - K[1, 2]) / K[1, 1]
    uu, vv = u, v
    if k1 or k2:
        for _ in range(6):
            r2 = uu * uu + vv * vv
            d = 1.0 + k1 * r2 + k2 * r2 * r2
            uu, vv = u / d, v / d
    return np.stack([uu, vv, np.ones_like(uu)], 1)


def essential_rotation(E, na, nb):
    """The rotation of camera b relative to camera a (x_b = R x_a + t) in an essential
    matrix: of its four decompositions, the one that puts most matches in front of both
    cameras."""
    np = _np()
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt) < 0:
        Vt = -Vt
    W = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    step = max(1, len(na) // CHEIRALITY_SAMPLES)
    a, b = na[::step], nb[::step]
    bb = np.einsum("ij,ij->i", b, b)
    best, best_n = None, -1
    for R in (U @ W @ Vt, U @ W.T @ Vt):
        Ra = a @ R.T
        aa = np.einsum("ij,ij->i", Ra, Ra)
        ab = np.einsum("ij,ij->i", Ra, b)
        det = aa * bb - ab * ab
        ok = np.abs(det) > 1e-12
        for t in (U[:, 2], -U[:, 2]):
            # depths la, lb with lb * b = la * Ra + t, in the least-squares sense
            r1, r2 = -(Ra @ t), b @ t
            la = (bb * r1 + ab * r2) / np.where(ok, det, 1.0)
            lb = (ab * r1 + aa * r2) / np.where(ok, det, 1.0)
            n = int(np.count_nonzero(ok & (la > 0) & (lb > 0)))
            if n > best_n:
                best, best_n = R, n
    return best


def homography_residual(Hn, na, nb, R):
    """How far a homography (between normalized coordinates) is from every plane seen
    with relative rotation R: with H = R + t n^T scaled so its middle singular value is
    1, H - R has rank one; its second singular value grows with the rotation error
    (about the error in radians when small, near 2 for half a turn)."""
    np = _np()
    s = np.linalg.svd(Hn, compute_uv=False)
    if s[1] <= 0:
        return None
    H = Hn / s[1]
    if np.median(np.einsum("ij,ij->i", nb, na @ H.T)) < 0:     # points in front of both
        H = -H
    return float(np.linalg.svd(H - R, compute_uv=False)[1])


def rotation_errors(model_dir, db_path, skip=()):
    """For every verified pair whose two photos are solved in ``model_dir``: how much the
    pair's own geometry disagrees with the rotation between the two solved cameras
    (radians, see module doc). {(a, b): (error, matches)}; photos in ``skip`` and pairs
    without a usable geometry are left out."""
    np = _np()
    cams, images = colmap_io.read_model(model_dir)
    names = image_names(db_path)
    by_name = {n: i for i, n in names.items()}
    pose = {}
    for im in images.values():
        iid = by_name.get(im["name"])
        if iid is None or iid in skip or im["camera_id"] not in cams:
            continue
        pose[iid] = (np.array(colmap_io.qvec_to_rotmat(im["qvec"]), float),
                     _intrinsics(cams[im["camera_id"]]))
    kps = keypoints(db_path, pose)
    out = {}
    for a, b, m, config, F, H in geometries(db_path):
        if a not in pose or b not in pose or a not in kps or b not in kps:
            continue
        (Ra, ia), (Rb, ib) = pose[a], pose[b]
        if m[:, 0].max() >= len(kps[a]) or m[:, 1].max() >= len(kps[b]):
            continue
        R = Rb @ Ra.T
        na, nb = _rays(kps[a][m[:, 0]], ia), _rays(kps[b][m[:, 1]], ib)
        err = None
        if config in (2, 3):                                      # calibrated / uncalibrated
            if F is not None:
                Rp = essential_rotation(ib[0].T @ F @ ia[0], na, nb)
                if Rp is not None:
                    err = _angle(Rp @ R.T)
        elif H is not None:                     # planar, panoramic, watermark: a homography
            err = homography_residual(np.linalg.inv(ib[0]) @ H @ ia[0], na, nb, R)
        if err is not None:
            out[(a, b)] = (err, len(m))
    return out


NEIGHBOUR_AGREE_DEG = 8.0         # a weak photo turned like its neighbours stays (see below)


def neighbour_agreement(model_dir, db_path, order, reach=2):
    """{photo name: (median rotation error in degrees, pairs)} over each solved photo's
    pairs with the photos at most ``reach`` shots away. After the pairs that contradict
    the walk are left out, a photo with few shared points (a dark or plain side of the
    object) that is still turned like the photos taken next to it is placed right; on
    the synthetic scene the misplaced photos measured 10 to 26 degrees, the weak but
    right photos of the gas mask 1.5 to 5."""
    np = _np()
    names = image_names(db_path)
    pos = {i: order[n] for i, n in names.items() if n in order}
    errs = {}
    for (a, b), (e, _) in rotation_errors(model_dir, db_path).items():
        if a in pos and b in pos and shot_distance(pos, a, b) <= reach:
            errs.setdefault(names[a], []).append(e)
            errs.setdefault(names[b], []).append(e)
    return {n: (math.degrees(float(np.median(v))), len(v)) for n, v in errs.items()}


def jump_photos(model_dir, order, jumps):
    """Photo ids next to the jumps of a solved model (solve_check.assess jump_list): their
    placement is doubtful, so their pairs are not checked against it."""
    names = {pos: name for name, pos in order.items()}
    out = set()
    for a, b, *_ in jumps:
        out.update(x for x in (names.get(a), names.get(b)) if x)
    return out


def walk_pairs(db_path, reference, order, width, limit=ROTATION_LIMIT, doubtful=()):
    """The pairs to solve with after the band solve ``reference``.

    Returns (keep, report): ``keep`` holds every verified pair except those that
    contradict the reference cameras and the long-range pairs of photos the reference
    does not place (they could not be checked); ``report`` counts them and lists the
    contradicting pairs as (photo a, photo b, error, matches), worst first."""
    names = image_names(db_path)
    pos = {i: order[n] for i, n in names.items() if n in order}
    by_name = {n: i for i, n in names.items()}
    skip = {by_name[n] for n in doubtful if n in by_name}
    errors = rotation_errors(reference, db_path, skip)
    _, ref_images = colmap_io.read_model(reference)
    placed = {by_name[im["name"]] for im in ref_images.values() if im["name"] in by_name} - skip
    keep, bad, unchecked = set(), [], 0
    for a, b in verified_pairs(db_path):
        e = errors.get((a, b))
        if e is not None:
            if e[0] > limit:
                bad.append((names[a], names[b], e[0], e[1]))
            else:
                keep.add((a, b))
            continue
        near = a in pos and b in pos and shot_distance(pos, a, b) <= width
        if near or (a in placed and b in placed):
            keep.add((a, b))                 # short-range, or both placed but no usable geometry
        else:
            unchecked += 1
    bad.sort(key=lambda r: -r[3])
    return keep, {"checked": len(errors), "contradicting": bad, "unchecked": unchecked,
                  "kept": len(keep)}


# ---------------------------------------------------------------------------
# guides: cameras placed by hand or by a guess, to check the pairs against (0.3.11)
# ---------------------------------------------------------------------------

TURN_MAX_FOLDS = 6                # more folds than this: no guess at which stretches to turn


def vertical(rotations, gravity=None):
    """The up direction of a set of solved cameras: the mean of their upright cameras'
    up axes (solve_check.upright_rotation; ``gravity`` {name: gravity} for photos stored on
    their side). ``rotations``: {name: world-to-camera rotation}. None when they disagree."""
    np = _np()
    from . import solve_check
    gravity = gravity or {}
    ups = [-np.array(solve_check.upright_rotation(R, gravity.get(n)), float)[1] for n, R in rotations.items()]
    if not ups:
        return None
    up = np.mean(ups, axis=0)
    norm = float(np.linalg.norm(up))
    return up / norm if norm > 0.5 else None


def turned_guide(model_dir, order, folds, dest, gravity=None):
    """A guess for a solve whose walk folds: its cameras, with the stretches of the walk
    between its folds turned by half a turn about the vertical, every second stretch, so
    that the stretch the solver laid over the opposite side faces its own side again (look-
    alike sides of a symmetric object: the starboard photos of the Mi-8 were laid over the
    port side, facing the same way as the port photos). ``folds``: [(position a, position b)]
    from solve_check.assess. Writes the guide as a text model to ``dest``; returns the names
    of the cameras turned (empty: nothing written)."""
    np = _np()
    cams, images = colmap_io.read_model(model_dir)
    cuts = sorted({b for _, b in folds})
    rot = {im["name"]: np.array(colmap_io.qvec_to_rotmat(im["qvec"]), float) for im in images.values()}
    up = vertical(rot, gravity)
    if not cuts or up is None:
        return 0
    G = 2.0 * np.outer(up, up) - np.eye(3)            # half a turn about up
    poses, turned = [], set()
    for iid, im in sorted(images.items()):
        R = rot[im["name"]]
        pos = order.get(im["name"])
        if pos is not None and sum(1 for c in cuts if c <= pos) % 2:
            R = R @ G.T
            turned.add(im["name"])
        poses.append((iid, im["name"], im["camera_id"], R, colmap_io.camera_center(im)))
    if not turned or len(turned) == len(poses):
        return set()
    colmap_io.write_text_model(dest, cams, poses)
    return turned


def far_links(pairs, names, order, turned, width):
    """The pairs of ``pairs`` that link a turned photo with one not turned, taken more
    than ``width`` shots apart: after a fold is turned back, the turned part may join the
    rest only along the walk. The rotation check cannot judge look-alike photos that face
    along the object's plane of symmetry (front or back views from either side): their
    look-alike rotation is nearly the true one. On the Mi-8 photos 12 such pairs (front
    views from the starboard side matched to front views from the port side, 132 shots
    later) passed the check and laid 28 starboard photos over the port side again; without
    them the incremental solver placed 247 of 260 photos right."""
    out = set()
    for a, b in pairs:
        na, nb = names.get(a), names.get(b)
        if na is None or nb is None or na not in order or nb not in order:
            continue
        if (na in turned) != (nb in turned) and abs(order[na] - order[nb]) > width:
            out.add(pair_key(a, b))
    return out
