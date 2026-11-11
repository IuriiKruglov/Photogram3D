# SPDX-License-Identifier: GPL-3.0-or-later
# Photogram3D - photos or video to a textured mesh with COLMAP + OpenMVS
# Copyright (C) 2026 Kruglov Iurii
# Authors: Kruglov Iurii, Anthropic Claude AI
# License: GNU General Public License (GPL) v3 or later, https://www.blender.org/about/license/
"""Sanity checks for a solved camera set.

Photos and video frames are taken in order while walking, so consecutive
shots must be close to each other, and a walk around an object keeps turning
the same way. When an object looks alike from different sides (a car's left
and right side, repeated wheels), matching links photos from opposite sides
and the solver can fold the walk onto itself: the mesh then contains several
copies of the object. Folds show up as consecutive shots that jump far apart
or turn back around the object.

(Matching only neighbouring shots was tried as a remedy: on the WRX photos it
split the walk in two, and on the synthetic scene it produced a fold-free
model with half its cameras wrong, so it is not used.)

Nothing here imports bpy.
"""

import math
import os
import re

from . import colmap_io

FOLD_JUMP_FACTOR = 5.0          # a step this many times the median step is a jump ...
GAP_ALLOWANCE = 2.0             # ... plus this many median steps per shot missing in between
# steps shorter than this share of a long step (the 90th percentile) are shots from the same
# spot and stay out of the median step (0.3.11)
SAME_SPOT_SHARE = 0.1
TURN_LIMIT_DEG = 30.0           # a backward turn around the object larger than this is a fold
GAP_TURN_MAX_DEG = 90.0         # no turn check across missing shots expected to turn more


def natural_key(name):
    """Sort key that puts 'img2' before 'img10'."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def shooting_order(names):
    """{image name: position}; file names from phones, cameras and our frame
    extractor sort in capture order."""
    return {name: i for i, name in enumerate(sorted(names, key=natural_key))}


def _points_count(model_dir):
    path = os.path.join(model_dir, "points3D.bin")
    if os.path.isfile(path):
        return colmap_io.read_points3d_count_binary(path)
    path = os.path.join(model_dir, "points3D.txt")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for ln in f if ln.strip() and not ln.startswith("#"))
    return 0


def _observations_count(model_dir):
    path = os.path.join(model_dir, "points3D.bin")
    if os.path.isfile(path):
        return colmap_io.read_observations_binary(path)
    path = os.path.join(model_dir, "points3D.txt")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return sum((len(ln.split()) - 8) // 2 for ln in f if ln.strip() and not ln.startswith("#"))
    return 0


def camera_path(model_dir, order):
    """[(order position, camera centre)] of the registered images, in shooting order."""
    return [(pos, c) for pos, c, _ in _cameras(model_dir, order)]


def upright_rotation(R, gravity):
    """The world-to-camera rotation of a photo turned upright about its viewing axis:
    ``gravity`` is the downward direction in the stored picture's camera frame (COLMAP's
    gravity prior, colmap_io.gravity_priors), None for a photo stored upright."""
    if not gravity:
        return R
    gx, gy = float(gravity[0]), float(gravity[1])
    n = math.hypot(gx, gy)
    if n < 1e-9:
        return R
    gx, gy = gx / n, gy / n
    y = [gx * R[0][c] + gy * R[1][c] for c in range(3)]          # down, in the world
    z = [float(v) for v in R[2]]
    x = [y[1] * z[2] - y[2] * z[1], y[2] * z[0] - y[0] * z[2], y[0] * z[1] - y[1] * z[0]]
    return [x, y, z]


def _cameras(model_dir, order, gravity=None):
    """[(order position, centre, world-to-camera rotation)] in shooting order. With
    ``gravity`` ({photo name: gravity}, colmap_io.gravity_priors), photos stored on their
    side get the rotation of their upright camera (upright_rotation): the turn check takes
    up from the cameras' down axes, and until 0.3.10 it took a horizontal direction for up
    on photos stored on their side (the Mi-8 photos: all 260, 89 degrees off)."""
    _, images = colmap_io.read_model(model_dir)
    gravity = gravity or {}
    seq = []
    for im in images.values():
        pos = order.get(im["name"])
        if pos is not None:
            R = upright_rotation(colmap_io.qvec_to_rotmat(im["qvec"]), gravity.get(im["name"]))
            seq.append((pos, colmap_io.camera_center(im), R))
    seq.sort(key=lambda s: s[0])
    return seq


def path_jumps(seq):
    """Steps between consecutive solved shots that are too long: ([(pos_a, pos_b, step per
    shot / median)], median step per shot).

    A step may be FOLD_JUMP_FACTOR times the median step, plus GAP_ALLOWANCE median steps
    for each shot missing in between (not solved, or dropped as weakly connected). Until
    0.3.9 the whole step was divided by the shots it spans; a fold hidden behind five
    dropped frames then passed as a step of three medians (GAZ-66, 0.3.9).

    The median is that of the steps between spots: steps shorter than SAME_SPOT_SHARE of
    the 90th-percentile step are shots taken from one spot (several photos per spot, a few
    centimetres apart) and are left out. Until 0.3.10 the median of all steps was used; on
    the Mi-8 photos, about half of them taken from the spot of the shot before, that median
    was 0.21 m against 0.41 m between spots, and six of the eight folds reported were plain
    walks to the next spot."""
    steps = [(math.dist(c0, c1), max(1, i1 - i0), i0, i1)
             for (i0, c0, _), (i1, c1, _) in zip(seq, seq[1:])]
    if len(steps) < 5:
        return [], 0.0
    per_shot = sorted(d / g for d, g, _, _ in steps)
    long_step = per_shot[int(0.9 * (len(per_shot) - 1))]
    moves = [s for s in per_shot if s >= SAME_SPOT_SHARE * long_step] or per_shot
    median = moves[len(moves) // 2]
    if median <= 0:
        return [], 0.0
    return ([(i0, i1, d / (g * median)) for d, g, i0, i1 in steps
             if d > median * (FOLD_JUMP_FACTOR + GAP_ALLOWANCE * (g - 1))], median)


def orbit_backsteps(seq):
    """Large backward turns of a walk around an object.

    Walking around something turns steadily one way around the point all
    cameras look at. A fold puts some shots on top of earlier ones, so the
    walk turns back. Returns [(pos_a, pos_b, degrees)]; empty when the
    cameras do not look at a common centre or turn less than 90 degrees.

    The way round is the way most steps turn, and the turn covered is what the
    walk turns that way: a fold turns back over what it covered, so the net turn
    of a folded walk can be small (GAZ-66, 0.3.9: 70 degrees, which until then
    meant "no orbit" and no check). Across missing shots a backward turn counts
    only while the turn expected over them is at most GAP_TURN_MAX_DEG.

    Up is taken from the cameras' down axes, so photos stored on their side need their
    upright rotation (_cameras with gravity).
    """
    import numpy as np
    if len(seq) < 8:
        return []
    C = np.array([s[1] for s in seq], dtype=float)
    R = [np.array(s[2], dtype=float) for s in seq]
    V = np.array([r[2] for r in R])                      # viewing directions (world)
    up = -np.mean([r[1] for r in R], axis=0)              # photos are upright
    if np.linalg.norm(up) < 1e-6:
        return []
    up /= np.linalg.norm(up)
    A = sum(np.eye(3) - np.outer(v, v) for v in V)
    b = sum((np.eye(3) - np.outer(v, v)) @ c for c, v in zip(C, V))
    if np.linalg.eigvalsh(A)[0] < 0.05 * len(V):         # rays nearly parallel: not an orbit
        return []
    target = np.linalg.solve(A, b)
    e1 = np.cross(up, [1.0, 0.0, 0.0])
    if np.linalg.norm(e1) < 0.1:
        e1 = np.cross(up, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    rel = C - target
    az = np.degrees(np.arctan2(rel @ e2, rel @ e1))
    daz = (np.diff(az) + 180.0) % 360.0 - 180.0
    shots = np.maximum(1, np.diff([s[0] for s in seq]))
    ahead = int((daz > 0).sum()) - int((daz < 0).sum())
    if ahead == 0:
        return []
    direction = 1.0 if ahead > 0 else -1.0
    if float(np.clip(direction * daz, 0.0, None).sum()) < 90.0:
        return []
    per_shot = float(np.median(np.abs(daz) / shots))
    limit = max(TURN_LIMIT_DEG, 2.0 * per_shot)
    return [(seq[k][0], seq[k + 1][0], float(abs(d))) for k, d in enumerate(daz)
            if direction * d < -limit and shots[k] * per_shot <= GAP_TURN_MAX_DEG]


def longest_gap(seq):
    """The most shots missing in a row between two solved ones (not counting the ends)."""
    return max((b - a - 1 for (a, *_), (b, *_) in zip(seq, seq[1:])), default=0)


def assess(model_dir, order, gravity=None):
    """Quality of a solved model: registered images, 3D points and path errors.

    Path errors are (a) jumps: steps between consecutive shots that are too long
    (path_jumps), and (b) large backward turns around the object (orbit_backsteps).
    ``gap``: the most shots missing in a row inside the walk (longest_gap). ``gravity``:
    colmap_io.gravity_priors of the photos (see _cameras).
    """
    seq = _cameras(model_dir, order, gravity)
    jumps, median = path_jumps(seq)
    back = orbit_backsteps(seq)
    errors = {(a, b) for a, b, _ in jumps} | {(a, b) for a, b, _ in back}
    return {"registered": len(seq), "points": _points_count(model_dir),
            "observations": _observations_count(model_dir),
            "jumps": len(errors), "jump_list": jumps, "backsteps": back, "median_step": median,
            "gap": longest_gap(seq)}


FOLD_PENALTY = 10   # one fold (duplicated geometry) costs as much as 10 missing photos,
FOLD_SHARE = 0.2    # or as this share of the photos placed, when that is more (0.3.13: 0.1)


def fold_penalty(registered):
    """Missing photos one fold costs. "A few" missing photos grows with the set: on the Mi-8
    photos (260) COLMAP's global solver, turned back, placed 258 photos in two runs of 0.3.13's
    tests with five of them off the walk (one fold), against 256 or 257 without a fold in the
    runs before (the solver draws at random), and the incremental solve beside it placed 246
    without a fold. At 10 photos a fold the folded solve won (248 against 246). On a video of
    the AW-249 (69 frames, 0.3.13) every solve folded once, with 11 frames laid over the other
    side, and the solves with those 11 turned back placed 58 without a fold: at a tenth (10) the
    folded 69 still won (59 against 58), at a fifth (14) the clean 58 does (55 against 58)."""
    return max(FOLD_PENALTY, int(math.ceil(FOLD_SHARE * registered)))


def score(q):
    """Higher is better: a folded walk is worse than a few missing photos (fold_penalty).
    Between solves that place as many photos, the one whose points explain more feature
    positions in the photos (observations), then the one with more points. Until 0.3.10 the
    points decided: on the rendered walk of test_camera_track a global solve bent by up to
    1.4 m along the walk had more points than the right incremental one (7,936 against 7,670)
    but fewer observations (31,152 against 32,344) and a reprojection error of 0.55 px against
    0.30 px, and the scan sat 65 px off the footage (0.3.11)."""
    return (q["registered"] - fold_penalty(q["registered"]) * q["jumps"], q.get("observations", 0), q["points"])
