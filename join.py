# SPDX-License-Identifier: GPL-3.0-or-later
# Photogram3D - photos or video to a textured mesh with COLMAP + OpenMVS
# Copyright (C) 2026 Kruglov Iurii
# Authors: Kruglov Iurii, Anthropic Claude AI
# License: GNU General Public License (GPL) v3 or later, https://www.blender.org/about/license/
"""Joining two scans of one object (0.3.13): two videos, photos and a video, different days,
different sides.

Each source is solved on its own, in coordinates of its own (position, turn and scale are
arbitrary). Matching features across sources was tried on two videos of a helicopter, one of
each side, filmed on different days: the only matches were on the rotor head and on insignia
painted on both sides, and both a joint solve and a fit through those matches laid one side
over the other (DEVELOPMENT.md, section 4). A symmetric object fits its mirror image just as
well, so no automatic check tells such a fold from the truth. The scans are therefore joined
by control points: the same spots marked on both, from which the similarity transform
(scale, rotation, translation) is found by least squares (Umeyama 1991), each pair's distance
shown so that a wrong click stands out. Then the dense clouds refine it where they overlap
(ICP with scale), the control points kept in the fit so that a small overlap cannot slide.

numpy only; the nearest-neighbour search is passed in (Blender's mathutils.kdtree).
"""

MIN_POINTS = 3
LEAVE_ONE_OUT_MIN = 4       # with this many pairs, each is also checked against the others
SUSPECT_FACTOR = 3.0        # a pair this many times farther off than the others is doubtful
# Points spread across the line through them by less than this share of their spread along it
# leave the turn about that line uncertain: on the AW-249 videos, the nose and the tail boom
# gave 0.02-0.04, and the rotor hub with them 0.2
SPREAD_MIN = 0.1
ICP_SAMPLES = 60_000        # points of the moving cloud used per iteration
ICP_ITERATIONS = 40
ICP_TOL = 1e-5              # relative change of the transform that ends the refinement
CONTROL_WEIGHT = 0.05       # the control points together weigh this share of the ICP fit
REFINE_GAIN = 0.75          # a refinement counts when the clouds' median distance drops to this share


def _np():
    import numpy as np
    return np


def similarity(src, dst, weights=None, scale=True):
    """(s, R, t) with dst ~ s R src + t by weighted least squares (Umeyama 1991): src and
    dst are N x 3, N >= 3 and not all on one line."""
    np = _np()
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    w = np.ones(len(src)) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    mu_s, mu_d = w @ src, w @ dst
    a, b = src - mu_s, dst - mu_d
    cov = (b * w[:, None]).T @ a
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var = float(w @ (a ** 2).sum(1))
    if var <= 0:
        raise ValueError("the points are all at one place")
    s = float(np.trace(np.diag(D) @ S) / var) if scale else 1.0
    return s, R, mu_d - s * R @ mu_s


def matrix(s, R, t):
    """4 x 4 matrix of x -> s R x + t."""
    np = _np()
    M = np.eye(4)
    M[:3, :3] = s * np.asarray(R)
    M[:3, 3] = t
    return M


def apply(M, pts):
    np = _np()
    pts = np.asarray(pts, float)
    return pts @ M[:3, :3].T + M[:3, 3]


def spread(pts):
    """How far the points (N x 3) spread across the line through them, as a share of their
    spread along it: 0 on one line, 1 for a round cloud."""
    np = _np()
    pts = np.asarray(pts, float)
    if len(pts) < 3:
        return 0.0
    sv = np.linalg.svd(pts - pts.mean(0), compute_uv=False)
    return float(sv[1] / sv[0]) if sv[0] > 0 else 0.0


def collinear(pts, tol=1e-3):
    """Whether the points (N x 3) lie on one line (or one spot), relative to their spread."""
    return spread(pts) < tol


def fit_points(moving, fixed):
    """Join by control points: the similarity that takes the ``moving`` scan's points onto
    the ``fixed`` scan's (both N x 3, the same spots in the same order).

    Returns {"matrix": 4 x 4, "scale", "residuals": distance of each pair after the fit,
    "rms", "loo": each pair's distance when the fit is made without it (None below
    LEAVE_ONE_OUT_MIN pairs), "suspect": the pair that looks misplaced, if one does, "spread":
    spread() of the points}. A pair is suspect when the others agree among themselves
    without it (their distances drop to under half) and it lies more than SUSPECT_FACTOR
    times farther off than they do: a single wrong click pulls the least-squares fit towards
    itself, so its own distance in the fit with it says little. Others nearly on one line
    (spread below SPREAD_MIN) agree among themselves whatever the turn about that line, so
    they judge no pair."""
    np = _np()
    P, Q = np.asarray(moving, float), np.asarray(fixed, float)
    if len(P) != len(Q) or len(P) < MIN_POINTS:
        raise ValueError(f"at least {MIN_POINTS} points placed on both scans are needed")
    if collinear(P) or collinear(Q):
        raise ValueError("the points lie on one line: place at least one away from the others")
    s, R, t = similarity(P, Q)
    M = matrix(s, R, t)
    res = np.linalg.norm(apply(M, P) - Q, axis=1)
    rms = float(np.sqrt((res ** 2).mean()))
    loo = None
    suspect = []
    if len(P) >= LEAVE_ONE_OUT_MIN:
        loo = np.zeros(len(P))
        rest_rms = np.full(len(P), np.inf)
        rest_max = np.zeros(len(P))
        for i in range(len(P)):
            keep = np.arange(len(P)) != i
            if collinear(P[keep]) or collinear(Q[keep]):
                loo[i] = res[i]
                continue
            narrow = spread(Q[keep]) < SPREAD_MIN
            Mi = matrix(*similarity(P[keep], Q[keep]))
            loo[i] = float(np.linalg.norm(apply(Mi, P[i:i + 1])[0] - Q[i]))
            rest = np.linalg.norm(apply(Mi, P[keep]) - Q[keep], axis=1)
            if not narrow:
                rest_rms[i] = float(np.sqrt((rest ** 2).mean()))
                rest_max[i] = float(rest.max())
        i = int(np.argmin(rest_rms))
        if rest_rms[i] < 0.5 * rms and loo[i] > SUSPECT_FACTOR * max(rest_max[i], 1e-12):
            suspect = [i]
    return {"matrix": M, "scale": s, "residuals": res, "rms": rms, "loo": loo, "suspect": suspect,
            "spread": spread(Q)}


def icp(moving, nearest, M0, scale_ref, controls=None, samples=ICP_SAMPLES, iterations=ICP_ITERATIONS,
        start_distance=None, seed=0):
    """Refine the join ``M0`` (4 x 4, moving -> fixed) on the overlap of two clouds.

    ``moving``: N x 3 points of the scan being moved (its own coordinates); ``nearest``:
    function (K x 3 fixed-frame points, limit) -> (K x 3 nearest fixed points, K distances;
    a point with none within the limit may get an infinite distance); ``scale_ref``: the size
    of the scan (for the distances); ``controls``: (moving points, fixed points) of the
    control points, kept in the fit with CONTROL_WEIGHT of the weight. Pairs farther apart
    than a limit are left out (parts that moved, or that only one scan has); the limit starts
    at ``start_distance`` and follows three times the median distance of the pairs kept.

    Returns {"matrix", "rms_before", "rms", "overlap", "iterations", "limit", "median_before",
    "median_after", "improved"}: rms and overlap over the pairs kept (overlap their share of
    the samples); median_before and median_after the median distance of the pairs within the
    starting limit, before and after, for the same samples. ``improved`` says whether the
    clouds came clearly closer (median_after at most REFINE_GAIN of median_before). When they
    overlap little, nearest points are mostly not the same spots, and the fit can slide a long
    way without bringing the clouds closer (on two videos of a helicopter, one of each side:
    the median went from 0.847% to 0.841% of the size while the control points moved from
    0.3% to 1.6% apart): such a result is not an improvement on the points' join."""
    np = _np()
    rng = np.random.default_rng(seed)
    P = np.asarray(moving, float)
    if len(P) > samples:
        P = P[rng.choice(len(P), samples, replace=False)]
    M = np.array(M0, float)
    limit = float(start_distance) if start_distance else 0.02 * scale_ref
    limit0 = limit
    floor = 1e-4 * scale_ref
    cp = cq = None
    if controls is not None and len(controls[0]) >= MIN_POINTS:
        cp, cq = np.asarray(controls[0], float), np.asarray(controls[1], float)
    stats = {}
    it = 0
    for it in range(1, iterations + 1):
        X = apply(M, P)
        Y, d = nearest(X, limit)
        keep = d < limit
        if keep.sum() < 10:
            raise ValueError("the two scans do not overlap where the points put them: check the points")
        if it == 1:
            stats["rms_before"] = float(np.sqrt((d[keep] ** 2).mean()))
            stats["median_before"] = float(np.median(d[keep]))
        src, dst = P[keep], Y[keep]
        w = np.ones(len(src))
        if cp is not None:
            w_each = CONTROL_WEIGHT * len(src) / ((1.0 - CONTROL_WEIGHT) * len(cp))
            src = np.vstack([src, cp])
            dst = np.vstack([dst, cq])
            w = np.concatenate([w, np.full(len(cp), w_each)])
        s, R, t = similarity(src, dst, weights=w)
        new = matrix(s, R, t)
        change = np.abs(new - M).max() / max(1e-12, np.abs(M[:3, :3]).max())
        M = new
        limit = max(floor, min(limit, 3.0 * float(np.median(d[keep]))))
        if change < ICP_TOL:
            break
    X = apply(M, P)
    Y, d = nearest(X, limit0)
    near = d < limit0
    after = float(np.median(d[near])) if near.any() else float("inf")
    keep = d < limit
    stats.update(matrix=M, rms=float(np.sqrt((d[keep] ** 2).mean())) if keep.any() else float("nan"),
                 overlap=float(keep.mean()), iterations=it, limit=limit, median_after=after,
                 improved=bool(after <= REFINE_GAIN * stats["median_before"]))
    return stats


def kdtree_nearest(points):
    """A nearest() for icp() over ``points`` (N x 3) with Blender's mathutils.kdtree.

    A k-d tree search for a point far from every point visits much of the tree: 240 µs
    against 11 µs near the cloud (one AW-249 video's 342,585 points), and in a join most
    of the moved cloud is far from the other (the other side of the object, the
    surroundings). So only the points with a fixed point in their grid cell of the limit's
    size, or in a neighbouring one, are searched; the others are farther than the limit."""
    np = _np()
    from mathutils import kdtree
    pts = np.asarray(points, float)
    tree = kdtree.KDTree(len(pts))
    for i, p in enumerate(pts):
        tree.insert(p, i)
    tree.balance()
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    grid = {}

    def cells(cell):
        """(sorted occupied cell keys, cells per axis) for a cell size."""
        if grid.get("cell") != cell:
            n = np.floor((hi - lo) / cell).astype(np.int64) + 1
            ijk = np.floor((pts - lo) / cell).astype(np.int64)
            grid.update(cell=cell, n=n, keys=np.unique((ijk[:, 0] * n[1] + ijk[:, 1]) * n[2] + ijk[:, 2]))
        return grid["keys"], grid["n"]

    def candidates(X, limit):
        cell = max(float(limit), float(np.max(hi - lo)) / 2000.0)      # at most 2000 cells per axis
        keys, n = cells(cell)
        ijk = np.floor((X - lo) / cell).astype(np.int64)
        found = np.zeros(len(X), bool)
        for off in ((a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)):
            q = ijk + off
            inside = np.all((q >= 0) & (q < n), axis=1)
            k = (q[:, 0] * n[1] + q[:, 1]) * n[2] + q[:, 2]
            pos = np.clip(np.searchsorted(keys, k), 0, len(keys) - 1)
            found |= inside & (keys[pos] == k)
        return found

    def nearest(X, limit=None):
        X = np.asarray(X, float)
        out = np.full((len(X), 3), np.nan)
        dist = np.full(len(X), np.inf)
        idx = np.flatnonzero(candidates(X, limit)) if limit else np.arange(len(X))
        for k in idx:
            co, _, d = tree.find(X[k])
            out[k] = co
            dist[k] = d
        return out, dist
    return nearest
