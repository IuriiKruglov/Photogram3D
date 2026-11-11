# SPDX-License-Identifier: GPL-3.0-or-later
# Photogram3D - photos or video to a textured mesh with COLMAP + OpenMVS
# Copyright (C) 2026 Kruglov Iurii
# Authors: Kruglov Iurii, Anthropic Claude AI
# License: GNU General Public License (GPL) v3 or later, https://www.blender.org/about/license/
"""Feature-preserving mesh polishing (normal filtering + vertex update).

Method: Sun et al. 2007, "Fast and Effective Feature-Preserving Mesh
Denoising". Face normals are averaged only with similar neighbours
(weight (n_i.n_j - T)^2 above a threshold T), then vertices are moved so
the faces agree with the filtered normals. The vertex update also accepts
normals from another source (for example a learned normal predictor).

The core works on numpy arrays and does not import bpy.
"""

import numpy as np


def face_data(verts, faces):
    a = verts[faces[:, 0]]
    b = verts[faces[:, 1]]
    c = verts[faces[:, 2]]
    cross = np.cross(b - a, c - a)
    area2 = np.linalg.norm(cross, axis=1)
    normals = cross / np.maximum(area2, 1e-30)[:, None]
    centroids = (a + b + c) / 3.0
    return normals, centroids, area2 * 0.5


def edge_adjacent_pairs(faces):
    """Index pairs (i, j) of faces sharing an edge, both directions."""
    m = len(faces)
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e.sort(axis=1)
    fid = np.tile(np.arange(m, dtype=np.int64), 3)
    key = e[:, 0].astype(np.int64) * (int(faces.max()) + 1) + e[:, 1]
    order = np.argsort(key, kind="stable")
    key_s, fid_s = key[order], fid[order]
    same = key_s[1:] == key_s[:-1]
    i = fid_s[:-1][same]
    j = fid_s[1:][same]
    return np.concatenate([i, j]), np.concatenate([j, i])


def _accumulate(index, values, size):
    out = np.zeros((size, values.shape[1]))
    for k in range(values.shape[1]):
        out[:, k] = np.bincount(index, weights=values[:, k], minlength=size)
    return out


def filter_normals(normals, areas, pairs, threshold=0.6, iterations=8):
    """Iterative feature-preserving face normal filter."""
    i, j = pairs
    n = normals.copy()
    self_w = (1.0 - threshold) ** 2
    for _ in range(iterations):
        dot = np.einsum("ij,ij->i", n[i], n[j])
        w = np.where(dot > threshold, (dot - threshold) ** 2, 0.0) * areas[j]
        acc = _accumulate(i, n[j] * w[:, None], len(n))
        acc += n * (self_w * areas)[:, None]
        n = acc / np.maximum(np.linalg.norm(acc, axis=1), 1e-30)[:, None]
    return n


def update_vertices(verts, faces, target_normals, iterations=10, strength=1.0, pinned=None):
    """Move vertices so faces become perpendicular to ``target_normals``."""
    v = verts.copy()
    nv = len(v)
    counts = np.bincount(faces.ravel(), minlength=nv).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    for _ in range(iterations):
        cent = (v[faces[:, 0]] + v[faces[:, 1]] + v[faces[:, 2]]) / 3.0
        delta = np.zeros_like(v)
        for k in range(3):
            vid = faces[:, k]
            d = np.einsum("ij,ij->i", target_normals, cent - v[vid])
            delta += _accumulate(vid, target_normals * d[:, None], nv)
        step = delta / counts[:, None] * strength
        if pinned is not None:
            step[pinned] = 0.0
        v += step
    return v


def boundary_vertices(faces, nverts):
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e.sort(axis=1)
    key = e[:, 0].astype(np.int64) * (nverts + 1) + e[:, 1]
    uniq, cnt = np.unique(key, return_counts=True)
    boundary_keys = uniq[cnt == 1]
    mask = np.zeros(nverts, dtype=bool)
    if len(boundary_keys):
        a = boundary_keys // (nverts + 1)
        b = boundary_keys % (nverts + 1)
        mask[a] = True
        mask[b] = True
    return mask


def polish(verts, faces, iterations=8, keep_edges=0.6, strength=0.5, normals=None,
           progress=None):
    """Denoise ``verts`` (N,3 float) of triangle mesh ``faces`` (M,3 int).

    keep_edges in [0,1] maps to the similarity threshold T in [0.2, 0.95]:
    higher keeps more creases. ``normals`` (M,3) replaces the filtered face
    normals (e.g. learned predictions). Boundary vertices stay fixed.
    """
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    pinned = boundary_vertices(faces, len(verts))
    threshold = 0.2 + 0.75 * float(keep_edges)
    v = verts
    pairs = edge_adjacent_pairs(faces)
    for it in range(max(1, iterations)):
        n, _, areas = face_data(v, faces)
        if normals is None:
            target = filter_normals(n, areas, pairs, threshold=threshold, iterations=4)
        else:
            target = normals
        v = update_vertices(v, faces, target, iterations=3, strength=strength, pinned=pinned)
        if progress:
            progress((it + 1) / max(1, iterations))
    return v
