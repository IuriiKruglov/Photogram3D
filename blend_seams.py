# SPDX-License-Identifier: GPL-3.0-or-later
# Photogram3D - photos or video to a textured mesh with COLMAP + OpenMVS
# Copyright (C) 2026 Kruglov Iurii
# Authors: Kruglov Iurii, Anthropic Claude AI
# License: GNU General Public License (GPL) v3 or later, https://www.blender.org/about/license/
"""Blend the seams of a textured mesh (0.3.12).

TextureMesh builds the texture from patches, each cut from one photo. Where two patches meet,
the photos' exposure, white balance, vignetting and reflections differ, and the edge shows as
a step in brightness or colour. OpenMVS evens such steps out itself ("seam leveling"), but
2.4.0 reads the photos wrongly there and paints blotches, so Photogram3D turns it off
(DEVELOPMENT.md, section 5) and blends the seams here instead, after the global seam leveling
of Waechter, Moehrle and Goesele, "Let There Be Color! Large-Scale Texturing of 3D
Reconstructions" (ECCV 2014), the method OpenMVS implements. A colour correction g, the log2
of a gain on linear light per colour channel, is found at every texture vertex (a mesh vertex
in one patch: a vertex on a seam has one per patch) such that

    sum over seam samples        w (f_a + g_a - f_b - g_b)^2        the two sides meet
  + SMOOTH * sum over the edges inside a patch  (g_i - g_j)^2       g varies smoothly
  + EPS * sum g^2                                                   and stays small

is least. f_a and f_b are the colours on either side, sampled along the seam's edges near
each vertex; w are Huber weights, so that a reflection or an occlusion seen on one side only
counts less. It is solved in two parts by conjugate gradients: one gain per patch first (the
patch-wide exposure, which would take long to spread inwards from the seams vertex by vertex),
then what is left at the seams, over the vertices. The corrections are interpolated over each
triangle into the texture pages, strip by strip. Faces TextureMesh found no photo for (all
three texture coordinates on one texel) take no part.

Run by pipeline.step_blend in Blender's Python outside Blender, with Pillow from the image
libraries' folder (--site); numpy and Pillow only. The pages named in --mtl-in are read and
not changed; the blended pages are written next to them with --suffix added to their names,
and --mtl-out names them. Prints "Read ...", "Seams ...", "Before ...", "Solved ...",
"Page i/n", "After ..." and "Done ..." lines.
"""

import argparse
import os
import sys
import time

SMOOTH = 0.1          # weight of the smoothness inside a patch, against 1 per seam sample
EPS = 0.01            # pull towards no correction (per patch: times its vertices); DEVELOPMENT.md 5
HUBER = 0.2           # stops: seam differences beyond this weigh less
SAMPLES = 4           # samples along each half of a seam edge
DARK = 0.005          # added to linear values before the log: dark noise weighs less
MAX_GAIN = 3.0        # stops: corrections are clipped to this
PATCH_ROUNDS = 2      # Huber rounds of the per-patch part
PATCH_ITERS = 500
VERTEX_ITERS = 150
TOL = 1e-4
STRIP = 512           # rows of a texture page done at a time
DILATE = 3            # texels the corrections reach beyond the triangles (patch borders)
CHUNK = 4_000_000     # candidate texels rasterized at a time
BLOCK = 1 << 26       # bytes of the OBJ file parsed at a time
LUMA = (0.2126, 0.7152, 0.0722)
PATCH_STAGE = True    # the two parts of the solve (switches for testing)
VERTEX_STAGE = True


def _init(site):
    if site and site not in sys.path:
        sys.path.insert(0, site)


def say(text):
    print(text, flush=True)


# ------------------------------------------------------------------------------- reading

def _parse_block(data, state):
    """Parse the complete lines in ``data`` (bytes) into ``state``: the lines of one kind come
    in runs (TextureMesh writes all vertices, all texture coordinates, then the faces of each
    material), and each run is parsed in one go."""
    import numpy as np
    buf = np.frombuffer(data, np.uint8)
    ends = np.flatnonzero(buf == 10)
    if not len(ends) or ends[-1] != len(data) - 1:
        ends = np.r_[ends, len(data)]
    starts = np.r_[0, ends[:-1] + 1]
    c0 = buf[np.minimum(starts, len(buf) - 1)]
    c1 = buf[np.minimum(starts + 1, len(buf) - 1)]
    kind = np.zeros(len(starts), np.int8)
    kind[(c0 == ord("v")) & (c1 == ord("t"))] = 2
    kind[(c0 == ord("f")) & ((c1 == ord(" ")) | (c1 == ord("\t")))] = 3
    kind[c0 == ord("u")] = 4
    del c0, c1, buf
    change = np.flatnonzero(np.diff(kind)) + 1
    for s, e in zip(np.r_[0, change], np.r_[change, len(kind)]):
        k = int(kind[s])
        if k == 0:
            continue
        text = data[starts[s]:ends[e - 1]]
        if k == 2:
            if state["per_vt"] is None:
                state["per_vt"] = len(data[starts[s]:ends[s]].split()) - 1
            per = state["per_vt"]
            vals = np.fromstring(text.replace(b"vt", b"  "), sep=" ")
            if per < 2 or len(vals) != (e - s) * per:
                raise ValueError("unexpected texture coordinate lines")
            state["vt"].append(vals.reshape(-1, per)[:, :2].astype(np.float32))
        elif k == 4:
            for line in text.splitlines():
                a = line.split(None, 1)
                if a and a[0] == b"usemtl":
                    name = a[1].strip().decode("utf-8", "replace") if len(a) > 1 else ""
                    if name not in state["names"]:
                        state["names"].append(name)
                    state["cur"] = state["names"].index(name)
        elif k == 3:
            if state["per_f"] is None:
                first = data[starts[s]:ends[s]].split()[1:]
                if len(first) != 3:
                    raise ValueError("the mesh has faces that are not triangles")
                if b"//" in first[0] or b"/" not in first[0]:
                    raise ValueError("the mesh has no texture coordinates")
                state["per_f"] = first[0].count(b"/") + 1
            per = state["per_f"]
            vals = np.fromstring(text.replace(b"f", b" ").replace(b"/", b" "), dtype=np.int64, sep=" ")
            if len(vals) != (e - s) * 3 * per:
                raise ValueError("the mesh has faces that are not triangles or lack texture coordinates")
            vals = vals.reshape(-1, 3, per)
            state["fv"].append((vals[:, :, 0] - 1).astype(np.int32))
            state["ft"].append((vals[:, :, 1] - 1).astype(np.int32))
            state["fm"].append(np.full(len(vals), state["cur"], np.int16))


def read_obj(path):
    """(texture coordinates [n, 2] float32, face vertices [F, 3], face texture coordinates
    [F, 3] (0-based, int32), face material [F] (-1: none), material names) of a triangle
    mesh with texture coordinates, read BLOCK bytes at a time."""
    import numpy as np
    state = {"vt": [], "fv": [], "ft": [], "fm": [], "names": [], "cur": -1, "per_vt": None, "per_f": None}
    rest = b""
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(BLOCK)
            data = rest + chunk if rest else chunk
            if not data:
                break
            if chunk:
                cut = data.rfind(b"\n") + 1
                if cut == 0:
                    rest = data
                    continue
                data, rest = data[:cut], data[cut:]
            else:
                rest = b""
            _parse_block(data, state)
            if not chunk:
                break
    if not state["fv"] or not state["vt"]:
        raise ValueError("no textured faces in the mesh")
    vt = np.concatenate(state["vt"])
    fv, ft = np.concatenate(state["fv"]), np.concatenate(state["ft"])
    if fv.min() < 0 or ft.min() < 0 or ft.max() >= len(vt):
        raise ValueError("unexpected face indices")
    return vt, fv, ft, np.concatenate(state["fm"]).astype(np.int32), state["names"]


def read_mtl(path):
    """(lines, {material name: texture file name}) of a .mtl file."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    maps, cur = {}, None
    for line in lines:
        a = line.split(None, 1)
        if a and a[0] == "newmtl":
            cur = a[1].strip() if len(a) > 1 else ""
        elif a and a[0] == "map_Kd" and cur is not None and len(a) > 1:
            maps[cur] = a[1].strip()
    return lines, maps


def write_mtl(path, lines, renamed):
    """The .mtl ``lines`` with the texture files renamed ({old: new})."""
    out = []
    for line in lines:
        a = line.split(None, 1)
        if a and a[0] == "map_Kd" and len(a) > 1 and a[1].strip() in renamed:
            line = line[:len(line) - len(line.lstrip())] + "map_Kd " + renamed[a[1].strip()]
        out.append(line)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out) + "\n")
    os.replace(tmp, path)


def load_page(path):
    import numpy as np
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))


def save_page(arr, path):
    from PIL import Image
    tmp = path + ".tmp"
    ext = os.path.splitext(path)[1].lower()
    fmt = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG"}.get(ext, "PNG")
    opts = {"quality": 95} if fmt == "JPEG" else {"compress_level": 4}
    Image.fromarray(arr).save(tmp, fmt, **opts)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------- the mesh

def linear_lut():
    import numpy as np
    x = np.arange(256) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def to_srgb8(lin):
    import numpy as np
    lin = np.clip(lin, 0.0, 1.0)
    s = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1 / 2.4) - 0.055)
    return np.clip(s * 255.0 + 0.5, 0, 255).astype(np.uint8)


def texture_vertices(fv, ft, fm, vt):
    """Per face corner, its texture vertex [F, 3] int32, and their number: one per mesh
    vertex, page and texture position (TextureMesh repeats equal texture coordinates under
    new indices for every face)."""
    import numpy as np
    uvk = (np.round(vt[:, 0].astype(np.float64) * (1 << 20)).astype(np.int64) * (1 << 22)
           + np.round(vt[:, 1].astype(np.float64) * (1 << 20)).astype(np.int64))
    uv = uvk[ft.ravel()]
    del uvk
    vm = fv.ravel().astype(np.int64) * 256 + np.repeat(fm, 3) + 1      # fm is -1 without a page
    order = np.lexsort((uv, vm))
    vs, us = vm[order], uv[order]
    del vm, uv
    new = np.empty(len(order), bool)
    new[:1] = True
    new[1:] = (vs[1:] != vs[:-1]) | (us[1:] != us[:-1])
    del vs, us
    ids = np.cumsum(new) - 1
    tv = np.empty(len(order), np.int32)
    tv[order] = ids
    return tv.reshape(fv.shape), (int(ids[-1]) + 1 if len(ids) else 0)


def seen_faces(vt, ft):
    """Faces with a texture: TextureMesh puts the faces it found no photo for on one texel."""
    import numpy as np
    uv = vt[ft].astype(np.float64)
    area = ((uv[:, 1, 0] - uv[:, 0, 0]) * (uv[:, 2, 1] - uv[:, 0, 1])
            - (uv[:, 2, 0] - uv[:, 0, 0]) * (uv[:, 1, 1] - uv[:, 0, 1]))
    return np.abs(area) > 1e-12


def seam_edges(fv, ft, fm, vt, seen):
    """Mesh edges with exactly two textured faces whose texture positions differ along them:
    (face a, face b, corner of the edge's lower vertex in a, of its higher one in a, and the
    same in b)."""
    import numpy as np
    F = len(fv)
    face = np.tile(np.arange(F, dtype=np.int32), 3)
    ca = np.repeat(np.arange(3, dtype=np.int8), F)
    cb = (ca + 1) % 3
    va, vb = fv[face, ca], fv[face, cb]
    swap = va > vb
    lo_c, hi_c = np.where(swap, cb, ca), np.where(swap, ca, cb)
    key = np.minimum(va, vb).astype(np.int64) * (int(fv.max()) + 1) + np.maximum(va, vb)
    del va, vb, swap, ca, cb
    order = np.argsort(key, kind="stable")
    ks = key[order]
    del key
    starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    counts = np.diff(np.r_[starts, len(ks)])
    del ks
    two = starts[counts == 2]
    a, b = order[two], order[two + 1]
    del order, starts, counts, two
    fa, fb = face[a], face[b]
    both = seen[fa] & seen[fb]
    a, b, fa, fb = a[both], b[both], fa[both], fb[both]
    la, ha, lb, hb = lo_c[a], hi_c[a], lo_c[b], hi_c[b]
    differ = ((np.abs(vt[ft[fa, la]] - vt[ft[fb, lb]]).max(1) > 1e-7)
              | (np.abs(vt[ft[fa, ha]] - vt[ft[fb, hb]]).max(1) > 1e-7) | (fm[fa] != fm[fb]))
    return fa[differ], fb[differ], la[differ], ha[differ], lb[differ], hb[differ]


def inner_edges(tv, n):
    """The edges between the texture vertices of each face, once each: (i, j) int32."""
    import numpy as np
    ei = np.concatenate([tv[:, 0], tv[:, 1], tv[:, 2]])
    ej = np.concatenate([tv[:, 1], tv[:, 2], tv[:, 0]])
    lo, hi = np.minimum(ei, ej).astype(np.int64), np.maximum(ei, ej).astype(np.int64)
    del ei, ej
    key = lo * n + hi
    key = np.unique(key[lo != hi])
    del lo, hi
    return (key // n).astype(np.int32), (key % n).astype(np.int32)


def components(n, a, b):
    """Connected components of the graph of ``n`` nodes and edges (a, b): labels 0 .. k-1."""
    import numpy as np
    parent = np.arange(n)
    while len(a):
        ra, rb = parent[a], parent[b]
        diff = ra != rb
        if not diff.any():
            break
        np.minimum.at(parent, np.maximum(ra[diff], rb[diff]), np.minimum(ra[diff], rb[diff]))
        while True:
            nxt = parent[parent]
            if np.array_equal(nxt, parent):
                break
            parent = nxt
        a, b = a[diff], b[diff]
    return np.unique(parent, return_inverse=True)[1]


# ---------------------------------------------------------------------------- the solve

def _system(n, pa, pb, w, ei, ej, smooth, eps):
    """The normal matrix of sum w (x_a - x_b - d)^2 + smooth sum (x_i - x_j)^2 + eps x^2,
    in compressed rows (row starts, column indices, values) with every row's diagonal, and
    the diagonal."""
    import numpy as np
    diag = np.broadcast_to(np.asarray(eps, float), (n,)).astype(float)
    diag += np.bincount(pa, w, n) + np.bincount(pb, w, n)
    if len(ei):
        diag += smooth * (np.bincount(ei, None, n) + np.bincount(ej, None, n))
    itype = np.int32 if n < 2 ** 31 else np.int64
    rows = np.concatenate([pa, pb, ei, ej, np.arange(n)]).astype(itype)
    cols = np.concatenate([pb, pa, ej, ei, np.arange(n)]).astype(itype)
    vals = np.concatenate([-w, -w, np.full(2 * len(ei), -smooth), diag]).astype(np.float32)
    order = np.argsort(rows, kind="stable")
    indices, data = cols[order], vals[order]
    starts = np.r_[0, np.cumsum(np.bincount(rows, minlength=n))[:-1]]
    return starts, indices, data, diag


def solve(n, pa, pb, w, d, ei, ej, smooth, eps, x0=None, iters=200, tol=TOL):
    """Least squares of sum w (x_a - x_b - d)^2 + smooth sum (x_i - x_j)^2 + eps x^2 for x
    [n, 3], by conjugate gradients with a Jacobi preconditioner: (x, iterations)."""
    import numpy as np
    starts, indices, data, diag = _system(n, pa, pb, w, ei, ej, smooth, eps)
    inv_diag = 1.0 / diag

    def mv(x):                                          # x: [3, n]
        return np.stack([np.add.reduceat(x[c][indices] * data, starts) for c in range(3)])

    rhs = np.stack([np.bincount(pa, w * d[:, c], n) - np.bincount(pb, w * d[:, c], n) for c in range(3)])
    x = np.zeros((3, n)) if x0 is None else np.ascontiguousarray(x0.T, dtype=float)
    r = rhs - mv(x)
    z = r * inv_diag
    p = z.copy()
    rz = (r * z).sum(1)
    norm0 = np.sqrt((rhs ** 2).sum(1)) + 1e-30
    it = 0
    for it in range(1, iters + 1):
        ap = mv(p)
        alpha = rz / ((p * ap).sum(1) + 1e-30)
        x += alpha[:, None] * p
        r -= alpha[:, None] * ap
        if (np.sqrt((r ** 2).sum(1)) / norm0).max() < tol:
            break
        z = r * inv_diag
        rz_new = (r * z).sum(1)
        p = z + (rz_new / (rz + 1e-30))[:, None] * p
        rz = rz_new
    return np.ascontiguousarray(x.T), it


def luma_stops(d):
    import numpy as np
    return np.abs(d @ np.array(LUMA))


# ------------------------------------------------------------------------- the pages

def _rasterize(rows, width, px, py, vals):
    """Interpolate per-corner values ``vals`` [F, 3, 3] over triangles in texel coordinates
    (``px``, ``py`` [F, 3]) into the rows ``rows`` = (first, last + 1) of a page: (values
    [h, width, 3] float32, covered [h, width] bool)."""
    import numpy as np
    r0, r1 = rows
    h = r1 - r0
    out = np.zeros((h, width, 3), np.float32)
    cov = np.zeros((h, width), bool)
    if not len(px):
        return out, cov
    x0 = np.clip(np.floor(px.min(1)), 0, width - 1).astype(np.int64)
    x1 = np.clip(np.ceil(px.max(1)), 0, width - 1).astype(np.int64)
    y0 = np.clip(np.floor(py.min(1)), r0, r1 - 1).astype(np.int64)
    y1 = np.clip(np.ceil(py.max(1)), r0, r1 - 1).astype(np.int64)
    nx, ny = x1 - x0 + 1, y1 - y0 + 1
    ax, ay = px[:, 0], py[:, 0]
    e1x, e1y = px[:, 1] - ax, py[:, 1] - ay
    e2x, e2y = px[:, 2] - ax, py[:, 2] - ay
    det = e1x * e2y - e2x * e1y
    ok = (np.abs(det) > 1e-9) & (nx > 0) & (ny > 0)
    idx_all = np.flatnonzero(ok)
    cnt_all = (nx * ny)[idx_all]
    csum = np.cumsum(cnt_all)
    start = 0
    while start < len(idx_all):
        base = csum[start - 1] if start else 0
        end = max(int(np.searchsorted(csum, base + CHUNK, side="right")), start + 1)
        idx, c = idx_all[start:end], cnt_all[start:end]
        f = np.repeat(idx, c)
        k = np.arange(int(c.sum())) - np.repeat(np.cumsum(c) - c, c)
        X = x0[f] + k % nx[f]
        Y = y0[f] + k // nx[f]
        del k
        dx = (X - ax[f]).astype(np.float32)
        dy = (Y - ay[f]).astype(np.float32)
        b1 = (dx * e2y[f] - e2x[f] * dy) / det[f]
        b2 = (e1x[f] * dy - dx * e1y[f]) / det[f]
        b0 = 1.0 - b1 - b2
        inside = (b0 >= -1e-4) & (b1 >= -1e-4) & (b2 >= -1e-4)
        f, X, Y = f[inside], X[inside], Y[inside]
        b0, b1, b2 = b0[inside, None], b1[inside, None], b2[inside, None]
        out[Y - r0, X] = vals[f, 0] * b0 + vals[f, 1] * b1 + vals[f, 2] * b2
        cov[Y - r0, X] = True
        start = end
    return out, cov


def _dilate(vals, cov, steps):
    """Spread values into the uncovered texels next to covered ones (the patches' borders,
    which texture filtering reads too)."""
    import numpy as np
    for _ in range(steps):
        acc = np.zeros_like(vals)
        n = np.zeros(cov.shape, np.float32)
        for to, frm in (((slice(1, None),), (slice(None, -1),)), ((slice(None, -1),), (slice(1, None),)),
                        ((slice(None), slice(1, None)), (slice(None), slice(None, -1))),
                        ((slice(None), slice(None, -1)), (slice(None), slice(1, None)))):
            acc[to] += vals[frm] * cov[frm][..., None]
            n[to] += cov[frm]
        grow = (~cov) & (n > 0)
        vals[grow] = acc[grow] / n[grow][:, None]
        cov = cov | grow
    return vals, cov


def apply_page(img, faces_uv, corner_g, lut):
    """Apply the corrections ``corner_g`` [F, 3, 3] (stops) of the triangles ``faces_uv``
    [F, 3, 2] to the page ``img`` [h, w, 3] uint8 in place, strip by strip; returns the
    share of texels changed."""
    import numpy as np
    h, w = img.shape[:2]
    px = (faces_uv[..., 0].astype(np.float32) * w - 0.5)
    py = ((1.0 - faces_uv[..., 1].astype(np.float32)) * h - 0.5)
    fy0 = np.floor(py.min(1))
    fy1 = np.ceil(py.max(1))
    changed = 0
    for s0 in range(0, h, STRIP):
        s1 = min(h, s0 + STRIP)
        r0, r1 = max(0, s0 - DILATE), min(h, s1 + DILATE)
        sel = np.flatnonzero((fy1 >= r0) & (fy0 < r1))
        corr, cov = _rasterize((r0, r1), w, px[sel], py[sel], corner_g[sel])
        corr, cov = _dilate(corr, cov, DILATE)
        corr, cov = corr[s0 - r0:s1 - r0], cov[s0 - r0:s1 - r0]
        if cov.any():
            block = img[s0:s1]
            block[cov] = to_srgb8(lut[block[cov]] * np.exp2(corr[cov]))
            changed += int(cov.sum())
    return changed / float(h * w)


# ---------------------------------------------------------------------------- the whole

def blend(obj_path, mtl_in, mtl_out, suffix):
    import numpy as np
    t0 = time.time()
    vt, fv, ft, fm, names = read_obj(obj_path)
    lines, maps = read_mtl(mtl_in)
    src_dir = os.path.dirname(os.path.abspath(mtl_in))
    out_dir = os.path.dirname(os.path.abspath(mtl_out))
    pages = {i: maps[name] for i, name in enumerate(names)
             if name in maps and os.path.isfile(os.path.join(src_dir, maps[name]))}
    if not pages:
        raise ValueError("the texture pages named in the .mtl file are missing")
    fm = np.where(np.isin(fm, list(pages)), fm, -1).astype(np.int32)
    seen = seen_faces(vt, ft) & (fm >= 0)
    tv, n = texture_vertices(fv, ft, fm, vt)
    say(f"Read {len(fv):,} faces ({int(seen.sum()):,} textured), {n:,} texture vertices, "
        f"{len(pages)} texture page(s)")
    ei, ej = inner_edges(tv, n)
    fa, fb, la, ha, lb, hb = seam_edges(fv, ft, fm, vt, seen)
    del fv
    # seam samples: the half of each seam edge next to each of its vertices, from both sides
    t = (np.arange(SAMPLES) + 0.5) / (2 * SAMPLES)
    lut = linear_lut()
    col = np.zeros((2, 2, len(fa), 3))                 # [end, side, edge, channel]
    for page, name in pages.items():
        img = load_page(os.path.join(src_dir, name))
        hgt, wid = img.shape[:2]
        for k, ends in enumerate(((la, ha, lb, hb), (ha, la, hb, lb))):
            for side, (faces, c0, c1) in enumerate(((fa, ends[0], ends[1]), (fb, ends[2], ends[3]))):
                sel = np.flatnonzero(fm[faces] == page)
                if not len(sel):
                    continue
                uv0 = vt[ft[faces[sel], c0[sel]]].astype(np.float64)
                uv1 = vt[ft[faces[sel], c1[sel]]].astype(np.float64)
                acc = np.zeros((len(sel), 3))
                for s in t:
                    uv = uv0 * (1 - s) + uv1 * s
                    x = np.clip(uv[:, 0] * wid - 0.5, 0, wid - 1.001)
                    y = np.clip((1.0 - uv[:, 1]) * hgt - 0.5, 0, hgt - 1.001)
                    x0, y0 = x.astype(np.int64), y.astype(np.int64)
                    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
                    acc += ((lut[img[y0, x0]] * (1 - fx) + lut[img[y0, x0 + 1]] * fx) * (1 - fy)
                            + (lut[img[y0 + 1, x0]] * (1 - fx) + lut[img[y0 + 1, x0 + 1]] * fx) * fy)
                col[k, side, sel] = np.log2(acc / len(t) + DARK)
        del img
    pa = np.concatenate([tv[fa, la], tv[fa, ha]]).astype(np.int64)
    pb = np.concatenate([tv[fb, lb], tv[fb, hb]]).astype(np.int64)
    d = np.concatenate([col[0, 1] - col[0, 0], col[1, 1] - col[1, 0]])     # g_a - g_b should be this
    del col
    patch = components(n, ei, ej)
    k = int(patch.max()) + 1 if n else 0
    say(f"Seams: {len(fa):,} edges between {len(np.unique(patch[pa])) if len(pa) else 0:,} of {k:,} patches")
    if not len(d):
        say("No seams to blend")
        g = np.zeros((n, 3))
    else:
        before = luma_stops(d)
        say(f"Before: steps across the seams median {np.median(before):.2f} stops, "
            f"90% {np.percentile(before, 90):.2f}")
        # 1. one gain per patch
        size = np.bincount(patch, minlength=k).astype(float)
        Pa, Pb = patch[pa], patch[pb]
        across = Pa != Pb
        w = np.ones(len(d))
        c = np.zeros((k, 3))
        none = np.zeros(0, np.int64)
        its1 = 0
        for _ in range(PATCH_ROUNDS if PATCH_STAGE else 0):
            c, its = solve(k, Pa[across], Pb[across], w[across], d[across], none, none, 0.0, EPS * size,
                           x0=c, iters=PATCH_ITERS)
            its1 += its
            r = luma_stops(c[Pa] - c[Pb] - d)
            w = np.where(r <= HUBER, 1.0, HUBER / np.maximum(r, 1e-9))
        # 2. what is left, spread from the seams into each side
        rest = d - (c[Pa] - c[Pb])
        h, its2 = (solve(n, pa, pb, w, rest, ei, ej, SMOOTH, EPS, iters=VERTEX_ITERS) if VERTEX_STAGE
                   else (np.zeros((n, 3)), 0))
        g = np.clip(c[patch] + h, -MAX_GAIN, MAX_GAIN)
        del h, rest
        say(f"Solved: {k:,} patch gains in {its1} iterations, {n:,} vertex corrections in {its2}")
    del ei, ej
    # the pages
    out_names = {}
    for i, (page, name) in enumerate(sorted(pages.items())):
        sel = np.flatnonzero((fm == page) & seen)
        stem, ext = os.path.splitext(name)
        new_name = stem + suffix + ext
        img = load_page(os.path.join(src_dir, name))
        apply_page(img, vt[ft[sel]], g[tv[sel]].astype(np.float32), lut)
        save_page(img, os.path.join(out_dir, new_name))
        del img
        out_names[name] = new_name
        say(f"Page {i + 1}/{len(pages)} {new_name}")
    if len(d):
        after = luma_stops(d - (g[pa] - g[pb]))
        say(f"After: steps across the seams median {np.median(after):.2f} stops, "
            f"90% {np.percentile(after, 90):.2f}; corrections up to {np.percentile(np.abs(g), 99):.2f} "
            f"stops (99%)")
    write_mtl(mtl_out, lines, out_names)
    say(f"Done in {time.time() - t0:.0f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="")
    ap.add_argument("--obj", required=True)
    ap.add_argument("--mtl-in", required=True)
    ap.add_argument("--mtl-out", required=True)
    ap.add_argument("--suffix", default="_blended")
    args = ap.parse_args()
    _init(args.site)
    blend(args.obj, args.mtl_in, args.mtl_out, args.suffix)


if __name__ == "__main__":
    main()
