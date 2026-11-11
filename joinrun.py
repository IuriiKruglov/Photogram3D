# SPDX-License-Identifier: GPL-3.0-or-later
# Photogram3D - photos or video to a textured mesh with COLMAP + OpenMVS
# Copyright (C) 2026 Kruglov Iurii
# Authors: Kruglov Iurii, Anthropic Claude AI
# License: GNU General Public License (GPL) v3 or later, https://www.blender.org/about/license/
"""One mesh from two joined scans (0.3.13).

The second scan has been moved onto the first by control points (join.py); the similarity
that takes its reconstruction's coordinates into the first one's is known. A run of its own
is then put together from both, without computing depth maps again:

- its photos: both runs' undistorted photos, linked (the second's names prefixed);
- its cameras: one COLMAP model with the first run's cameras and points as they are, then
  the second run's moved into the first's coordinates. OpenMVS numbers a scene's photos in
  the order of this model, so the first run's photos keep their numbers and the second's
  follow them;
- its dense cloud: both runs' OpenMVS clouds in one file, the second's points moved and the
  photos that see each point renumbered. ReconstructMesh needs those photo lists: they say
  which space between a camera and a point is empty.

Mesh, refinement, Even Out Photos, texture and Blend Seams then run as in any run, inside the
first scan's crop box. numpy only; OpenMVS's dense cloud files (binary PLY with a photo list
and a weight list per point) are read and written a block at a time.
"""

import hashlib
import json
import os
import shutil
import struct

from . import colmap_io

PREFIX = "j2_"                  # the second scan's photos in the joined run
BLOCK = 1 << 26                 # bytes of a dense cloud handled at a time


def _np():
    import numpy as np
    return np


def join_id(solve_a, solve_b, matrix):
    """The joined run's camera solve id: it changes with either scan's solve or the join."""
    m = ",".join(f"{v:.9g}" for row in matrix for v in row)
    return "joined-" + hashlib.sha1(f"{solve_a}|{solve_b}|{m}".encode()).hexdigest()[:16]


def second_prefix(model_a):
    """The prefix of the second scan's photo names: PREFIX, or j3_, j4_, ... when the first
    scan is itself a joined run with photos named so. Two videos name their frames alike
    (by frame number), so the names are always prefixed."""
    names = [im["name"] for im in colmap_io.read_images_binary(os.path.join(model_a, "images.bin")).values()]
    k = int(PREFIX[1:-1])
    while any(n.startswith(f"j{k}_") for n in names):
        k += 1
    return f"j{k}_"


# --------------------------------------------------------------------------- COLMAP model

def read_points3d(path):
    """{point id: (xyz, rgb, error, [(image id, point2D index)])} of a points3D.bin."""
    out = {}
    with open(path, "rb") as f:
        (num,) = struct.unpack("<Q", f.read(8))
        for _ in range(num):
            pid, x, y, z, r, g, b, err, length = struct.unpack("<Q3d3BdQ", f.read(51))
            track = struct.unpack(f"<{2 * length}I", f.read(8 * length))
            out[pid] = ((x, y, z), (r, g, b), err, list(zip(track[0::2], track[1::2])))
    return out


def write_model(folder, cameras, images, points):
    """A binary COLMAP model. ``cameras`` {id: read_cameras_binary entry}; ``images``
    [(id, qvec, tvec, camera id, name, [(x, y, point id)])] in the order to write;
    ``points`` [(id, xyz, rgb, error, [(image id, point2D index)])]."""
    os.makedirs(folder, exist_ok=True)
    models = {name: (mid, n) for mid, (name, n) in colmap_io.CAMERA_MODELS.items()}
    with open(os.path.join(folder, "cameras.bin"), "wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for cid, c in sorted(cameras.items()):
            mid, n = models[c["model"]]
            f.write(struct.pack("<IiQQ", cid, mid, c["width"], c["height"]))
            f.write(struct.pack(f"<{n}d", *c["params"]))
    with open(os.path.join(folder, "images.bin"), "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for iid, q, t, cid, name, pts in images:
            f.write(struct.pack("<I7dI", iid, *q, *t, cid))
            f.write(name.encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", len(pts)))
            if pts:
                f.write(b"".join(struct.pack("<ddq", x, y, pid) for x, y, pid in pts))
    with open(os.path.join(folder, "points3D.bin"), "wb") as f:
        f.write(struct.pack("<Q", len(points)))
        for pid, xyz, rgb, err, track in points:
            f.write(struct.pack("<Q3d3BdQ", pid, *xyz, *rgb, err, len(track)))
            if track:
                f.write(struct.pack(f"<{2 * len(track)}I", *[v for pair in track for v in pair]))


def _moved_pose(qvec, tvec, M):
    """A camera's (qvec, tvec) after the similarity M (4 x 4, world x -> s R x + t)."""
    np = _np()
    A = np.asarray(M, float)[:3, :3]
    s = float(np.cbrt(np.linalg.det(A)))
    R = A / s
    Rc = np.asarray(colmap_io.qvec_to_rotmat(qvec))
    C = -Rc.T @ np.asarray(tvec, float)
    C2 = s * R @ C + np.asarray(M, float)[:3, 3]
    Rc2 = Rc @ R.T
    return tuple(colmap_io.rotmat_to_qvec(Rc2.tolist())), tuple((-Rc2 @ C2).tolist())


def joined_model(model_a, model_b, M, out):
    """Write the joined COLMAP model (binary) into ``out``: the first model's cameras,
    photos and points as they are, then the second's moved by ``M`` (its coordinates into
    the first's), its photos renamed with second_prefix(). Photos are written in the order
    of each model's images.bin: InterfaceCOLMAP numbers a scene's photos in that order
    (OpenMVS apps/InterfaceCOLMAP, ImportScene), and so do the photo lists of each run's
    dense cloud. Returns (the first model's photo count, the second's, {second's name:
    joined name})."""
    np = _np()
    Mf = np.asarray(M, float)
    prefix = second_prefix(model_a)
    cams_a = colmap_io.read_cameras_binary(os.path.join(model_a, "cameras.bin"))
    cams_b = colmap_io.read_cameras_binary(os.path.join(model_b, "cameras.bin"))
    imgs_a = colmap_io.read_images_binary(os.path.join(model_a, "images.bin"), read_points=True)
    imgs_b = colmap_io.read_images_binary(os.path.join(model_b, "images.bin"), read_points=True)
    pts_a = read_points3d(os.path.join(model_a, "points3D.bin"))
    pts_b = read_points3d(os.path.join(model_b, "points3D.bin"))
    cam_off = max(cams_a) if cams_a else 0
    img_off = max(imgs_a) if imgs_a else 0
    pt_off = max(pts_a) if pts_a else 0
    cameras = dict(cams_a)
    cameras.update({cid + cam_off: c for cid, c in cams_b.items()})
    images = [(iid, im["qvec"], im["tvec"], im["camera_id"], im["name"], im["points2D"])
              for iid, im in imgs_a.items()]
    renamed = {}
    for iid, im in imgs_b.items():
        q, t = _moved_pose(im["qvec"], im["tvec"], Mf)
        name = prefix + im["name"]
        renamed[im["name"]] = name
        pts = [(x, y, pid + pt_off if pid >= 0 else -1) for x, y, pid in im["points2D"]]
        images.append((iid + img_off, q, t, im["camera_id"] + cam_off, name, pts))
    points = [(pid,) + rest for pid, rest in pts_a.items()]
    for pid, (xyz, rgb, err, track) in pts_b.items():
        X = (Mf[:3, :3] @ np.asarray(xyz) + Mf[:3, 3]).tolist()
        points.append((pid + pt_off, X, rgb, err, [(i + img_off, k) for i, k in track]))
    write_model(out, cameras, images, points)
    return len(imgs_a), len(imgs_b), renamed


def link_photos(src, dst, renamed=None):
    """Hard links (copies across drives) of the photos in ``src`` in ``dst``, renamed by
    ``renamed`` {name: new name} where given."""
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        path = os.path.join(src, name)
        if not os.path.isfile(path):
            continue
        target = os.path.join(dst, (renamed or {}).get(name, name))
        if os.path.exists(target):
            os.remove(target)
        try:
            os.link(path, target)
        except OSError:
            shutil.copy2(path, target)


# --------------------------------------------------------------------- the dense clouds

_PLY_TYPES = {"char": "b", "int8": "b", "uchar": "B", "uint8": "B", "short": "h", "int16": "h",
              "ushort": "H", "uint16": "H", "int": "i", "int32": "i", "uint": "I", "uint32": "I",
              "float": "f", "float32": "f", "double": "d", "float64": "d"}


def read_ply_header(path):
    """(header bytes, vertex count, [(name, struct type)] of the fixed properties, [(name,
    count type, item type)] of the lists after them) of a binary little-endian PLY whose
    only element is vertex."""
    with open(path, "rb") as f:
        head = b""
        while not head.endswith(b"end_header\n"):
            line = f.readline()
            if not line:
                raise ValueError(f"{os.path.basename(path)}: no PLY header")
            head += line
    lines = head.decode("ascii", "replace").splitlines()
    if "format binary_little_endian 1.0" not in lines:
        raise ValueError(f"{os.path.basename(path)}: not a binary little-endian PLY")
    count, fixed, lists, elements = 0, [], [], 0
    for ln in lines:
        a = ln.split()
        if not a:
            continue
        if a[0] == "element":
            elements += 1
            if a[1] == "vertex":
                count = int(a[2])
        elif a[0] == "property" and a[1] == "list":
            lists.append((a[4], _PLY_TYPES[a[2]], _PLY_TYPES[a[3]]))
        elif a[0] == "property":
            if lists:
                raise ValueError(f"{os.path.basename(path)}: unexpected property after the lists")
            fixed.append((a[2], _PLY_TYPES[a[1]]))
    if elements != 1:
        raise ValueError(f"{os.path.basename(path)}: unexpected PLY elements")
    return head, count, fixed, lists


def _layout(path):
    head, count, fixed, lists = read_ply_header(path)
    names = [n for n, _ in fixed]
    if names[:3] != ["x", "y", "z"] or any(t != "f" for _, t in fixed[:3]):
        raise ValueError(f"{os.path.basename(path)}: unexpected point layout")
    if [n for n, _, _ in lists][:1] != ["view_indices"] or lists[0][1] != "B" or lists[0][2] not in "IiHh":
        raise ValueError(f"{os.path.basename(path)}: the points carry no photo lists")
    return head, count, fixed, lists


def _records(buf, start, fixed_size, lists):
    """Offsets of the complete records in ``buf`` from ``start``: (offsets, end of the last)."""
    sizes = [struct.calcsize(item) for _, _, item in lists]
    offs = []
    p = start
    n = len(buf)
    while True:
        q = p + fixed_size
        ok = True
        for size in sizes:
            if q >= n:
                ok = False
                break
            q += 1 + buf[q] * size
        if not ok or q > n:
            break
        offs.append(p)
        p = q
    return offs, p


def merge_dense(ply_a, ply_b, M, offset, out, progress=None):
    """Write ``out``: the dense cloud ``ply_a`` as it is, then ``ply_b`` moved by ``M`` (4 x 4)
    with its photo numbers raised by ``offset``. Both must have OpenMVS's layout (x, y, z,
    colour, normals, a photo list and a weight list). Returns the point count."""
    np = _np()
    head_a, na, fixed_a, lists_a = _layout(ply_a)
    head_b, nb, fixed_b, lists_b = _layout(ply_b)
    if fixed_a != fixed_b or lists_a != lists_b:
        raise ValueError("the two dense clouds are stored differently")
    Mf = np.asarray(M, float)
    A = Mf[:3, :3]
    s = float(np.cbrt(np.linalg.det(A)))
    R = A / s
    fixed_size = struct.calcsize("<" + "".join(t for _, t in fixed_b))
    names = [n for n, _ in fixed_b]
    has_normals = names[3:] and all(k in names for k in ("nx", "ny", "nz"))
    pos = {}
    off = 0
    for n_, t in fixed_b:
        pos[n_] = off
        off += struct.calcsize("<" + t)
    idx_size = struct.calcsize("<" + lists_b[0][2])
    idx_dtype = {"I": "<u4", "i": "<i4", "H": "<u2", "h": "<i2"}[lists_b[0][2]]
    head = head_a.replace(f"element vertex {na}\n".encode(), f"element vertex {na + nb}\n".encode(), 1)
    tmp = out + ".tmp"
    done = 0
    with open(tmp, "wb") as fo:
        fo.write(head)
        with open(ply_a, "rb") as fa:
            fa.seek(len(head_a))
            shutil.copyfileobj(fa, fo, BLOCK)
        with open(ply_b, "rb") as fb:
            fb.seek(len(head_b))
            rest = b""
            while True:
                chunk = fb.read(BLOCK)
                buf = rest + chunk
                if not buf:
                    break
                offs, end = _records(buf, 0, fixed_size, lists_b)
                if not offs:
                    if not chunk:
                        raise ValueError(f"{os.path.basename(ply_b)} ends inside a point")
                    rest = buf
                    continue
                arr = np.frombuffer(buf[:end], np.uint8).copy()
                o = np.asarray(offs, np.int64)
                # positions (and normals): float32 triples at fixed places in each record
                for keys, move in ((("x", "y", "z"), True), (("nx", "ny", "nz"), False)):
                    if not move and not has_normals:
                        continue
                    cols = [o + pos[k] for k in keys]
                    vals = np.stack([arr[np.add.outer(c, np.arange(4))].reshape(-1).view("<f4") for c in cols], 1)
                    vals = vals.astype(float)
                    vals = vals @ (A.T if move else R.T) + (Mf[:3, 3] if move else 0.0)
                    for k, c in enumerate(cols):
                        arr[np.add.outer(c, np.arange(4))] = vals[:, k].astype("<f4").view(np.uint8).reshape(-1, 4)
                # the photo numbers: the first list after the fixed part
                cnt = arr[o + fixed_size].astype(np.int64)
                starts = o + fixed_size + 1
                tot = int(cnt.sum())
                if tot:
                    first = np.repeat(starts, cnt) + idx_size * (np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt))
                    where = np.add.outer(first, np.arange(idx_size))
                    v = arr[where].reshape(-1).view(idx_dtype).astype(np.int64) + offset
                    arr[where] = v.astype(idx_dtype).view(np.uint8).reshape(-1, idx_size)
                fo.write(arr.tobytes())
                done += len(offs)
                if progress:
                    progress(done, nb)
                rest = buf[end:]
                if not chunk:
                    break
    if done != nb:
        os.remove(tmp)
        raise ValueError(f"{os.path.basename(ply_b)}: {done} of {nb} points read")
    os.replace(tmp, out)
    return na + nb


SHARE_SPACINGS = 2.0      # the two sources' points this many point spacings apart share photos


def share_photos(path, n_first, spacings=SHARE_SPACINGS, progress=None):
    """Where the two sources of a joined dense cloud overlap, give each point the photos (and
    weights) of the other source's nearest point too, when that lies within ``spacings``
    times the first source's median point spacing and faces the same way. ``n_first``: the
    first source's points, which come first. Rewrites ``path``; returns the points that got
    photos.

    Without it, a surface both sources saw is two interleaved sets of points, each seen only
    by its own source's photos, and on such clouds ReconstructMesh's graph cut (OpenMVS 2.4,
    IBFS with float capacities) can run without end: the synthetic scene's photos split in
    two (even, odd) and joined back stalled for over half an hour; with the photos shared
    (91% of the points) its graph cut took 1.2 s. DEVELOPMENT.md, section 4."""
    np = _np()
    head, total, fixed, lists = _layout(path)
    if total <= n_first or n_first < 2:
        return 0
    names = [n for n, _ in fixed]
    fixed_size = struct.calcsize("<" + "".join(t for _, t in fixed))
    pos, off = {}, 0
    for n_, t in fixed:
        pos[n_] = off
        off += struct.calcsize("<" + t)
    with open(path, "rb") as f:
        data = f.read()
    offs, end = _records(data, len(head), fixed_size, lists)
    if len(offs) != total:
        raise ValueError(f"{os.path.basename(path)}: {len(offs)} points read of {total}")
    o = np.asarray(offs, np.int64)
    arr = np.frombuffer(data, np.uint8)

    def floats(keys):
        return np.stack([arr[np.add.outer(o + pos[k], np.arange(4))].reshape(-1).view("<f4") for k in keys],
                        1).astype(float)

    X = floats(("x", "y", "z"))
    N = floats(("nx", "ny", "nz")) if all(k in names for k in ("nx", "ny", "nz")) else None
    from mathutils import kdtree

    def tree(P):
        t = kdtree.KDTree(len(P))
        for i, p in enumerate(P):
            t.insert(p, i)
        t.balance()
        return t

    first, second = X[:n_first], X[n_first:]
    trees = (tree(first), tree(second))
    rng = np.random.default_rng(0)
    sample = rng.choice(n_first, min(5000, n_first), replace=False)
    gaps = [trees[0].find_n(first[i], 2)[1][2] for i in sample]
    radius = spacings * float(np.median(gaps)) if gaps else 0.0
    if not radius > 0.0:
        return 0
    # only the points with a point of the other source in their grid cell or a neighbour
    lo = X.min(axis=0)
    ijk = np.floor((X - lo) / radius).astype(np.int64)
    n = ijk.max(axis=0) + 2
    key = (ijk[:, 0] * n[1] + ijk[:, 1]) * n[2] + ijk[:, 2]
    near = np.zeros(total, bool)
    for part, other in ((slice(0, n_first), slice(n_first, total)), (slice(n_first, total), slice(0, n_first))):
        keys = np.unique(key[other])
        for d in ((a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)):
            q = ijk[part] + d
            k = (q[:, 0] * n[1] + q[:, 1]) * n[2] + q[:, 2]
            at = np.clip(np.searchsorted(keys, k), 0, len(keys) - 1)
            near[part] |= keys[at] == k
    item_sizes = [struct.calcsize("<" + it) for _, _, it in lists]

    def photo_lists(i):
        q = offs[i] + fixed_size
        out = []
        for (_, ct, it), size in zip(lists, item_sizes):
            cnt = data[q]
            out.append(list(struct.unpack_from(f"<{cnt}{it}", data, q + 1)))
            q += 1 + cnt * size
        return out

    extra = {}
    for i in np.flatnonzero(near):
        mine = i < n_first
        _, j, d = trees[1 if mine else 0].find(X[i])
        if j is None or d > radius:
            continue
        j = j + n_first if mine else j
        if N is not None and float(N[i] @ N[j]) <= 0.0:
            continue
        extra[int(i)] = int(j)
    if not extra:
        return 0
    tmp = path + ".tmp"
    with open(tmp, "wb") as fo:
        fo.write(head)
        bounds = offs + [end]
        done = 0
        for i in range(total):
            j = extra.get(i)
            if j is None:
                fo.write(data[bounds[i]:bounds[i + 1]])
                continue
            mine, theirs = photo_lists(i), photo_lists(j)
            room = 255 - len(mine[0])
            fo.write(data[bounds[i]:bounds[i] + fixed_size])
            for (_, ct, it), a, b in zip(lists, mine, theirs):
                both = a + b[:room]
                fo.write(struct.pack("<" + ct, len(both)) + struct.pack(f"<{len(both)}{it}", *both))
            done += 1
            if progress and done % 100000 == 0:
                progress(done / len(extra))
        fo.write(data[end:])
    os.replace(tmp, path)
    return len(extra)


def merge_previews(prev_a, prev_b, M, out):
    """The crop stage's cloud previews of both scans in one (the second moved by M)."""
    np = _np()
    from . import roi
    xa, ca = roi.load_preview(prev_a)
    xb, cb = roi.load_preview(prev_b)
    Mf = np.asarray(M, float)
    xb = xb.astype(float) @ Mf[:3, :3].T + Mf[:3, 3]
    roi.save_preview(out, np.vstack([xa, xb]), np.vstack([ca, cb]))


def save_transform(path, M):
    """The 4 x 4 join (the second scan's coordinates into the first's) as JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"matrix": [[float(v) for v in row] for row in M]}, f, indent=2)
