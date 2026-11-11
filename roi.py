# SPDX-License-Identifier: GPL-3.0-or-later
# Photogram3D - photos or video to a textured mesh with COLMAP + OpenMVS
# Copyright (C) 2026 Kruglov Iurii
# Authors: Kruglov Iurii, Anthropic Claude AI
# License: GNU General Public License (GPL) v3 or later, https://www.blender.org/about/license/
"""Region of interest (crop box) and dense point clouds, without bpy.

OpenMVS keeps a region of interest as an oriented box. Its text form (``fs << obb``
in OpenMVS, read back by ``Scene::LoadROI``) is 15 numbers:

    r00 r01 r02 / r10 r11 r12 / r20 r21 r22   rotation, world to box (rows = box axes)
    px py pz                                  centre, world coordinates
    ex ey ez                                  half lengths along the box axes

A point p is inside when |R (p - c)| <= e on every axis (``TOBB::Intersects``).
"""

import math
import os
import struct

import numpy as np

_PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "<i2", "int16": "<i2", "ushort": "<u2", "uint16": "<u2",
    "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4",
    "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
}


# ---------------------------------------------------------------------------
# Oriented box
# ---------------------------------------------------------------------------

class OBB:
    """rot: 3x3 world-to-box rotation (rows are the box axes), center, half extents."""

    def __init__(self, rot, center, ext):
        self.rot = np.asarray(rot, dtype=np.float64).reshape(3, 3)
        self.center = np.asarray(center, dtype=np.float64).reshape(3)
        self.ext = np.abs(np.asarray(ext, dtype=np.float64).reshape(3))

    def to_text(self):
        rows = [" ".join(f"{v:.9g}" for v in r) for r in self.rot]
        return "\n".join(rows + [" ".join(f"{v:.9g}" for v in self.center),
                                 " ".join(f"{v:.9g}" for v in self.ext)]) + "\n"

    @classmethod
    def from_text(cls, text):
        vals = [float(v) for v in text.split()]
        if len(vals) == 6:  # OpenMVS also accepts an axis-aligned box: min xyz, max xyz
            lo, hi = np.array(vals[:3]), np.array(vals[3:])
            return cls(np.eye(3), (lo + hi) / 2, (hi - lo) / 2)
        if len(vals) < 15:
            raise ValueError("not an OpenMVS region-of-interest file")
        return cls(np.array(vals[:9]).reshape(3, 3), vals[9:12], vals[12:15])

    def save(self, path):
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="ascii") as f:
            f.write(self.to_text())
        os.replace(tmp, path)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="ascii") as f:
            return cls.from_text(f.read())

    def same_as(self, other, tol=1e-5):
        """Same box up to round-off (Blender keeps object matrices in 32-bit floats),
        measured against the box size."""
        scale = max(float(self.ext.max()), float(other.ext.max()), 1e-12)
        return bool(np.abs(self.matrix() - other.matrix()).max() <= tol * scale)

    def save_if_changed(self, path, tol=1e-5):
        """Write the box unless ``path`` already holds the same box; returns True if written.

        The file's text is part of the meshing step's signature, so a box that was
        not moved must keep its old text or the mesh would be rebuilt for nothing."""
        try:
            if self.same_as(OBB.load(path), tol):
                return False
        except (OSError, ValueError):
            pass
        self.save(path)
        return True

    def inside(self, pts):
        """Boolean mask of the points (N x 3) inside the box."""
        local = (np.asarray(pts, dtype=np.float64) - self.center) @ self.rot.T
        return np.all(np.abs(local) <= self.ext, axis=1)

    def matrix(self):
        """4x4 matrix that maps the cube [-1, 1]^3 onto the box (box to world)."""
        m = np.eye(4)
        m[:3, :3] = self.rot.T * self.ext[None, :]
        m[:3, 3] = self.center
        return m

    @classmethod
    def from_matrix(cls, m):
        """Inverse of matrix(): any affine map of the unit cube without shear."""
        m = np.asarray(m, dtype=np.float64)
        axes = m[:3, :3]
        ext = np.linalg.norm(axes, axis=0)
        if np.any(ext <= 0):
            raise ValueError("the crop box has zero size")
        rot = (axes / ext[None, :]).T
        # re-orthonormalise (user transforms can leave tiny shear)
        u, _, vt = np.linalg.svd(rot)
        rot = u @ vt
        return cls(rot, m[:3, 3], ext)

    def corners(self):
        """The eight corners (8 x 3)."""
        cube = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)
        return (cube * self.ext) @ self.rot + self.center

    def moved(self, m):
        """The box after the similarity ``m`` (4 x 4: scale, rotation, translation)."""
        m = np.asarray(m, dtype=np.float64)
        a = m[:3, :3]
        s = float(np.cbrt(np.linalg.det(a)))
        return OBB(self.rot @ (a / s).T, a @ self.center + m[:3, 3], s * self.ext)

    def enclosing(self, pts):
        """The smallest box with this box's axes that holds this box and the points (N x 3)."""
        local = (np.vstack([self.corners(), np.asarray(pts, dtype=np.float64).reshape(-1, 3)])
                 - self.center) @ self.rot.T
        lo, hi = local.min(axis=0), local.max(axis=0)
        return OBB(self.rot, self.center + ((lo + hi) / 2) @ self.rot, (hi - lo) / 2)

    def as_dict(self):
        return {"rot": self.rot.tolist(), "center": self.center.tolist(), "ext": self.ext.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(d["rot"], d["center"], d["ext"])


def fit_box(pts, up=(0.0, -1.0, 0.0), lo=2.0, hi=98.0):
    """Box around the bulk of the points, upright along ``up``.

    COLMAP's model_orientation_aligner puts gravity on +Y, so up is -Y; the two
    horizontal axes follow the principal directions of the points.
    """
    pts = np.asarray(pts, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    up = up / np.linalg.norm(up)
    helper = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    a = np.cross(up, helper)
    a /= np.linalg.norm(a)
    b = np.cross(up, a)
    flat = np.stack([pts @ a, pts @ b], axis=1)
    flat -= flat.mean(axis=0)
    w, v = np.linalg.eigh(np.cov(flat.T))
    major = v[:, np.argmax(w)]
    ax0 = major[0] * a + major[1] * b
    ax0 /= np.linalg.norm(ax0)
    ax2 = up
    ax1 = np.cross(ax2, ax0)
    rot = np.stack([ax0, ax1, ax2])          # rows: box axes
    local = pts @ rot.T
    lo_v = np.percentile(local, lo, axis=0)
    hi_v = np.percentile(local, hi, axis=0)
    center_local = (lo_v + hi_v) / 2
    ext = np.maximum((hi_v - lo_v) / 2, 1e-6)
    return OBB(rot, center_local @ rot, ext)


# ---------------------------------------------------------------------------
# Dense point clouds (OpenMVS scene_dense.ply, COLMAP fused.ply)
# ---------------------------------------------------------------------------

def _ply_header(f):
    lines = []
    while True:
        raw = f.readline()
        if not raw:
            raise ValueError("PLY header not terminated")
        line = raw.decode("ascii", "replace").strip()
        lines.append(line)
        if line == "end_header":
            break
    if not lines or lines[0] != "ply":
        raise ValueError("not a PLY file")
    fmt = next((ln.split()[1] for ln in lines if ln.startswith("format")), "")
    if fmt != "binary_little_endian":
        raise ValueError(f"unsupported PLY format {fmt!r}")
    elements = []
    for ln in lines:
        parts = ln.split()
        if parts[0] == "element":
            elements.append({"name": parts[1], "count": int(parts[2]), "props": []})
        elif parts[0] == "property" and elements:
            if parts[1] == "list":
                elements[-1]["props"].append((parts[4], "list", (parts[2], parts[3])))
            else:
                elements[-1]["props"].append((parts[2], "scalar", parts[1]))
    return elements


def ply_vertex_count(path):
    with open(path, "rb") as f:
        elements = _ply_header(f)
    return next((e["count"] for e in elements if e["name"] == "vertex"), 0)


def read_points(path, max_points=0):
    """(xyz float32 N x 3, rgb uint8 N x 3 or None) of a binary PLY point cloud.

    Handles OpenMVS clouds, whose vertices end in variable-length view lists, and
    every-k-th subsampling to at most ``max_points``.
    """
    with open(path, "rb") as f:
        elements = _ply_header(f)
        data = f.read()
    if not elements or elements[0]["name"] != "vertex":
        raise ValueError("the PLY file has no leading vertex element")
    vert = elements[0]
    n = vert["count"]
    fixed, lists, size = {}, [], 0
    for name, kind, typ in vert["props"]:
        if kind == "scalar":
            if lists:
                raise ValueError("scalar PLY properties after a list are not supported")
            dt = np.dtype(_PLY_TYPES[typ])
            fixed[name] = (size, dt)
            size += dt.itemsize
        else:
            lists.append((np.dtype(_PLY_TYPES[typ[0]]), np.dtype(_PLY_TYPES[typ[1]])))
    step = max(1, math.ceil(n / max_points)) if max_points and n > max_points else 1

    if not lists:
        offsets = np.arange(0, n, step, dtype=np.int64) * size
    else:
        offs = []
        off = 0
        simple = all(c.itemsize == 1 for c, _ in lists)
        if simple:
            items = [it.itemsize for _, it in lists]
            for i in range(n):
                if i % step == 0:
                    offs.append(off)
                off += size
                for isz in items:
                    off += 1 + data[off] * isz
        else:
            for i in range(n):
                if i % step == 0:
                    offs.append(off)
                off += size
                for cnt, it in lists:
                    c = struct.unpack_from(cnt.str, data, off)[0]
                    off += cnt.itemsize + c * it.itemsize
        offsets = np.asarray(offs, dtype=np.int64)

    raw = np.frombuffer(data, dtype=np.uint8)

    def gather(names):
        out = []
        for name in names:
            start, dt = fixed[name]
            cols = np.empty(len(offsets), dtype=dt)
            for a in range(0, len(offsets), 1 << 20):
                o = offsets[a:a + (1 << 20)]
                idx = o[:, None] + (start + np.arange(dt.itemsize))[None, :]
                cols[a:a + len(o)] = raw[idx].view(dt).reshape(-1)
            out.append(cols)
        return np.stack(out, axis=1)

    xyz = gather(("x", "y", "z")).astype(np.float32)
    rgb = None
    if all(k in fixed for k in ("red", "green", "blue")):
        rgb = gather(("red", "green", "blue"))
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return xyz, rgb


def save_preview(path, xyz, rgb):
    """Compact cloud for the Blender preview (loaded with numpy on the main thread)."""
    tmp = path + ".tmp.npz"
    np.savez(tmp, xyz=xyz.astype(np.float32),
             rgb=(rgb if rgb is not None else np.full((len(xyz), 3), 200, np.uint8)))
    os.replace(tmp, path)


def load_preview(path):
    with np.load(path) as d:
        return d["xyz"], d["rgb"]
