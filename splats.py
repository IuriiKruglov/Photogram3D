# SPDX-License-Identifier: GPL-3.0-or-later
# Photogram3D - photos or video to a textured mesh with COLMAP + OpenMVS
# Copyright (C) 2026 Kruglov Iurii
# Authors: Kruglov Iurii, Anthropic Claude AI
# License: GNU General Public License (GPL) v3 or later, https://www.blender.org/about/license/
"""Gaussian splats (0.3.5) with Brush (Apache-2.0, github.com/ArthurBrussee/brush).

Brush trains 3D Gaussian splats from a COLMAP scene on any GPU through WebGPU (DirectX 12
or Vulkan on Windows), so it runs on NVIDIA, AMD and Intel cards alike. It reads pinhole
cameras only, which is what COLMAP's undistorted workspace (``dense/``) holds; the
photos there are the ones the depth maps use.

Brush 0.3.0 facts this module relies on (read in its source, crates/brush-dataset and
crates/brush-process, and checked by running it):

* it finds ``cameras.bin`` anywhere in the folder it is given, ``images.bin`` beside it,
  each photo by its name anywhere in the folder, and ``points3D.bin`` for the starting
  splats; a single ``.ply`` file anywhere in the folder replaces those starting splats,
  so the training folder holds nothing else;
* ``--export-every`` / ``--export-path`` / ``--export-name`` write the splats as a
  standard 3DGS PLY (x y z, scale_0..2 as logs, opacity as a logit, rot_0..3 with w
  first, f_dc_0..2 and f_rest_* spherical harmonics, binary little endian, properties
  in alphabetical order) and always at the last step;
* without a terminal its progress bar is hidden; with RUST_LOG it logs
  "Refine iter N, M splats." every 200 steps until the end.

Nothing here imports bpy.
"""

import hashlib
import json
import math
import os
import re
import shutil
import zipfile

import numpy as np

from .common import IS_WINDOWS, current, exe_name
from .runner import StepError

BRUSH_VERSION = "0.3.0"
BRUSH_URL = ("https://github.com/ArthurBrussee/brush/releases/download/v0.3.0/"
             "brush-app-x86_64-pc-windows-msvc.zip")
BRUSH_SHA256 = "b68e3e9cf052d51bf3ee30776fa5a364de7f2ba13b58443128ff797bb7bcfcd6"
BRUSH_ZIP_SIZE = 158_791_348
BRUSH_EXE = "brush_app"
MARKER = "photogram3d_brush.json"

SPLATS_PLY = "splats.ply"              # cropped to the box (what is imported)
SPLATS_FULL_PLY = "splats_full.ply"    # as trained
RUST_LOG = "warn,brush_cli=info,brush_process=info"
_RE_REFINE = re.compile(r"Refine iter (\d+), (\d+) splats")     # every --refine-every steps
_RE_EVAL = re.compile(r"Eval iter (\d+): PSNR ([0-9.]+), ssim ([0-9.]+)")

SH_C0 = 0.28209479177387814           # degree-0 spherical harmonic


# ---------------------------------------------------------------------------
# The tool
# ---------------------------------------------------------------------------

def paths(tools_root):
    folder = os.path.join(tools_root, "brush")
    return {"dir": folder, "exe": os.path.join(folder, exe_name(BRUSH_EXE)),
            "marker": current(os.path.join(folder, MARKER))}


def find_brush(tools_root, override=""):
    """Path of brush_app: the preference override, the installed copy, or PATH."""
    if override and os.path.isfile(override):
        return override
    p = paths(tools_root) if tools_root else None
    if p and os.path.isfile(p["exe"]) and os.path.isfile(p["marker"]):
        return p["exe"]
    return shutil.which(BRUSH_EXE) or ""


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def install_brush(job, tools_root, log, download):
    """Download Brush's Windows build (159 MB), check its SHA-256 and unpack brush_app.exe.

    ``download(url, dest, progress)`` downloads a file (tools.download with the job's
    cancel event, passed in so that tests can stand in for it)."""
    if not IS_WINDOWS:
        raise StepError("Brush is installed automatically on Windows only; on this system put "
                        "brush_app on the PATH or set its path in the add-on preferences")
    p = paths(tools_root)
    os.makedirs(p["dir"], exist_ok=True)
    zip_path = os.path.join(tools_root, "_downloads", os.path.basename(BRUSH_URL))

    def progress(done, total):
        total = total or BRUSH_ZIP_SIZE
        job.set_fraction(0.02 + 0.08 * done / total,
                         f"downloading Brush ({done / 1e6:.0f} of {total / 1e6:.0f} MB)")

    download(BRUSH_URL, zip_path, progress)
    digest = _sha256(zip_path)
    if digest != BRUSH_SHA256:
        os.remove(zip_path)
        raise StepError(f"The Brush download is not the published file (SHA-256 {digest[:12]}..., "
                        f"expected {BRUSH_SHA256[:12]}...); try again later")
    with zipfile.ZipFile(zip_path) as zf:
        for name in ("brush_app.exe", "LICENSE"):
            if name in zf.namelist():
                with zf.open(name) as src, open(os.path.join(p["dir"], name + ".part"), "wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
                os.replace(os.path.join(p["dir"], name + ".part"), os.path.join(p["dir"], name))
    if not os.path.isfile(p["exe"]):
        raise StepError("brush_app.exe was not in the Brush download")
    with open(p["marker"], "w", encoding="utf-8") as f:
        json.dump({"version": BRUSH_VERSION, "sha256": BRUSH_SHA256}, f)
    try:
        os.remove(zip_path)
    except OSError:
        pass
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"[Photogram3D] Brush {BRUSH_VERSION} installed in {p['dir']}\n")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def prepare_dataset(dense_dir, data_dir, images=None):
    """A folder with only what Brush should read: images/ (links to the undistorted
    photos, or to their evened-out copies in ``images``) and sparse/0 (the undistorted,
    pinhole COLMAP model). Returns the photo count."""
    shutil.rmtree(data_dir, ignore_errors=True)
    os.makedirs(os.path.join(data_dir, "images"))
    os.makedirs(os.path.join(data_dir, "sparse", "0"))
    src_sparse = os.path.join(dense_dir, "sparse")
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        src = os.path.join(src_sparse, name)
        if not os.path.isfile(src):
            raise StepError(f"The undistorted scene has no {name} ({src_sparse})")
        shutil.copy2(src, os.path.join(data_dir, "sparse", "0", name))
    count = 0
    for entry in os.scandir(images or os.path.join(dense_dir, "images")):
        # the evened-out photos' folder also holds what the evening read (observations.npz)
        if entry.is_file() and entry.name.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")) \
                and not entry.name.endswith(".tmp.jpg"):
            link_or_copy(entry.path, os.path.join(data_dir, "images", entry.name))
            count += 1
    return count


def max_splats_for(vram_mb):
    """Splat budget for the GPU memory: training keeps the parameters, their gradients and
    two optimizer moments (about 1 KB per splat with degree-3 colours) plus the images and
    render buffers. Brush's own default is 10 million."""
    if not vram_mb:
        return 3_000_000
    return int(max(500_000, min(10_000_000, (vram_mb - 1500) * 1024 * 1024 / 1600)) // 100_000 * 100_000)


def train_args(exe, data_dir, out_dir, steps, max_resolution, max_splats, sh_degree=3):
    return [exe, data_dir, "--total-steps", str(int(steps)),
            "--max-resolution", str(int(max_resolution)), "--max-splats", str(int(max_splats)),
            "--sh-degree", str(int(sh_degree)),
            "--export-path", out_dir, "--export-name", SPLATS_FULL_PLY,
            "--export-every", str(int(steps))]


def make_parser(steps, lo=0.12, hi=0.95):
    """Progress from Brush's log: 'Refine iter N, M splats.' every 200 steps."""
    state = {"splats": 0}

    def parse(text):
        m = _RE_REFINE.search(text)
        if m:
            it, count = int(m.group(1)), int(m.group(2))
            state["splats"] = count
            return lo + (hi - lo) * min(1.0, it / max(1, steps)), \
                f"step {it:,} of {steps:,}, {count:,} splats"
        return None
    parse.state = state
    return parse


# ---------------------------------------------------------------------------
# PLY files
# ---------------------------------------------------------------------------

_PLY_TYPES = {"float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
              "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1", "short": "<i2",
              "ushort": "<u2", "int": "<i4", "uint": "<u4", "int32": "<i4", "uint32": "<u4"}


def read_ply(path):
    """(comments, structured array) of a binary little-endian PLY whose first element is
    'vertex' with scalar properties only (a 3DGS splat file)."""
    with open(path, "rb") as f:
        header = []
        while True:
            raw = f.readline()
            if not raw:
                raise ValueError(f"{path}: PLY header not terminated")
            line = raw.decode("ascii", "replace").strip()
            header.append(line)
            if line == "end_header":
                break
        if header[0] != "ply" or not any(h == "format binary_little_endian 1.0" for h in header):
            raise ValueError(f"{path}: not a binary little-endian PLY")
        count, props, comments, seen_vertex = 0, [], [], False
        for line in header:
            parts = line.split()
            if parts[0] == "comment":
                comments.append(line[8:])
            elif parts[0] == "element":
                if seen_vertex:
                    break                       # only the vertex element is read
                if parts[1] != "vertex":
                    raise ValueError(f"{path}: the first element is not 'vertex'")
                seen_vertex, count = True, int(parts[2])
            elif parts[0] == "property" and seen_vertex:
                if parts[1] == "list":
                    raise ValueError(f"{path}: list properties are not splat data")
                props.append((parts[2], _PLY_TYPES[parts[1]]))
        data = np.fromfile(f, dtype=np.dtype(props), count=count)
    if len(data) != count:
        raise ValueError(f"{path}: {len(data)} of {count} splats in the file")
    return comments, data


def write_ply(path, comments, data):
    names = {"<f4": "float", "<f8": "double", "|u1": "uchar", "|i1": "char", "<i2": "short",
             "<u2": "ushort", "<i4": "int", "<u4": "uint"}
    lines = ["ply", "format binary_little_endian 1.0"]
    lines += [f"comment {c}" for c in comments]
    lines.append(f"element vertex {len(data)}")
    for name in data.dtype.names:
        lines.append(f"property {names[data.dtype[name].str]} {name}")
    lines.append("end_header")
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(("\n".join(lines) + "\n").encode("ascii"))
        data.tofile(f)
    os.replace(tmp, path)


def splat_fields(data):
    """Positions, linear-light colours, opacities (0-1), scales and unit quaternions (w x y z)."""
    xyz = np.stack([data["x"], data["y"], data["z"]], 1).astype(np.float32)
    srgb = np.clip(0.5 + SH_C0 * np.stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]], 1), 0, 1)
    linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4).astype(np.float32)
    opacity = (1.0 / (1.0 + np.exp(-data["opacity"].astype(np.float64)))).astype(np.float32)
    scale = np.exp(np.stack([data["scale_0"], data["scale_1"], data["scale_2"]], 1)).astype(np.float32)
    quat = np.stack([data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]], 1).astype(np.float32)
    quat /= np.maximum(np.linalg.norm(quat, axis=1, keepdims=True), 1e-12)
    return xyz, linear, opacity, scale, quat


def crop(data, obb, min_opacity=1.0 / 255.0):
    """Keep the splats whose centre is inside the box (roi.OBB) and that are not invisible."""
    keep = np.ones(len(data), bool)
    if obb is not None:
        xyz = np.stack([data["x"], data["y"], data["z"]], 1).astype(np.float64)
        keep &= obb.inside(xyz)
    if min_opacity > 0:
        keep &= data["opacity"] > math.log(min_opacity / (1.0 - min_opacity))
    return data[keep]
