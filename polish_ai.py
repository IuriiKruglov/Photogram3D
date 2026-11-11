# SPDX-License-Identifier: GPL-3.0-or-later
# Photogram3D - photos or video to a textured mesh with COLMAP + OpenMVS
# Copyright (C) 2026 Kruglov Iurii
# Authors: Kruglov Iurii, Anthropic Claude AI
# License: GNU General Public License (GPL) v3 or later, https://www.blender.org/about/license/
"""AI polish: fuse learned per-photo normals into the reconstructed mesh.

1. A monocular normal network (MoGe-2, ONNX) predicts a normal map for each
   undistorted photo, in a separate process (ai_normals.py).
2. Mesh vertices are projected into every photo with the known COLMAP
   cameras; a splatted z-buffer plus a back-face test keep only the surface
   the photo actually sees.
3. Predicted normals are rotated to world space and averaged per vertex,
   weighted by how frontally the photo sees the surface. Vertices whose
   views disagree, or that only one photo sees, are left alone.
4. The axis convention of the network is checked against the mesh itself
   and the whole result is rejected if they disagree.
5. Vertices move so faces follow the fused normals (polish.update_vertices),
   with boundaries pinned and implausible targets ignored.

Multi-view stereo provides the shape; the network provides fine
orientation, which helps on glossy or weakly textured surfaces where
stereo is noisy. Nothing here imports bpy.
"""

import itertools
import json
import os
import shutil
import sys

import numpy as np

from . import colmap_io, tools
from .common import IS_WINDOWS, current, describe_returncode, human_bytes
from .polish import boundary_vertices, face_data, update_vertices
from .runner import Job, JobCancelled, Step, StepError

MODELS = {
    "SMALL": {"label": "Small (ViT-S, 141 MB)", "repo": "Ruicheng/moge-2-vits-normal-onnx",
              "file": "moge-2-vits-normal.onnx", "size": 141_000_000},
    "BASE": {"label": "Base (ViT-B, 419 MB)", "repo": "Ruicheng/moge-2-vitb-normal-onnx",
             "file": "moge-2-vitb-normal.onnx", "size": 419_000_000},
}
DETAIL_TOKENS = {"STANDARD": 1600, "HIGH": 2500}

# Wheels installed into the tools folder on first use (no dependencies:
# importing onnxruntime only needs numpy, which Blender ships).
# DirectML runs on any DirectX 12 GPU, so no CUDA/cuDNN download is needed.
# Each candidate is installed and then loaded once in Blender's Python; one that does not
# load gives way to the next (on one laptop Blender's Python crashed with an access violation
# while loading 1.24.4, 0.3.9): older DirectML builds, then the CPU-only runtime. All of
# them run the opset 17 models used here (ALIKED, LightGlue, MoGe-2).
RUNTIME_CANDIDATES_WINDOWS = (
    ("onnxruntime-directml==1.24.4", "pillow==12.3.0"),
    ("onnxruntime-directml==1.22.0", "pillow==12.3.0"),
    ("onnxruntime-directml==1.20.1", "pillow==12.3.0"),
    ("onnxruntime==1.24.4", "pillow==12.3.0"),          # CPU only
    ("onnxruntime==1.20.1", "pillow==12.3.0"),          # CPU only, last resort
)
RUNTIME_CANDIDATES_OTHER = (("onnxruntime", "pillow"),)
RUNTIME_MARKER = "photogram3d_runtime.json"
AI_KEY = "Photogram3D AI"


def model_url(key):
    return f"https://huggingface.co/{MODELS[key]['repo']}/resolve/main/model.onnx"


def ai_paths(tools_root, python_version=None):
    ver = python_version or sys.version_info[:2]
    root = os.path.join(tools_root, "ai")
    return {"root": root,
            "site": os.path.join(root, f"site-py{ver[0]}{ver[1]}"),
            "models": os.path.join(root, "models")}


def runtime_info(site):
    path = current(os.path.join(site, RUNTIME_MARKER))
    if not (os.path.isfile(path) and os.path.isfile(os.path.join(site, "onnxruntime", "__init__.py"))):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def model_file(models_dir, key):
    path = os.path.join(models_dir, MODELS[key]["file"])
    return path if os.path.isfile(path) and os.path.getsize(path) > 1_000_000 else None


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------

def vertex_normals(verts, faces):
    fn, _, areas = face_data(verts, faces)
    acc = np.zeros_like(verts)
    for k in range(3):
        for c in range(3):
            acc[:, c] += np.bincount(faces[:, k], weights=fn[:, c] * areas, minlength=len(verts))
    return acc / np.maximum(np.linalg.norm(acc, axis=1), 1e-30)[:, None]


def project(verts, cam, image, out_w, out_h):
    """Pixel coords in an (out_w x out_h) map, camera-space points and R."""
    r = np.array(colmap_io.qvec_to_rotmat(image["qvec"]))
    t = np.array(image["tvec"])
    xc = verts @ r.T + t
    fx, fy, cx, cy = colmap_io.pinhole_intrinsics(cam)
    sx = out_w / float(cam["width"])
    sy = out_h / float(cam["height"])
    z = xc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = (fx * xc[:, 0] / z + cx) * sx
        v = (fy * xc[:, 1] / z + cy) * sy
    return u, v, z, r, xc


def visible_mask(u, v, z, w, h, tol=0.01):
    """Front-most vertices per pixel via a splatted, 1-pixel dilated z-buffer."""
    ok = np.isfinite(u) & np.isfinite(v) & (z > 1e-6) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    idx = np.flatnonzero(ok)
    ui = np.floor(u[idx]).astype(np.int64)
    vi = np.floor(v[idx]).astype(np.int64)
    pix = vi * w + ui
    zbuf = np.full(w * h, np.inf)
    np.minimum.at(zbuf, pix, z[idx])
    zimg = zbuf.reshape(h, w)
    pad = np.pad(zimg, 1, constant_values=np.inf)
    zmin = zimg.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            zmin = np.minimum(zmin, pad[1 + dy:1 + dy + h, 1 + dx:1 + dx + w])
    vis = np.zeros(len(z), dtype=bool)
    front = z[idx] <= zmin.ravel()[pix] * (1.0 + tol) + 1e-9
    vis[idx[front]] = True
    return vis


def calibrate_convention(samples):
    """Axis signs (sx, sy, sz) that best match mesh normals.

    ``samples`` is a list of (pred_cam (N,3), mesh_cam (N,3)). Returns
    (signs, agreement) where agreement is the median dot product.
    """
    pred = np.concatenate([s[0] for s in samples])
    mesh = np.concatenate([s[1] for s in samples])
    best = (np.ones(3), -2.0)
    for signs in itertools.product((1.0, -1.0), repeat=3):
        sg = np.array(signs)
        med = float(np.median(np.einsum("ij,ij->i", pred * sg, mesh)))
        if med > best[1]:
            best = (sg, med)
    return best


def load_normal_map(path):
    """(H, W, 3) unit normals and (H, W) validity from an ai_normals.py output."""
    with np.load(path) as d:
        n = d["normal"].astype(np.float32)
        m = d["mask"].astype(bool) if "mask" in d.files else np.ones(n.shape[:2], dtype=bool)
    length = np.linalg.norm(n, axis=-1, keepdims=True)
    m &= length[..., 0] > 63.5  # int8 storage: a unit vector has length ~127
    return n / np.maximum(length, 1e-6), m


def _view_samples(verts, mesh_n, cam, im, w, h, orient):
    """Visible, front-facing vertex ids of one view with pixel coords and weights."""
    u, v, z, r, xc = project(verts, cam, im, w, h)
    ids = np.flatnonzero(visible_mask(u, v, z, w, h))
    if len(ids) == 0:
        return None
    mesh_cam = (mesh_n[ids] @ r.T) * orient
    ray = xc[ids] / np.maximum(np.linalg.norm(xc[ids], axis=1), 1e-12)[:, None]
    facing = -np.einsum("ij,ij->i", mesh_cam, ray)
    keep = facing > 0.1
    ids, mesh_cam, facing = ids[keep], mesh_cam[keep], facing[keep]
    ui = np.clip(np.floor(u[ids]).astype(np.int64), 0, w - 1)
    vi = np.clip(np.floor(v[ids]).astype(np.int64), 0, h - 1)
    return ids, ui, vi, mesh_cam, facing, r


def mesh_orientation(verts, mesh_n, views, size=(512, 384)):
    """+1 if face winding gives outward normals (facing the cameras), else -1."""
    votes = []
    for cam, im, _ in views[:6]:
        w, h = size
        u, v, z, r, xc = project(verts, cam, im, w, h)
        ids = np.flatnonzero(visible_mask(u, v, z, w, h))
        if len(ids) < 50:
            continue
        ray = xc[ids] / np.maximum(np.linalg.norm(xc[ids], axis=1), 1e-12)[:, None]
        votes.append(float(np.median(-np.einsum("ij,ij->i", mesh_n[ids] @ r.T, ray))))
    return -1.0 if votes and np.median(votes) < 0 else 1.0


def fuse_normals(verts, faces, views, load=load_normal_map, progress=None, min_agreement=0.5,
                 min_views=2, min_consistency=0.75, check_cancel=None, log=print):
    """Fused per-vertex normals from per-photo normal maps.

    ``views``: list of (cam_dict, image_dict, normal_map_path); normal maps are
    in the network's camera convention (OpenCV for MoGe; verified here).
    Returns (targets (N,3) in the mesh's winding orientation, valid (N,) bool, stats).
    """
    mesh_n = vertex_normals(verts, faces)
    orient = mesh_orientation(verts, mesh_n, views)
    acc = np.zeros_like(verts)
    wsum = np.zeros(len(verts))
    count = np.zeros(len(verts), dtype=np.int32)
    signs = None
    pending, calib = [], []
    used = 0
    for i, (cam, im, path) in enumerate(views):
        if check_cancel:
            check_cancel()
        nmap, valid = load(path)
        h, w = nmap.shape[:2]
        smp = _view_samples(verts, mesh_n, cam, im, w, h, orient)
        if smp is not None:
            ids, ui, vi, mesh_cam, facing, r = smp
            ok = valid[vi, ui]
            pred = nmap[vi[ok], ui[ok]].astype(np.float64)
            norm = np.linalg.norm(pred, axis=1)
            good = norm > 0.5
            sel = (ids[ok][good], pred[good] / norm[good][:, None], facing[ok][good], r)
            if len(sel[0]):
                used += 1
                if signs is None:
                    calib.append((sel[1], mesh_cam[ok][good]))
                    pending.append(sel)
                    if len(calib) >= min(4, len(views)):
                        signs, agree = _check_convention(calib, min_agreement, log)
                        for p in pending:
                            _accumulate(acc, wsum, count, p, signs)
                        pending = []
                else:
                    _accumulate(acc, wsum, count, sel, signs)
        if progress:
            progress((i + 1) / len(views))
    if signs is None:
        if not calib:
            raise RuntimeError("the mesh is not visible in any photo; is it a Photogram3D scan?")
        signs, agree = _check_convention(calib, min_agreement, log)
        for p in pending:
            _accumulate(acc, wsum, count, p, signs)
    norm = np.linalg.norm(acc, axis=1)
    consistency = norm / np.maximum(wsum, 1e-12)
    valid = (count >= min_views) & (consistency >= min_consistency) & (norm > 1e-9)
    targets = np.zeros_like(verts)
    targets[valid] = acc[valid] / norm[valid][:, None] * orient
    stats = {"views_used": used, "signs": signs.tolist(), "orientation": orient,
             "covered": float((count > 0).mean()), "valid": float(valid.mean()),
             "median_views": float(np.median(count[count > 0])) if (count > 0).any() else 0.0}
    return targets, valid, stats


def _check_convention(calib, min_agreement, log):
    signs, agree = calibrate_convention(calib)
    log(f"AI normal convention {signs.astype(int).tolist()} (OpenCV = [1, 1, 1]), "
        f"median agreement with the scan {agree:.2f}")
    if agree < min_agreement:
        raise RuntimeError(f"the AI normals disagree with the scan (median cos {agree:.2f} < "
                           f"{min_agreement}); nothing was changed")
    return signs, agree


def _accumulate(acc, wsum, count, sel, signs):
    ids, pred_cam, facing, r = sel
    n_world = (pred_cam * signs) @ r  # R^T n for row vectors
    wgt = facing ** 2
    acc[ids] += n_world * wgt[:, None]
    wsum[ids] += wgt
    count[ids] += 1


def polish_with_normals(verts, faces, targets, valid, influence=0.6, iterations=8,
                        max_angle_deg=50.0, progress=None):
    """Move vertices so faces follow the fused normals.

    Face targets are the mean of the three vertex targets, blended with the
    current face normal by ``influence`` (0 = no change). Faces with a vertex
    lacking a reliable target, or whose target is more than ``max_angle_deg``
    away from the current normal, keep their current normal.
    """
    verts = np.asarray(verts, dtype=np.float64)
    pinned = boundary_vertices(faces, len(verts))
    face_ok = valid[faces].all(axis=1)
    ft = targets[faces].sum(axis=1)
    ft /= np.maximum(np.linalg.norm(ft, axis=1), 1e-30)[:, None]
    cos_max = np.cos(np.radians(max_angle_deg))
    v = verts.copy()
    for it in range(iterations):
        fn, _, _ = face_data(v, faces)
        use = face_ok & (np.einsum("ij,ij->i", ft, fn) > cos_max)
        mix = fn.copy()
        mix[use] = influence * ft[use] + (1.0 - influence) * fn[use]
        mix /= np.maximum(np.linalg.norm(mix, axis=1), 1e-30)[:, None]
        v = update_vertices(v, faces, mix, iterations=2, strength=0.5, pinned=pinned)
        if progress:
            progress((it + 1) / iterations)
    return v


# ---------------------------------------------------------------------------
# job steps (worker thread)
# ---------------------------------------------------------------------------

# loaded the way the runners load it (-I, the site folder appended); the steps are printed so
# that a crash while loading (no Python error then) shows in the log where it happened, and
# faulthandler adds the Python stack of a crash
VERIFY_CODE = ("import sys; sys.path.append(sys.argv[1]); print('loading numpy', flush=True); "
               "import numpy; print('loading onnxruntime', flush=True); import onnxruntime as o; "
               "print('loading Pillow', flush=True); import PIL; "
               "print('ORT', o.__version__, 'PIL', PIL.__version__, 'EP', ','.join(o.get_available_providers()), "
               "flush=True)")


def _verify_runtime(job, py, folder, log):
    """Load ONNX Runtime from ``folder`` in Blender's Python: (report line or None, why not)."""
    found = {}

    def grab(text):
        if text.startswith("ORT "):
            found["report"] = text.strip()
        elif text.startswith("loading "):
            found["stage"] = text.strip()
        return None

    code = job.run([py, "-X", "faulthandler", "-I", "-c", VERIFY_CODE, folder], log, parser=grab,
                   ok_codes=None)
    if code == 0 and found.get("report"):
        return found["report"], ""
    where = found.get("stage", "starting")
    return None, f"{where} failed ({describe_returncode(code)})"


def step_runtime(job, ctx):
    py, site, log = ctx["python"], ctx["site"], ctx["log"]
    job.set_fraction(0.05, "checking pip")
    if job.run([py, "-E", "-m", "pip", "--version"], log, ok_codes=None) != 0:
        job.set_fraction(0.1, "bootstrapping pip")
        job.run([py, "-E", "-m", "ensurepip", "--user"], log, ok_codes=None)
        if job.run([py, "-E", "-m", "pip", "--version"], log, ok_codes=None) != 0:
            raise StepError("Blender's Python has no pip and ensurepip failed; see logs/ai_polish.log")
    staging = site + ".staging"
    candidates = RUNTIME_CANDIDATES_WINDOWS if IS_WINDOWS else RUNTIME_CANDIDATES_OTHER
    installed, report, failed, downloaded = None, "", [], False
    for k, pkgs in enumerate(candidates):
        shutil.rmtree(staging, ignore_errors=True)
        job.set_fraction(0.15 + 0.8 * k / len(candidates), "downloading " + ", ".join(pkgs))
        code = job.run([py, "-E", "-m", "pip", "install", "--no-deps", "--no-input",
                        "--disable-pip-version-check", "--no-warn-script-location",
                        "--progress-bar", "off", "--target", staging, *pkgs], log, ok_codes=None)
        if code != 0:
            continue
        downloaded = True
        job.set_fraction(0.15 + 0.8 * (k + 0.8) / len(candidates), "loading " + pkgs[0])
        report, why = _verify_runtime(job, py, staging, log)
        if report:
            installed = pkgs
            break
        failed.append(f"{pkgs[0]}: {why}")
        job.add_line(f"{pkgs[0]} does not load in Blender's Python here: {why}; trying the next one")
    if installed is None:
        shutil.rmtree(staging, ignore_errors=True)
        if not downloaded:
            raise StepError("could not install onnxruntime with pip; check the internet connection "
                            "(logs/ai_polish.log)")
        raise StepError("ONNX Runtime does not load in Blender's Python on this computer ("
                        + "; ".join(failed[:2]) + (" ..." if len(failed) > 2 else "") + ")")
    if os.path.isdir(site):
        shutil.rmtree(site, ignore_errors=True)
    os.replace(staging, site)
    with open(os.path.join(site, RUNTIME_MARKER), "w", encoding="utf-8") as f:
        json.dump({"packages": list(installed), "report": report,
                   "python": list(sys.version_info[:3])}, f)
    job.add_line(f"AI runtime installed: {report}")


def step_model(job, ctx):
    key = ctx["model_key"]
    dest = os.path.join(ctx["models_dir"], MODELS[key]["file"])

    def prog(done, total):
        total = total or MODELS[key]["size"]
        job.set_fraction(done / total, f"{human_bytes(done)} / {human_bytes(total)}")

    try:
        tools.download(model_url(key), dest, progress=prog, cancel=job.cancel_event, magic=b"\x08")
    except tools.Cancelled as exc:
        raise JobCancelled() from exc
    except tools.InstallError as exc:
        raise StepError(f"AI model download failed: {exc}") from exc
    ctx["model_path"] = dest


def select_views(model_dir, images_dir, max_views):
    cams, images = colmap_io.read_model(model_dir)
    views = []
    for iid in sorted(images, key=lambda k: images[k]["name"]):
        im = images[iid]
        path = os.path.join(images_dir, im["name"])
        if os.path.isfile(path):
            views.append((cams[im["camera_id"]], im, path))
    if max_views and len(views) > max_views:
        pick = np.linspace(0, len(views) - 1, max_views).round().astype(int)
        views = [views[i] for i in sorted(set(pick.tolist()))]
    return views


def _predict_parser(text):
    if text.startswith("Normals "):
        part = text.split()[1]
        if "/" in part:
            a, b = part.split("/", 1)
            if a.isdigit() and b.isdigit() and int(b):
                return int(a) / int(b), f"photo {a}/{b}"
    return None


def step_predict(job, ctx):
    views = select_views(ctx["model_dir"], ctx["images_dir"], ctx["max_views"])
    if len(views) < 2:
        raise StepError("fewer than 2 undistorted photos found in the project's dense folder")
    cache = os.path.join(ctx["project"], "ai_normals", f"{ctx['model_key'].lower()}_{ctx['tokens']}")
    os.makedirs(cache, exist_ok=True)
    todo, out_views = [], []
    for cam, im, path in views:
        out = os.path.join(cache, im["name"].replace("/", "_").replace("\\", "_") + ".npz")
        out_views.append((cam, im, out))
        if not (os.path.isfile(out) and os.path.getmtime(out) >= os.path.getmtime(path)):
            todo.append({"image": path, "out": out})
    if todo:
        jobs_path = os.path.join(cache, "jobs.json")
        with open(jobs_path, "w", encoding="utf-8") as f:
            json.dump({"tokens": ctx["tokens"], "items": todo}, f)
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_normals.py")
        job.run([ctx["python"], "-X", "faulthandler", "-I", script, "--site", ctx["site"], "--model", ctx["model_path"],
                 "--jobs", jobs_path, "--device", ctx["device"]], ctx["log"],
                parser=_predict_parser, below_normal=True)
    else:
        job.add_line(f"Using {len(views)} cached normal maps")
    ctx["views"] = [v for v in out_views if os.path.isfile(v[2])]
    if len(ctx["views"]) < 2:
        raise StepError("the AI model produced fewer than 2 normal maps; see logs/ai_polish.log")


def step_fuse(job, ctx):
    verts, faces = ctx["verts"], ctx["faces"]

    def log(msg):
        job.add_line(msg)

    try:
        targets, valid, stats = fuse_normals(
            verts, faces, ctx["views"], progress=lambda f: job.set_fraction(0.6 * f, "fusing views"),
            check_cancel=job.check_cancel, log=log)
    except RuntimeError as exc:
        raise StepError(str(exc)) from exc
    job.add_line(f"Fused {stats['views_used']} views: {stats['valid'] * 100:.0f}% of vertices have a "
                 f"reliable AI normal (median {stats['median_views']:.0f} views each)")
    if stats["valid"] < 0.05:
        raise StepError("too few vertices have reliable AI normals; nothing was changed")
    ctx["targets"], ctx["valid_mask"] = targets, valid
    new = polish_with_normals(verts, faces, targets, valid, influence=ctx["influence"],
                              progress=lambda f: job.set_fraction(0.6 + 0.4 * f, "moving vertices"))
    moved = np.linalg.norm(new - verts, axis=1)[valid]  # guided vertices only
    stats["moved_median"] = float(np.median(moved)) if len(moved) else 0.0
    stats["moved_p99"] = float(np.percentile(moved, 99)) if len(moved) else 0.0
    ctx["new_verts"] = new
    with job.lock:
        job.result.update(stats)


def build_job(spec):
    """``spec``: object_name, project, model_dir, images_dir, python, tools_root,
    model_key, tokens, device ('auto'/'cpu'), max_views, influence, verts, faces."""
    paths = ai_paths(spec["tools_root"])
    ctx = dict(spec)
    ctx.update(site=paths["site"], models_dir=paths["models"],
               log=os.path.join(spec["project"], "logs", "ai_polish.log"),
               model_path=model_file(paths["models"], spec["model_key"]))
    steps = []
    if runtime_info(paths["site"]) is None:
        steps.append(Step("ai_runtime", "Install AI runtime (once)", step_runtime, 1.0))
    if ctx["model_path"] is None:
        steps.append(Step("ai_model", "Download AI model (once)", step_model, 1.5))
    steps.append(Step("ai_predict", "AI normal maps", step_predict, 3.0))
    steps.append(Step("ai_fuse", "Fuse normals and polish", step_fuse, 1.0))
    job = Job(spec["object_name"], steps, spec["project"])
    job.kind = "ai_polish"
    return job, ctx
