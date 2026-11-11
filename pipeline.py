# SPDX-License-Identifier: GPL-3.0-or-later
# Photogram3D - photos or video to a textured mesh with COLMAP + OpenMVS
# Copyright (C) 2026 Kruglov Iurii
# Authors: Kruglov Iurii, Anthropic Claude AI
# License: GNU General Public License (GPL) v3 or later, https://www.blender.org/about/license/
"""Reconstruction pipeline: COLMAP 4 (SfM) + OpenMVS (dense, mesh, texture).

The plan is built from a plain settings dict so it runs without bpy
(the add-on UI, or a headless test, provides the dict).
"""

import glob
import json
import math
import os
import re
import shutil
import sqlite3
import struct
import sys
import threading
import time

from . import ai_backend, camera_track, colmap_io, errors, formats, roi, solve_check, tools, video, views, walk
from .common import (IS_WINDOWS, SOLVE_FILE, TRANSFORM_FILE, current, describe_returncode, list_images,
                     stable_hash)
from .hardware import best_gpu, openmvs_cuda_compatible, preset_params
from .runner import Job, Step, StepError
from .tools import openmvs_exe

EXHAUSTIVE_LIMIT = 300      # images; above this use sequential/vocab matching
MIN_REGISTERED_RATIO = 0.6  # below this share of registered images the other solver is tried


# ---------------------------------------------------------------------------
# Progress parsers: each returns fraction (0..1), (fraction, note) or None
# ---------------------------------------------------------------------------

_RE_PROCESSED = re.compile(r"Processed file \[(\d+)/(\d+)\]")
_RE_BLOCK = re.compile(r"Processing block \[(\d+)/(\d+), (\d+)/(\d+)\]")
_RE_BLOCK1 = re.compile(r"Processing block \[(\d+)/(\d+)\]")
_RE_IMAGE = re.compile(r"Processing image \[(\d+)/(\d+)\]")
_RE_UNDIST = re.compile(r"Undistorting image \[(\d+)/(\d+)\]")
_RE_REGISTER = re.compile(r"Registering image #\d+ \(num_reg_frames=(\d+)\)")
_RE_VIEW = re.compile(r"Processing view (\d+) / (\d+)")
_RE_FUSE = re.compile(r"Fusing image \[(\d+)/(\d+)\]")
_RE_OMVS = re.compile(r"^(?P<msg>[A-Za-z][A-Za-z \-]+?) (?P<done>\d+) \((?P<pct>[\d.]+)%")

GLOBAL_STAGES = [
    ("Running rotation averaging", 0.15),
    ("Running track establishment", 0.3),
    ("Running global positioning", 0.45),
    ("Running iterative bundle adjustment", 0.65),
    ("Running iterative retriangulation", 0.85),
]


def parse_feature(line):
    m = _RE_PROCESSED.search(line)
    if m:
        i, n = int(m.group(1)), int(m.group(2))
        return i / max(n, 1), f"image {i}/{n}"
    return None


def parse_matching(line):
    m = _RE_BLOCK.search(line)
    if m:
        # logged when a block starts, so count it as not yet done
        i, n, j, n2 = (int(x) for x in m.groups())
        return ((i - 1) * n2 + (j - 1)) / max(n * n2, 1), f"block {i},{j} of {n}x{n2}"
    m = _RE_BLOCK1.search(line) or _RE_IMAGE.search(line)
    if m:
        i, n = int(m.group(1)), int(m.group(2))
        return i / max(n, 1), f"{i}/{n}"
    return None


def make_mapper_parser(num_images):
    state = {"frac": 0.0}

    def parse(line):
        m = _RE_REGISTER.search(line)
        if m:
            k = int(m.group(1))
            state["frac"] = max(state["frac"], min(0.95, k / max(num_images, 1)))
            return state["frac"], f"{k}/{num_images} images registered"
        for text, frac in GLOBAL_STAGES:
            if text in line:
                state["frac"] = max(state["frac"], frac)
                return state["frac"], text.replace("Running ", "")
        return None
    return parse


def parse_undistort(line):
    m = _RE_UNDIST.search(line)
    if m:
        i, n = int(m.group(1)), int(m.group(2))
        return i / max(n, 1), f"image {i}/{n}"
    return None


def make_openmvs_parser(stages, milestones=None):
    """``stages`` maps OpenMVS progress-message prefixes to (start, end) fractions,
    or to a list of ranges for a stage that runs several times with the same
    message (a new run is detected when its percentage drops).
    ``milestones`` maps log substrings (phases without a counter) to a fraction."""
    state = {"prefix": None, "pct": 0.0, "run": 0}

    def parse(line):
        m = _RE_OMVS.search(line)
        if not m:
            for text, frac in (milestones or {}).items():
                if text in line:
                    return frac, text
            return None
        msg = m.group("msg").strip()
        pct = float(m.group("pct")) / 100.0
        for prefix, ranges in stages.items():
            if not msg.startswith(prefix):
                continue
            if state["prefix"] != prefix:
                state.update(prefix=prefix, run=0)
            elif pct + 0.2 < state["pct"]:
                state["run"] += 1
            state["pct"] = pct
            if isinstance(ranges, list):
                a, b = ranges[min(state["run"], len(ranges) - 1)]
                note = f"{msg} ({state['run'] + 1}/{len(ranges)}) {pct * 100:.0f}%"
            else:
                (a, b), note = ranges, f"{msg} {pct * 100:.0f}%"
            return a + (b - a) * pct, note
        return None
    return parse


# DensifyPointCloud runs the geometric-consistency pass twice (--geometric-iters 2).
DENSIFY_STAGES = {
    "Estimated depth-maps": (0.0, 0.5),
    "Geometric-consistent estimated depth-maps": [(0.5, 0.67), (0.67, 0.84)],
    "Filtered depth-maps": (0.84, 0.87),
    "Fused depth-maps": (0.87, 0.98),
    "Dense fused depth-maps": (0.87, 0.98),
    "Merged depth-maps": (0.87, 0.98),
}


def make_monotonic(*parsers):
    """Combine parsers; progress never moves backwards."""
    state = {"frac": 0.0}

    def parse(line):
        for prs in parsers:
            res = prs(line)
            if res is not None:
                frac, note = res if isinstance(res, tuple) else (res, None)
                state["frac"] = max(state["frac"], frac)
                return state["frac"], note
        return None
    return parse


def make_densify_parser(num_images, passes=3):
    """Console percentages (Linux) or one log line per depth map (Windows).

    DensifyPointCloud estimates every depth map once and refines it in two
    geometric-consistency passes, logging "Depth-map for image N estimated" each time.
    """
    total = max(1, num_images * passes)
    count = {"maps": 0}

    def from_log(line):
        if "Depth-map for image" in line and "estimated" in line:
            count["maps"] += 1
            return 0.84 * min(1.0, count["maps"] / total), f"depth map {count['maps']}/{total}"
        if "fused and filtered" in line:
            return 0.97, "fusing depth maps"
        return None
    return make_monotonic(make_openmvs_parser(DENSIFY_STAGES), from_log)


_RE_REFINE_ITER = re.compile(r"\]\s+(\d+)\. f:")


def make_refine_parser():
    """RefineMesh: pre-decimation, then iterations per scale (45, then 22 by default)."""
    state = {"scale": -1}
    iters = (45, 22)
    ranges = ((0.05, 0.5), (0.5, 0.97))

    def from_log(line):
        if "Mesh subdivided:" in line:
            state["scale"] = min(state["scale"] + 1, 1)
            return ranges[state["scale"]][0], f"scale {state['scale'] + 1}/2"
        m = _RE_REFINE_ITER.search(line)
        if m and state["scale"] >= 0:
            a, b = ranges[state["scale"]]
            i = int(m.group(1))
            return a + (b - a) * min(1.0, i / iters[state["scale"]]), \
                f"scale {state['scale'] + 1}/2, iteration {i}"
        return None
    stdout = make_openmvs_parser({"Decimated faces": (0.0, 0.05),
                                  "Processed iterations": [(0.05, 0.5), (0.5, 0.97)]})
    return make_monotonic(stdout, from_log)


def _tail(ctx, app):
    """Follow OpenMVS's own log on Windows (its console output cannot be piped there)."""
    return (ctx["paths"]["mvs"], app) if IS_WINDOWS else None


def make_patchmatch_parser(passes=2):
    """COLMAP patch_match_stereo and stereo_fusion progress.

    With geometric consistency, patch_match_stereo runs every image photometrically and
    then every image again geometrically, logging "Processing view i / n" in both passes
    (images it finds finished are skipped silently); stereo_fusion logs
    "Fusing image [i/n]"."""
    state = {"pass": 0, "last": 0}

    def parse(line):
        m = _RE_VIEW.search(line)
        if m:
            i, n = int(m.group(1)), int(m.group(2))
            if i < state["last"]:                 # numbering restarted: next pass
                state["pass"] = min(passes - 1, state["pass"] + 1)
            state["last"] = i
            k = state["pass"]
            note = f"depth map {i}/{n}" + (f", pass {k + 1} of {passes}" if passes > 1 else "")
            return 0.85 * (k + i / max(n, 1)) / passes, note
        m = _RE_FUSE.search(line)
        if m:
            i, n = int(m.group(1)), int(m.group(2))
            return 0.85 + 0.15 * i / max(n, 1), f"fusing {i}/{n}"
        return None
    return make_monotonic(parse)


# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------

def result_path(project_dir):
    return current(os.path.join(project_dir, "photogram3d_result.json"))


# ---------------------------------------------------------------------------
# Attempts (0.3.4): a run with Resume off goes to a new folder next to the output
# folder (Output_2, Output_3, ...), so earlier results are kept. Resume continues the
# newest attempt.
# ---------------------------------------------------------------------------

def attempt_folder(base, k):
    base = os.path.normpath(base)
    return base if k <= 1 else f"{base}_{k}"


def _attempt_used(folder):
    return (os.path.isfile(current(os.path.join(folder, "photogram3d_state.json")))
            or os.path.isfile(result_path(folder)))


def attempts(base):
    """[(k, folder)] of the runs made for the output folder ``base``: base itself is
    attempt 1, base_2 attempt 2, and so on. Folders Photogram3D never wrote are ignored."""
    if not base:
        return []
    base = os.path.normpath(base)
    out = [(1, base)] if _attempt_used(base) else []
    parent, name = os.path.split(base)
    try:
        entries = os.listdir(parent or ".")
    except OSError:
        return out
    pattern = re.compile(re.escape(name) + r"_(\d+)$", re.IGNORECASE if os.name == "nt" else 0)
    for entry in entries:
        m = pattern.match(entry)
        if m and int(m.group(1)) >= 2:
            folder = os.path.join(parent, entry)
            if _attempt_used(folder):
                out.append((int(m.group(1)), folder))
    return sorted(out)


def fork_target(base, src):
    """(folder, k) for a new run forked from the run ``src`` (Re-solve Without Selected, Re-solve
    From Cameras): the next attempt of ``base``, never ``src`` itself (a run folder whose state
    file is gone does not count as an attempt)."""
    folder, k = choose_attempt(base, resume=False)
    while os.path.normcase(os.path.normpath(folder)) == os.path.normcase(os.path.normpath(src)) or \
            (os.path.isdir(folder) and os.listdir(folder)):
        k += 1
        folder = attempt_folder(base, k)
    return folder, k


def choose_attempt(base, resume):
    """(folder, k) for a new run: the newest attempt when resuming, else the next one."""
    done = attempts(base)
    if not done:
        return os.path.normpath(base), 1
    k, folder = done[-1]
    if resume:
        return folder, k
    k += 1
    # a folder of that name that holds something else is left alone
    while os.path.isdir(attempt_folder(base, k)) and os.listdir(attempt_folder(base, k)):
        k += 1
    return attempt_folder(base, k), k


def scan_name(cfg):
    """Short name of the input: the photo folder's or the video's (a joined run's is given)."""
    if cfg.get("scan_name"):
        return cfg["scan_name"]
    if cfg.get("input_mode") == "VIDEO" and cfg.get("video_path"):
        return os.path.splitext(os.path.basename(cfg["video_path"]))[0] or "scan"
    folder = os.path.basename(os.path.normpath(cfg.get("images_dir") or "")) if cfg.get("images_dir") else ""
    return folder or os.path.basename(os.path.normpath(cfg.get("project_dir") or "scan")) or "scan"


def project_paths(project_dir):
    p = {"project": project_dir}
    p["frames"] = os.path.join(project_dir, "frames")
    p["photos"] = os.path.join(project_dir, "photos")      # converted photos (formats.py)
    p["prepared"] = os.path.join(project_dir, "prepared")  # evened-out photos (step_even)
    p["colmap"] = os.path.join(project_dir, "colmap")
    p["db"] = os.path.join(p["colmap"], "database.db")
    p["image_list"] = os.path.join(p["colmap"], "image_list.txt")
    p["sparse"] = os.path.join(p["colmap"], "sparse")
    p["sparse_inc"] = os.path.join(p["colmap"], "sparse_incremental")
    p["model"] = os.path.join(p["colmap"], "model")
    p["guide"] = os.path.join(p["colmap"], "guide")        # cameras placed by hand (Re-solve From Cameras)
    p["dense"] = os.path.join(project_dir, "dense")
    p["mvs"] = os.path.join(project_dir, "mvs")
    p["logs"] = os.path.join(project_dir, "logs")
    p["scene"] = os.path.join(p["mvs"], "scene.mvs")
    p["scene_dense"] = os.path.join(p["mvs"], "scene_dense.mvs")
    p["mesh"] = os.path.join(p["mvs"], "scene_dense_mesh.ply")
    p["mesh_refined"] = os.path.join(p["mvs"], "scene_dense_mesh_refine.ply")
    p["textured"] = os.path.join(p["mvs"], "scene_textured.obj")
    p["state"] = current(os.path.join(project_dir, "photogram3d_state.json"))
    return p


# ---------------------------------------------------------------------------
# Resume bookkeeping
# ---------------------------------------------------------------------------

def _load_state(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_state(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _step_signature(ctx, key):
    """Hash of everything a step's output depends on (its inputs + settings)."""
    return stable_hash({"key": key, "sig": ctx["sig_chain"].get(key)})


def _resumable(job, ctx, key, outputs):
    if not ctx["cfg"].get("resume", True):
        return False
    state = ctx["state"]
    if state.get(key) != _step_signature(ctx, key):
        return False
    if not all(os.path.exists(o) for o in outputs):
        return False
    job.add_line("(resumed: outputs from a previous run are up to date)")
    job.skipped.append(key)
    job.set_fraction(1.0, "reused")
    return True


def _mark_done(ctx, key):
    ctx["state"][key] = _step_signature(ctx, key)
    _save_state(ctx["paths"]["state"], ctx["state"])


# Steps whose results fix the cameras and their coordinates. Build Mesh continues a run
# from its crop stage: the crop box (and the alignment) were placed on the dense cloud of
# this solve, so none of these may be computed again then (0.3.5; in 0.3.4 a lost photo
# exclusion made Build Mesh solve the cameras again and mesh in other coordinates).
FRAME_STEPS = ("frames", "photos", "features", "matching", "mapping", "orient")
_FRAME_STEP_WORDS = {"frames": "the video frames", "photos": "the photos",
                     "features": "the image features", "matching": "the matches",
                     "mapping": "the cameras", "orient": "the up-axis of the cameras"}


# Later steps that do not depend on these earlier ones: the splats train on the
# undistorted photos only, so a new crop box (a new mesh) must not train them again (the
# step crops the trained splats every time).
_NOT_DEPENDING_ON = {"splats": {"colmap_dense", "mvs_import", "densify", "crop_prepare", "mesh",
                                "refine", "texture", "blend"},
                     # the photos evened out for the texture depend on the cameras only (0.3.12)
                     "even": {"colmap_dense", "mvs_import", "densify", "crop_prepare", "mesh", "refine"}}


def _invalidate_from(ctx, key):
    """Forget this step and every later one that depends on it."""
    if ctx.get("crop_continuation") and key in FRAME_STEPS:
        raise StepError(
            f"Build Mesh would have to compute {_FRAME_STEP_WORDS[key]} again: the photos or a "
            "setting they depend on (quality, solver, AI features, photo options) changed since the "
            "dense cloud was made. New cameras would move the cloud away from the crop box. Set the "
            "settings back as they were, or press Reconstruct to make a new dense cloud and crop box")
    order = ctx["step_order"]
    if key in order:
        for k in order[order.index(key):]:
            if key in _NOT_DEPENDING_ON.get(k, ()):
                continue
            ctx["state"].pop(k, None)
        _save_state(ctx["paths"]["state"], ctx["state"])


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def _log(ctx, key):
    idx = ctx["step_order"].index(key) + 1 if key in ctx["step_order"] else 0
    return os.path.join(ctx["paths"]["logs"], f"{idx:02d}_{key}.log")


def _colmap(ctx):
    return ctx["ts"]["colmap"]


def _gpu_flag(ctx):
    return "1" if ctx["use_gpu"] else "0"


def step_frames(job, ctx):
    p = ctx["paths"]
    key = "frames"
    if _resumable(job, ctx, key, [p["frames"]]) and list_images(p["frames"]):
        ctx["images_dir"] = p["frames"]
        return
    _invalidate_from(ctx, key)
    ts = ctx["ts"]
    if not ts.get("ffmpeg") or not ts.get("ffprobe"):
        raise StepError("FFmpeg is not installed (needed for video). Install it from the add-on preferences.")
    n = video.extract(job, ctx["cfg"], ts["ffmpeg"], ts["ffprobe"], ctx["cfg"]["video_path"],
                      p["frames"], _log(ctx, key))
    job.add_line(f"{n} frames extracted")
    ctx["images_dir"] = p["frames"]
    _mark_done(ctx, key)


def step_photos(job, ctx):
    """Photos in formats COLMAP cannot read are converted to JPEG (formats.py); of a RAW +
    JPEG pair of one shot only one is used. With nothing to convert the photo folder is
    used as it is."""
    p, cfg = ctx["paths"], ctx["cfg"]
    key = "photos"
    src = cfg["images_dir"]
    prefer_raw = cfg.get("raw_pairs", "JPEG") == "RAW"
    max_side = int(cfg.get("photo_max_size") or 0)
    pl = formats.plan(src, prefer_raw)
    if max_side:       # JPEG and PNG photos larger than Max Photo Size are scaled down too
        for n in pl["use"]:
            size = None if n in pl["convert"] else formats.image_size(os.path.join(src, n))
            if size and max(size) > max_side:
                pl["convert"].append(n)
    stats = []
    for n in pl["use"]:
        try:
            st = os.stat(os.path.join(src, n))
            stats.append([n, st.st_size, int(st.st_mtime)])
        except OSError:
            stats.append([n, 0, 0])
    ctx["sig_chain"][key] = {"src": os.path.normpath(src), "photos": stats, "raw": prefer_raw}
    if max_side:
        ctx["sig_chain"][key]["max"] = max_side
    if pl["skipped"]:
        what = "JPEG/HEIC twins of RAW photos" if prefer_raw else "RAW twins of JPEG photos"
        job.add_line(f"Skipped {len(pl['skipped'])} {what} (one photo per shot)")
    if not pl["convert"]:
        ctx["images_dir"], ctx["image_names"] = src, pl["use"]
        job.set_fraction(1.0, "nothing to convert")
        return
    out = p["photos"]
    names = [formats.converted_name(n) if n in pl["convert"] else n for n in pl["use"]]
    if _resumable(job, ctx, key, [out]) and all(os.path.isfile(os.path.join(out, n)) for n in names):
        ctx["images_dir"], ctx["image_names"] = out, names
        return
    _invalidate_from(ctx, key)
    log = _log(ctx, key)
    site = formats.paths(cfg.get("tools_root") or p["project"])["site"]
    python = cfg.get("python") or sys.executable
    if formats.runtime_info(site) is None:
        formats.install_runtime(job, python, site, log)
    os.makedirs(out, exist_ok=True)
    # photos converted with another Max Photo Size are converted again
    marker = current(os.path.join(out, "photogram3d_conversion.json"))
    settings = {"max_side": max_side}
    try:
        with open(marker, encoding="utf-8") as f:
            same = json.load(f) == settings
    except (OSError, ValueError):
        same = not max_side           # folders converted before 0.3.5 had no size limit
    if not same:
        _remove_files([os.path.join(out, formats.converted_name(n)) for n in pl["convert"]])
    with open(marker, "w", encoding="utf-8") as f:
        json.dump(settings, f)
    todo = [n for n in pl["convert"]
            if not formats.is_converted(os.path.join(src, n), os.path.join(out, formats.converted_name(n)))]
    kinds = {}
    for n in pl["convert"]:
        kinds[formats.ext(n).lstrip(".").upper()] = kinds.get(formats.ext(n).lstrip(".").upper(), 0) + 1
    _note(job, log, f"Converting {len(pl['convert'])} photos to JPEG ("
                    + ", ".join(f"{v} {k}" for k, v in sorted(kinds.items())) + ")"
                    + (f", scaled down to {max_side} px where larger" if max_side else ""))
    failed = formats.convert(job, python, site, src, todo, out, log, max_side=max_side,
                             twins=pl.get("twins")) if todo else []
    for n in pl["use"]:
        if n not in pl["convert"]:
            formats.link_or_copy(os.path.join(src, n), os.path.join(out, n))
    keep = set(names) | {os.path.basename(marker)}
    for f in os.listdir(out):
        if f not in keep and os.path.isfile(os.path.join(out, f)):
            os.remove(os.path.join(out, f))
    names = [n for n in names if os.path.isfile(os.path.join(out, n))]
    if failed:
        msg = (f"{len(failed)} photo(s) could not be converted and are left out: "
               + ", ".join(failed[:6]) + (" ..." if len(failed) > 6 else ""))
        _note(job, log, msg)
        ctx["result"].setdefault("warnings", []).append(msg)
    ctx["images_dir"], ctx["image_names"] = out, names
    _mark_done(ctx, key)


def _write_image_list(ctx):
    names = ctx.get("image_names") or list_images(ctx["images_dir"])
    if len(names) < 3:
        raise StepError(f"need at least 3 images, found {len(names)} in {ctx['images_dir']}")
    with open(ctx["paths"]["image_list"], "w", encoding="utf-8") as f:
        f.write("\n".join(names) + "\n")
    ctx["num_images"] = len(names)
    return names


def step_features(job, ctx):
    p, cfg, prm = ctx["paths"], ctx["cfg"], ctx["params"]
    key = "features"
    if (cfg["ai_features"] and ai_device(ctx) == "DIRECTML" and ai_setup_ready(cfg.get("tools_root") or "")
            and not ai_backend.runtime_directml(cfg["tools_root"])):
        # only ONNX Runtime's CPU build loads here (polish_ai.step_runtime): COLMAP on the CPU
        _ai_to_cpu(job, ctx, _log(ctx, key), "only ONNX Runtime's CPU build loads in Blender's Python here")
        cfg = ctx["cfg"]
    os.makedirs(p["colmap"], exist_ok=True)
    names = _write_image_list(ctx)
    single = bool(cfg["single_camera"])
    if single:
        # one lens for photos of different sizes (portrait and landscape HEIC, say) makes
        # COLMAP skip every photo whose size differs from the first one's
        sizes = {formats.image_size(os.path.join(ctx["images_dir"], n)) for n in names} - {None}
        if len(sizes) > 1:
            single = False
            job.add_line(f"The photos come in {len(sizes)} sizes ("
                         + ", ".join(f"{w}x{h}" for w, h in sorted(sizes)[:4])
                         + "): each photo gets its own lens instead of one for all")
    # the photos step only matters when it converted the photos (a folder of JPEGs is used
    # as it is, and projects made before 0.3.4 then keep their finished steps)
    converted = os.path.normcase(os.path.normpath(ctx["images_dir"])) == \
        os.path.normcase(os.path.normpath(p["photos"]))
    fmax = prm["feat_max_image"]
    if cfg["ai_features"]:
        fmax = min(fmax, ai_max_size(ctx))
    ctx["sig_chain"][key] = {"images": names, "prev": ctx["sig_chain"].get("frames") or
                             (ctx["sig_chain"].get("photos") if converted else None),
                             "ai": cfg["ai_features"], "fmax": fmax,
                             "model": cfg["camera_model"], "single": single}
    if cfg["ai_features"]:          # new key only when used, so SIFT runs still resume
        ctx["sig_chain"][key]["points"] = prm["aliked_features"]
    if _resumable(job, ctx, key, [p["db"]]):
        return
    _invalidate_from(ctx, key)
    if os.path.exists(p["db"]):
        os.remove(p["db"])
    job.add_line(f"{len(names)} images")
    if cfg["ai_features"] and ai_device(ctx) == "DIRECTML":
        try:
            _features_directml(job, ctx, single, fmax, _log(ctx, key))
        except (StepError, OSError, ValueError, sqlite3.Error) as exc:
            _ai_to_cpu(job, ctx, _log(ctx, key), f"finding the features failed ({exc})")
            return step_features(job, ctx)
        _mark_done(ctx, key)
        return
    args = [_colmap(ctx), "feature_extractor",
            "--database_path", p["db"],
            "--image_path", ctx["images_dir"],
            "--image_list_path", p["image_list"],
            "--ImageReader.camera_model", cfg["camera_model"],
            "--ImageReader.single_camera", "1" if single else "0",
            "--FeatureExtraction.max_image_size", str(fmax)]
    if cfg["ai_features"]:
        args += ["--FeatureExtraction.type", "ALIKED_N16ROT",
                 "--AlikedExtraction.max_num_features", str(prm["aliked_features"])]
    else:
        args += ["--FeatureExtraction.type", "SIFT",
                 "--SiftExtraction.max_num_features", str(prm["sift_features"])]
    _run_with_gpu_fallback(job, ctx, args, "--FeatureExtraction.use_gpu", _log(ctx, key),
                           parse_feature, ai=cfg["ai_features"])
    _log_notes(job, ctx, key)
    _mark_done(ctx, key)


def _log_notes(job, ctx, key):
    """Remarks from errors.notes() on the step's log, once per run (AI Features on the CPU)."""
    notes = ctx["result"].setdefault("notes", [])
    for note in errors.notes(key, errors.read_tail(_log(ctx, key)), ctx["cfg"]):
        if note not in notes:
            notes.append(note)
            job.add_line(note)


AI_ON_CPU = ("AI Features run on the CPU: COLMAP's AI models need NVIDIA's cuBLAS and cuDNN on "
             "the GPU, which the COLMAP download does not include. For the GPU: Preferences › AI "
             "Features On › GPU (DirectML), or Download NVIDIA Libraries")


AI_NVIDIA_MISSING = ("AI Features run on the CPU: NVIDIA's libraries were chosen but are not "
                     "installed (Preferences › AI Features › Download NVIDIA Libraries)")


def _ai_on_gpu(job, ctx, log):
    """Whether COLMAP's AI models (ALIKED, LightGlue) run on the GPU: not when the CPU was
    chosen, and only with NVIDIA's CUDA libraries (tools.onnx_gpu_ready). When they are not
    there it is said once per run, in the step's log and in the run's notes (the panel's lines
    scroll away during a long step)."""
    if _gpu_flag(ctx) != "1" or ai_device(ctx) == "CPU":
        return False
    if "ai_gpu" not in ctx:
        ctx["ai_gpu"] = tools.onnx_gpu_ready(ctx["ts"]["colmap"], ctx["ts"]["colmap_env"])
        if not ctx["ai_gpu"]:
            text = AI_NVIDIA_MISSING if ai_device(ctx) == "NVIDIA" else AI_ON_CPU
            _note(job, log, text)
            notes = ctx["result"].setdefault("notes", [])
            if text not in notes:
                notes.append(text)
    return ctx["ai_gpu"]


def _run_with_gpu_fallback(job, ctx, args, gpu_opt, log, parser, ai=False, cpu_args=()):
    """Run with the GPU when there is one; an AI (ONNX) step that fails there is run again on
    the CPU. ``cpu_args`` are added when it runs on the CPU."""
    use_gpu = _gpu_flag(ctx)
    if ai and use_gpu == "1" and not _ai_on_gpu(job, ctx, log):
        use_gpu = "0"
    extra = list(cpu_args) if use_gpu == "0" else []
    try:
        job.run(args + [gpu_opt, use_gpu] + extra, log, env=ctx["ts"]["colmap_env"], parser=parser)
    except StepError:
        if not (ai and use_gpu == "1"):
            raise
        job.add_line("AI step failed on the GPU; retrying on the CPU (slower)")
        if os.path.exists(ctx["paths"]["db"]) and gpu_opt.startswith("--FeatureExtraction"):
            os.remove(ctx["paths"]["db"])
        job.run(args + [gpu_opt, "0"] + list(cpu_args), log, env=ctx["ts"]["colmap_env"], parser=parser)


def ai_device(ctx):
    """Where AI Features run (ai_backend): DIRECTML (the runner, any DirectX 12 GPU), NVIDIA
    or CPU (COLMAP itself); COLMAP (settings from before 0.3.9) lets COLMAP use the GPU when
    NVIDIA's libraries are there, as 0.3.7 did."""
    return ctx["cfg"].get("ai_device") or "COLMAP"


def ai_max_size(ctx):
    """ALIKED's photo size: on the CPU by the RAM (hardware.preset_params), on a GPU the
    largest size the quick test ran there (ai_backend), 1600 px without a test."""
    if ai_device(ctx) in ("DIRECTML", "NVIDIA"):
        return int(ctx["cfg"].get("ai_max_size") or ai_backend.GPU_DEFAULT_SIZE)
    return ctx["params"]["aliked_max_image"]


AI_FEATURES_DIR = "aliked_features"   # in the colmap folder: the runner's feature files
AI_MARKER = "photogram3d_ai.json"     # what the runner's files in a folder were made for


def _runner_cmd(ctx, cmd):
    from . import polish_ai
    cfg = ctx["cfg"]
    site = polish_ai.ai_paths(cfg["tools_root"])["site"]
    return [ai_backend.python_for(cfg), "-X", "faulthandler", "-I", ai_backend.runner_script(), cmd,
            "--site", site, "--device", "cpu" if cfg.get("ai_runner_cpu") else "auto"]


def _fresh_for(folder, signature):
    """Keep the runner's files in ``folder`` only when they were made for ``signature`` (a
    cancelled run continues; files made with other settings are removed)."""
    marker = os.path.join(folder, AI_MARKER)
    try:
        with open(marker, "r", encoding="utf-8") as f:
            same = json.load(f).get("sig") == signature
    except (OSError, ValueError):
        same = False
    if not same:
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder, exist_ok=True)
        with open(marker, "w", encoding="utf-8") as f:
            json.dump({"sig": signature}, f)
    return same


def _runner_parser(prefix, total, lo=0.0, hi=1.0):
    def parse(line):
        if line.startswith(prefix + " "):
            part = line.split()[1]
            if "/" in part:
                a, b = part.split("/", 1)
                if a.isdigit() and b.isdigit():
                    n = max(int(b), 1)
                    return lo + (hi - lo) * int(a) / n, f"{a}/{b}"
        return None
    return parse


def _features_directml(job, ctx, single, fmax, log):
    """ALIKED on any DirectX 12 GPU with Photogram3D's runner (ai_features.py), then COLMAP
    reads the photos as its own extractor would (cameras, EXIF, gravity) and takes the
    keypoints from the runner's files; the descriptors go into the database afterwards."""
    p, cfg, prm = ctx["paths"], ctx["cfg"], ctx["params"]
    feats = os.path.join(p["colmap"], AI_FEATURES_DIR)
    _fresh_for(feats, _step_signature(ctx, "features"))
    job.set_fraction(0.0, "ALIKED on the GPU")
    job.run(_runner_cmd(ctx, "extract") + [
        "--model", ai_backend.model_path(cfg["tools_root"], "aliked"), "--images", ctx["images_dir"],
        "--list", p["image_list"], "--out", feats, "--max-size", str(fmax),
        "--max-points", str(prm["aliked_features"])], log,
        parser=_runner_parser("Features", ctx["num_images"], 0.0, 0.9), below_normal=False)
    job.set_fraction(0.9, "handing the features to COLMAP")
    job.run([_colmap(ctx), "feature_importer", "--database_path", p["db"], "--image_path", ctx["images_dir"],
             "--image_list_path", p["image_list"], "--import_path", feats,
             "--ImageReader.camera_model", cfg["camera_model"],
             "--ImageReader.single_camera", "1" if single else "0"],
            log, env=ctx["ts"]["colmap_env"])
    try:
        n = ai_backend.set_descriptors(p["db"], feats)
    except (OSError, ValueError, sqlite3.Error) as exc:
        raise StepError(f"the features could not be put into COLMAP's database ({exc})") from exc
    if n == 0:
        raise StepError("the AI runner produced no features; see the features log")
    _note(job, log, f"AI Features on the GPU (DirectML): {n} photos at up to {fmax} px")
    shutil.rmtree(feats, ignore_errors=True)


def _matching_directml(job, ctx, sequential, log):
    """LightGlue on any DirectX 12 GPU with the runner, written into COLMAP's database; COLMAP
    then checks every matched pair (geometric_verifier) as its matcher would."""
    p, cfg = ctx["paths"], ctx["cfg"]
    work = os.path.join(p["colmap"], "ai_matching")
    if not _fresh_for(work, _step_signature(ctx, "matching")):
        con = sqlite3.connect(p["db"], timeout=60)
        try:
            con.execute("DELETE FROM matches")
            con.execute("DELETE FROM two_view_geometries")
            con.commit()
        finally:
            con.close()
    pairs = os.path.join(work, "pairs.txt")
    n_pairs = ai_backend.write_pairs(p["db"], pairs, sequential)
    job.add_line(f"{n_pairs} photo pairs to match")
    job.run(_runner_cmd(ctx, "match") + [
        "--model", ai_backend.model_path(cfg["tools_root"], "lightglue"), "--db", p["db"], "--pairs", pairs],
        log, parser=_runner_parser("Matches", n_pairs, 0.0, 0.9), below_normal=False)
    job.set_fraction(0.9, "checking the matched pairs")
    job.run([_colmap(ctx), "geometric_verifier", "--database_path", p["db"]], log,
            env=ctx["ts"]["colmap_env"])


def _ai_to_cpu(job, ctx, log, why):
    """The GPU path (DirectML) failed: the run continues with COLMAP on the CPU."""
    ctx["cfg"] = dict(ctx["cfg"], ai_device="CPU")
    runner_name = os.path.basename(ai_backend.python_for(ctx["cfg"]))
    why = why.replace(f"{runner_name} failed", "the AI runner failed")      # it runs in Blender's Python
    text = f"AI Features on the GPU (DirectML): {why}; continuing on the CPU"
    _note(job, log, text)
    ctx["result"].setdefault("notes", []).append(text)


def ai_setup_ready(tools_root):
    """ONNX Runtime and the two model files are there (AI Features with DirectML)."""
    return ai_backend.setup_ready(tools_root)


def step_ai_test(job, ctx):
    """Automatic, first run on this computer: Test AI Speed (ai_backend), then this run uses the
    fastest device that worked, with the largest photo size the GPU handled."""
    cfg, hw, ts = ctx["cfg"], ctx["hw"], ctx["ts"]
    names = list_images(cfg["images_dir"]) if cfg.get("input_mode") != "VIDEO" and cfg.get("images_dir") else []
    sample = next((os.path.join(cfg["images_dir"], n) for n in names
                   if n.lower().endswith((".jpg", ".jpeg", ".png"))), "")
    prm = ctx["params"]
    spec = {"tools_root": cfg["tools_root"], "python": ai_backend.python_for(cfg), "sample": sample,
            "cpu_size": prm["aliked_max_image"], "cpu_matchers": ai_matchers(hw, prm["aliked_features"]),
            "colmap": ts.get("colmap"), "colmap_env": ts.get("colmap_env") or {},
            "has_nvidia_gpu": bool(hw.get("gpus")), "hw_text": "", "n_photos": len(names) or 42,
            "video": cfg.get("input_mode") == "VIDEO"}
    sub_job, sub_ctx = ai_backend.build_test_job(spec)
    sub_ctx["log"] = _log(ctx, "ai_test")
    try:
        for step in sub_job.steps:
            if step.key == "ai_setup":
                continue                      # done by this run's own setup step
            step.func(job, sub_ctx)
    except (StepError, OSError, ValueError) as exc:
        _ai_to_cpu(job, ctx, _log(ctx, "ai_test"), f"the speed test failed ({exc})")
        return
    test = ai_backend.read_test(cfg["tools_root"])
    device = ai_backend.resolve("AUTO", cfg["tools_root"], test, bool(hw.get("gpus")))
    if device == "AUTO" or device not in ai_backend.LABELS:
        device = "CPU"
    update = {"ai_device": device}
    res = ((test or {}).get("results") or {}).get(device)
    if device in ("DIRECTML", "NVIDIA") and res and res.get("extract"):
        update["ai_max_size"] = ai_backend.max_gpu_size(res)
    ctx["cfg"] = dict(cfg, **update)
    _, lines = ai_backend.suggest(test, ai_backend.nvidia_ready(cfg["tools_root"]), len(names) or 42,
                                  cfg.get("input_mode") == "VIDEO")
    _note(job, _log(ctx, "ai_test"), "AI Features: tested this computer (" + "; ".join(lines) + "); using "
          + ai_backend.LABELS[device])


def step_ai_setup(job, ctx):
    """Once per computer: ONNX Runtime with DirectML (as AI Polish uses) and COLMAP's two
    AI model files, for AI Features on the GPU (DirectML)."""
    cfg = ctx["cfg"]
    try:
        ai_backend.setup_runtime_and_models(job, cfg["tools_root"], ai_backend.python_for(cfg),
                                            _log(ctx, "ai_setup"))
    except StepError as exc:
        _ai_to_cpu(job, ctx, _log(ctx, "ai_setup"), f"the setup failed ({exc})")


def step_matching(job, ctx):
    p, cfg, prm = ctx["paths"], ctx["cfg"], ctx["params"]
    key = "matching"
    # COLMAP 4.2.1's LightGlue matcher cannot do guided matching: it throws "Guided
    # matching not supported for LightGlue", which kills exhaustive_matcher (Windows
    # reports 0xC0000409). Guided matching is a SIFT refinement only.
    guided = bool(prm["guided_matching"]) and not cfg["ai_features"]
    ctx["sig_chain"][key] = {"prev": ctx["sig_chain"].get("features"),
                             "guided": guided, "mode": cfg["input_mode"]}
    if _resumable(job, ctx, key, [p["db"]]):
        return
    _invalidate_from(ctx, key)
    n = ctx["num_images"]
    sequential = cfg["input_mode"] == "VIDEO" and n > EXHAUSTIVE_LIMIT
    if n > EXHAUSTIVE_LIMIT and not sequential:
        job.add_line(f"{n} images: exhaustive matching will be slow")
    if cfg["ai_features"] and ai_device(ctx) == "DIRECTML":
        try:
            _matching_directml(job, ctx, sequential, _log(ctx, key))
            _mark_done(ctx, key)
            return
        except (StepError, OSError, ValueError, sqlite3.Error) as exc:
            # the features are COLMAP's own format: its matcher continues with them, keeping
            # the pairs the runner matched
            _ai_to_cpu(job, ctx, _log(ctx, key), f"matching failed ({exc})")
    args = [_colmap(ctx), "sequential_matcher" if sequential else "exhaustive_matcher",
            "--database_path", p["db"],
            "--FeatureMatching.type", "ALIKED_LIGHTGLUE" if cfg["ai_features"] else "SIFT_BRUTEFORCE",
            "--FeatureMatching.guided_matching", "1" if guided else "0"]
    if sequential:
        args += ["--SequentialMatching.overlap", "20",
                 "--SequentialMatching.quadratic_overlap", "1"]
    cpu_args = []
    if cfg["ai_features"]:
        if not sequential:
            # COLMAP logs one line per block of photo pairs (50 photos by default: a 42 photo
            # set was one silent block); blocks of 10 photos keep the progress moving
            args += ["--ExhaustiveMatching.block_size", str(AI_MATCH_BLOCK)]
        cpu_args = ["--FeatureMatching.num_threads", str(ai_matchers(ctx["hw"], prm["aliked_features"]))]
    _run_with_gpu_fallback(job, ctx, args, "--FeatureMatching.use_gpu", _log(ctx, key),
                           parse_matching, ai=cfg["ai_features"], cpu_args=cpu_args)
    _log_notes(job, ctx, key)
    _mark_done(ctx, key)


AI_MATCH_BLOCK = 10
LIGHTGLUE_GB = {2048: 0.45, 4096: 1.3}     # one LightGlue model on the CPU, measured


def ai_matchers(hw, points):
    """LightGlue models COLMAP may run at once on the CPU (one per thread by default): as
    many as fit in half the RAM, at most one per CPU thread."""
    ram = hw.get("ram_gb") or 16
    each = LIGHTGLUE_GB.get(points, 1.3 * (points / 4096) ** 2)
    threads = hw.get("threads") or os.cpu_count() or 4
    return max(1, min(threads, int(0.5 * ram / each)))


WEAK_IMAGE_RATIO = 0.3       # drop images with < 30% of the median 3D observations ...
WEAK_IMAGE_MAX_SHARE = 0.2    # ... unless that would remove more than 20% of the images
# 3D points per solved photo below which the set is weakly matched: the gas mask (0.3.3)
# had 42-55 and folded; the WRX photos have 260, the wall 237, the synthetic scene 179
WEAK_POINTS_PER_PHOTO = 80


def _drop_weak_images(job, ctx, model_dir, log, out=None, db=None):
    """Remove weakly connected (likely mis-registered) images from the model.

    On ambiguous scenes (symmetric objects, repeated texture) the mappers can
    place a few cameras wrongly. Those images share far fewer 3D points with the
    rest than normal images do, and a wrong camera later projects wrong texture
    and depth. With ``db`` (a solve along the shooting order, 0.3.5) a weak image that
    is turned like the photos taken next to it stays (walk.neighbour_agreement): it is
    weak because its side of the object has little texture, not because it is misplaced.
    Returns the model folder to continue with.
    """
    p = ctx["paths"]
    obs = colmap_io.image_observations(model_dir)
    if len(obs) < 10:
        return model_dir
    counts = sorted(obs.values())
    median = counts[len(counts) // 2]
    weak = sorted(name for name, c in obs.items() if c < WEAK_IMAGE_RATIO * median)
    if weak and db:
        order = solve_check.shooting_order(_db_image_names(db))
        agree = walk.neighbour_agreement(model_dir, db, order)
        kept = [n for n in weak if n in agree and agree[n][1] >= 2
                and agree[n][0] <= walk.NEIGHBOUR_AGREE_DEG]
        if kept:
            _note(job, log, f"Kept {len(kept)} weakly connected photo(s) turned like the photos "
                            "taken next to them: " + ", ".join(kept[:6]) + (" ..." if len(kept) > 6 else ""))
            weak = [n for n in weak if n not in kept]
    if not weak:
        return model_dir
    if len(weak) > WEAK_IMAGE_MAX_SHARE * len(obs):
        _note(job, log, f"{len(weak)} images are weakly connected; keeping all of them")
        return model_dir
    out = out or os.path.join(p["colmap"], "filtered")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out, exist_ok=True)
    names_path = out + "_weak_images.txt"
    with open(names_path, "w", encoding="utf-8") as f:
        f.write("\n".join(weak) + "\n")
    try:
        job.run([_colmap(ctx), "image_deleter", "--input_path", model_dir, "--output_path", out,
                 "--image_names_path", names_path], log, env=ctx["ts"]["colmap_env"])
    except StepError as exc:
        _note(job, log, f"could not drop weak images ({exc}); keeping all")
        return model_dir
    if colmap_io.count_registered(out) != len(obs) - len(weak):
        return model_dir
    _note(job, log, f"Dropped {len(weak)} weakly connected image(s), likely mis-registered "
                 f"(< {WEAK_IMAGE_RATIO:.0%} of the median {median} shared points): "
                 + ", ".join(weak[:8]) + (" ..." if len(weak) > 8 else ""))
    return out


SOLVER_NAMES = {"GLOBAL": "Global solver", "INCREMENTAL": "Incremental solver"}


def _db_image_names(db_path):
    import sqlite3
    con = sqlite3.connect(db_path, timeout=30)
    try:
        return [r[0] for r in con.execute("SELECT name FROM images")]
    finally:
        con.close()


# COLMAP's image pair id: kMaxNumImages * smaller id + larger id (util/types.h)
_MAX_IMAGES = 2147483647
EXCLUDED_FILE = "excluded_images.txt"      # in the colmap folder: photos left out of the solve


def _db_without(p, names):
    """A copy of the match database in which ``names`` match nothing: the solvers then
    leave those photos out, and features and matches are reused."""
    import sqlite3
    out = os.path.join(p["colmap"], "database_solve.db")
    for suffix in ("", "-wal", "-shm"):
        try:
            os.remove(out + suffix)
        except OSError:
            pass
    src = sqlite3.connect(p["db"], timeout=30)
    dst = sqlite3.connect(out)
    try:
        src.backup(dst)
    finally:
        src.close()
    try:
        want = set(names)
        ids = [r[0] for r in dst.execute("SELECT image_id, name FROM images") if r[1] in want]
        for iid in ids:
            for table in ("matches", "two_view_geometries"):
                dst.execute(f"DELETE FROM {table} WHERE pair_id / ? = ? OR pair_id % ? = ?",
                            (_MAX_IMAGES, iid, _MAX_IMAGES, iid))
        dst.commit()
    finally:
        dst.close()
    return out


def _write_excluded(p, names):
    path = os.path.join(p["colmap"], EXCLUDED_FILE)
    if names:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(names) + "\n")
    elif os.path.exists(path):
        os.remove(path)


def read_excluded(project):
    try:
        with open(os.path.join(project_paths(project)["colmap"], EXCLUDED_FILE), encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    except OSError:
        return []


FORK_STEPS = ("frames", "photos", "features", "matching")


def fork_attempt(src_project, dst_project):
    """Start a new run from an existing one's photos, features and matches (Re-solve
    Without Selected): they are linked (or copied) and their steps marked done, so the
    new run starts at the camera solve."""
    src, dst = project_paths(src_project), project_paths(dst_project)
    for key in ("project", "colmap", "logs"):
        os.makedirs(dst[key], exist_ok=True)
    for name in ("database.db", "image_list.txt"):
        path = os.path.join(src["colmap"], name)
        if os.path.isfile(path):
            shutil.copy2(path, os.path.join(dst["colmap"], name))   # the solver writes to it
    for sub in ("frames", "photos"):
        if os.path.isdir(src[sub]):
            os.makedirs(dst[sub], exist_ok=True)
            for entry in os.scandir(src[sub]):
                if entry.is_file():
                    formats.link_or_copy(entry.path, os.path.join(dst[sub], entry.name))
    state = _load_state(src["state"])
    keep = {k: v for k, v in state.items() if k in FORK_STEPS}
    _save_state(dst["state"], keep)
    return dst_project


def _run_solver(job, ctx, mapper, db, out_dir, log):
    """Run one COLMAP mapper; returns the largest model folder or None."""
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    exe = "global_mapper" if mapper == "GLOBAL" else "mapper"
    try:
        job.run([_colmap(ctx), exe, "--database_path", db, "--image_path", ctx["images_dir"],
                 "--output_path", out_dir], log, env=ctx["ts"]["colmap_env"],
                parser=make_mapper_parser(ctx["num_images"]))
    except StepError as exc:
        job.add_line(f"{SOLVER_NAMES[mapper]} failed: {exc}")
        return None
    models = colmap_io.find_models(out_dir)
    return models[0][1] if models else None


def fold_spots(q, order, most=3):
    """Where a solved walk breaks, by photo: 'the walk breaks between A and B, ...' for
    solve_check.assess's jumps and backward turns (shooting order positions)."""
    names = {pos: name for name, pos in order.items()}

    def short(pos):
        name = names.get(pos, str(pos))
        return name.split(".", 1)[0] or name

    where = sorted({(a, b) for a, b, *_ in q.get("jump_list", [])} |
                   {(a, b) for a, b, *_ in q.get("backsteps", [])})
    if not where:
        return "the walk folds back"
    text = "; ".join(f"{short(a)} and {short(b)}" for a, b in where[:most])
    return "the walk breaks between " + text + (" ..." if len(where) > most else "")


def _note(job, log, text):
    """A summary line for the panel that also stays in the step log."""
    job.add_line(text)
    try:
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"[Photogram3D] {text}\n")
    except OSError:
        pass


MAPPING_FILE = "mapping.json"      # in colmap/sparse: what the chosen solve looked like (0.3.5)
WALK_MIN_PHOTOS = 8                # fewer photos than this: no solve along the shooting order
# consecutive photos missing inside the walk (not solved or dropped as weakly connected) from
# which the other solver is tried too: on GAZ-66 (0.3.9) the global solver dropped five frames
# in a row exactly where it folded the walk, and the fold passed the path check
GAP_RETRY = 3
# video up to this many frames: both solvers always run and the better solve is kept
# (solve_check.score), 0.3.10. On the rendered walk of test_camera_track the global solver's
# result changed with the random draws of COLMAP's match check, between right, folded, and bent
# by up to 1.3 m with no jump or backward turn (it passed the path check, 6,624 points, 0.59 px);
# the incremental solver was right every time (7,612 points, 0.29 px). The second solve takes
# seconds to a few minutes at this size. Photos keep the rule above: there the incremental
# solver comes first, and on the synthetic scene it once placed 47 photos with 9 of them wrong,
# which would outscore a right global solve of 44.
BOTH_SOLVERS_MAX = 300


def _exclusion(cfg, p):
    """The photos this run leaves out: the list Re-solve Without Selected gave the run when
    it started, else the one saved with the run (Build Mesh and Reconstruct continue a run
    with its own list; in 0.3.4 they lost it and solved the cameras again with every photo)."""
    names = cfg.get("exclude_images")
    if names is None:
        names = read_excluded(p["project"])
    return sorted(set(names or []))


def _short(name):
    return name.split(".", 1)[0] or name


class _Candidate:
    """One camera solve that step_mapping considers (``walk``: solved along the shooting
    order, from the pairs that passed walk.walk_pairs)."""

    def __init__(self, label, path, q, walk_solve=False):
        self.label, self.path, self.q, self.walk = label, path, q, walk_solve
        self.score = solve_check.score(q)


def _best(tried, prefer_walk=False):
    if prefer_walk:
        clean = [c for c in tried if c.walk and c.q["jumps"] == 0]
        if clean:
            return max(clean, key=lambda c: c.score)
    return max(tried, key=lambda c: c.score) if tried else None


def _read_mapping_info(p):
    try:
        with open(os.path.join(p["sparse"], MAPPING_FILE), "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _fold_left_unrepaired(ctx, order):
    """A finished solve with folds that the repairs have not seen yet (the solve along the
    shooting order, and since 0.3.11 the fold turned back): it is solved again once. The
    path is checked again with this version's check, so a fold an earlier check let pass is
    found too (GAZ-66, 0.3.9). Never for a solve guided by placed cameras, and never while
    continuing to the mesh from the crop stage: the crop box was placed on this solve's
    cloud."""
    if ctx.get("crop_continuation"):
        return False
    info = _read_mapping_info(ctx["paths"])
    if info is not None and (info.get("guided") is not None or (info.get("walk") and "turned" in info)):
        return False
    try:
        gravity = colmap_io.gravity_priors(ctx["paths"]["db"])
        return solve_check.assess(ctx["model_in"], order, gravity)["jumps"] > 0
    except (OSError, ValueError, KeyError):
        return bool(info and info.get("jumps"))


def _solve_along_walk(job, ctx, db, hint, order, n, log, attempt, tried):
    """The repair for folded camera paths (walk.py): orientations from the photo pairs
    taken shortly one after the other, a check of every photo pair against them, and the
    usual solvers again without the pairs that contradict them. The candidates go into
    ``tried``; returns a summary for mapping.json ("failed" says why when the pairs could
    not be judged: then no candidate is added)."""
    p = ctx["paths"]
    cams = colmap_io.read_model(hint)[0] if hint else None
    turn = walk.step_turn(hint, order) if hint else None
    width = walk.band_width(turn)
    names = walk.image_names(db)
    pos = {i: order[nm] for i, nm in names.items() if nm in order}
    band = walk.band_pairs(walk.pair_inliers(db), pos, width)
    _note(job, log, f"Solving along the shooting order: the cameras are first turned using only "
                    f"the photos taken up to {width} shots apart"
                    + (f" (about {turn * width:.0f} degrees)" if turn else ""))
    job.set_fraction(0.0, "solving along the shooting order")
    band_db = walk.copy_database(db, os.path.join(p["colmap"], "database_walk_band.db"),
                                 keep=band, cameras=cams)
    ref = os.path.join(p["sparse"], "walk_rotations")
    shutil.rmtree(ref, ignore_errors=True)
    os.makedirs(ref, exist_ok=True)
    summary = {"width": width, "turn_deg": turn, "contradicting": 0, "band_pairs": len(band)}
    try:
        job.run([_colmap(ctx), "rotation_averager", "--database_path", band_db, "--output_path", ref],
                log, env=ctx["ts"]["colmap_env"])
    except StepError as exc:
        _note(job, log, f"The cameras could not be turned along the shooting order ({exc})")
        return dict(summary, failed="rotation averaging")
    placed = colmap_io.count_registered(ref) if os.path.isfile(os.path.join(ref, "images.bin")) else 0
    jumps = walk.rotation_jumps(ref, order) if placed else []
    if placed < MIN_REGISTERED_RATIO * n or jumps:
        _note(job, log, f"Turning the cameras along the shooting order placed {placed}/{n} photos"
                        + (f" with {len(jumps)} sudden turn(s)" if jumps else "")
                        + ": not reliable enough to check the photo pairs against")
        return dict(summary, failed="orientations", placed=placed, rotation_jumps=len(jumps))
    keep, rep = walk.walk_pairs(db, ref, order, width)
    bad = rep["contradicting"]
    if bad:
        _note(job, log, f"{len(bad)} photo pair(s) contradict the walk: look-alike views of "
                        "different sides were matched to each other (e.g. "
                        + "; ".join(f"{_short(a)} and {_short(b)}" for a, b, *_ in bad[:3])
                        + "). Solving again without them")
    else:
        _note(job, log, "No photo pair contradicts the walk; solving again with the checked pairs")
    clean = walk.copy_database(db, os.path.join(p["colmap"], "database_walk.db"), keep=keep,
                               cameras=cams)
    for mapper in ("GLOBAL", "INCREMENTAL"):
        attempt(mapper, db=clean, walk_solve=True, label=SOLVER_NAMES[mapper] + " along the shooting order",
                folder=os.path.join(p["sparse"], "walk_" + mapper.lower()))
    summary.update(placed=placed, contradicting=len(bad), unchecked=rep["unchecked"],
                   checked=rep["checked"], examples=[[a, b, round(e, 2), m] for a, b, e, m in bad[:20]])
    return summary


def _check_against(job, ctx, db, guide, order, log, name, what, turned=()):
    """The photo pairs that agree with the cameras of ``guide`` (walk.walk_pairs), less
    those that link the ``turned`` photos with the rest from far apart in the walk
    (walk.far_links), as a copy of the match database (colmap/``name``); returns (database,
    pairs left out as [(photo a, photo b, error, matches)], far links left out)."""
    p = ctx["paths"]
    width = walk.band_width(walk.step_turn(guide, order))
    keep, rep = walk.walk_pairs(db, guide, order, width)
    bad = rep["contradicting"]
    far = walk.far_links(keep, walk.image_names(db), order, set(turned), width) if turned else set()
    if bad:
        _note(job, log, f"{len(bad)} photo pair(s) disagree with {what} by more than "
                        f"{math.degrees(walk.ROTATION_LIMIT):.0f} degrees (e.g. "
                        + "; ".join(f"{_short(a)} and {_short(b)}" for a, b, *_ in bad[:3])
                        + "): left out")
    else:
        _note(job, log, f"No photo pair disagrees with {what}")
    if far:
        _note(job, log, f"{len(far)} more pair(s) link turned photos with the rest from more than "
                        f"{width} shots apart: left out, the turned part joins the rest along the walk")
    cams = colmap_io.read_model(guide)[0]
    clean = walk.copy_database(db, os.path.join(p["colmap"], name), keep=keep - far, cameras=cams)
    return clean, bad, far


def _remove_database(path):
    """A copy of the match database the solvers no longer need (as large as the original)."""
    for suffix in ("", "-wal", "-shm"):
        try:
            os.remove(path + suffix)
        except OSError:
            pass


def _solve_turned(job, ctx, db, base, order, gravity, n, log, attempt):
    """The repair when every solve still folds (0.3.11): the stretches of the walk between
    the folds of the best solve are turned by half a turn about the vertical as a guess
    (walk.turned_guide), the photo pairs that disagree with it are left out, and both
    solvers run again; the turned part joins the rest only along the walk (walk.far_links).
    A symmetric object's look-alike sides fold the walk that way: on the Mi-8 photos the
    starboard stretch lay over the port side, facing the same way, and neither solver nor the
    solve along the shooting order (its bands of nearby photos were not connected) got it
    right; turned back, the global solver placed 256 of 260 photos without a fold and the
    incremental one 246. Returns a summary for mapping.json."""
    p = ctx["paths"]
    folds = sorted({(a, b) for a, b, *_ in base.q["jump_list"]} | {(a, b) for a, b, *_ in base.q["backsteps"]})
    if len(folds) > walk.TURN_MAX_FOLDS:
        _note(job, log, f"The {base.label.lower()} folds in {len(folds)} places: too many to guess which "
                        "parts of the walk to turn back")
        return {"failed": "too many folds", "folds": len(folds)}
    guide = os.path.join(p["sparse"], "turned_guide")
    shutil.rmtree(guide, ignore_errors=True)
    turned = walk.turned_guide(base.path, order, folds, guide, gravity)
    if not turned:
        return {"failed": "nothing to turn", "folds": len(folds)}
    _note(job, log, f"Turning back the {len(turned)} camera(s) between the folds of the "
                    f"{base.label.lower()} (half a turn about the vertical: a side laid over the "
                    "opposite side) and checking every photo pair against them")
    clean, bad, far = _check_against(job, ctx, db, guide, order, log, "database_turned.db",
                                     "the turned cameras", turned)
    for mapper in ("GLOBAL", "INCREMENTAL"):
        attempt(mapper, db=clean, walk_solve=True, label=SOLVER_NAMES[mapper] + " with the fold turned back",
                folder=os.path.join(p["sparse"], "turned_" + mapper.lower()))
    _remove_database(clean)
    return {"base": base.label, "turned": len(turned), "folds": [list(f) for f in folds],
            "contradicting": len(bad), "far_links": len(far),
            "examples": [[a, b, round(e, 2), m] for a, b, e, m in bad[:20]]}


GUIDE_TURNED = "turned.txt"     # in the guide folder: the photos whose cameras were turned by hand
GUIDE_TURNED_DEG = 20.0         # a camera turned more than this from where the solve put it is turned


def has_guide(p):
    return os.path.isfile(os.path.join(p["guide"], "images.txt"))


def turned_from(R, image):
    """Whether a camera's rotation ``R`` is turned more than GUIDE_TURNED_DEG from where the
    solve put it (``image``: colmap_io.read_model's entry)."""
    import numpy as np
    solved = np.asarray(colmap_io.qvec_to_rotmat(image["qvec"]), float)
    return math.degrees(walk._angle(np.asarray(R, float) @ solved.T)) > GUIDE_TURNED_DEG


def write_guide(src_project, dst_project, poses):
    """Re-solve From Cameras: the cameras as placed in Blender (``poses``: {photo name:
    (world-to-camera rotation, centre)}, importer.scene_camera_poses) written as the new run's
    guide, with the lens and photo ids of the run they come from, and the photos whose camera
    was turned by hand (more than GUIDE_TURNED_DEG from the solve). Returns (cameras, turned
    photo names)."""
    import numpy as np
    src = project_paths(src_project)["model"]
    cams, images = colmap_io.read_model(src)
    out, turned = [], []
    for iid, im in sorted(images.items()):
        pose = poses.get(im["name"])
        if pose is None:
            continue
        R = np.asarray(pose[0], float)
        if turned_from(R, im):
            turned.append(im["name"])
        out.append((iid, im["name"], im["camera_id"], R, pose[1]))
    dst = project_paths(dst_project)["guide"]
    shutil.rmtree(dst, ignore_errors=True)
    colmap_io.write_text_model(dst, cams, out)
    with open(os.path.join(dst, GUIDE_TURNED), "w", encoding="utf-8") as f:
        f.write("".join(n + "\n" for n in turned))
    return len(out), turned


def read_turned(p):
    try:
        with open(os.path.join(p["guide"], GUIDE_TURNED), "r", encoding="utf-8") as f:
            return {ln.strip() for ln in f if ln.strip()}
    except OSError:
        return set()


def step_mapping(job, ctx):
    """Solve the cameras, check the result, and try the other solver when it is wrong.

    Each candidate is first cleaned of weakly connected images (_drop_weak_images),
    then its camera path in shooting order is checked (solve_check.assess). On the
    WRX photos the global solver folded a two-lap walk onto itself with a low
    reprojection error, so only the path revealed it; on the synthetic scene the
    incremental solver mirrored a block of cameras, which the turn check caught,
    and the cleaned global result was right. When the best candidate still folds (a
    symmetric object), the cameras are solved along the shooting order (walk.py, 0.3.5);
    Solver: Shooting Order does that every time. Best candidate: solve_check.score.
    """
    p, cfg = ctx["paths"], ctx["cfg"]
    key = "mapping"
    exclude = _exclusion(cfg, p)
    mapper_setting = cfg.get("mapper") or "AUTO"
    ctx["sig_chain"][key] = {"prev": ctx["sig_chain"].get("matching"), "mapper": mapper_setting,
                             "weak_filter": WEAK_IMAGE_RATIO, "path_check": 2}
    if exclude:
        ctx["sig_chain"][key]["exclude"] = exclude
    guide = p["guide"] if has_guide(p) else None
    if guide:
        with open(os.path.join(guide, "images.txt"), "r", encoding="utf-8") as f:
            ctx["sig_chain"][key]["guide"] = stable_hash([f.read(), sorted(read_turned(p))])
    order = solve_check.shooting_order(_db_image_names(p["db"])) if os.path.isfile(p["db"]) else {}
    gravity = colmap_io.gravity_priors(p["db"])        # photos stored on their side: their up
    if _resumable(job, ctx, key, [os.path.join(p["sparse"], "chosen.txt")]):
        ctx["model_in"] = read_chosen_model(p)
        if os.path.isdir(ctx["model_in"]):
            if not _fold_left_unrepaired(ctx, order):
                return
            job.add_line("The camera path of this run folds back on itself; solving the cameras "
                         "again")
    _invalidate_from(ctx, key)
    n = ctx["num_images"]
    for old in (p["sparse"], p["sparse_inc"], os.path.join(p["colmap"], "filtered")):
        shutil.rmtree(old, ignore_errors=True)
    os.makedirs(p["sparse"], exist_ok=True)
    log = _log(ctx, key)
    db = p["db"]
    if exclude:
        db = _db_without(p, exclude)
        n -= len(exclude)
        _note(job, log, f"Solving without {len(exclude)} photo(s): " + ", ".join(exclude[:8])
              + (" ..." if len(exclude) > 8 else ""))
    _write_excluded(p, exclude)
    first = mapper_setting if mapper_setting in ("GLOBAL", "INCREMENTAL") else "AUTO"
    if first == "AUTO":
        first = "GLOBAL" if cfg["input_mode"] == "VIDEO" or n > EXHAUSTIVE_LIMIT else "INCREMENTAL"
    second = "INCREMENTAL" if first == "GLOBAL" else "GLOBAL"
    tried = []

    def attempt(mapper, db=db, label=None, folder=None, walk_solve=False):
        label = label or SOLVER_NAMES[mapper]
        folder = folder or os.path.join(p["sparse"], mapper.lower())
        path = _run_solver(job, ctx, mapper, db, folder, log)
        if path is None:
            return None
        path = _drop_weak_images(job, ctx, path, log, out=folder + "_clean",
                                 db=db if walk_solve else None)
        q = solve_check.assess(path, order, gravity)
        tried.append(_Candidate(label, path, q, walk_solve))
        _note(job, log, f"{label}: {q['registered']}/{n} photos, {q['points']} points "
              f"({q['observations']} observations), "
              + (f"{q['jumps']} fold(s) in the camera path ({fold_spots(q, order)})"
                 if q["jumps"] else "camera path consistent"))
        return q

    walk_info = turn_info = guide_info = None
    if guide:
        # Re-solve From Cameras: the cameras as the user left them are the guide
        _note(job, log, "Solving along the cameras as placed in Blender: every photo pair is checked "
                        "against their orientations")
        turned = read_turned(p)
        clean, bad, far = _check_against(job, ctx, db, guide, order, log, "database_guided.db",
                                         "the cameras as placed", turned)
        guide_info = {"contradicting": len(bad), "far_links": len(far), "turned": len(turned),
                      "examples": [[a, b, round(e, 2), m] for a, b, e, m in bad[:20]]}
        for mapper in (first, second):
            attempt(mapper, db=clean, walk_solve=True, label=SOLVER_NAMES[mapper] + " along the placed cameras",
                    folder=os.path.join(p["sparse"], "guided_" + mapper.lower()))
        _remove_database(clean)
    else:
        q = attempt(first)
        gap = q.get("gap", 0) if q is not None else 0
        both = cfg["input_mode"] == "VIDEO" and n <= BOTH_SOLVERS_MAX
        if both or not (q is not None and q["jumps"] == 0 and q["registered"] >= MIN_REGISTERED_RATIO * n
                        and gap < GAP_RETRY):
            if q is not None and q["jumps"]:
                _note(job, log, "Some photos taken seconds apart were placed far apart or on top of "
                      "earlier ones: views from different sides were matched to each other "
                      "(typical for symmetric objects). Trying the " + SOLVER_NAMES[second].lower())
            elif q is not None and gap >= GAP_RETRY:
                _note(job, log, f"{gap} photos taken one after another are missing from this solve: the "
                      "walk may break there. Trying the " + SOLVER_NAMES[second].lower() + " as well")
            elif q is not None:
                _note(job, log, "Solving with the " + SOLVER_NAMES[second].lower() + " as well, to keep "
                      "the better of the two")
            attempt(second)
    best = _best(tried)
    # along the shooting order: asked for, or the best solve folds the walk (look-alike
    # sides) or places too few photos (a fold may have cost the rest)
    if not guide and n >= WALK_MIN_PHOTOS and (mapper_setting == "WALK" or best is None or best.q["jumps"]
                                               or best.q["registered"] < MIN_REGISTERED_RATIO * n):
        hint = max(tried, key=lambda c: c.q["registered"]).path if tried else None
        walk_info = _solve_along_walk(job, ctx, db, hint, order, n, log, attempt, tried)
    # every solve still folds: turn back what lies between the folds of the best one (0.3.11)
    best = _best(tried)
    if n >= WALK_MIN_PHOTOS and best is not None and best.q["jumps"]:
        turn_info = _solve_turned(job, ctx, db, best, order, gravity, n, log, attempt)
    if not tried:
        raise StepError("COLMAP could not reconstruct the cameras. Typical causes: too little "
                        "overlap between photos, motion blur, or a reflective/featureless subject.")
    best = _best(tried, prefer_walk=mapper_setting == "WALK")
    label, path, q = best.label, best.path, best.q
    count = q["registered"]
    _note(job, log, f"Using: {label} ({count}/{n} photos)")
    per_photo = q["points"] / max(1, count)
    if q["jumps"]:
        msg = (f"The camera path still has {q['jumps']} fold(s) ({fold_spots(q, order)}): parts "
               "of the scan may be doubled. Turn the misplaced cameras the way they should face "
               "(a side laid over the other side: R Z 180) and use Re-solve From Cameras, or "
               "select them and use Re-solve Without Selected, or shoot again with a textured, "
               "asymmetric surface (a newspaper) under the object")
        _note(job, log, "WARNING: " + msg)
        ctx.setdefault("result", {}).setdefault("warnings", []).append(msg)
    elif best.walk and guide_info is not None and best.label.endswith("along the placed cameras"):
        msg = (f"Solved along the cameras as placed: {guide_info['contradicting']} photo pair(s) that "
               "disagreed with them were left out")
        ctx.setdefault("result", {}).setdefault("notes", []).append(msg)
    elif best.walk and turn_info and best.label.endswith("with the fold turned back"):
        msg = (f"The solve folded; the part between the folds was turned back and "
               f"{turn_info['contradicting']} photo pair(s) that matched look-alike sides of the "
               "object to each other were left out")
        ctx.setdefault("result", {}).setdefault("notes", []).append(msg)
    elif best.walk and walk_info and walk_info["contradicting"]:
        msg = (f"Solved along the shooting order: {walk_info['contradicting']} photo pair(s) that "
               "matched look-alike sides of the object to each other were left out")
        ctx.setdefault("result", {}).setdefault("notes", []).append(msg)
    if per_photo < WEAK_POINTS_PER_PHOTO:
        msg = (f"Only {per_photo:.0f} 3D points per photo (well matched sets have 150 or more): "
               "the photos share few features. Use High or Ultra (finer features) or AI Features, "
               "and give the object a textured background")
        _note(job, log, "WARNING: " + msg)
        ctx.setdefault("result", {}).setdefault("warnings", []).append(msg)
    if count < 0.5 * n:
        _note(job, log, "WARNING: fewer than half of the images were registered; "
              "the mesh will be incomplete")
    ctx["model_in"] = path
    ctx["registered"] = count
    with open(os.path.join(p["sparse"], "chosen.txt"), "w", encoding="utf-8") as f:
        f.write(os.path.relpath(path, p["project"]))     # relative: the project can be moved
    info = {"chosen": label, "registered": count, "jumps": q["jumps"], "walk": walk_info is not None,
            "candidates": [{"label": c.label, "registered": c.q["registered"], "points": c.q["points"],
                            "observations": c.q["observations"], "jumps": c.q["jumps"]} for c in tried]}
    if walk_info:
        info["walk_check"] = walk_info
    if turn_info is not None:
        info["turned"] = turn_info
    if guide_info is not None:
        info["guided"] = guide_info
    with open(os.path.join(p["sparse"], MAPPING_FILE), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)
    _mark_done(ctx, key)


_RE_ABS_PATH = re.compile(r"^(?:[A-Za-z]:)?[\\/]")


def read_chosen_model(p):
    """The camera model the mapping step chose, as a folder of this project.

    chosen.txt holds a path relative to the project folder; before 0.3.2 it held an
    absolute one, which went stale when the project was moved or copied (the cameras were
    then solved again and every later step, depth maps included, was redone). Such a path
    is re-rooted: its longest tail that exists in this project wins."""
    try:
        with open(os.path.join(p["sparse"], "chosen.txt"), "r", encoding="utf-8") as f:
            stored = f.read().strip()
    except OSError:
        return ""
    if not stored:
        return ""
    if not _RE_ABS_PATH.match(stored):
        return os.path.join(p["project"], stored)
    parts = [x for x in re.split(r"[\\/]+", stored) if x]
    for k in range(1, len(parts)):
        candidate = os.path.join(p["project"], *parts[k:])
        if os.path.isdir(candidate):
            return candidate
    return stored


def step_orient(job, ctx):
    p, cfg = ctx["paths"], ctx["cfg"]
    key = "orient"
    ctx["sig_chain"][key] = {"prev": ctx["sig_chain"].get("mapping"), "align": cfg["align_up"]}
    if _resumable(job, ctx, key, [os.path.join(p["model"], "images.bin")]):
        return
    _invalidate_from(ctx, key)
    shutil.rmtree(p["model"], ignore_errors=True)
    os.makedirs(p["model"], exist_ok=True)
    ok = False
    gravity = colmap_io.gravity_priors(p["db"]) if cfg["align_up"] else {}
    sideways = sum(1 for g in gravity.values() if colmap_io.rot90_from_gravity((g[0], g[1])))
    if cfg["align_up"] and sideways:
        # COLMAP's IMAGE-ORIENTATION takes every stored picture's y axis as down: level for a
        # portrait photo of a phone, which is stored on its side (0.3.9)
        try:
            ok = _align_with_gravity(job, ctx, gravity, sideways, _log(ctx, key))
        except (StepError, OSError, ValueError) as exc:
            job.add_line(f"orientation alignment failed ({exc}); keeping the original orientation")
    elif cfg["align_up"]:
        try:
            job.run([_colmap(ctx), "model_orientation_aligner", "--image_path", ctx["images_dir"],
                     "--input_path", ctx["model_in"], "--output_path", p["model"],
                     "--method", "IMAGE-ORIENTATION"],
                    _log(ctx, key), env=ctx["ts"]["colmap_env"])
            ok = os.path.isfile(os.path.join(p["model"], "images.bin"))
        except StepError as exc:
            job.add_line(f"orientation alignment failed ({exc}); keeping the original orientation")
    if not ok:
        for name in os.listdir(ctx["model_in"]):
            src = os.path.join(ctx["model_in"], name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(p["model"], name))
    _new_solve(job, ctx, _log(ctx, key))
    _mark_done(ctx, key)


def solve_id(p):
    """The id of the run's current camera solve (common.SOLVE_FILE); None for a run solved
    before 0.3.5."""
    try:
        with open(current(os.path.join(p["model"], SOLVE_FILE)), "r", encoding="utf-8") as f:
            return str(json.load(f)["id"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


# ---------------------------------------------------------------------------
# The camera on every frame of a video (0.3.9, camera_track.py)
# ---------------------------------------------------------------------------

TRACK_EVERY = {"EVERY": 1, "HALF": 2}
_RE_REGISTERING = re.compile(r"Registering image #\d+ \((\d+)\)")


def step_track(job, ctx):
    """Video: the camera on every frame (camera_track.py). It never stops the run: when the
    track fails, the tracked camera follows the solved frames, with a note."""
    p, cfg = ctx["paths"], ctx["cfg"]
    key = "track"
    every = TRACK_EVERY.get(cfg.get("video_track", "EVERY"), 1)
    t = camera_track.paths(p["project"])
    ctx["sig_chain"][key] = {"prev": ctx["sig_chain"].get("orient"), "every": every, "v": 1}
    if _resumable(job, ctx, key, [t["json"]]):
        ctx["result"]["track_json"] = t["json"]
        return
    log = _log(ctx, key)
    try:
        track_video(job, ctx, every, log)
    except (StepError, OSError, ValueError, KeyError, IndexError, sqlite3.Error) as exc:
        msg = (f"The camera could not be tracked on every frame ({exc}); the tracked camera follows "
               "the solved frames only")
        _note(job, log, msg)
        ctx["result"].setdefault("notes", []).append(msg)
        os.makedirs(t["root"], exist_ok=True)
        with open(t["json"], "w", encoding="utf-8") as f:
            json.dump({"version": 1, "failed": str(exc), "frames": []}, f)
    finally:
        _track_cleanup(t)
    ctx["result"]["track_json"] = t["json"]
    _mark_done(ctx, key)


def _track_cleanup(t):
    """Keep track.json and the logs: the frames and the tracking database are large."""
    for name in ("frames", "seed", "tri", "reg", "seed2", "tri2", "reg2"):
        shutil.rmtree(os.path.join(t["root"], name), ignore_errors=True)
    for name in ("track.db", "track.db-shm", "track.db-wal", "pairs.txt"):
        try:
            os.remove(os.path.join(t["root"], name))
        except OSError:
            pass


def track_video(job, ctx, every, log, ranges=(0.0, 0.25, 0.4, 0.75, 0.85, 1.0)):
    """Place the camera on every ``every``-th frame of the video part the project used.
    Returns the track.json data. ``ranges``: progress at the start of each stage."""
    p, ts = ctx["paths"], ctx["ts"]
    t = camera_track.paths(p["project"])
    meta = video.load_frames_json(p["frames"])
    if not meta or not meta.get("frames"):
        raise StepError("frames.json is missing")
    vpath = ctx["cfg"].get("video_path") or meta.get("video") or ""
    if not os.path.isfile(vpath):
        raise StepError(f"the video is not at {vpath}")
    if not ts.get("ffmpeg") or not ts.get("ffprobe"):
        raise StepError("FFmpeg is not installed")
    cams, images = colmap_io.read_model(p["model"])
    solved = {im["name"] for im in images.values()}
    if len(solved) < 2:
        raise StepError("fewer than two frames were solved")
    _track_cleanup(t)
    os.makedirs(t["frames"], exist_ok=True)
    info = video.probe(ts["ffprobe"], vpath)
    for k in ("clip_fps", "stream_start"):
        meta.setdefault(k, info.get(k))
    start = float(meta.get("start") or 0.0)
    end = float(meta.get("end") or 0.0) or video.media_end(info)
    in_range = max(1, int(round((end - start) * (meta.get("clip_fps") or info["fps"]))))
    if len(meta["frames"]) >= 0.98 * in_range:
        # the reconstruction used every frame (Frames: Every Frame): they are the track
        frames = [(f["file"], int(f["n"]), float(f["t"])) for f in meta["frames"]]
        poses = {im["name"]: (list(im["qvec"]), list(im["tvec"])) for im in images.values()}
        data = camera_track.save(t["json"], meta, frames, poses, {n: n for n in solved}, 1)
        _note(job, log, f"The reconstruction solved every frame: the camera track is its {data['placed']:,} "
                        "solved frames")
        return data

    # 1. every frame of the range, named by its frame number like the solved frames
    cam = next(iter(cams.values()))
    size = (int(cam["width"]), int(cam["height"]))     # the solved frames' size (Max Photo Size)
    frames = video.write_range(job, ts["ffmpeg"], vpath, t["frames"], info, start, end, log, size=size,
                               progress=ranges[0:2])
    frames = [f for f in frames if f[1] % every == 0 or f[0] in solved]
    keep = {f[0] for f in frames}
    for name in os.listdir(t["frames"]):
        if name not in keep:
            os.remove(os.path.join(t["frames"], name))
    keyframes = {f[0]: f[0] for f in frames if f[0] in solved}
    if len(keyframes) < 2:
        raise StepError(f"only {len(keyframes)} solved frame(s) were found again in the video")
    sizes = {formats.image_size(os.path.join(t["frames"], f[0])) for f in frames[:3]}
    if sizes != {size}:
        raise StepError(f"the frames came out {sizes}, the solved ones are {size[0]}x{size[1]}")
    _note(job, log, f"Tracking {len(frames):,} frames ({len(keyframes)} of them solved)")

    # 2. SIFT features with the solved lens, matches along the video and to the solved frames
    with open(os.path.join(t["root"], "list.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(n for n, _, _ in frames) + "\n")
    _, params = camera_track.camera_line(1, cam, int(cam["width"]), int(cam["height"]))
    args = [_colmap(ctx), "feature_extractor", "--database_path", t["db"], "--image_path", t["frames"],
            "--image_list_path", os.path.join(t["root"], "list.txt"),
            "--ImageReader.camera_model", cam["model"], "--ImageReader.single_camera", "1",
            "--ImageReader.camera_params", ",".join(f"{v:.17g}" for v in params),
            "--FeatureExtraction.type", "SIFT", "--FeatureExtraction.max_image_size", str(camera_track.SIFT_MAX_SIDE),
            "--SiftExtraction.max_num_features", str(camera_track.SIFT_FEATURES)]
    lo, hi = ranges[1:3]
    _run_with_gpu_fallback(job, ctx, args, "--FeatureExtraction.use_gpu", log,
                           _scaled_parser(job, parse_feature, lo, hi, "features"))
    names = [n for n, _, _ in frames]
    keys = [i for i, n in enumerate(names) if n in keyframes]
    n_pairs = camera_track.write_pairs(t["pairs"], camera_track.make_pairs(names, keys))
    lo, hi = ranges[2:4]
    _run_with_gpu_fallback(job, ctx, [_colmap(ctx), "matches_importer", "--database_path", t["db"],
                                      "--match_list_path", t["pairs"], "--match_type", "pairs",
                                      "--FeatureMatching.type", "SIFT_BRUTEFORCE"],
                           "--FeatureMatching.use_gpu", log, _scaled_parser(job, parse_matching, lo, hi,
                                                                           f"matching {n_pairs:,} pairs"))

    # 3. the solved frames with fixed poses: points from their matches, then every frame on them
    camera_track.write_seed(t["seed"], p["model"], keyframes, t["db"], size)
    fixed = ["--Mapper.ba_refine_focal_length", "0", "--Mapper.ba_refine_principal_point", "0",
             "--Mapper.ba_refine_extra_params", "0"]
    total = len(frames)
    poses = {}
    for rnd in range(2):
        src, tri, reg = ((t["seed"], t["tri"], t["reg"]) if rnd == 0 else
                         (os.path.join(t["root"], n) for n in ("seed2", "tri2", "reg2")))
        if rnd:
            if len(poses) >= total:
                break
            # the frames still missing: points from the placed frames next to them (and the
            # solved ones), no bundle adjustment of those points; the placed frames stay
            camera_track.write_posed(src, t["seed"], {n: poses[n] for n in camera_track.near_missing(
                names, poses, keyframes)}, t["db"])
        os.makedirs(tri, exist_ok=True)
        os.makedirs(reg, exist_ok=True)
        lo, hi = ranges[3 + rnd:5 + rnd] if rnd == 0 else (ranges[4], ranges[5])
        job.set_fraction(lo, "3D points from the solved frames" if rnd == 0 else
                         f"more 3D points for the {total - len(poses):,} frames left")
        no_ba = ["--Mapper.ba_global_max_refinements", "0"] if rnd else []
        job.run([_colmap(ctx), "point_triangulator", "--database_path", t["db"], "--image_path", t["frames"],
                 "--input_path", src, "--output_path", tri, "--clear_points", "1",
                 "--refine_intrinsics", "0"] + no_ba, log, env=ts["colmap_env"])
        base = len(read_registered(tri))

        def reg_parser(line, base=base, lo=lo, hi=hi):
            m = _RE_REGISTERING.search(line)
            if m:
                done = int(m.group(1)) - base
                return lo + (hi - lo) * min(1.0, done / max(1, total - base)), f"frame {int(m.group(1))}/{total}"
            return None
        job.run([_colmap(ctx), "image_registrator", "--database_path", t["db"], "--input_path", tri,
                 "--output_path", reg] + fixed, log, env=ts["colmap_env"], parser=reg_parser)
        before = len(poses)
        for name, pose in camera_track.read_poses(reg).items():
            poses.setdefault(name, pose)
        if rnd:
            _note(job, log, f"A second round placed {len(poses) - before:,} more frame(s)")
    missing = [n for n in keyframes if n not in poses]
    if missing:
        raise StepError(f"{len(missing)} solved frame(s) lost their place: {', '.join(missing[:3])}")
    data = camera_track.save(t["json"], meta, frames, poses, keyframes, every)
    placed = data["placed"]
    gaps = _longest_gap(frames, poses)
    _note(job, log, f"Camera tracked on {placed:,} of {total:,} frames"
                    + (f" (every {every}nd frame)" if every == 2 else "")
                    + (f"; the longest stretch without a place is {gaps} frames, interpolated" if gaps > 1 else ""))
    return data


def build_track_job(project, video_path, every, ts, hw, use_gpu=True, name="Camera track"):
    """A job that only tracks the camera on every frame of a video project solved before:
    projects made before 0.3.9 or with Camera Track set to Solved Frames. Its result has
    stage "track" (the panel then keys the tracked camera again instead of importing)."""
    paths = project_paths(project)
    os.makedirs(paths["logs"], exist_ok=True)
    cfg = {"project_dir": project, "video_path": video_path, "input_mode": "VIDEO", "use_gpu": use_gpu,
           "video_track": "HALF" if every == 2 else "EVERY"}
    ctx = {"cfg": cfg, "ts": ts, "hw": hw, "paths": paths, "use_gpu": bool(hw.get("gpus")) and use_gpu,
           "state": _load_state(paths["state"]), "sig_chain": {}, "step_order": ["track"],
           "result": {"stage": "track", "project": project}}

    def step(job, ctx):
        log = os.path.join(paths["logs"], "track.log")
        try:
            data = track_video(job, ctx, every, log)
        finally:
            _track_cleanup(camera_track.paths(project))
        ctx["result"]["track_json"] = camera_track.paths(project)["json"]
        with job.lock:
            job.result.update(ctx["result"], placed=data["placed"], total=data["total"])

    job = Job(name, [Step("track", "Tracking the camera on every frame", step, 1.0)], project)
    return job, ctx


def read_registered(model_dir):
    try:
        return camera_track.read_poses(model_dir) if os.path.isfile(os.path.join(model_dir, "images.bin")) else {}
    except (OSError, ValueError):
        return {}


def _longest_gap(frames, poses):
    run = best = 0
    for name, _, _ in frames:
        run = 0 if name in poses else run + 1
        best = max(best, run)
    return best


def _scaled_parser(job, parser, lo, hi, what):
    """A COLMAP progress parser mapped to [lo, hi] of the step."""
    def scaled(line):
        r = parser(line)
        if r is None:
            return None
        return lo + (hi - lo) * r[0], f"{what}: {r[1]}"
    return scaled


def _align_with_gravity(job, ctx, gravity, sideways, log):
    """Up from each photo's own down (its EXIF orientation, which COLMAP keeps as gravity):
    COLMAP's IMAGE-ORIENTATION consensus with those directions, applied with
    model_transformer. Returns True when the aligned model was written."""
    p = ctx["paths"]
    axes = colmap_io.downward_axes(ctx["model_in"], gravity)
    down = colmap_io.consensus_axis(axes)
    if not down or not any(down):
        raise StepError("no consensus on the photos' down direction")
    q = colmap_io.rotation_between(down, (0.0, 1.0, 0.0))          # COLMAP: +y is down
    tform = os.path.join(p["colmap"], "up_from_photos.txt")
    with open(tform, "w", encoding="utf-8") as f:
        f.write("1 %.17g %.17g %.17g %.17g 0 0 0\n" % q)        # Sim3d: scale, qw qx qy qz, t
    job.run([_colmap(ctx), "model_transformer", "--input_path", ctx["model_in"], "--output_path", p["model"],
             "--transform_path", tform], log, env=ctx["ts"]["colmap_env"])
    angle = math.degrees(2.0 * math.acos(max(-1.0, min(1.0, q[0]))))
    _note(job, log, f"Up-axis from the photos' EXIF orientation ({sideways} photo(s) stored on their side, "
                    f"turned {angle:.0f} degrees)")
    return os.path.isfile(os.path.join(p["model"], "images.bin"))


def _new_solve(job, ctx, log):
    """New cameras: a new solve id, and the crop box and alignment saved for the run's
    earlier solve are dropped (they were placed in that solve's coordinates)."""
    import uuid
    p = ctx["paths"]
    stale = [path for path in (os.path.join(p["mvs"], CROP_ROI), current(os.path.join(p["project"], TRANSFORM_FILE)))
             if os.path.isfile(path)]
    for path in stale:
        try:
            os.remove(path)
        except OSError:
            pass
    if stale:
        _note(job, log, "The cameras were solved again: the crop box and the alignment made for the "
                        "earlier solve no longer fit and were reset")
    sid = uuid.uuid4().hex
    with open(current(os.path.join(p["model"], SOLVE_FILE)), "w", encoding="utf-8") as f:
        json.dump({"id": sid, "time": time.time()}, f)
    ctx["solve_id"] = sid


# Spread photos (0.3.4). A video can give thousands of solved frames a fraction of a degree
# apart: depth maps for all of them would take hours and add little. Above
# DENSE_SUBSET_MIN photos the depth maps use photos at least DENSE_VIEW_ANGLE apart (the
# camera path, the tracked camera and the static cameras keep every frame); above
# TEXTURE_SUBSET_MIN photos the texture uses photos TEXTURE_VIEW_ANGLE apart.
DENSE_SUBSET_MIN = 300
DENSE_VIEW_ANGLE = 5.0
TEXTURE_SUBSET_MIN = 60
TEXTURE_VIEW_ANGLE = 20.0


def _reduced_model(job, ctx, model, keep, out, log):
    """A copy of ``model`` with only the photos in ``keep`` (COLMAP's image_deleter)."""
    _, images = colmap_io.read_model(model)
    keep = set(keep)
    drop = sorted(im["name"] for im in images.values() if im["name"] not in keep)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out, exist_ok=True)
    names_path = out.rstrip("\\/") + "_dropped.txt"
    with open(names_path, "w", encoding="utf-8") as f:
        f.write("\n".join(drop) + "\n")
    job.run([_colmap(ctx), "image_deleter", "--input_path", model, "--output_path", out,
             "--image_names_path", names_path], log, env=ctx["ts"]["colmap_env"])
    if colmap_io.count_registered(out) != len(images) - len(drop):
        raise StepError(f"image_deleter kept {colmap_io.count_registered(out)} photos, "
                        f"expected {len(images) - len(drop)}")
    return out


def step_even(job, ctx):
    """Even out exposure and vignetting across the undistorted photos (prepare_images.py), for
    the texture and the splats. The depth maps use the photos as they are: their matching
    compares patches with their own mean and spread taken out, which a photo's exposure or a
    smooth vignetting does not change. So turning this on for a finished run redoes only the
    texture (0.3.12; until then the photos were evened before the depth maps, and turning it
    on redid them as well: four hours of the Mi-8 run at Ultra, against 40 minutes of texture)."""
    import numpy as np
    p, cfg = ctx["paths"], ctx["cfg"]
    key = "even"
    sharpen = round(float(cfg.get("sharpen") or 0.0), 3)
    ctx["sig_chain"][key] = {"prev": ctx["sig_chain"].get("undistort"), "v": 2, "sharpen": sharpen}
    out = p["prepared"]
    src = os.path.join(p["dense"], "images")
    points = colmap_io.image_points(os.path.join(p["dense"], "sparse"))
    names = sorted(points)
    if _resumable(job, ctx, key, [out]) and all(os.path.isfile(os.path.join(out, n)) for n in names):
        ctx["texture_images_dir"] = out
        return
    _invalidate_from(ctx, key)
    log = _log(ctx, key)
    site = formats.paths(cfg.get("tools_root") or p["project"])["site"]
    python = cfg.get("python") or sys.executable
    if formats.runtime_info(site) is None:
        formats.install_runtime(job, python, site, log)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    ids = np.unique(np.concatenate([points[n][1] for n in names])) if names else np.zeros(0, np.int64)
    image, point, xs, ys = [], [], [], []
    for i, n in enumerate(names):
        xy, pid = points[n]
        image.append(np.full(len(pid), i, np.int32))
        point.append(np.searchsorted(ids, pid).astype(np.int32))
        xs.append(xy[:, 0].astype(np.float32))
        ys.append(xy[:, 1].astype(np.float32))
    obs = os.path.join(out, "observations.npz")
    np.savez(obs, image=np.concatenate(image), point=np.concatenate(point), x=np.concatenate(xs),
             y=np.concatenate(ys))
    jobs = os.path.join(out, "prepare_jobs.json")
    with open(jobs, "w", encoding="utf-8") as f:
        json.dump({"images": [{"src": os.path.join(src, n), "dst": os.path.join(out, n)}
                              for n in names], "obs": obs, "sharpen": sharpen}, f)
    _note(job, log, f"Evening out exposure and vignetting over {len(names)} photos "
                    f"({len(ids):,} 3D points seen in them)")
    total = len(names)
    failed = []

    def parse(text):
        if text.startswith(("Exposure:", "Vignetting at")):
            job.add_line(text)
        elif text.startswith("Failed "):
            failed.append(text[7:].split(":", 1)[0])
        m = re.match(r"(Sampled|Prepared) (\d+)/(\d+)", text)
        if m:
            k = int(m.group(2)) / max(1, total)
            if m.group(1) == "Sampled":
                return 0.05 + 0.35 * k, f"reading photo {m.group(2)}/{total}"
            return 0.45 + 0.5 * k, f"writing photo {m.group(2)}/{total}"
        return None

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prepare_images.py")
    job.run([python, "-I", script, "--site", site, "--jobs", jobs], log, parser=parse)
    if failed:
        raise StepError(f"{len(failed)} photo(s) could not be prepared: " + ", ".join(failed[:4]))
    ctx["texture_images_dir"] = out
    _mark_done(ctx, key)


def step_undistort(job, ctx):
    p, prm, cfg = ctx["paths"], ctx["params"], ctx["cfg"]
    key = "undistort"
    model = p["model"]
    n = colmap_io.count_registered(model)
    spread = DENSE_VIEW_ANGLE if cfg.get("spread_views", True) and n > DENSE_SUBSET_MIN else 0.0
    ctx["sig_chain"][key] = {"prev": ctx["sig_chain"].get("orient"), "max": prm["undist_max_image"]}
    if spread:      # only then: runs of up to DENSE_SUBSET_MIN photos made before 0.3.4 resume
        ctx["sig_chain"][key]["spread"] = [spread, DENSE_SUBSET_MIN]
    if _resumable(job, ctx, key, [os.path.join(p["dense"], "sparse", "images.bin")]):
        return
    _invalidate_from(ctx, key)
    log = _log(ctx, key)
    shutil.rmtree(p["dense"], ignore_errors=True)
    if spread:
        keep = views.select_views(model, spread)
        if len(keep) < 0.9 * n:
            model = _reduced_model(job, ctx, model, keep, os.path.join(p["colmap"], "model_dense"), log)
            msg = (f"Depth maps from {len(keep)} of {n} photos (at least {spread:g}° apart); the "
                   "camera path keeps every photo")
            _note(job, log, msg)
            ctx["result"].setdefault("notes", []).append(msg)
    args = [_colmap(ctx), "image_undistorter", "--image_path", ctx["images_dir"],
            "--input_path", model, "--output_path", p["dense"], "--output_type", "COLMAP"]
    if prm["undist_max_image"] > 0:
        args += ["--max_image_size", str(prm["undist_max_image"])]
    job.run(args, log, env=ctx["ts"]["colmap_env"], parser=parse_undistort)
    _mark_done(ctx, key)


# GPU depth maps: COLMAP's CUDA PatchMatch writes the depth maps, InterfaceCOLMAP turns
# them into OpenMVS depth maps while it imports the scene, and DensifyPointCloud fuses
# them without estimating anything (--geometric-iters 0, every map already exists), so
# the cloud comes from the same fusion as the CPU path. On the test scene, COLMAP's own
# stereo_fusion of the same depth maps kept 167k points at a median 1.9 mm from the true
# surface, OpenMVS's fusion 430k points at 0.8 mm.
COLMAP_DENSE_DONE = "photogram3d_depth_maps.json"     # in dense/stereo, written last


def colmap_dense_size(ctx):
    """COLMAP's depth-map size, one OpenMVS also gets for every photo (see agreed_long_side)."""
    want = int(ctx["params"]["colmap_dense_max_image"])
    try:
        sizes = {(w, h) for _, _, _, w, h in _scene_images(ctx["paths"])}
    except (OSError, ValueError, KeyError):
        return want
    return agreed_long_side(sizes, want)


def step_colmap_dense(job, ctx):
    """GPU depth maps with COLMAP CUDA (patch_match_stereo)."""
    p = ctx["paths"]
    key = "colmap_dense"
    done = current(os.path.join(p["dense"], "stereo", COLMAP_DENSE_DONE))
    ctx["sig_chain"][key] = {"prev": ctx["sig_chain"].get("undistort"), "max": colmap_dense_size(ctx),
                             "fuse": "openmvs"}
    if _resumable(job, ctx, key, [done]):
        return
    _invalidate_from(ctx, key)
    _remove_files([done, os.path.join(p["dense"], "fused.ply"), os.path.join(p["dense"], "fused.ply.vis")])
    _prepare_depth_maps(job, ctx, key, _colmap_depth_files(p["dense"]), _colmap_map_complete)
    log = _log(ctx, key)
    size = str(colmap_dense_size(ctx))
    gpu = best_gpu(ctx["hw"])
    _note(job, log, f"Depth maps on {gpu['name'] if gpu else 'the NVIDIA GPU'} (COLMAP CUDA), "
                    f"up to {size} px")
    # COLMAP keeps images and maps in a RAM cache of 32 GB by default; keep it within RAM
    ram = ctx["hw"].get("ram_gb") or 16
    job.run([_colmap(ctx), "patch_match_stereo", "--workspace_path", p["dense"],
             "--workspace_format", "COLMAP", "--PatchMatchStereo.geom_consistency", "1",
             "--PatchMatchStereo.max_image_size", size,
             "--PatchMatchStereo.cache_size", f"{max(2.0, ram * 0.3):.1f}"],
            log, env=ctx["ts"]["colmap_env"], parser=make_patchmatch_parser())
    stereo = os.path.join(p["dense"], "stereo")
    maps = sorted(os.path.basename(f)[:-len(".geometric.bin")]
                  for f in glob.glob(os.path.join(stereo, "depth_maps", "*.geometric.bin")))
    if not maps:
        raise StepError("patch_match_stereo produced no depth maps (see the log)")
    # the photometric pass only fed the geometric one (about 120 MB per photo at 3200 px)
    _remove_files(glob.glob(os.path.join(stereo, "*_maps", "*.photometric.bin")))
    n_images = colmap_io.count_registered(os.path.join(p["dense"], "sparse"))
    _note(job, log, f"{len(maps)} of {n_images} photos have a GPU depth map"
                    + ("" if len(maps) >= n_images else "; OpenMVS computes the others on the CPU"))
    with open(done + ".tmp", "w", encoding="utf-8") as f:
        json.dump({"size": int(size), "maps": maps}, f)
    os.replace(done + ".tmp", done)
    _mark_done(ctx, key)


def step_mvs_import(job, ctx):
    p = ctx["paths"]
    key = "mvs_import"
    ctx["sig_chain"][key] = {"prev": ctx["sig_chain"].get("undistort"),
                             "dense": ctx["sig_chain"].get("colmap_dense")}
    if _resumable(job, ctx, key, [p["scene"]]):
        return
    _invalidate_from(ctx, key)
    shutil.rmtree(p["mvs"], ignore_errors=True)
    os.makedirs(p["mvs"], exist_ok=True)
    if ctx["dense_engine"] != "COLMAP":
        # InterfaceCOLMAP converts every COLMAP depth map it finds; only the GPU path wants that
        _remove_files(_colmap_depth_files(p["dense"]))
    job.run([openmvs_exe(ctx["ts"], "InterfaceCOLMAP"), "-i", p["dense"], "-o", p["scene"],
             "--image-folder", os.path.join(p["dense"], "images"), "-w", p["mvs"]],
            _log(ctx, key), tail=_tail(ctx, "InterfaceCOLMAP"))
    if not os.path.isfile(p["scene"]):
        raise StepError("InterfaceCOLMAP did not write scene.mvs")
    _mark_done(ctx, key)


def _omvs_common(ctx):
    args = ["-w", ctx["paths"]["mvs"], "--max-threads", "0"]
    if ctx.get("openmvs_cuda"):
        args += ["--cuda-device", "-1"]
    return args


# Partner photos for depth maps (0.3.5). OpenMVS scores each candidate partner by the 3D
# points the two photos share, their angles and how much of the photo they cover, and
# drops partners scoring below "View Min Score" (2.0). With few points per photo every
# partner can fall below it: 24 of the gas mask's 37 photos (42 points per photo) got
# none and no depth map. On the synthetic scene thinned to 39 points per photo, 13 of 44
# photos kept one partner at 2.0; with 0 (no absolute minimum, only OpenMVS's 3% of the
# best partner) every photo had 4 to 8, the cloud covered more of the true surface
# within 1 cm (76.6% against 72.2%) and was closer to it (median 1.01 mm against 1.16).
# With all 197 points per photo both settings gave the same cloud (76.2% / 76.0%,
# 1.02 mm), so the lower minimum is only used where it changes something.
WEAK_VIEW_MIN_SCORE = 0.0


def weak_points_per_photo(ctx):
    """3D points per photo of the undistorted model when it is a weakly matched set
    (below WEAK_POINTS_PER_PHOTO), else None."""
    model = os.path.join(ctx["paths"]["dense"], "sparse")
    n = colmap_io.count_registered(model)
    try:
        points = colmap_io.read_points3d_count_binary(os.path.join(model, "points3D.bin"))
    except (OSError, ValueError):
        return None
    per = points / max(1, n)
    return per if n and per < WEAK_POINTS_PER_PHOTO else None


# DensifyPointCloud (OpenMVS 2.x) guesses a region of interest from where the cameras look
# (--estimate-roi 1.1: the box of the sparse points they aim at, scaled by 1.1), weights the
# choice of each photo's partners by it and at the end trims the dense cloud to it
# (--crop-to-roi 1); the scene keeps it, and ReconstructMesh crops to it as well. On the
# second AW-249 video, which looks at the cockpit and sees the tail far off, the guess ended
# before the tail: 86% of COLMAP's points were inside it, and the trim removed 60,277 of the
# 337,180 fused points. Beyond the last fitting on the tail boom COLMAP had about 450 points
# and the dense cloud none; without the guess it kept 334,083 points, 11,134 of them there.
# Dense Region (0.3.14): Full, the default, makes no guess, so nothing is trimmed and the crop
# box is the region of interest; Automatic is the guess, scaled by Region Size.
DENSE_REGION_SIZE = 1.1          # OpenMVS's own scale of its guess (the behaviour up to 0.3.13)


def dense_region_scale(cfg):
    """--estimate-roi for the settings: 0 for Full (no region), the size for Automatic."""
    if cfg.get("dense_region", "FULL") != "AUTO":
        return 0.0
    try:
        size = float(cfg.get("dense_region_size", DENSE_REGION_SIZE))
    except (TypeError, ValueError):
        size = DENSE_REGION_SIZE
    return round(max(0.1, size), 3)


def dense_region_args(cfg):
    return ["--estimate-roi", f"{dense_region_scale(cfg):g}"]


def _sign_dense_region(sig, cfg):
    """The region in the densify signature. Runs made up to 0.3.13 have none in theirs and
    were made with OpenMVS's guess at 1.1, so Automatic at 1.1 adds nothing: it reuses
    their cloud, and Full makes theirs again."""
    scale = dense_region_scale(cfg)
    if scale != DENSE_REGION_SIZE:
        sig["region"] = scale
    return sig


def dense_config_args(job, ctx, log, weak):
    """DensifyPointCloud options for a weakly matched set: a config file that lowers View
    Min Score (OpenMVS reads "Name = value" lines)."""
    if weak is None:
        return []
    path = os.path.join(ctx["paths"]["mvs"], "densify_weak_set.ini")
    with open(path, "w", encoding="ascii") as f:
        f.write(f"View Min Score = {WEAK_VIEW_MIN_SCORE:g}\n")
    _note(job, log, f"Only {weak:.0f} 3D points per photo: depth maps also accept partner photos "
                    f"that share few points (OpenMVS View Min Score {WEAK_VIEW_MIN_SCORE:g} instead "
                    "of 2), or many photos would get none")
    return ["--dense-config-file", path]


# ---------------------------------------------------------------------------
# Depth maps left in a working folder
# ---------------------------------------------------------------------------
# DensifyPointCloud and COLMAP's patch_match_stereo both skip every image whose depth
# map already exists in their working folder, whatever settings made it. Switching a
# project between High and Ultra changes only the depth-map settings, so the folder is
# kept: DensifyPointCloud then loaded the old maps and crashed (SIGSEGV within seconds
# on the test scene, 640 px maps in a 960 px run; 0xC0000005 on Windows).

_RE_OMVS_DEPTH = re.compile(r"^depth\d+\.", re.IGNORECASE)
_DMAP_HEADER = 28      # HeaderDepthDataRaw in OpenMVS Interface.h


def _omvs_depth_files(mvs_dir):
    try:
        return [os.path.join(mvs_dir, n) for n in os.listdir(mvs_dir) if _RE_OMVS_DEPTH.match(n)]
    except OSError:
        return []


def _dmap_complete(path):
    """An OpenMVS .dmap is as long as its header says (a kill can cut one short).
    Other depth* files (confidence maps) are not checked."""
    if not path.lower().endswith(".dmap"):
        return True
    try:
        with open(path, "rb") as f:
            head = f.read(_DMAP_HEADER + 2)
            if len(head) < _DMAP_HEADER + 2 or head[:2] != b"DR":
                return False
            kind = head[2]
            width, height = struct.unpack_from("<II", head, 12)
            (name_len,) = struct.unpack_from("<H", head, _DMAP_HEADER)
            f.seek(name_len, os.SEEK_CUR)
            raw = f.read(4)
            if len(raw) < 4:
                return False
            (n_ids,) = struct.unpack("<I", raw)
            size = os.fstat(f.fileno()).st_size
    except OSError:
        return False
    # depth float, then normal 3 floats, confidence float, views 4 bytes when present
    per_pixel = 4 + 12 * bool(kind & 2) + 4 * bool(kind & 4) + 4 * bool(kind & 8)
    expected = _DMAP_HEADER + 2 + name_len + 4 + 4 * n_ids + 21 * 8 + width * height * per_pixel
    return size == expected


def _colmap_depth_files(dense_dir):
    out = []
    for sub in ("depth_maps", "normal_maps", "consistency_graphs"):
        folder = os.path.join(dense_dir, "stereo", sub)
        try:
            out += [os.path.join(folder, n) for n in os.listdir(folder)]
        except OSError:
            pass
    return out


def _colmap_map_complete(path):
    """A COLMAP depth or normal map ("W&H&C&" then 32-bit floats) is as long as its
    header says. Consistency graphs are not checked."""
    if os.path.basename(os.path.dirname(path)) == "consistency_graphs":
        return True
    try:
        with open(path, "rb") as f:
            head = f.read(64)
            size = os.fstat(f.fileno()).st_size
    except OSError:
        return False
    parts = head.split(b"&", 3)
    if len(parts) < 4:
        return False
    try:
        width, height, channels = (int(v) for v in parts[:3])
    except ValueError:
        return False
    return size == sum(len(v) for v in parts[:3]) + 3 + width * height * channels * 4


def _prepare_depth_maps(job, ctx, key, files, complete):
    """Delete the depth maps the tool must not reuse before running step ``key``.

    They are kept only for a rerun with the same signature (resume after a cancel or a
    crash), and then only those that are complete."""
    tag = key + ":depth_maps"
    sig = _step_signature(ctx, key)
    same = ctx["cfg"].get("resume", True) and ctx["state"].get(tag) == sig
    stale = [f for f in files if not (same and complete(f))]
    _remove_files(stale)
    left = [f for f in stale if os.path.exists(f)]
    if left:
        raise StepError(f"Cannot delete {len(left)} old depth maps, for example "
                        f"{os.path.basename(left[0])}. Is a sync program (Google Drive, OneDrive, "
                        "Dropbox) or another app using the project folder?")
    if stale:
        job.add_line(f"Removed {len(stale)} depth-map files "
                     + ("cut short by the interruption" if same else
                        "left by an earlier run (the tool would reuse them whatever their settings)"))
    kept = len(files) - len(stale)
    if kept:
        job.add_line(f"Reusing {kept} depth-map files from the interrupted run")
    ctx["state"][tag] = sig
    _save_state(ctx["paths"]["state"], ctx["state"])


# At the end of each consistency pass every photo has its .dmap (depth, normal and
# confidence: 20 bytes per pixel) and the pass's .geo.dmap next to it.
DEPTH_DISK_BYTES_PER_PIXEL = 40


def _gb(n):
    return f"{n / 1024 ** 3:.1f} GB"


def check_depth_disk(job, ctx, level, max_res, min_res, log=None):
    """Stop before the depth maps when the output drive cannot hold them at their peak
    (about 13 GB for 42 photos at 3200 px, 27 GB at 4624 px)."""
    p = ctx["paths"]
    images = _scene_images(p)
    pixels, longest = 0, 0
    for _, _, _, w, h in images:
        side = omvs_max_resolution(max(w, h), level, min_res, max_res)
        sw, sh = _scaled_size(w, h, side)
        pixels += sw * sh
        longest = max(longest, sw, sh)
    need = pixels * DEPTH_DISK_BYTES_PER_PIXEL
    try:
        free = shutil.disk_usage(p["mvs"]).free
    except OSError:
        return need
    # maps already on disk are replaced or reused, so their space counts as free
    for f in _omvs_depth_files(p["mvs"]) + _colmap_depth_files(p["dense"]):
        try:
            free += os.path.getsize(f)
        except OSError:
            pass
    if free < need:
        raise StepError(f"The depth maps need about {_gb(need)} of free disk space at their peak "
                        f"({len(images)} photos at up to {longest} px), and the output folder's drive "
                        f"has {_gb(free)}. Free some space, choose an output folder on another drive, "
                        "or use a lower Quality")
    if free < 1.5 * need and log:
        _note(job, log, f"The depth maps need about {_gb(need)} of disk space at their peak; "
                        f"{_gb(free)} is free")
    return need


def step_densify(job, ctx):
    """OpenMVS depth maps and fusion; with GPU depth maps, the fusion only."""
    p, prm = ctx["paths"], ctx["params"]
    key = "densify"
    gpu_maps = ctx["dense_engine"] == "COLMAP"
    if gpu_maps:
        # the converted maps have COLMAP's size: same scaling, no other resolution level
        level, max_res = 0, colmap_dense_size(ctx)
    else:
        level, max_res = prm["dense_res_level"], prm["dense_max_res"]
    ctx["sig_chain"][key] = _sign_dense_region({"prev": ctx["sig_chain"].get("mvs_import"),
                                                "p": [level, max_res, prm["dense_views"]]}, ctx["cfg"])
    if gpu_maps:
        ctx["sig_chain"][key]["fuse_only"] = True
    weak = weak_points_per_photo(ctx)
    if weak is not None:
        ctx["sig_chain"][key]["view_min_score"] = WEAK_VIEW_MIN_SCORE
    dense_ply = os.path.join(p["mvs"], "scene_dense.ply")
    if _resumable(job, ctx, key, [p["scene_dense"], dense_ply]):
        return
    _invalidate_from(ctx, key)
    log = _log(ctx, key)
    args = [openmvs_exe(ctx["ts"], "DensifyPointCloud"), p["scene"], "-o", p["scene_dense"],
            "--resolution-level", str(level), "--max-resolution", str(max_res),
            "--min-resolution", str(min(640, max_res)),
            "--number-views", str(prm["dense_views"]), "--tower-mode", "0"] + dense_region_args(ctx["cfg"]) + [
            # the depth maps (about 150 MB each at 3200 px) are only needed until they
            # are fused; an interrupted run keeps them for the resume
            "--remove-dmaps", "1"] + dense_config_args(job, ctx, log, weak)
    if gpu_maps:
        # mvs_import rebuilt the folder with the maps converted from COLMAP's: keep them all
        n = len([f for f in _omvs_depth_files(p["mvs"]) if f.lower().endswith(".dmap")])
        _note(job, log, f"Fusing {n} GPU depth maps with OpenMVS")
        args += ["--geometric-iters", "0"]
    else:
        _prepare_depth_maps(job, ctx, key, _omvs_depth_files(p["mvs"]), _dmap_complete)
        # COLMAP maps an interrupted GPU or CPU + GPU run left (about 110 MB each at 3200 px)
        _remove_files(_colmap_depth_files(p["dense"]))
        check_depth_disk(job, ctx, level, max_res, min(640, max_res), log)
    job.run(args + _omvs_common(ctx), log, tail=_tail(ctx, "DensifyPointCloud"),
            parser=make_densify_parser(colmap_io.count_registered(os.path.join(p["dense"], "sparse"))))
    if not os.path.isfile(dense_ply):
        raise StepError("DensifyPointCloud produced no point cloud")
    if gpu_maps:
        _remove_files(_colmap_depth_files(p["dense"]))   # fused now (about 120 MB per photo at 3200 px)
    _mark_done(ctx, key)


def _dense_cloud(ctx):
    """(scene .mvs, dense point cloud .ply) that meshing starts from."""
    p = ctx["paths"]
    return p["scene_dense"], os.path.join(p["mvs"], "scene_dense.ply")


# ---------------------------------------------------------------------------
# Depth maps on the CPU and the GPU at once
# ---------------------------------------------------------------------------
# OpenMVS (CPU) and COLMAP (NVIDIA GPU) share the first depth-map pass, the slow one:
# 27.5 of 46 min on the WRX photos at 3200 px, with OpenMVS at about 39 s per photo on
# a 5900HX and COLMAP at about 30 s on the laptop RTX 3070. Both tools skip a photo
# whose output already exists, so when one starts a photo the other gets a placeholder
# for it: OpenMVS's "Reference image N paired" puts empty COLMAP map files in place,
# COLMAP's "Processing view i / n for NAME" an OpenMVS .dmap holding GPU_CLAIM. The
# faster tool simply takes more photos.
#
# OpenMVS runs with its usual settings, so its photos get exactly the CPU path's first
# pass (with consistency passes to come it keeps weaker matches for them to refine), and
# it is stopped as soon as it starts its first consistency pass: that pass writes
# .geo.dmap files and replaces the .dmap files only at its end, so the first-pass maps
# are complete and untouched. InterfaceCOLMAP converts the GPU maps, and a second
# DensifyPointCloud run does the two consistency passes and the fusion for every photo,
# exactly as in the CPU path.

GPU_CLAIM = b"PHOTOGRAM3D GPU CLAIM\n"
HYBRID_FIRST_PASS = 0.45        # share of the step's bar for the first pass
HYBRID_GPU_WAIT = 600           # s; a GPU photo still unfinished this long after the CPU is done goes to the CPU
_RE_OMVS_REF = re.compile(r"Reference image\s+(\d+) paired with")
_RE_OMVS_EST = re.compile(r"Depth-map for image\s+(\d+)\b.*?estimated[^:]*:\s*(\d+)x(\d+)")
# COLMAP 4.2.1 writes "... patch_match.cc:419] === Processing view 1 / 42 for NAME ==="
_RE_COLMAP_START = re.compile(r"Processing view (\d+) / (\d+) for (.+?)(?:\s+=+)?\s*$")
_RE_COLMAP_WRITE = re.compile(r"Writing (?:photometric|geometric) output for (.+?)\s*$")


def omvs_max_resolution(image_size, level, min_size, max_size):
    """OpenMVS's TImage::computeMaxResolution: the long side its depth maps get."""
    if level == 0:
        return min(image_size, max_size)
    size = image_size >> level
    if size < min_size:
        level = 0
        while (image_size >> (level + 1)) >= min_size:
            level += 1
        size = image_size >> level
    return min(size, max_size)


def _scene_images(p):
    """[(scene index, image id, name, width, height)] in OpenMVS's scene order, which
    InterfaceCOLMAP takes from images.bin, with the undistorted image size."""
    cams, images = colmap_io.read_model(os.path.join(p["dense"], "sparse"))
    return [(i, int(iid), im["name"], cams[im["camera_id"]]["width"], cams[im["camera_id"]]["height"])
            for i, (iid, im) in enumerate(images.items())]


def _scaled_size(w, h, long_side):
    """OpenMVS's image size at --max-resolution long_side (Image::ResizeImage: double
    scale, cvRound, which rounds half to even like Python's round)."""
    scale = min(1.0, long_side / max(w, h))
    return round(w * scale), round(h * scale)


def colmap_scaled_size(w, h, max_size):
    """COLMAP's depth-map size at --PatchMatchStereo.max_image_size (mvs::Image::Downsize:
    a 32-bit float factor, std::round, which rounds half away from zero)."""
    import numpy as np
    if w <= max_size and h <= max_size:
        return w, h
    f32 = np.float32
    factor = min(f32(max_size) / f32(w), f32(max_size) / f32(h))
    return tuple(int(math.floor(float(f32(v) * factor) + 0.5)) for v in (w, h))


def agreed_long_side(sizes, want, reach=32):
    """The long side nearest ``want`` at which COLMAP and OpenMVS scale every photo to the
    same size. They can differ by a pixel: Pixel 7a photos undistort to 2401 x 3200, and at
    1600 px COLMAP makes 1201 x 1600 maps, OpenMVS 1200 x 1600 (1200.5 rounded up, and to
    even). A map of the wrong size cannot be used, so the GPU's maps were thrown away."""
    sizes = set(sizes)

    def agree(side):
        return all(colmap_scaled_size(w, h, side) == _scaled_size(w, h, side) for w, h in sizes)

    for d in range(reach + 1):
        for side in ((want,) if d == 0 else (want - d, want + d)):
            if side > 0 and agree(side):
                return side
    return want


def _dmap_info(path):
    """(width, height, offset of the depth data) of an OpenMVS .dmap, or None."""
    try:
        with open(path, "rb") as f:
            head = f.read(_DMAP_HEADER + 2)
            if len(head) < _DMAP_HEADER + 2 or head[:2] != b"DR":
                return None
            width, height = struct.unpack_from("<II", head, 12)
            (name_len,) = struct.unpack_from("<H", head, _DMAP_HEADER)
            f.seek(_DMAP_HEADER + 2 + name_len)
            (n_ids,) = struct.unpack("<I", f.read(4))
    except (OSError, struct.error):
        return None
    return width, height, _DMAP_HEADER + 2 + name_len + 4 + 4 * n_ids + 21 * 8


def _dmap_has_conf(path):
    """True for OpenMVS's own maps (they carry a confidence map); converted ones do not."""
    try:
        with open(path, "rb") as f:
            head = f.read(3)
        return len(head) == 3 and head[:2] == b"DR" and bool(head[2] & 4)
    except OSError:
        return False


def _fix_depth_range(path):
    """InterfaceCOLMAP stores the depth range of a converted map including its empty
    pixels, so it starts at 0; OpenMVS draws random depths from that range while it
    refines. Store the valid range with OpenMVS's own 10% margins instead."""
    import numpy as np
    info = _dmap_info(path)
    if info is None:
        return False
    width, height, offset = info
    with open(path, "r+b") as f:
        f.seek(offset)
        depth = np.frombuffer(f.read(4 * width * height), dtype="<f4")
        valid = depth[np.isfinite(depth) & (depth > 0)]
        if valid.size == 0:
            return False
        f.seek(20)
        f.write(struct.pack("<ff", float(valid.min()) * 0.9, float(valid.max()) * 1.1))
    return True


def _is_claim(path):
    try:
        with open(path, "rb") as f:
            return f.read(len(GPU_CLAIM) + 1) == GPU_CLAIM
    except OSError:
        return False


class _FirstPassShare:
    """Which tool does which photo in the shared first pass. Each tool's log lines
    arrive on their own thread."""

    def __init__(self, job, p, images):
        self.job, self.p = job, p
        self.lock = threading.Lock()
        self.ids = {i: iid for i, iid, _, _, _ in images}
        self.names = {i: name for i, _, name, _, _ in images}
        self.index_of = {name: i for i, _, name, _, _ in images}
        self.index_of_id = {iid: i for i, iid, _, _, _ in images}
        self.src_size = {i: (w, h) for i, _, _, w, h in images}
        self.n = len(images)
        self.cpu, self.gpu = set(), set()            # photos each tool took
        self.cpu_done, self.gpu_done = set(), set()  # photos each tool finished
        self.seen = set()                            # photos OpenMVS reached in this run
        self.sizes = {}                              # image size -> OpenMVS depth-map size
        self.gpu_started = {}                        # photo -> time COLMAP started it
        self.gpu_times = []                          # seconds per GPU photo
        self.first_pass_done = threading.Event()     # OpenMVS began its consistency passes
        self.best = 0.0

    def dmap(self, i):
        return os.path.join(self.p["mvs"], f"depth{self.ids[i]:04d}.dmap")

    def colmap_files(self, i):
        stereo = os.path.join(self.p["dense"], "stereo")
        return [os.path.join(stereo, sub, self.names[i] + ".photometric.bin")
                for sub in ("depth_maps", "normal_maps")]

    def hold_for_cpu(self, i):
        """COLMAP skips a photo whose depth and normal maps exist."""
        for path in self.colmap_files(i):
            if not os.path.exists(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                open(path, "wb").close()

    def hold_for_gpu(self, i):
        """OpenMVS skips a photo whose .dmap exists (without reading it)."""
        if not os.path.exists(self.dmap(i)):
            with open(self.dmap(i), "wb") as f:
                f.write(GPU_CLAIM)

    def take_over(self):
        """Maps kept from an interrupted run: claim their photos for the tool that made them."""
        for i in self.ids:
            if os.path.isfile(self.dmap(i)):          # complete; _prepare_depth_maps removed the others
                self.cpu.add(i)
                self.cpu_done.add(i)
                self.hold_for_cpu(i)
            elif all(os.path.isfile(f) for f in self.colmap_files(i)):
                self.gpu.add(i)
                self.gpu_done.add(i)
                self.hold_for_gpu(i)

    def progress(self):
        with self.lock:
            cpu = len(self.cpu_done)
            gpu = len(self.gpu_done - self.cpu_done)
            frac = HYBRID_FIRST_PASS * min(1.0, (cpu + gpu) / max(self.n, 1))
            self.best = max(self.best, frac)
            best = self.best
        self.job.set_fraction(best, f"depth map {cpu + gpu}/{self.n} (CPU {cpu}, GPU {gpu})")

    def on_openmvs(self, line):
        if self.first_pass_done.is_set():
            return None
        m = _RE_OMVS_EST.search(line)
        if m:
            i = self.index_of_id.get(int(m.group(1)))
            if i is not None:
                with self.lock:
                    self.cpu_done.add(i)
                    self.sizes.setdefault(self.src_size[i], (int(m.group(2)), int(m.group(3))))
                self.progress()
            return None
        m = _RE_OMVS_REF.search(line)
        if not m:
            return None
        i = int(m.group(1))
        with self.lock:
            if i in self.seen:
                # every photo is logged once per pass, skipped ones too: this is the
                # first consistency pass starting, so the first pass is complete
                self.first_pass_done.set()
                return None
            self.seen.add(i)
            if i not in self.ids or i in self.gpu or i in self.cpu:
                return None
            self.cpu.add(i)
        self.hold_for_cpu(i)
        return None

    def on_colmap(self, line):
        m = _RE_COLMAP_START.search(line)
        if m:
            i = self.index_of.get(m.group(3).strip())
            if i is None:
                return
            with self.lock:
                if i in self.gpu:
                    return
                self.gpu.add(i)
                self.gpu_started[i] = time.time()
                taken = i in self.cpu
            if not taken:
                self.hold_for_gpu(i)
            return
        m = _RE_COLMAP_WRITE.search(line)
        if m:
            i = self.index_of.get(m.group(1).strip())
            if i is None:
                return
            with self.lock:
                self.gpu_done.add(i)
                if i in self.gpu_started:
                    self.gpu_times.append(time.time() - self.gpu_started[i])
            self.progress()

    def gpu_wait_limit(self):
        with self.lock:
            longest = max(self.gpu_times, default=0.0)
        return max(HYBRID_GPU_WAIT, 5 * longest)


# COLMAP's first pass (20 source photos) held about 2.6 GB of the laptop RTX 3070 at 3200 px
# (Task Manager: 3.2 GB in use during the 0.3.2 CPU + GPU run, 0.5 GB of it Blender's, the
# rest the desktop's). Its per-pixel buffers grow with the pixel count: about 5.0 GB at
# 4624 px. Blender, the desktop and other apps keep HYBRID_VRAM_RESERVE_GB.
HYBRID_VRAM_BASE_GB = 0.4
HYBRID_VRAM_GB_AT_3200 = 2.2
HYBRID_VRAM_RESERVE_GB = 1.5


def hybrid_vram_gb(long_side):
    return HYBRID_VRAM_BASE_GB + HYBRID_VRAM_GB_AT_3200 * (long_side / 3200.0) ** 2


def hybrid_gpu_ok(ctx, long_side):
    """The GPU has room for COLMAP's first pass at this size: an 8 GB GPU up to about
    5300 px, a 4 GB GTX 1650 Ti up to about 3100 px (Normal and High, not Ultra)."""
    gpu = best_gpu(ctx["hw"])
    vram_gb = (gpu["vram_mb"] / 1024.0) if gpu else 0.0
    return hybrid_vram_gb(long_side) <= vram_gb - HYBRID_VRAM_RESERVE_GB


def step_densify_hybrid(job, ctx):
    """Depth maps on the CPU (OpenMVS) and the NVIDIA GPU (COLMAP) at once."""
    p, prm = ctx["paths"], ctx["params"]
    key = "densify"
    images = _scene_images(p)
    min_res = min(640, prm["dense_max_res"])
    preset_side = omvs_max_resolution(max(max(w, h) for _, _, _, w, h in images),
                                      prm["dense_res_level"], min_res, prm["dense_max_res"])
    long_side = agreed_long_side({(w, h) for _, _, _, w, h in images}, preset_side)
    if not hybrid_gpu_ok(ctx, long_side):
        msg = (f"The GPU's memory is too small for {long_side} px depth maps; "
               "the CPU computes them alone")
        job.add_line(msg)
        ctx["result"].setdefault("notes", []).append(msg)
        return step_densify(job, ctx)
    sig = {"prev": ctx["sig_chain"].get("mvs_import"), "p": [0, long_side, prm["dense_views"]],
           "hybrid": 1}
    weak = weak_points_per_photo(ctx)
    if weak is not None:
        sig["view_min_score"] = WEAK_VIEW_MIN_SCORE
    ctx["sig_chain"][key] = _sign_dense_region(dict(sig), ctx["cfg"])
    ctx["sig_chain"]["densify_gpu"] = sig        # tags the COLMAP maps of an interrupted run
    dense_ply = os.path.join(p["mvs"], "scene_dense.ply")
    if _resumable(job, ctx, key, [p["scene_dense"], dense_ply]):
        return
    _invalidate_from(ctx, key)
    log = _log(ctx, key)
    gpu_log = os.path.splitext(log)[0] + "_gpu.log"
    _prepare_depth_maps(job, ctx, key, _omvs_depth_files(p["mvs"]), _dmap_complete)
    _prepare_depth_maps(job, ctx, "densify_gpu", _colmap_depth_files(p["dense"]), _colmap_map_complete)
    # consistency-pass output of an interrupted run is never reused
    _remove_files([f for f in _omvs_depth_files(p["mvs"]) if f.lower().endswith(".geo.dmap")])
    _remove_files([dense_ply])
    check_depth_disk(job, ctx, 0, long_side, min(min_res, long_side), log)
    share = _FirstPassShare(job, p, images)
    share.take_over()
    share.progress()
    gpu = best_gpu(ctx["hw"])
    _note(job, log, f"First pass on the CPU and on {gpu['name'] if gpu else 'the NVIDIA GPU'} "
                    f"at once, {long_side} px")
    if long_side != preset_side:
        _note(job, log, f"{long_side} px rather than {preset_side}: at {preset_side} px COLMAP and "
                        "OpenMVS round these photos' size differently, and the GPU maps could not "
                        "be used")

    ram = ctx["hw"].get("ram_gb") or 16
    gpu_stop = threading.Event()
    gpu_state = {}

    def gpu_worker():
        try:
            # filter 0: unfiltered maps, as COLMAP itself feeds its consistency pass
            gpu_state["code"] = job.run_side(
                [_colmap(ctx), "patch_match_stereo", "--workspace_path", p["dense"],
                 "--workspace_format", "COLMAP", "--PatchMatchStereo.geom_consistency", "0",
                 "--PatchMatchStereo.filter", "0", "--PatchMatchStereo.max_image_size", str(long_side),
                 "--PatchMatchStereo.cache_size", f"{max(2.0, ram * 0.25):.1f}"],
                gpu_log, env=ctx["ts"]["colmap_env"], on_line=share.on_colmap, stop=gpu_stop,
                below_normal=False)      # its one CPU thread feeds the GPU: ahead of OpenMVS
        except Exception as exc:  # noqa: BLE001 - the CPU carries on without the GPU
            gpu_state["error"] = str(exc)

    worker = threading.Thread(target=gpu_worker, name="Photogram3DGPU", daemon=True)
    worker.start()
    densify = openmvs_exe(ctx["ts"], "DensifyPointCloud")
    scene_args = ["--resolution-level", "0", "--max-resolution", str(long_side),
                  "--min-resolution", str(min(min_res, long_side)),
                  "--number-views", str(prm["dense_views"]), "--tower-mode", "0"
                  ] + dense_region_args(ctx["cfg"]) + dense_config_args(job, ctx, log, weak)
    first_scene = os.path.join(p["mvs"], "scene_first_pass.mvs")
    try:
        # OpenMVS's own log is followed on every system: its console is paused while
        # it estimates, so the per-photo lines are only there
        code = job.run([densify, p["scene"], "-o", first_scene] + scene_args
                       + ["--fusion-mode", "1"] + _omvs_common(ctx),
                       log, tail=(p["mvs"], "DensifyPointCloud"), parser=share.on_openmvs,
                       ok_codes=None, stop=share.first_pass_done)
        if code != 0 and not share.first_pass_done.is_set():
            raise StepError(f"DensifyPointCloud failed ({describe_returncode(code)}). "
                            + " | ".join(list(job.lines)[-3:]))
    except BaseException:
        gpu_stop.set()
        worker.join(timeout=30)
        raise
    _remove_files([first_scene, os.path.splitext(first_scene)[0] + ".ply"]
                  + [f for f in _omvs_depth_files(p["mvs"]) if f.lower().endswith(".geo.dmap")])
    if share.first_pass_done.is_set():
        job.add_line("OpenMVS: first pass done (stopped at its consistency passes, which run "
                     "once the GPU maps are in)")
    else:
        # it never logged the start of a consistency pass and finished them itself
        job.add_line("OpenMVS: finished without the expected log lines; its maps will be "
                     "refined once more")

    # COLMAP finishes the photo it is on; every other photo is taken by now
    t_cpu_done = time.time()
    while worker.is_alive():
        worker.join(timeout=1.0)
        job.check_cancel()
        if worker.is_alive() and time.time() - t_cpu_done > share.gpu_wait_limit() and not gpu_stop.is_set():
            _note(job, log, "COLMAP did not finish its photo in time; the CPU computes it")
            gpu_stop.set()
    job.check_cancel()
    code = gpu_state.get("code")
    if gpu_state.get("error") or code not in (0, None):
        _note(job, log, "COLMAP stopped early (" + (gpu_state.get("error") or describe_returncode(code))
              + f"); the CPU computes its unfinished photos. See {os.path.basename(gpu_log)}")

    # COLMAP's placeholders must not reach InterfaceCOLMAP, which reads every map file
    _remove_files([f for f in _colmap_depth_files(p["dense"]) if not _colmap_map_complete(f)])
    gpu_maps = []
    for i in share.ids:
        if not all(os.path.isfile(f) for f in share.colmap_files(i)):
            continue
        if _dmap_has_conf(share.dmap(i)):
            _remove_files(share.colmap_files(i))  # both tools did it: the CPU's map stays
        else:
            gpu_maps.append(i)
    if gpu_maps:
        tmp_scene = os.path.join(p["mvs"], "scene_gpu_maps.mvs")
        try:
            job.run([openmvs_exe(ctx["ts"], "InterfaceCOLMAP"), "-i", p["dense"], "-o", tmp_scene,
                     "--image-folder", os.path.join(p["dense"], "images"), "-w", p["mvs"]],
                    log, tail=_tail(ctx, "InterfaceCOLMAP"))
        except StepError as exc:
            _note(job, log, f"The GPU depth maps could not be converted ({exc}); the CPU computes them")
        _remove_files([tmp_scene, os.path.splitext(tmp_scene)[0] + ".ply"])
    _remove_files([f for f in _omvs_depth_files(p["mvs"]) if _is_claim(f)])
    _remove_files(_colmap_depth_files(p["dense"]))
    for i in share.ids:
        path = share.dmap(i)
        if not os.path.isfile(path) or _dmap_has_conf(path):
            continue
        # a converted GPU map (now, or by the interrupted run): OpenMVS must find the
        # size it would have made, or it reads past the map
        info = _dmap_info(path)
        want = share.sizes.get(share.src_size[i]) or _scaled_size(*share.src_size[i], long_side)
        if (info is None or (info[0], info[1]) != want or not _dmap_complete(path)
                or not _fix_depth_range(path)):
            _remove_files([path])                # the CPU computes it in the next run
    missing = [i for i in share.ids if not os.path.isfile(share.dmap(i))]
    on_cpu = sum(1 for i in share.ids if _dmap_has_conf(share.dmap(i)))
    on_gpu = share.n - on_cpu - len(missing)      # converted now or by the interrupted run
    _note(job, log, f"First pass: {on_cpu} photos on the CPU, {on_gpu} on the GPU"
                    + (f", {len(missing)} left for the CPU" if missing else ""))
    ctx["result"]["depth_maps_split"] = {"cpu": on_cpu, "gpu": on_gpu, "cpu_after": len(missing)}

    # consistency passes and fusion for all photos, as in the CPU path
    total = 2 * len(images) + len(missing)
    count = {"maps": 0}

    def second_parser(line):
        if "Depth-map for image" in line and "estimated" in line:
            count["maps"] += 1
            frac = HYBRID_FIRST_PASS + (0.97 - HYBRID_FIRST_PASS) * min(1.0, count["maps"] / total)
            return frac, f"refining depth map {count['maps']}/{total}"
        if "fused and filtered" in line:
            return 0.97, "fusing depth maps"
        return None

    job.run([densify, p["scene"], "-o", p["scene_dense"]] + scene_args
            + ["--remove-dmaps", "1"] + _omvs_common(ctx),
            log, tail=(p["mvs"], "DensifyPointCloud"), parser=make_monotonic(second_parser))
    if not os.path.isfile(dense_ply):
        raise StepError("DensifyPointCloud produced no point cloud")
    # --remove-dmaps deletes the maps it fused; a photo OpenMVS left out of the dense
    # reconstruction (no usable neighbours) may still have a GPU map
    _remove_files([f for f in _omvs_depth_files(p["mvs"]) if f.lower().endswith(".dmap")])
    _mark_done(ctx, key)


# ReconstructMesh's Delaunay step needed 7.2 GB for 4.0M points on the WRX photos
# (1.8 GB per million). Above this share of RAM the points are thinned first.
MESH_GB_PER_MPOINT = 1.8
MESH_RAM_SHARE = 0.55
MESH_POINT_DISTANCE = 1.5      # ReconstructMesh default (px)


def mesh_point_distance(points, ram_gb):
    """--min-point-distance that keeps ReconstructMesh within the RAM budget."""
    budget = (ram_gb or 16) * MESH_RAM_SHARE / MESH_GB_PER_MPOINT * 1e6
    if points <= budget:
        return MESH_POINT_DISTANCE
    # points merged within d px: the count falls roughly with d^2
    return round(min(6.0, MESH_POINT_DISTANCE * math.sqrt(points / budget)), 2)


def _mesh_points(ctx, cloud):
    """Points ReconstructMesh will triangulate: all of them, or those in the crop box."""
    roi_path = ctx["cfg"].get("crop_roi")
    preview = os.path.join(ctx["paths"]["mvs"], CLOUD_PREVIEW)
    try:
        if roi_path and os.path.isfile(preview):
            xyz, _ = roi.load_preview(preview)
            total = roi.ply_vertex_count(cloud)
            return int(roi.OBB.load(roi_path).inside(xyz).mean() * total)
        return roi.ply_vertex_count(cloud)
    except (OSError, ValueError, KeyError):
        return 0


def step_mesh(job, ctx):
    p, cfg, prm = ctx["paths"], ctx["cfg"], ctx["params"]
    key = "mesh"
    faces = int(cfg.get("target_faces") or prm["target_faces"])
    scene, cloud = _dense_cloud(ctx)
    roi_path = cfg.get("crop_roi") if cfg.get("crop_roi") and os.path.isfile(cfg["crop_roi"]) else None
    roi_text = open(roi_path, encoding="ascii").read() if roi_path else ""
    points = _mesh_points(ctx, cloud)
    dist = mesh_point_distance(points, ctx["hw"].get("ram_gb"))
    sig = {"prev": ctx["sig_chain"].get("densify") or ctx["sig_chain"].get("mvs_import"),
           "faces": faces, "fs": cfg["free_space"]}
    if roi_text:   # new keys only when used, so older projects still resume
        sig["roi"] = roi_text
    if dist != MESH_POINT_DISTANCE:
        sig["dist"] = dist
    ctx["sig_chain"][key] = sig
    if _resumable(job, ctx, key, [p["mesh"]]):
        return
    _invalidate_from(ctx, key)
    log = _log(ctx, key)
    args = [openmvs_exe(ctx["ts"], "ReconstructMesh"), scene, "-p", cloud,
            "-o", os.path.splitext(p["mesh"])[0] + ".mvs",
            "--target-face-num", str(faces)]
    if roi_path:
        args += ["--import-roi-file", roi_path, "--integrate-only-roi", "1", "--crop-to-roi", "1"]
        _note(job, log, f"Meshing inside the crop box: about {points / 1e6:.1f}M points")
    if dist != MESH_POINT_DISTANCE:
        args += ["--min-point-distance", str(dist)]
        _note(job, log, f"{points / 1e6:.1f}M points would not fit in RAM for meshing; merging "
                        f"points closer than {dist} px")
    if cfg["free_space"]:
        args += ["--free-space-support", "1"]
    job.run(args + _omvs_common(ctx), log, tail=_tail(ctx, "ReconstructMesh"),
            parser=make_openmvs_parser(
                {"Points inserted": (0.0, 0.3), "Points weighted": (0.35, 0.6),
                 "Decimated faces": (0.85, 0.98)},
                {"Delaunay tetrahedralization completed": 0.32,
                 "Delaunay tetrahedras weighting completed": 0.6,
                 "Delaunay tetrahedras graph-cut completed": 0.75}))
    if not os.path.isfile(p["mesh"]):
        raise StepError("ReconstructMesh produced no mesh")
    _mark_done(ctx, key)


REFINE_MIN_SIDE = 1000         # AUTO refinement needs refinement images at least this large
# Image pixels RefineMesh may hold per GB of RAM. Measured on the WRX run (16 GB): 42
# images of 2312 x 1736 (168.6 MP) peaked at 7.8 GB, about 46 bytes per pixel with the
# mesh. 13 MP per GB keeps RefineMesh near 60% of RAM (25 MP per GB, used up to 0.2.2,
# would have let it need more than all of it on large photo sets).
REFINE_PIXELS_PER_GB = 13e6
REFINE_FACE_AREA = 32          # RefineMesh --max-face-area default (px); Ultra halves it


def refine_plan(ctx):
    """(resolution level, long side in px) for RefineMesh.

    As sharp as RAM allows, capped per preset for speed. Measured on the test
    scene (960 px images, refined at OpenMVS's 640 px floor): the refined mesh
    was coarser and less accurate (0.53 -> 0.93 mm median), so AUTO skips
    refinement when the images would be smaller than REFINE_MIN_SIDE.
    """
    cams, images = colmap_io.read_model(os.path.join(ctx["paths"]["dense"], "sparse"))
    if not images:
        return 1, 0
    side = max(max(c["width"], c["height"]) for c in cams.values())
    pixels = sum(cams[im["camera_id"]]["width"] * cams[im["camera_id"]]["height"]
                 for im in images.values())
    budget = (ctx["hw"].get("ram_gb") or 16) * REFINE_PIXELS_PER_GB
    cap = ctx["params"]["refine_max_side"]
    level = 0
    while level < 4 and (side / 2 ** level > cap or pixels / 4 ** level > budget):
        level += 1
    return level, int(side / 2 ** level)


def step_refine(job, ctx):
    p = ctx["paths"]
    key = "refine"
    level, side = refine_plan(ctx)
    if ctx["cfg"].get("refine", "AUTO") == "AUTO" and side < REFINE_MIN_SIDE:
        job.add_line(f"Refinement skipped: the photos would be {side} px at a memory-safe "
                     f"resolution (< {REFINE_MIN_SIDE} px), where it adds noise rather than detail")
        ctx["do_refine"] = False
        return
    face_area = int(ctx["params"].get("refine_face_area", REFINE_FACE_AREA))
    ctx["sig_chain"][key] = {"prev": ctx["sig_chain"].get("mesh"), "rl": level}
    if face_area != REFINE_FACE_AREA:
        ctx["sig_chain"][key]["area"] = face_area
    if _resumable(job, ctx, key, [p["mesh_refined"]]):
        return
    _invalidate_from(ctx, key)
    job.add_line(f"Refining with {side} px images (resolution level {level}, faces up to "
                 f"{face_area} px)")
    scene = p["scene_dense"]
    job.run([openmvs_exe(ctx["ts"], "RefineMesh"), scene, "-m", p["mesh"],
             "-o", os.path.splitext(p["mesh_refined"])[0] + ".mvs",
             "--resolution-level", str(level), "--max-views", "8",
             "--scales", "2", "--max-face-area", str(face_area)] + _omvs_common(ctx),
            _log(ctx, key), tail=_tail(ctx, "RefineMesh"), parser=make_refine_parser())
    if not os.path.isfile(p["mesh_refined"]):
        raise StepError("RefineMesh produced no mesh")
    _mark_done(ctx, key)


TEXTURE_MAX_SIZE = 16384     # largest texture page TextureMesh is asked for
TEXTURE_MAX_RUNS = 4         # TextureMesh runs per texturing step, retries included
_RE_PATCH_TOO_BIG = re.compile(r"patch of size (\d+) x (\d+) does not fit")

# OpenMVS 2.4.0 samples the 8-bit photos with a float-pixel stride while it levels
# the seams (TImage::sample passes the result type as the pixel type; fixed upstream
# in October 2026, after 2.4.0). The seam colours it reads are garbage, so the
# leveling paints black and rainbow blotches over the texture (42% of the texels on
# a 16 MP test scene, most of the first WRX atlas), and near the bottom of a large
# photo the read runs past the image buffer: the 0xC0000005 crash of the WRX HIGH run.
# Both leveling passes stay off until Photogram3D ships a fixed OpenMVS.
SEAM_LEVELING = False


def texture_attempts(level, size):
    """TextureMesh settings to try, in order: as requested, then with half-size photos
    (a quarter of the memory) in case the first run fails for another reason."""
    lev = 1 if SEAM_LEVELING else 0
    attempts = [{"level": level, "size": size, "local": lev, "global": lev, "why": ""}]
    if SEAM_LEVELING:
        attempts.append({"level": level, "size": size, "local": 0, "global": 0,
                         "why": "seam leveling off"})
    attempts.append({"level": level + 1, "size": size, "local": 0, "global": 0,
                     "why": "half-resolution photos"})
    return attempts


def _texture_outputs(textured):
    """The OBJ, its .mtl, the texture pages and the .mvs TextureMesh writes, and what the
    seam blending adds (the blended pages and TextureMesh's own .mtl, step_blend)."""
    stem = os.path.splitext(textured)[0]
    folder, base = os.path.split(stem)
    out = [stem + ".mvs", stem + ".mtl", textured, stem + UNBLENDED_MTL]
    try:
        out += [os.path.join(folder, n) for n in os.listdir(folder)
                if n.startswith(base + "_material_")]
    except OSError:
        pass
    return out


def _remove_files(paths):
    for path in paths:
        try:
            os.remove(path)
        except OSError:
            pass


def _watch_texture(parser, seen):
    """Progress parser that also remembers a patch too large for the texture page."""
    def parse(line):
        m = _RE_PATCH_TOO_BIG.search(line)
        if m:
            seen["patch"] = (int(m.group(1)), int(m.group(2)))
        return parser(line)
    return parse


def _next_pow2(n):
    size = 1
    while size < n:
        size *= 2
    return size


def _photo_side(ctx, level, min_side=640):
    """Long side in px of the largest undistorted photo at an OpenMVS resolution level."""
    try:
        cams, _ = colmap_io.read_model(os.path.join(ctx["paths"]["dense"], "sparse"))
        side = max(max(c["width"], c["height"]) for c in cams.values())
    except (OSError, ValueError):
        return 0
    while level > 0 and side // 2 >= min_side:
        side //= 2
        level -= 1
    return side


def step_texture(job, ctx):
    p, cfg, prm = ctx["paths"], ctx["cfg"], ctx["params"]
    key = "texture"
    src_mesh = p["mesh_refined"] if ctx["do_refine"] else p["mesh"]
    tex_size = int(cfg.get("texture_size") or prm["texture_size"])
    dense_model = os.path.join(p["dense"], "sparse")
    n = colmap_io.count_registered(dense_model)
    spread = TEXTURE_VIEW_ANGLE if cfg.get("spread_views", True) and n > TEXTURE_SUBSET_MIN else 0.0
    # "leveling" makes Resume redo textures made with the broken seam leveling (0.2.0)
    ctx["sig_chain"][key] = {"prev": ctx["sig_chain"].get("refine") or ctx["sig_chain"].get("mesh"),
                             "size": tex_size, "rl": prm["texture_res_level"],
                             "leveling": SEAM_LEVELING}
    if spread:
        ctx["sig_chain"][key]["spread"] = [spread, TEXTURE_SUBSET_MIN]
    evened = ctx.get("texture_images_dir")
    if evened:
        ctx["sig_chain"][key]["even"] = ctx["sig_chain"].get("even")
    if _resumable(job, ctx, key, [p["textured"]]):
        ctx["result"]["mesh"] = p["textured"]
        return
    _invalidate_from(ctx, key)
    log = _log(ctx, key)
    scene = p["scene_dense"]
    folder = None
    if spread:
        keep = views.select_views(dense_model, spread)
        if len(keep) < 0.9 * n:
            folder = os.path.join(p["project"], "texture_views")
            _reduced_model(job, ctx, dense_model, keep, os.path.join(folder, "sparse"), log)
            msg = f"Texture from {len(keep)} of {n} photos (about {spread:g}° apart)"
            _note(job, log, msg)
            ctx["result"].setdefault("notes", []).append(msg)
    if evened and folder is None:
        # a scene of its own that reads the evened-out photos (the dense scene reads the others)
        folder = os.path.join(p["project"], "texture_views")
        shutil.rmtree(folder, ignore_errors=True)
        shutil.copytree(dense_model, os.path.join(folder, "sparse"))
    if folder is not None:
        scene = os.path.join(p["mvs"], "scene_texture_views.mvs")
        job.run([openmvs_exe(ctx["ts"], "InterfaceCOLMAP"), "-i", folder, "-o", scene,
                 "--image-folder", evened or os.path.join(p["dense"], "images"), "-w", p["mvs"]],
                log, tail=_tail(ctx, "InterfaceCOLMAP"))
    if evened:
        _note(job, log, "Texture from the evened-out photos")
    attempts = texture_attempts(prm["texture_res_level"], tex_size)
    failures = []
    too_big = None
    i = runs = 0
    while i < len(attempts) and runs < TEXTURE_MAX_RUNS:
        a = attempts[i]
        runs += 1
        if a["why"]:
            _note(job, log, f"Texturing again with {a['why']} (attempt {runs})")
        _remove_files(_texture_outputs(p["textured"]))
        seen = {}
        code = job.run(
            [openmvs_exe(ctx["ts"], "TextureMesh"), scene, "-m", src_mesh,
             "-o", os.path.splitext(p["textured"])[0] + ".mvs", "--export-type", "obj",
             "--resolution-level", str(a["level"]), "--max-texture-size", str(a["size"]),
             "--local-seam-leveling", str(a["local"]), "--global-seam-leveling", str(a["global"])]
            + _omvs_common(ctx),
            log, tail=_tail(ctx, "TextureMesh"), ok_codes=None,
            parser=_watch_texture(make_openmvs_parser(
                {"Initialized views": (0.0, 0.25)},
                {"Assigning the best view to each face completed": 0.6,
                 "Generating texture atlas and image completed": 0.95}), seen))
        if code == 0 and os.path.isfile(p["textured"]):
            if a["why"]:
                ctx["result"].setdefault("warnings", []).append(
                    f"Textured with {a['why']}: TextureMesh failed with the full settings")
            if a["size"] != tex_size and too_big:
                ctx["result"].setdefault("notes", []).append(
                    f"Texture pages are {a['size']} px instead of {tex_size} px: "
                    f"a {too_big[0]} x {too_big[1]} px patch did not fit")
            ctx["result"]["mesh"] = p["textured"]
            _mark_done(ctx, key)
            return
        failures.append(describe_returncode(code) if code else "no OBJ written")
        _note(job, log, f"TextureMesh failed ({failures[-1]})")
        if "patch" in seen:
            # A patch is larger than a texture page and OpenMVS stops instead of splitting
            # it. It only reports the first such patch, but no patch is larger than a
            # photo, so the pages are sized for the photos in one step.
            w, h = too_big = seen["patch"]
            need = _next_pow2(max(w, h, _photo_side(ctx, a["level"])))
            old = a["size"]
            for later in attempts[i:]:
                if need <= TEXTURE_MAX_SIZE:
                    later["size"] = max(later["size"], need)
                else:
                    later["level"] += 1
            _note(job, log, f"A {w} x {h} px patch does not fit a {old} px texture page; using "
                            + (f"{need} px pages" if need <= TEXTURE_MAX_SIZE else "smaller photos"))
            continue
        i += 1
    # Every attempt failed: keep the untextured mesh so the result can still be imported.
    _remove_files(_texture_outputs(p["textured"]))
    msg = (f"Texturing failed ({failures[-1] if failures else 'not run'}); "
           "the untextured mesh was kept. See the texture log.")
    _note(job, log, msg)
    ctx["result"].setdefault("warnings", []).append(msg)
    ctx["result"]["mesh"] = src_mesh


BLEND_SUFFIX = "_blended"              # the blended texture pages: <page>_blended.jpg
UNBLENDED_MTL = "_unblended.mtl"       # TextureMesh's own .mtl, kept while the pages are blended


def unblend(textured):
    """Put the texture back as TextureMesh made it: its own .mtl, no blended pages."""
    stem = os.path.splitext(textured)[0]
    folder, base = os.path.split(stem)
    raw = stem + UNBLENDED_MTL
    if os.path.isfile(raw):
        os.replace(raw, stem + ".mtl")
    try:
        names = os.listdir(folder)
    except OSError:
        names = []
    _remove_files([os.path.join(folder, n) for n in names
                   if n.startswith(base + "_material_") and os.path.splitext(n)[0].endswith(BLEND_SUFFIX)])


def step_blend(job, ctx):
    """Blend the texture's seams (blend_seams.py, 0.3.12). TextureMesh's own seam leveling is
    off (the OpenMVS 2.4.0 bug, DEVELOPMENT.md section 5), so where patches from photos taken
    at different exposures meet, the texture showed light and dark patches. The pages
    TextureMesh made stay as they are: the blended pages are written next to them
    (``..._blended.jpg``), the .mtl names those, and TextureMesh's own .mtl is kept as
    ``..._unblended.mtl``. Off, it puts that .mtl back. Turning it on or off for a finished
    run redoes only this step. If it fails, the texture stays as TextureMesh made it."""
    p, cfg = ctx["paths"], ctx["cfg"]
    key = "blend"
    on = bool(cfg.get("blend_seams", True))
    ctx["sig_chain"][key] = {"prev": ctx["sig_chain"].get("texture"), "v": 1, "on": on}
    textured = p["textured"]
    stem = os.path.splitext(textured)[0]
    mtl, raw = stem + ".mtl", stem + UNBLENDED_MTL
    if ctx["result"].get("mesh") != textured or not os.path.isfile(mtl):
        job.add_line("No texture to blend")
        return
    if _resumable(job, ctx, key, [mtl]):
        return
    _invalidate_from(ctx, key)
    if not on:
        unblend(textured)
        _mark_done(ctx, key)
        return
    log = _log(ctx, key)
    if not os.path.isfile(raw):
        shutil.copy2(mtl, raw)                     # TextureMesh's own, kept (the .mtl is replaced last)
    site = formats.paths(cfg.get("tools_root") or p["project"])["site"]
    python = cfg.get("python") or sys.executable
    if formats.runtime_info(site) is None:
        try:                                       # Pillow: downloaded once (about 16 MB)
            formats.install_runtime(job, python, site, log)
        except StepError as exc:
            unblend(textured)
            msg = (f"Seams not blended: {exc}. The texture is as TextureMesh made it; Blend Seams "
                   "off skips this step")
            _note(job, log, msg)
            ctx["result"].setdefault("warnings", []).append(msg)
            return
    seen = {}

    def parse(text):
        for head in ("Before:", "After:"):
            if text.startswith(head):
                seen[head] = text
        if text.startswith("Before:"):
            return 0.3, "solving the corrections"
        if text.startswith("Solved:"):
            return 0.5, "writing the texture"
        m = re.match(r"Page (\d+)/(\d+)", text)
        if m:
            return 0.5 + 0.48 * int(m.group(1)) / max(1, int(m.group(2))), f"page {m.group(1)}/{m.group(2)}"
        if text.startswith("Read "):
            return 0.1, "sampling the seams"
        return None

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blend_seams.py")
    code = job.run([python, "-I", script, "--site", site, "--obj", textured, "--mtl-in", raw,
                    "--mtl-out", mtl, "--suffix", BLEND_SUFFIX], log, parser=parse, ok_codes=None)
    if code != 0:
        unblend(textured)
        msg = (f"Seams not blended ({describe_returncode(code)}); the texture is as TextureMesh made it. "
               "See the blend log; Blend Seams off skips this step")
        _note(job, log, msg)
        ctx["result"].setdefault("warnings", []).append(msg)
        return
    m0 = re.search(r"median ([\d.]+) stops, 90% ([\d.]+)", seen.get("Before:", ""))
    m1 = re.search(r"median ([\d.]+) stops, 90% ([\d.]+)", seen.get("After:", ""))
    if m0 and m1:
        ctx["result"].setdefault("notes", []).append(
            f"Texture seams blended: steps across them {m0.group(1)} -> {m1.group(1)} stops "
            f"(median), {m0.group(2)} -> {m1.group(2)} (90%)")
    _mark_done(ctx, key)


JOIN_FILE = "photogram3d_join.json"    # in a joined run's folder: the two runs and the join


def step_join(job, ctx):
    """The photos, cameras and dense cloud of a run joining two scans (joinrun.py, 0.3.13):
    the second run's moved into the first's coordinates, no depth maps computed again."""
    from . import joinrun
    p, cfg = ctx["paths"], ctx["cfg"]
    key = "join"
    j = cfg["join"]
    pa, pb = project_paths(j["a"]), project_paths(j["b"])
    M = j["matrix"]
    ctx["result"]["joined_from"] = [j["a"], j["b"]]
    for q, solve in ((pa, j.get("solve_a")), (pb, j.get("solve_b"))):
        if solve and solve_id(q) != solve:
            raise StepError(f"{os.path.basename(os.path.normpath(q['project']))} was solved again since the "
                            "join: its cameras are elsewhere now. Import it, Join again and Complete Joining")
    dense_a, dense_b = (os.path.join(q["mvs"], "scene_dense.ply") for q in (pa, pb))
    for q, path in ((pa, dense_a), (pb, dense_b)):
        if not os.path.isfile(path) or not os.path.isfile(os.path.join(q["dense"], "sparse", "images.bin")):
            raise StepError(f"{os.path.basename(os.path.normpath(q['project']))} has no dense cloud: "
                            "Reconstruct it first (it may stop at the crop box)")

    def stamp(path):
        st = os.stat(path)
        return [st.st_size, int(st.st_mtime)]
    sig = {"a": [os.path.normcase(j["a"]), j.get("solve_a"), stamp(dense_a)],
           "b": [os.path.normcase(j["b"]), j.get("solve_b"), stamp(dense_b)],
           "m": [[round(float(v), 9) for v in row] for row in M], "v": 2}       # 2: photos shared (0.3.14)
    ctx["sig_chain"][key] = sig
    ctx["sig_chain"]["densify"] = sig                  # what the mesh step follows
    ctx["sig_chain"]["undistort"] = sig                # and the evening of the photos
    # photos stored on their side, in either run (the importer turns their cameras upright)
    prefix = joinrun.second_prefix(os.path.join(pa["dense"], "sparse"))
    ctx["upright"] = run_upright(pa)
    ctx["upright"].update({prefix + k: v for k, v in run_upright(pb).items()})
    # the videos the two runs were made from, for the joined scan's tracked cameras (0.3.15)
    ctx["result"]["videos"] = joined_videos(j)
    scene_ply = os.path.join(p["mvs"], "scene_dense.ply")
    if _resumable(job, ctx, key, [p["scene_dense"], scene_ply]):
        try:
            with open(os.path.join(p["project"], JOIN_FILE), "r", encoding="utf-8") as f:
                desc = json.load(f)
            ctx["registered"] = ctx["num_images"] = sum(desc["photos"])
            if "videos" not in desc:          # joined by 0.3.14: a scan joined onto it needs them
                desc["videos"] = ctx["result"]["videos"]
                with open(os.path.join(p["project"], JOIN_FILE), "w", encoding="utf-8") as f:
                    json.dump(desc, f, indent=2)
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return
    _invalidate_from(ctx, key)
    log = _log(ctx, key)
    box_path = os.path.join(p["mvs"], CROP_ROI)
    prev = joined_description(p["project"])
    user_box = None
    if prev is not None and os.path.normcase(prev["a"]) == os.path.normcase(j["a"]) \
            and prev.get("solve_a") == j.get("solve_a"):
        try:      # the box Build Mesh saved, in the first scan's coordinates: kept when a cloud changes
            with open(box_path, "rb") as f:
                user_box = f.read()
        except OSError:
            pass
    for folder in (p["dense"], p["mvs"]):
        shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(p["mvs"])
    if user_box is not None:
        with open(box_path, "wb") as f:
            f.write(user_box)
    na, nb, renamed = joinrun.joined_model(os.path.join(pa["dense"], "sparse"),
                                           os.path.join(pb["dense"], "sparse"), M,
                                           os.path.join(p["dense"], "sparse"))
    joinrun.link_photos(os.path.join(pa["dense"], "images"), os.path.join(p["dense"], "images"))
    joinrun.link_photos(os.path.join(pb["dense"], "images"), os.path.join(p["dense"], "images"), renamed)
    _note(job, log, f"Joining {na} photos of {os.path.basename(os.path.normpath(j['a']))} and {nb} of "
                    f"{os.path.basename(os.path.normpath(j['b']))}")
    job.set_fraction(0.1, "joining the dense clouds")

    def progress(done, total):
        job.set_fraction(0.1 + 0.75 * done / max(1, total), f"{done:,} of {total:,} points moved")
    n = joinrun.merge_dense(dense_a, dense_b, M, na, scene_ply, progress=progress)
    _note(job, log, f"Joined dense cloud: {n:,} points")
    # where both saw the surface, each point gets the other source's photos too (without, the
    # mesh step's graph cut can run without end on two interleaved sets of points)
    job.set_fraction(0.85, "sharing the photos where the two clouds overlap")
    shared = joinrun.share_photos(scene_ply, joinrun.read_ply_header(dense_a)[1])
    _note(job, log, f"Photos shared where the clouds overlap: {shared:,} points ({shared / max(1, n):.0%})")
    prev_a, prev_b = (os.path.join(q["mvs"], CLOUD_PREVIEW) for q in (pa, pb))
    if os.path.isfile(prev_a) and os.path.isfile(prev_b):
        joinrun.merge_previews(prev_a, prev_b, M, os.path.join(p["mvs"], CLOUD_PREVIEW))
    job.run([openmvs_exe(ctx["ts"], "InterfaceCOLMAP"), "-i", p["dense"], "-o", p["scene_dense"],
             "--image-folder", os.path.join(p["dense"], "images"), "-w", p["mvs"]],
            log, tail=_tail(ctx, "InterfaceCOLMAP"))
    if not os.path.isfile(p["scene_dense"]):
        raise StepError("InterfaceCOLMAP did not write the joined scene")
    os.makedirs(p["model"], exist_ok=True)
    with open(current(os.path.join(p["model"], SOLVE_FILE)), "w", encoding="utf-8") as f:
        json.dump({"id": j["solve"], "time": time.time()}, f)
    with open(os.path.join(p["project"], JOIN_FILE), "w", encoding="utf-8") as f:
        json.dump(dict(j, photos=[na, nb], scan_name=cfg.get("scan_name", ""), upright=ctx["upright"],
                       videos=ctx["result"]["videos"]), f, indent=2)
    # the box Complete Joining starts the joined scan with (both scans' boxes): the crop
    # stage shows it as the user's box, and Build Mesh saves the box over it
    if cfg.get("join_box") and os.path.isfile(cfg["join_box"]):
        shutil.copyfile(cfg["join_box"], box_path)
    ctx["registered"] = ctx["num_images"] = na + nb
    _mark_done(ctx, key)


def run_videos(q):
    """The videos of a run, for tracked cameras: [{"project", "video", "frames_json",
    "model_distorted", "matrix", "solve"}], "matrix" taking the video's solve (its id
    "solve") into the run's coordinates (None: they are the same). A video run has its own; a
    joined run the ones it was joined from; a photo run none."""
    desc = joined_description(q["project"])
    if desc is not None:
        return joined_videos(desc)
    meta = video.load_frames_json(q["frames"])
    if not meta or not os.path.isdir(q["model"]):
        return []
    return [{"project": q["project"], "video": meta.get("video", ""),
             "frames_json": os.path.join(q["frames"], video.FRAMES_JSON), "model_distorted": q["model"],
             "matrix": None, "solve": solve_id(q)}]


def joined_videos(join):
    """run_videos() of a joined run, from its join (cfg["join"] or the join file): the videos
    of the first run as they are, then the second's moved by the join. A join file written
    since 0.3.15 has them; for one joined before, they come from the two runs."""
    if isinstance(join.get("videos"), list):
        return [dict(v) for v in join["videos"]]
    return run_videos(project_paths(join["a"])) + [dict(v, matrix=_compose(join["matrix"], v.get("matrix")))
                                                   for v in run_videos(project_paths(join["b"]))]


def _compose(M, N):
    """M x N as nested lists, N None meaning the identity."""
    if N is None:
        return [list(map(float, row)) for row in M]
    return [[sum(float(M[i][k]) * float(N[k][j]) for k in range(4)) for j in range(4)] for i in range(4)]


def run_upright(q):
    """{photo: quarter turns} of the photos stored on their side in a run: from its COLMAP
    database, or for a joined run (a scan joined onto again) from its join file."""
    desc = joined_description(q["project"])
    if desc is not None:
        return {k: int(v) for k, v in (desc.get("upright") or {}).items()}
    return dict(colmap_io.upright_turns(q["db"]))


def joined_description(project_dir):
    """The join a joined run's folder was made from (step_join's JOIN_FILE), or None for
    any other run."""
    try:
        with open(os.path.join(project_dir, JOIN_FILE), "r", encoding="utf-8") as f:
            desc = json.load(f)
    except (OSError, ValueError, TypeError):
        return None
    keys = ("a", "b", "matrix", "solve_a", "solve_b", "solve")
    return desc if isinstance(desc, dict) and all(k in desc for k in keys) else None


def build_joined(cfg, ts, hw):
    """(job, ctx) of a run joining two scans: cfg["join"] = {"a": the first run's folder,
    "b": the second's, "matrix": the second's coordinates into the first's (4 x 4),
    "solve_a", "solve_b", "solve": the joined run's solve id}; cfg["join_box"] the box the
    joined scan starts with; the mesh and texture settings as for any run.

    With cfg["crop_stage"] (Complete Joining, 0.3.14) it stops at the crop stage: one cloud
    of both scans, the cameras of both and a crop box, imported as a scan of its own. Build
    Mesh on that scan continues it (build() hands such a folder here, cfg["crop_roi"] its box)."""
    params = preset_params(cfg["quality"], hw)
    paths = project_paths(cfg["project_dir"])
    for key in ("project", "logs"):
        os.makedirs(paths[key], exist_ok=True)
    try:
        os.remove(result_path(paths["project"]))
    except OSError:
        pass
    do_refine = cfg.get("refine", "AUTO")
    do_refine = params["refine"] if do_refine == "AUTO" else (do_refine == "ON")
    ctx = {"cfg": cfg, "ts": ts, "hw": hw, "params": params, "paths": paths, "use_gpu": False,
           "dense_engine": "OPENMVS", "do_refine": do_refine,
           "openmvs_cuda": bool(cfg.get("openmvs_cuda")) and openmvs_cuda_compatible(hw),
           "images_dir": "", "state": _load_state(paths["state"]), "sig_chain": {}, "result": {}}
    later = [Step("mesh", "Building mesh (OpenMVS)", step_mesh, 1.0)]
    if do_refine:
        later.append(Step("refine", "Refining mesh detail (OpenMVS)", step_refine, 2.5))
    if cfg.get("texture", True):
        if cfg.get("even_photos"):
            later.append(Step("even", "Evening out exposure and vignetting", step_even, 0.6))
        later += [Step("texture", "Texturing (OpenMVS)", step_texture, 1.5),
                  Step("blend", "Blending texture seams", step_blend, 0.3)]
    later.append(Step("finish", "Finishing", step_finish, 0.05))
    stop_for_crop = bool(cfg.get("crop_stage")) and not cfg.get("crop_roi") and not cfg.get("crop_skip")
    ctx["crop_stage"] = stop_for_crop
    ctx["crop_continuation"] = bool(cfg.get("crop_roi") or cfg.get("crop_skip"))
    join = Step("join", "Joining the two scans", step_join, 1.0)
    steps = [join] + ([Step("crop_prepare", "Preparing the crop box", step_crop_prepare, 0.2)]
                      if stop_for_crop else later)
    ctx["step_order"] = ["join", "crop_prepare"] + [s.key for s in later]
    job = Job(scan_name(cfg) + " joined", steps, cfg["project_dir"])
    ctx["t_build"] = time.time()
    return job, ctx


CLOUD_PREVIEW = "cloud_preview.npz"   # in the mvs folder
CROP_ROI = "crop_roi.txt"             # the user's box, OpenMVS region-of-interest format
PREVIEW_MAX_POINTS = 2_000_000


def up_vector(model_dir):
    """World up from the cameras: the mean of each photo's upward image axis."""
    _, images = colmap_io.read_model(model_dir)
    ups = []
    for im in images.values():
        r = colmap_io.qvec_to_rotmat(im["qvec"])          # world -> camera
        ups.append([-r[1][0], -r[1][1], -r[1][2]])         # camera -Y (image up) in world
    if not ups:
        return (0.0, -1.0, 0.0)
    u = [sum(c) / len(ups) for c in zip(*ups)]
    n = math.sqrt(sum(c * c for c in u)) or 1.0
    return tuple(c / n for c in u)


def _run_identity(ctx):
    """What the importer needs to name and place this run: its attempt number, the input's
    name and, for a video, the file that maps frames to video times (tracked camera)."""
    cfg, p = ctx["cfg"], ctx["paths"]
    out = {"attempt": int(cfg.get("attempt") or 1), "scan": scan_name(cfg),
           "base": cfg.get("base_dir") or p["project"], "solve_id": solve_id(p)}
    if cfg.get("input_mode") == "VIDEO":
        out["video"] = cfg.get("video_path", "")
        out["frames_json"] = os.path.join(p["frames"], video.FRAMES_JSON)
        out["model_distorted"] = p["model"]       # the lens with its distortion, video size
    else:
        # photos stored on their side (portrait shots of a phone): the quarter turns that show
        # them upright, which the importer gives their cameras (0.3.9); a joined run has them
        # from both runs (0.3.13)
        out["upright"] = ctx["upright"] if "upright" in ctx else colmap_io.upright_turns(p["db"])
    return out


def step_crop_prepare(job, ctx):
    """Stop after the dense cloud: a preview cloud and a starting crop box for Blender."""
    p = ctx["paths"]
    _, cloud = _dense_cloud(ctx)
    if not os.path.isfile(cloud):
        raise StepError("no dense point cloud to crop")
    log = _log(ctx, "crop_prepare")
    job.set_fraction(0.1, "reading the dense point cloud")
    total = roi.ply_vertex_count(cloud)
    xyz, rgb = roi.read_points(cloud, max_points=PREVIEW_MAX_POINTS)
    job.set_fraction(0.7, "fitting the crop box")
    preview = os.path.join(p["mvs"], CLOUD_PREVIEW)
    roi.save_preview(preview, xyz, rgb)
    # around everything but stray points; the user shrinks it to the object
    box = roi.fit_box(xyz, up=up_vector(os.path.join(p["dense"], "sparse")), lo=0.5, hi=99.5)
    box.ext *= 1.03
    user_box = None
    user_path = os.path.join(p["mvs"], CROP_ROI)
    if os.path.isfile(user_path):
        try:
            user_box = roi.OBB.load(user_path).as_dict()
        except (OSError, ValueError):
            user_box = None
    _note(job, log, f"Dense cloud: {total:,} points ({len(xyz):,} shown). Adjust the crop box, "
                    "then press Build Mesh")
    ctx["result"].update({
        "stage": "crop", "cloud_preview": preview, "cloud_points": total,
        "roi_fit": box.as_dict(), "roi_user": user_box,
        "model": os.path.join(p["dense"], "sparse"), "images": os.path.join(p["dense"], "images"),
        "project": p["project"], "registered": ctx.get("registered"),
        "num_images": ctx.get("num_images")})
    ctx["result"].update(_run_identity(ctx))
    with job.lock:
        job.result.update(ctx["result"])
    job.set_fraction(1.0, "waiting for the crop box")


# ---------------------------------------------------------------------------
# Gaussian splats (0.3.5)
# ---------------------------------------------------------------------------

SPLAT_STEPS_DEFAULT = 30000          # Brush's own default
SPLAT_RESOLUTION_DEFAULT = 1920      # Brush's own default (long side of the training photos)


def step_splats(job, ctx):
    """Gaussian splats with Brush on the undistorted photos (splats.py), cropped to the box.

    Training depends only on the photos and cameras (the undistort step) and the training
    settings. The crop is cheap and made every time, so a new box and Build Mesh again
    crop the trained splats again without training."""
    from . import splats, tools
    p, cfg = ctx["paths"], ctx["cfg"]
    key = "splats"
    steps = int(cfg.get("splat_steps") or SPLAT_STEPS_DEFAULT)
    res = int(cfg.get("splat_resolution") or SPLAT_RESOLUTION_DEFAULT)
    gpu = best_gpu(ctx["hw"])
    budget = splats.max_splats_for(gpu["vram_mb"] if gpu else 0)
    out = os.path.join(p["project"], "splats")
    full = os.path.join(out, splats.SPLATS_FULL_PLY)
    log = _log(ctx, key)
    ctx["sig_chain"][key] = {"prev": ctx["sig_chain"].get("undistort"), "steps": steps,
                             "res": res, "max": budget, "brush": splats.BRUSH_VERSION}
    evened = ctx.get("texture_images_dir")
    if evened:
        ctx["sig_chain"][key]["even"] = ctx["sig_chain"].get("even")
    if not _resumable(job, ctx, key, [full]):
        _invalidate_from(ctx, key)
        exe = splats.find_brush(cfg.get("tools_root") or "", cfg.get("brush_path") or "")
        if not exe:
            def download(url, dest, progress):
                try:
                    return tools.download(url, dest, progress=progress, cancel=job.cancel_event)
                except tools.Cancelled:
                    job.check_cancel()
                    raise
                except tools.InstallError as exc:
                    raise StepError(f"Could not download Brush ({exc})") from exc
            splats.install_brush(job, cfg.get("tools_root") or p["project"], log, download)
            exe = splats.find_brush(cfg.get("tools_root") or p["project"])
            if not exe:
                raise StepError("Brush could not be installed (see the log)")
        os.makedirs(out, exist_ok=True)
        for old in os.listdir(out):
            if old.lower().endswith(".ply"):
                os.remove(os.path.join(out, old))
        data = os.path.join(out, "data")
        n = splats.prepare_dataset(p["dense"], data, images=evened)
        _note(job, log, f"Training Gaussian splats on {n} photos: {steps:,} steps, photos up to "
                        f"{res} px, at most {budget:,} splats")
        parser = splats.make_parser(steps)
        # Brush keeps its GPU tuning cache in "target" under its working folder: the tools
        # folder, so that it is reused by every project
        work = splats.paths(cfg.get("tools_root") or out)["dir"]
        os.makedirs(work, exist_ok=True)
        try:
            job.run(splats.train_args(exe, data, out, steps, res, budget), log, cwd=work,
                    parser=parser, env={"RUST_LOG": splats.RUST_LOG})
        finally:
            shutil.rmtree(data, ignore_errors=True)
        if not os.path.isfile(full):
            raise StepError("Brush finished without writing the splats (see the log)")
        _mark_done(ctx, key)
    job.set_fraction(0.96, "cropping the splats")
    comments, data = splats.read_ply(full)
    roi_path = cfg.get("crop_roi") if cfg.get("crop_roi") and os.path.isfile(cfg["crop_roi"]) else None
    kept = splats.crop(data, roi.OBB.load(roi_path) if roi_path else None)
    if not len(kept):
        raise StepError("No splat is inside the crop box")
    splats.write_ply(os.path.join(out, splats.SPLATS_PLY), comments, kept)
    _note(job, log, f"Splats: {len(kept):,} of {len(data):,}"
                    + (" inside the crop box" if roi_path else " (no crop box)"))
    ctx["result"]["splats"] = os.path.join(out, splats.SPLATS_PLY)


def step_finish(job, ctx):
    p = ctx["paths"]
    res = ctx["result"]
    if "mesh" not in res:
        res["mesh"] = p["mesh_refined"] if ctx["do_refine"] else p["mesh"]
    res["model"] = os.path.join(p["dense"], "sparse")
    res["images"] = os.path.join(p["dense"], "images")
    res["project"] = p["project"]
    res["registered"] = ctx.get("registered") or colmap_io.count_registered(p["model"])
    res["num_images"] = ctx.get("num_images")
    res["timings"] = dict(job.timings)
    res.update(_run_identity(ctx))
    with open(result_path(p["project"]), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    with job.lock:
        job.result.update(res)
    job.set_fraction(1.0, "done")


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

def validate(cfg, ts):
    """Return a list of human-readable problems (empty if ready to run)."""
    problems = []
    if not ts.get("colmap"):
        problems.append("COLMAP is not installed")
    if not ts.get("openmvs_dir"):
        problems.append("OpenMVS is not installed")
    if cfg.get("join") or joined_description(cfg.get("project_dir") or "") is not None:
        return problems                  # its photos and clouds come from the two runs
    if cfg["input_mode"] == "VIDEO":
        if not os.path.isfile(cfg.get("video_path", "")):
            problems.append("Choose a video file")
        if not ts.get("ffmpeg"):
            problems.append("FFmpeg is not installed (needed for video)")
    else:
        folder = cfg.get("images_dir", "")
        if not os.path.isdir(folder):
            problems.append("Choose a folder with photos")
        elif len(formats.plan(folder, cfg.get("raw_pairs", "JPEG") == "RAW")["use"]) < 3:
            problems.append("The photo folder needs at least 3 photos (JPG, PNG, TIFF, HEIC, "
                            "camera RAW, EXR, WebP, ...)")
    if not cfg.get("project_dir"):
        problems.append("Choose an output folder")
    return problems


def build(cfg, ts, hw):
    """Create (job, ctx) for the given settings. A joined run's folder (Build Mesh on a
    joined scan) is continued by build_joined, whatever the panel's input."""
    desc = joined_description(cfg.get("project_dir") or "") if not cfg.get("join") else None
    if desc is not None:
        return build_joined(dict(cfg, join=desc, input_mode="IMAGES", scan_name=desc.get("scan_name") or
                                 scan_name(cfg)), ts, hw)
    params = preset_params(cfg["quality"], hw)
    paths = project_paths(cfg["project_dir"])
    for key in ("project", "colmap", "logs"):
        os.makedirs(paths[key], exist_ok=True)
    # A result file from an earlier run would point "Import Result" at meshes this
    # run is about to replace; the Finishing step writes a new one.
    try:
        os.remove(result_path(paths["project"]))
    except OSError:
        pass

    gpus = hw.get("gpus") or []
    use_gpu = bool(gpus) and cfg.get("use_gpu", True)
    # COLMAP's dense module needs the CUDA build and an NVIDIA GPU.
    dense_engine = cfg.get("dense_engine", "OPENMVS")
    engine_note = gpu_depth_unavailable(cfg, ts, hw) if dense_engine in ("COLMAP", "HYBRID") else ""
    if engine_note:
        dense_engine = "OPENMVS"
    do_refine = cfg.get("refine", "AUTO")
    do_refine = params["refine"] if do_refine == "AUTO" else (do_refine == "ON")

    ctx = {
        "cfg": cfg, "ts": ts, "hw": hw, "params": params, "paths": paths,
        "use_gpu": use_gpu, "dense_engine": dense_engine, "do_refine": do_refine,
        "openmvs_cuda": bool(cfg.get("openmvs_cuda")) and openmvs_cuda_compatible(hw),
        "images_dir": cfg.get("images_dir", ""), "state": _load_state(paths["state"]),
        "sig_chain": {}, "result": {},
    }

    steps = []
    if cfg["input_mode"] == "VIDEO":
        ctx["sig_chain"]["frames"] = {"video": cfg["video_path"],
                                      "mtime": os.path.getmtime(cfg["video_path"]) if os.path.exists(cfg["video_path"]) else 0,
                                      "fps": cfg.get("video_fps"), "all": bool(cfg.get("video_all_frames")),
                                      "start": float(cfg.get("video_start") or 0.0),
                                      "end": float(cfg.get("video_end") or 0.0),
                                      "blur": cfg.get("video_blur_filter"),
                                      "names": 2}   # 0.3.4: files named by frame number + frames.json
        if int(cfg.get("photo_max_size") or 0):
            ctx["sig_chain"]["frames"]["max"] = int(cfg["photo_max_size"])
        if video.frames_mode(cfg) == "AUTO":     # new keys only when used: older runs resume
            ctx["sig_chain"]["frames"].update(
                {"mode": "AUTO", "fps": None, "all": False, "auto": 1,
                 "max_frames": int(cfg.get("video_max_frames") or video.AUTO_MAX_FRAMES)})
        steps.append(Step("frames", "Extracting video frames", step_frames, 1.0))
    else:
        steps.append(Step("photos", "Checking photo formats", step_photos, 0.5))
    ai = bool(cfg.get("ai_features"))
    directml = ai and cfg.get("ai_device") == "DIRECTML"
    auto = ai and cfg.get("ai_device") == "AUTO"        # Automatic, not tested yet on this computer
    if (directml or auto) and not ai_setup_ready(cfg.get("tools_root") or ""):
        steps.append(Step("ai_setup", "Setting up AI Features on the GPU (once)", step_ai_setup, 0.5))
    if auto:
        steps.append(Step("ai_test", "Testing where AI Features run fastest (once)", step_ai_test, 0.5))
    feat_label = ("Finding AI features (GPU, DirectML)" if directml else
                  "Finding AI features" if auto else "Finding features (COLMAP)")
    match_label = ("Matching images (GPU, DirectML)" if directml else
                   "Matching images (AI)" if auto else "Matching images (COLMAP)")
    steps += [
        Step("features", feat_label, step_features, 1.0),
        Step("matching", match_label, step_matching, 1.5),
        Step("mapping", "Solving cameras (COLMAP)", step_mapping, 1.5),
        Step("orient", "Aligning up-axis", step_orient, 0.1),
    ]
    if cfg["input_mode"] == "VIDEO" and cfg.get("video_track", "EVERY") in TRACK_EVERY \
            and cfg.get("video_camera", True):
        steps.append(Step("track", "Tracking the camera on every frame", step_track, 1.5))
    steps += [
        Step("undistort", "Undistorting images", step_undistort, 0.4),
    ]
    if dense_engine == "COLMAP":
        steps.append(Step("colmap_dense", "Depth maps on the GPU (COLMAP CUDA)", step_colmap_dense, 4.0))
        steps.append(Step("mvs_import", "Preparing OpenMVS scene", step_mvs_import, 0.4))
        steps.append(Step("densify", "Fusing the GPU depth maps (OpenMVS)", step_densify, 0.4))
    elif dense_engine == "HYBRID":
        steps.append(Step("mvs_import", "Preparing OpenMVS scene", step_mvs_import, 0.2))
        steps.append(Step("densify", "Dense point cloud (OpenMVS on the CPU + COLMAP on the GPU)",
                          step_densify_hybrid, 3.0))
    else:
        steps.append(Step("mvs_import", "Preparing OpenMVS scene", step_mvs_import, 0.2))
        steps.append(Step("densify", "Dense point cloud (OpenMVS)", step_densify, 4.0))
    steps.append(Step("mesh", "Building mesh (OpenMVS)", step_mesh, 1.0))
    if do_refine:
        steps.append(Step("refine", "Refining mesh detail (OpenMVS)", step_refine, 2.5))
    if cfg.get("even_photos") and (cfg.get("texture", True) or cfg.get("splats")):
        steps.append(Step("even", "Evening out exposure and vignetting", step_even, 0.6))
    if cfg.get("texture", True):
        steps.append(Step("texture", "Texturing (OpenMVS)", step_texture, 1.5))
        steps.append(Step("blend", "Blending texture seams", step_blend, 0.3))
    if cfg.get("splats"):
        steps.append(Step("splats", "Gaussian splats (Brush)", step_splats, 3.0))
    steps.append(Step("finish", "Finishing", step_finish, 0.05))
    order = [s.key for s in steps]

    # Crop stage: stop after the dense cloud so the user can place the crop box.
    # The later steps stay in the bookkeeping order so invalidation still reaches them.
    stop_for_crop = bool(cfg.get("crop_stage")) and not cfg.get("crop_roi") and not cfg.get("crop_skip")
    ctx["crop_stage"] = stop_for_crop
    # Build Mesh and Without Crop continue from the crop stage: the cameras must stay
    ctx["crop_continuation"] = bool(cfg.get("crop_roi") or cfg.get("crop_skip"))
    cut = order.index("mesh")
    order = order[:cut] + ["crop_prepare"] + order[cut:]
    if stop_for_crop:
        steps = steps[:cut] + [Step("crop_prepare", "Preparing the crop box", step_crop_prepare, 0.2)]
    ctx["step_order"] = order

    k = int(cfg.get("attempt") or 1)
    name = scan_name(cfg) + (f" {k}" if k > 1 else "")
    job = Job(name, steps, cfg["project_dir"])
    if engine_note:
        msg = engine_note + "; depth maps are computed with OpenMVS on the CPU instead"
        job.add_line(msg)
        ctx["result"].setdefault("notes", []).append(msg)
    ctx["t_build"] = time.time()
    return job, ctx


def gpu_depth_unavailable(cfg, ts, hw):
    """Why COLMAP's GPU depth maps cannot run here ("" when they can)."""
    if not (hw.get("gpus") or []):
        return "COLMAP GPU depth maps need an NVIDIA GPU, and none was detected"
    if not cfg.get("use_gpu", True):
        return "COLMAP GPU depth maps need Use NVIDIA GPU switched on"
    if ts.get("colmap_variant") not in ("cuda", None):
        return "COLMAP GPU depth maps need the CUDA build of COLMAP (Install Tools picks it when it sees the GPU)"
    return ""


DENSE_ENGINE_NOTES = {
    "COLMAP": "Experimental: depth maps on the GPU with COLMAP, meshing with OpenMVS",
    "HYBRID": "Experimental: the CPU and the GPU share the first depth-map pass, "
              "then OpenMVS finishes as in the CPU path",
}
