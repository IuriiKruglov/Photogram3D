# SPDX-License-Identifier: GPL-3.0-or-later
# Photogram3D - photos or video to a textured mesh with COLMAP + OpenMVS
# Copyright (C) 2026 Kruglov Iurii
# Authors: Kruglov Iurii, Anthropic Claude AI
# License: GNU General Public License (GPL) v3 or later, https://www.blender.org/about/license/
"""Where AI Features (ALIKED + LightGlue) run (0.3.9): the GPU of any make, NVIDIA's GPU path
or the CPU, chosen by the user or from a quick test of the computer.

- DIRECTML: Photogram3D's own runner (ai_features.py) runs COLMAP's two model files with ONNX
  Runtime DirectML on any DirectX 12 GPU (the runtime AI Polish installs), and hands COLMAP
  the keypoints (`colmap feature_importer`) and the matches (`colmap geometric_verifier`).
- NVIDIA: COLMAP's own GPU path, which needs NVIDIA's cuBLAS, cuDNN and CUDA runtime next to
  COLMAP: downloaded from NVIDIA's packages on PyPI (1.3 GB, NVIDIA's license: free to use,
  not open source) into the tools folder, only when the user asks for them.
- CPU: COLMAP on the CPU, set up for it as in 0.3.7.

No Blender imports: the pipeline and the tests use this module directly.
"""

import hashlib
import json
import os
import shutil
import sqlite3
import sys
import time
import zipfile

from . import tools
from .common import IS_WINDOWS, human_bytes, human_time

DEVICES = ("AUTO", "DIRECTML", "NVIDIA", "CPU")
LABELS = {"DIRECTML": "GPU (DirectML)", "NVIDIA": "GPU (NVIDIA libraries)", "CPU": "CPU"}

# COLMAP 4's own model files (src/colmap/feature/resources.h), checked by their SHA-256
MODELS = {
    "aliked": {"file": "aliked-n16rot.onnx", "size": 2_997_054,
               "url": "https://github.com/colmap/colmap/releases/download/3.13.0/aliked-n16rot.onnx",
               "sha256": "39c423d0a6f03d39ec89d3d1d61853765c2fb6a8b8381376c703e5758778a547"},
    "lightglue": {"file": "aliked-lightglue.onnx", "size": 45_804_950,
                  "url": "https://github.com/colmap/colmap/releases/download/3.13.0/aliked-lightglue.onnx",
                  "sha256": "b9a5de7204648b18a8cf5dcac819f9d30de1a5961ef03756803c8b86c2dceb8d"},
}

# NVIDIA's libraries for COLMAP's ONNX Runtime CUDA provider (CUDA 12, cuDNN 9), as NVIDIA
# publishes them on PyPI for Windows. cuFFT, which the provider loads only for FFT
# operators, is not needed by these two models.
NVIDIA_WHEELS = (
    {"name": "nvidia-cuda-runtime-cu12", "version": "12.9.79", "size": 3_591_604,
     "url": "https://files.pythonhosted.org/packages/59/df/e7c3a360be4f7b93cee39271b792669baeb3846c58a4df6dfcf187a7ffab/"
            "nvidia_cuda_runtime_cu12-12.9.79-py3-none-win_amd64.whl",
     "sha256": "8e018af8fa02363876860388bd10ccb89eb9ab8fb0aa749aaf58430a9f7c4891",
     "bin": "nvidia/cuda_runtime/bin/", "check": ("cudart64_12.dll",)},
    {"name": "nvidia-cublas-cu12", "version": "12.9.2.10", "size": 553_162_896,
     "url": "https://files.pythonhosted.org/packages/20/e2/fc9a0e985249d873150276d5afb02e39a66817fedbf1a385724393e505ed/"
            "nvidia_cublas_cu12-12.9.2.10-py3-none-win_amd64.whl",
     "sha256": "623f43027d40d44ceadf0043f002bd25cf353e8f13ce90b9a87057019f560661",
     "bin": "nvidia/cublas/bin/", "skip": ("nvblas64_12.dll",), "check": ("cublas64_12.dll", "cublasLt64_12.dll")},
    {"name": "nvidia-cudnn-cu12", "version": "9.27.0.42", "size": 743_068_852,
     "url": "https://files.pythonhosted.org/packages/aa/38/f856579877f7c1c5066e61182e7de7bc27bf35a78c8d1b0fa592e6985bc4/"
            "nvidia_cudnn_cu12-9.27.0.42-py3-none-win_amd64.whl",
     "sha256": "06e9b0026f3bad97d2b58666330fabec04fe1672f776661ecb0ce0029c27f142",
     "bin": "nvidia/cudnn/bin/", "check": ("cudnn64_9.dll",)},
    # NVRTC, CUDA's runtime compiler (0.3.10): cuDNN 9 compiles some of its kernels when they
    # are needed and leaves those engines out without it ("no valid engine configs", as a
    # GTX 1650 Ti reported for ALIKED). ONNX Runtime's own GPU package lists it among the
    # CUDA libraries it needs (onnxruntime-gpu[cuda]).
    {"name": "nvidia-cuda-nvrtc-cu12", "version": "12.9.86", "size": 76_408_187,
     "url": "https://files.pythonhosted.org/packages/52/de/823919be3b9d0ccbf1f784035423c5f18f4267fb0123558d58b813c6ec86/"
            "nvidia_cuda_nvrtc_cu12-12.9.86-py3-none-win_amd64.whl",
     "sha256": "72972ebdcf504d69462d3bcd67e7b81edd25d0fb85a2c46d3ea3517666636349",
     "bin": "nvidia/cuda_nvrtc/bin/", "skip": ("nvrtc64_120_0.alt.dll",),
     "check": ("nvrtc64_120_0.dll", "nvrtc-builtins64_129.dll")},
)
NVIDIA_DOWNLOAD = sum(w["size"] for w in NVIDIA_WHEELS)
NVIDIA_LICENSE_URL = "https://docs.nvidia.com/cuda/eula/index.html"
NVIDIA_MARKER = "photogram3d_nvidia.json"

TEST_FILE = "ai_features_test.json"          # in the tools folder: the last quick test
TEST_SIZES = (1280, 1600, 2048, 2400)        # ALIKED photo sizes tried on the GPU
GPU_DEFAULT_SIZE = 1600                      # without a test
SEQUENTIAL_OVERLAP = 20                      # as the pipeline's sequential_matcher


# ---------------------------------------------------------------------------- files

def models_dir(tools_root):
    return os.path.join(tools_root, "ai", "models")


def model_path(tools_root, key):
    return os.path.join(models_dir(tools_root), MODELS[key]["file"])


def models_ready(tools_root):
    return all(os.path.isfile(model_path(tools_root, k)) and
               os.path.getsize(model_path(tools_root, k)) == MODELS[k]["size"] for k in MODELS)


def sha256_of(path, cancel=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            if cancel is not None and cancel.is_set():
                raise tools.Cancelled()
            chunk = f.read(1 << 22)
            if not chunk:
                return h.hexdigest()
            h.update(chunk)


def download_checked(url, dest, sha256, progress=None, cancel=None, magic=b"PK"):
    """tools.download, then the file must have the SHA-256 it is published with."""
    tools.download(url, dest, progress=progress, cancel=cancel, magic=magic)
    got = sha256_of(dest, cancel)
    if got != sha256:
        os.remove(dest)
        raise tools.InstallError(f"{os.path.basename(dest)}: checksum mismatch (got {got[:12]}..., "
                                 f"expected {sha256[:12]}...)")
    return dest


def download_models(tools_root, progress=None, cancel=None):
    """COLMAP's ALIKED and LightGlue models (49 MB), unless already there."""
    todo = [k for k in MODELS if not (os.path.isfile(model_path(tools_root, k)) and
                                      os.path.getsize(model_path(tools_root, k)) == MODELS[k]["size"])]
    total = sum(MODELS[k]["size"] for k in todo) or 1
    done = 0
    for k in todo:
        m = MODELS[k]

        def prog(n, _total, base=done):
            if progress:
                progress((base + n) / total, f"{m['file']}: {human_bytes(base + n)} / {human_bytes(total)}")

        download_checked(m["url"], model_path(tools_root, k), m["sha256"], prog, cancel, magic=b"\x08")
        done += m["size"]


def nvidia_dir(tools_root):
    return os.path.join(tools_root, "nvidia", "bin")


def nvidia_ready(tools_root):
    d = nvidia_dir(tools_root)
    return bool(tools_root) and all(os.path.isfile(os.path.join(d, dll)) for dll in tools.ONNX_CUDA_DLLS)


def _nvidia_marker(tools_root):
    try:
        with open(os.path.join(tools_root, "nvidia", NVIDIA_MARKER), "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def nvidia_missing(tools_root):
    """NVIDIA_WHEELS whose DLLs are not in the tools folder (NVRTC for an install made by
    0.3.9; all of them without an install)."""
    if not tools_root:
        return list(NVIDIA_WHEELS)
    d = nvidia_dir(tools_root)
    return [w for w in NVIDIA_WHEELS if not all(os.path.isfile(os.path.join(d, x)) for x in w["check"])]


def nvidia_summary(tools_root):
    """'cuBLAS 12.9, cuDNN 9.27, NVRTC 12.9' from the installed wheels."""
    labels = {"nvidia-cublas-cu12": "cuBLAS", "nvidia-cudnn-cu12": "cuDNN", "nvidia-cuda-nvrtc-cu12": "NVRTC"}
    missing = {w["name"] for w in nvidia_missing(tools_root)}
    wheels = _nvidia_marker(tools_root).get("wheels") or NVIDIA_WHEELS
    out = []
    for w in wheels:
        if w.get("name") in labels and w["name"] not in missing:
            out.append(labels[w["name"]] + " " + ".".join(str(w.get("version", "")).split(".")[:2]))
    return ", ".join(out)


def install_nvidia(tools_root, progress=None, cancel=None, keep_downloads=False):
    """Download NVIDIA's wheels that are not installed yet (checked by SHA-256), unpack their
    DLLs and license texts into tools/nvidia, and remove the wheels. A complete install is
    built next to tools/nvidia and then put in its place; a missing wheel of an existing
    install (NVRTC, added in 0.3.10) is added to it. Returns the list of DLLs."""
    root = os.path.join(tools_root, "nvidia")
    staging = root + ".staging"
    downloads = os.path.join(tools_root, "_downloads")
    wheels = nvidia_missing(tools_root)
    adding = len(wheels) < len(NVIDIA_WHEELS)
    marker = _nvidia_marker(tools_root) if adding else {}
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(os.path.join(staging, "bin"), exist_ok=True)
    os.makedirs(os.path.join(staging, "licenses"), exist_ok=True)
    total = sum(w["size"] for w in wheels) or 1
    done = 0
    dlls = []
    for w in wheels:
        dest = os.path.join(downloads, os.path.basename(w["url"]))

        def prog(n, _total, base=done, name=w["name"]):
            if progress:
                progress(0.95 * (base + n) / total,
                         f"{name}: {human_bytes(base + n)} / {human_bytes(total)}")

        if not (os.path.isfile(dest) and os.path.getsize(dest) == w["size"]
                and sha256_of(dest, cancel) == w["sha256"]):
            download_checked(w["url"], dest, w["sha256"], prog, cancel)
        done += w["size"]
        with zipfile.ZipFile(dest) as z:
            for info in z.infolist():
                name = info.filename
                base = name.rsplit("/", 1)[-1]
                if name.startswith(w["bin"]) and base.lower().endswith(".dll") and base not in w.get("skip", ()):
                    target = os.path.join(staging, "bin", base)
                elif name.endswith("licenses/License.txt") or name.endswith("/LICENSE.txt"):
                    target = os.path.join(staging, "licenses", f"{w['name']}-License.txt")
                else:
                    continue
                if cancel is not None and cancel.is_set():
                    raise tools.Cancelled()
                with z.open(info) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 22)
                if target.lower().endswith(".dll"):
                    dlls.append(base)
        if not keep_downloads:
            os.remove(dest)
    names = {w["name"] for w in wheels}
    kept = [w for w in marker.get("wheels", []) if w.get("name") not in names]
    if adding and not kept:       # no marker: the wheels whose DLLs are there
        kept = [{k: w[k] for k in ("name", "version", "sha256")} for w in NVIDIA_WHEELS if w["name"] not in names]
    info = {"wheels": kept + [{k: w[k] for k in ("name", "version", "sha256")} for w in wheels],
            "dlls": sorted(set(marker.get("dlls", [])) | set(dlls)), "license": NVIDIA_LICENSE_URL,
            "installed": time.strftime("%Y-%m-%d %H:%M")}
    missing = [d for d in tools.ONNX_CUDA_DLLS if d not in info["dlls"]]
    if missing:
        raise tools.InstallError("the NVIDIA packages lack " + ", ".join(missing))
    if adding:
        # into the existing install: each file under a temporary name first, then renamed
        for sub in ("bin", "licenses"):
            os.makedirs(os.path.join(root, sub), exist_ok=True)
            for entry in list(os.scandir(os.path.join(staging, sub))):
                target = os.path.join(root, sub, entry.name)
                shutil.move(entry.path, target + ".new")
                os.replace(target + ".new", target)
        with open(os.path.join(root, NVIDIA_MARKER + ".new"), "w", encoding="utf-8") as f:
            json.dump(info, f, indent=1)
        os.replace(os.path.join(root, NVIDIA_MARKER + ".new"), os.path.join(root, NVIDIA_MARKER))
        shutil.rmtree(staging, ignore_errors=True)
    else:
        with open(os.path.join(staging, NVIDIA_MARKER), "w", encoding="utf-8") as f:
            json.dump(info, f, indent=1)
        shutil.rmtree(root, ignore_errors=True)
        os.replace(staging, root)
    forget_test(tools_root)
    if progress:
        progress(1.0, "NVIDIA libraries installed")
    return sorted(dlls)


def remove_nvidia(tools_root):
    shutil.rmtree(os.path.join(tools_root, "nvidia"), ignore_errors=True)
    forget_test(tools_root)


# ---------------------------------------------------------------------------- the quick test

def read_test(tools_root):
    try:
        with open(os.path.join(tools_root, TEST_FILE), "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    for r in (data.get("results") or {}).values():        # JSON keys are strings
        for k in ("extract", "points"):
            r[k] = {int(size): v for size, v in (r.get(k) or {}).items()}
    return data


def forget_test(tools_root):
    """The last test no longer describes this computer (NVIDIA's libraries added or removed):
    Automatic tests again before its next run."""
    try:
        os.remove(os.path.join(tools_root, TEST_FILE))
    except OSError:
        pass


def write_test(tools_root, data):
    os.makedirs(tools_root, exist_ok=True)
    tmp = os.path.join(tools_root, TEST_FILE + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, os.path.join(tools_root, TEST_FILE))


def parse_bench(lines):
    """The runner's BENCH lines -> {"DIRECTML" | "CPU" | "CUDA": {"extract": {size: s},
    "points": {size: n}, "match": s, "failed": [...]}}"""
    names = {"DmlExecutionProvider": "DIRECTML", "CPUExecutionProvider": "CPU",
             "CUDAExecutionProvider": "CUDA"}
    out = {}
    for ln in lines:
        parts = ln.split()
        if len(parts) < 4 or parts[0] != "BENCH" or parts[1] not in names:
            continue
        r = out.setdefault(names[parts[1]], {"extract": {}, "points": {}, "match": None, "failed": []})
        if parts[2] == "extract" and len(parts) >= 5:
            if parts[4] == "FAILED":
                r["failed"].append(f"ALIKED at {parts[3]} px: " + " ".join(parts[5:]))
            else:
                r["extract"][int(parts[3])] = float(parts[4])
                if len(parts) > 5:
                    r["points"][int(parts[3])] = int(parts[5])
        elif parts[2] == "match":
            if parts[3] == "FAILED":
                r["failed"].append("LightGlue: " + " ".join(parts[4:]))
            elif len(parts) >= 5:
                r["match"] = float(parts[4])
    return out


EXHAUSTIVE_LIMIT = 300      # as pipeline.EXHAUSTIVE_LIMIT: longer videos are matched in sequence


def pair_count(n, video=False):
    """Photo pairs matched for n photos: every pair, or for a video longer than
    EXHAUSTIVE_LIMIT frames COLMAP's sequential pairs (each frame with the 1st, 2nd, 4th, ...
    next, 20 at most)."""
    if not video or n <= EXHAUSTIVE_LIMIT:
        return n * (n - 1) // 2
    return sum(min(SEQUENTIAL_OVERLAP, (n - 1 - i).bit_length()) for i in range(n))


def estimate(result, n, video=False, size=GPU_DEFAULT_SIZE):
    """Seconds for features and matching of n photos from one device's test, or None."""
    if not result or result.get("match") is None:
        return None
    ex = result["extract"]
    if size not in ex:
        if not ex:
            return None
        known = max(ex)
        t = ex[known] * (size / known) ** 2
    else:
        t = ex[size]
    return n * t + pair_count(n, video) * result["match"]


def max_gpu_size(result, ceiling=TEST_SIZES[-1]):
    """The largest photo size ALIKED ran at on the GPU in the test (else 1600)."""
    ok = [s for s in (result or {}).get("extract", {}) if s <= ceiling]
    return max(ok) if ok else GPU_DEFAULT_SIZE


def suggest(test, nvidia_libs, n=42, video=False):
    """(device, [lines]) from the last test: the fastest device that worked."""
    if not test:
        return None, ["Not tested yet"]
    res = test.get("results", {})
    est = {}
    if res.get("DIRECTML"):
        e = estimate(res["DIRECTML"], n, video, max_gpu_size(res["DIRECTML"]))
        if e is not None:
            est["DIRECTML"] = e
    if res.get("NVIDIA") and nvidia_libs:
        e = estimate(res["NVIDIA"], n, video, max_gpu_size(res["NVIDIA"]))
        if e is not None:
            est["NVIDIA"] = e
    if res.get("CPU"):
        e = estimate(res["CPU"], n, video, test.get("cpu_size", GPU_DEFAULT_SIZE))
        if e is not None:
            est["CPU"] = e
    if not est:
        return None, ["The test found no working device"]
    best = min(est, key=est.get)
    lines = [f"{LABELS[d]}: about {human_time(est[d])} for {n} photos" for d in sorted(est, key=est.get)]
    return best, lines


def resolve(pref, tools_root, test=None, has_nvidia_gpu=False, untested="guess"):
    """The device a run uses for AI Features: the user's choice, or for AUTO the test's
    suggestion. Not tested yet: "AUTO" with untested="test" (the run tests first), else the
    guess: NVIDIA's path when its libraries are installed, DirectML on Windows, the CPU
    elsewhere."""
    if pref in ("DIRECTML", "NVIDIA", "CPU"):
        return pref
    libs = nvidia_ready(tools_root)
    test = test if test is not None else read_test(tools_root)
    best, _ = suggest(test, libs)
    if best:
        return best
    if test:
        return "CPU"            # tested, and no GPU path worked here
    if untested == "test":
        return "AUTO"
    if libs and has_nvidia_gpu:
        return "NVIDIA"
    return "DIRECTML" if IS_WINDOWS else "CPU"


# ---------------------------------------------------------------------------- COLMAP database

def image_names(db):
    con = sqlite3.connect(db, timeout=60)
    try:
        return {iid: name for iid, name in con.execute("SELECT image_id, name FROM images")}
    finally:
        con.close()


def write_pairs(db, path, sequential=False, overlap=SEQUENTIAL_OVERLAP):
    """The photo pairs to match, one 'name1<TAB>name2' per line (names may hold spaces): every
    pair, or COLMAP's sequential pairs with quadratic overlap (in name order). Returns the number
    of pairs."""
    names = sorted(image_names(db).values())
    n = len(names)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            if sequential:
                partners = []
                for k in range(overlap):
                    j = i + (1 << k)
                    if j >= n:
                        break
                    partners.append(j)
            else:
                partners = range(i + 1, n)
            for j in partners:
                f.write(f"{names[i]}\t{names[j]}\n")
                count += 1
    return count


def set_descriptors(db, features_dir):
    """Put the ALIKED descriptors (float32, COLMAP's type ALIKED_N16ROT) the runner saved next
    to each feature file into the database, in place of the importer's zero SIFT ones.
    Returns the number of photos."""
    import numpy as np
    con = sqlite3.connect(db, timeout=60)
    try:
        rows = []
        for iid, name in con.execute("SELECT image_id, name FROM images").fetchall():
            path = os.path.join(features_dir, name + ".npy")
            if not os.path.isfile(path):
                continue
            d = np.ascontiguousarray(np.load(path), dtype=np.float32)
            n_kp = con.execute("SELECT rows FROM keypoints WHERE image_id = ?", (iid,)).fetchone()
            if n_kp is None or n_kp[0] != d.shape[0]:
                raise ValueError(f"{name}: {d.shape[0]} descriptors for {n_kp and n_kp[0]} keypoints")
            rows.append((1, d.shape[0], d.shape[1] * 4, d.tobytes(), iid))
        con.executemany("UPDATE descriptors SET type = ?, rows = ?, cols = ?, data = ? WHERE image_id = ?", rows)
        con.commit()
        return len(rows)
    finally:
        con.close()


def runner_script():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_features.py")


def python_for(cfg):
    return cfg.get("python") or sys.executable


# ---------------------------------------------------------------------------- jobs

def setup_runtime_and_models(job, tools_root, python, log):
    """ONNX Runtime (DirectML on Windows, as AI Polish installs it) and COLMAP's two model
    files, each only when missing. Raises StepError / JobCancelled."""
    from . import polish_ai
    from .runner import JobCancelled, StepError
    paths = polish_ai.ai_paths(tools_root)
    if polish_ai.runtime_info(paths["site"]) is None:
        polish_ai.step_runtime(job, {"python": python, "site": paths["site"], "log": log})
    if not models_ready(tools_root):
        try:
            download_models(tools_root, progress=lambda f, note: job.set_fraction(f, note),
                            cancel=job.cancel_event)
        except tools.Cancelled as exc:
            raise JobCancelled() from exc
        except tools.InstallError as exc:
            raise StepError(f"AI model download failed: {exc}") from exc


def setup_ready(tools_root):
    from . import polish_ai
    return bool(tools_root) and polish_ai.runtime_info(polish_ai.ai_paths(tools_root)["site"]) is not None \
        and models_ready(tools_root)


def runtime_directml(tools_root):
    """Whether the installed ONNX Runtime is a DirectML build (not the CPU-only one the setup
    falls back to when no DirectML build loads here). Elsewhere than Windows: True (the runner
    then runs on whatever it has, as before)."""
    from . import polish_ai
    if not IS_WINDOWS:
        return True
    info = polish_ai.runtime_info(polish_ai.ai_paths(tools_root)["site"]) if tools_root else None
    return bool(info) and "DmlExecutionProvider" in (info.get("report") or "")


def _bench_parser(ctx):
    lines = ctx.setdefault("bench_lines", [])
    steps = {"extract": 0}

    def parse(line):
        if line.startswith("BENCH "):
            lines.append(line.strip())
            parts = line.split()
            if len(parts) > 2 and parts[2] == "extract":
                steps["extract"] += 1
                return min(0.9, 0.15 * steps["extract"]), f"ALIKED at {parts[3]} px"
            if len(parts) > 2 and parts[2] == "match":
                return 0.95, "LightGlue"
        return None
    return parse


def _step_test_setup(job, ctx):
    """ONNX Runtime and the two model files for the runner. When they cannot be set up (no
    download, or the runtime does not load in Blender's Python: an access violation on one
    laptop, 0.3.9) the test goes on without the runner: DirectML is reported as not working,
    and the CPU (and NVIDIA's path) are timed with COLMAP itself."""
    from .runner import StepError
    try:
        setup_runtime_and_models(job, ctx["tools_root"], ctx["python"], ctx["log"])
    except StepError as exc:
        ctx["runner_error"] = str(exc)
        job.add_line(f"AI runner unavailable ({exc}); timing COLMAP itself instead")


def _fail(ctx, device, why):
    ctx.setdefault("failures", {}).setdefault(device, []).append(why)


def _step_test_bench(job, ctx):
    """The runner on the GPU (DirectML) and on the CPU, each in a process of its own so that a
    crash on one does not lose the other; without the runner the CPU is timed with COLMAP."""
    from . import polish_ai
    from .common import describe_returncode
    if ctx.get("runner_error") or not setup_ready(ctx["tools_root"]):
        _fail(ctx, "DIRECTML", "the AI runner is not available: " + (ctx.get("runner_error") or "not set up"))
        _colmap_device_test(job, ctx, "CPU")
        return
    site = polish_ai.ai_paths(ctx["tools_root"])["site"]
    base = [ctx["python"], "-X", "faulthandler", "-I", runner_script(), "bench", "--site", site,
            "--aliked", model_path(ctx["tools_root"], "aliked"),
            "--lightglue", model_path(ctx["tools_root"], "lightglue"),
            "--cpu-matchers", str(ctx.get("cpu_matchers") or 1)]
    if ctx.get("sample"):
        base += ["--image", ctx["sample"]]
    parser = _bench_parser(ctx)
    sizes = list(TEST_SIZES)
    code = job.run(base + ["--gpu-sizes", ",".join(str(s) for s in sizes), "--cpu-size", "0"],
                   ctx["log"], parser=parser, below_normal=False, ok_codes=None)
    if code != 0:
        worked = sorted(parse_bench(ctx.get("bench_lines", [])).get("DIRECTML", {}).get("extract", {}))
        if worked:
            # crashed at a larger size: time LightGlue again at the largest size that worked
            code = job.run(base + ["--gpu-sizes", str(worked[-1]), "--cpu-size", "0"], ctx["log"],
                           parser=parser, below_normal=False, ok_codes=None)
        if code != 0:
            _fail(ctx, "DIRECTML", f"the GPU test crashed ({describe_returncode(code)})")
    elif IS_WINDOWS and "BENCH auto no-gpu" in ctx.get("bench_lines", []):
        _fail(ctx, "DIRECTML", "ONNX Runtime found no DirectX 12 GPU" if runtime_directml(ctx["tools_root"])
              else "only ONNX Runtime's CPU build loads in Blender's Python here")
    code = job.run(base + ["--device", "cpu", "--cpu-size", str(ctx["cpu_size"])], ctx["log"],
                   parser=parser, below_normal=False, ok_codes=None)
    cpu = parse_bench(ctx.get("bench_lines", [])).get("CPU") or {}
    if code != 0 or cpu.get("match") is None:
        _colmap_device_test(job, ctx, "CPU")


def _glog_seconds(line):
    """Seconds of day from a glog line ("I20261021 18:37:02.746028 ...")."""
    try:
        hh, mm, ss = line.split()[1].split(":")
        return int(hh) * 3600 + int(mm) * 60 + float(ss)
    except (IndexError, ValueError):
        return None


def nvidia_test_images(sample, folder, size=1600, count=4):
    """``count`` overlapping views made from one photo (crops moved across it), for timing
    COLMAP's own GPU path."""
    from PIL import Image
    os.makedirs(folder, exist_ok=True)
    if sample:
        with Image.open(sample) as im:
            base = im.convert("RGB")
    else:
        import numpy as np
        from .ai_features import sample_image
        base = Image.fromarray(sample_image(np, (size, size * 3 // 4)))
    base.thumbnail((size, size), Image.Resampling.BILINEAR)
    w, h = base.size
    cw, ch = int(w * 0.85), int(h * 0.85)
    names = []
    for k in range(count):
        x = int((w - cw) * k / max(1, count - 1))
        crop = base.crop((x, (h - ch) // 2, x + cw, (h - ch) // 2 + ch))
        name = f"view_{k}.jpg"
        crop.save(os.path.join(folder, name), quality=92)
        names.append(name)
    return names


def onnx_error(lines):
    """A short reason from COLMAP's ONNX Runtime error lines, or ""."""
    for ln in lines:
        if "No valid engine configs" in ln or "HEURISTIC_QUERY_FAILED" in ln:
            return "NVIDIA's cuDNN found no kernel for ALIKED on this GPU"
        if "ONNX Runtime error" in ln:
            return ln.split("]", 1)[-1].strip()[:160]
    return ""


def _colmap_device_test(job, ctx, device):
    """COLMAP's own ALIKED and LightGlue on four views made from one photo: on the GPU with
    NVIDIA's libraries (device "NVIDIA") or on the CPU ("CPU", set up as a run sets it up)."""
    gpu = device == "NVIDIA"
    work = os.path.join(ctx["tools_root"], "_ai_test")
    shutil.rmtree(work, ignore_errors=True)
    images = os.path.join(work, "images")
    size = 1600 if gpu else int(ctx.get("cpu_size") or 1600)
    names = nvidia_test_images(ctx.get("sample"), images, size=size)
    db = os.path.join(work, "test.db")
    lines = []

    def grab(line):
        lines.append(line)
        return None

    env = ctx["colmap_env"]
    where = "with NVIDIA's libraries" if gpu else "on the CPU (COLMAP)"
    result = {"extract": {}, "points": {}, "match": None, "failed": []}
    try:
        job.set_fraction(0.1, "ALIKED " + where)
        t = time.time()
        job.run([ctx["colmap"], "feature_extractor", "--database_path", db, "--image_path", images,
                 "--FeatureExtraction.type", "ALIKED_N16ROT", "--FeatureExtraction.use_gpu",
                 "1" if gpu else "0", "--FeatureExtraction.max_image_size", str(size)],
                ctx["log"], env=env, parser=grab, below_normal=False)
        # one ALIKED extractor works through the photos (COLMAP 4.2 uses all CPU threads in it):
        # the time between two photos, else the whole time per photo (with loading the model)
        stamps = [_glog_seconds(ln) for ln in lines if "Processed file [" in ln]
        stamps = [s for s in stamps if s is not None]
        if len(stamps) >= 2:
            result["extract"][size] = max(0.001, (stamps[-1] - stamps[0]) / (len(stamps) - 1))
        else:
            result["extract"][size] = max(0.001, (time.time() - t) / len(names))
        lines.clear()
        job.set_fraction(0.6, "LightGlue " + where)
        t = time.time()
        threads = [] if gpu else ["--FeatureMatching.num_threads", str(ctx.get("cpu_matchers") or 1)]
        job.run([ctx["colmap"], "exhaustive_matcher", "--database_path", db,
                 "--FeatureMatching.type", "ALIKED_LIGHTGLUE", "--FeatureMatching.use_gpu",
                 "1" if gpu else "0", "--FeatureMatching.guided_matching", "0"] + threads,
                ctx["log"], env=env, parser=grab, below_normal=False)
        pairs = len(names) * (len(names) - 1) // 2
        result["match"] = (time.time() - t) / pairs        # includes loading the model: an upper bound
    except Exception as exc:  # noqa: BLE001 - a failed test is a result too
        result["failed"].append(onnx_error(lines) or str(exc)[:200])
    ctx.setdefault("colmap_results", {})[device] = result
    shutil.rmtree(work, ignore_errors=True)


def _step_test_nvidia(job, ctx):
    """COLMAP's own ALIKED and LightGlue on the GPU with NVIDIA's libraries, on four views."""
    _colmap_device_test(job, ctx, "NVIDIA")


def test_results(ctx):
    """The test's results by device: the runner's (DirectML, CPU), COLMAP's (NVIDIA, and the
    CPU when the runner could not time it) and why a device did not work."""
    results = parse_bench(ctx.get("bench_lines", []))
    for device, res in (ctx.get("colmap_results") or {}).items():
        if device == "CPU" and (results.get("CPU") or {}).get("match") is not None:
            continue
        results[device] = res
    for device, why in (ctx.get("failures") or {}).items():
        r = results.setdefault(device, {"extract": {}, "points": {}, "match": None, "failed": []})
        r["failed"] = list(why) + list(r.get("failed") or [])
    return results


def _step_test_save(job, ctx):
    results = test_results(ctx)
    data = {"date": time.strftime("%Y-%m-%d %H:%M"), "hardware": ctx.get("hw_text", ""),
            "sample": os.path.basename(ctx.get("sample") or "") or "a test picture",
            "cpu_size": ctx["cpu_size"], "results": results}
    write_test(ctx["tools_root"], data)
    best, lines = suggest(read_test(ctx["tools_root"]), nvidia_ready(ctx["tools_root"]),
                          ctx.get("n_photos") or 42, ctx.get("video", False))
    ctx.update(suggested=best, lines=lines)
    if getattr(job, "kind", "") == "ai_test":         # the test on its own (not inside a run)
        job.result.update({"suggested": best, "lines": lines})
        for ln in lines:
            job.add_line(ln)


def build_test_job(spec):
    """``spec``: tools_root, python, sample (a photo or ""), cpu_size, cpu_matchers (LightGlue
    models COLMAP runs at once on the CPU), colmap, colmap_env, has_nvidia_gpu, hw_text,
    n_photos, video."""
    from .runner import Job, Step
    ctx = dict(spec)
    ctx["log"] = os.path.join(spec["tools_root"], "logs", "ai_features_test.log")
    os.makedirs(os.path.dirname(ctx["log"]), exist_ok=True)
    steps = []
    if not setup_ready(spec["tools_root"]):
        steps.append(Step("ai_setup", "Setting up AI Features on the GPU (once)", _step_test_setup, 1.0))
    steps.append(Step("ai_bench", "Testing the GPU (DirectML) and the CPU", _step_test_bench, 1.0))
    if spec.get("colmap") and spec.get("has_nvidia_gpu") and nvidia_ready(spec["tools_root"]):
        steps.append(Step("ai_nvidia", "Testing NVIDIA's libraries", _step_test_nvidia, 1.0))
    steps.append(Step("ai_save", "Saving the result", _step_test_save, 0.1))
    job = Job("AI Features test", steps, spec["tools_root"])
    job.kind = "ai_test"
    return job, ctx


def _step_nvidia_install(job, ctx):
    from .runner import JobCancelled, StepError
    try:
        dlls = install_nvidia(ctx["tools_root"], progress=lambda f, note: job.set_fraction(f, note),
                              cancel=job.cancel_event)
    except tools.Cancelled as exc:
        raise JobCancelled() from exc
    except (tools.InstallError, OSError, zipfile.BadZipFile) as exc:
        raise StepError(f"NVIDIA libraries: {exc}") from exc
    job.add_line(f"NVIDIA libraries installed: {len(dlls)} DLLs in {nvidia_dir(ctx['tools_root'])}")


def nvidia_download_size(tools_root):
    return sum(w["size"] for w in nvidia_missing(tools_root))


def build_nvidia_job(tools_root):
    from .runner import Job, Step
    size = human_bytes(nvidia_download_size(tools_root) or NVIDIA_DOWNLOAD)
    job = Job("NVIDIA libraries", [Step("nvidia", f"Downloading NVIDIA's libraries ({size})",
                                        _step_nvidia_install, 1.0)], tools_root)
    job.kind = "nvidia"
    return job, {"tools_root": tools_root}
