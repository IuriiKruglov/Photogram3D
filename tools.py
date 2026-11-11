# SPDX-License-Identifier: GPL-3.0-or-later
# Photogram3D - photos or video to a textured mesh with COLMAP + OpenMVS
# Copyright (C) 2026 Kruglov Iurii
# Authors: Kruglov Iurii, Anthropic Claude AI
# License: GNU General Public License (GPL) v3 or later, https://www.blender.org/about/license/
"""Tool discovery and automatic installation of COLMAP, OpenMVS and FFmpeg.

Everything here is bpy-free so it can run in a worker thread.
"""

import ctypes
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.request
import zipfile

from .common import (
    IS_WINDOWS, describe_returncode, exe_name, human_bytes, is_startup_failure,
    popen_flags,
)

# Pinned, tested versions. Newer releases may change command-line options.
COLMAP_VERSION = "4.2.1"
OPENMVS_VERSION = "v2.4.0"
FFMPEG_BRANCH = "8.1"

COLMAP_URL = ("https://github.com/colmap/colmap/releases/download/"
              "{ver}/colmap-x64-windows-{variant}.zip")
OPENMVS_URL = ("https://github.com/cdcseacave/openMVS/releases/download/"
               "{ver}/OpenMVS_Windows_x64.zip")
FFMPEG_URLS = (
    "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/"
    "ffmpeg-n{br}-latest-win64-gpl-{br}.zip",
    "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/"
    "ffmpeg-master-latest-win64-gpl.zip",
)
VC_REDIST_URL = "https://aka.ms/vc14/vc_redist.x64.exe"

APPROX_SIZES = {  # compressed download sizes, for the UI
    "colmap-cuda": 395 * 1024 ** 2,
    "colmap-nocuda": 122 * 1024 ** 2,
    "openmvs": 21 * 1024 ** 2,
    "ffmpeg": 160 * 1024 ** 2,
}

OPENMVS_APPS = ("InterfaceCOLMAP", "DensifyPointCloud", "ReconstructMesh",
                "RefineMesh", "TextureMesh")

# COLMAP 4.2.1 is linked with MSVC 14.51, OpenMVS 2.4.0 with 14.44.
MSVC_RUNTIME_MIN = (14, 44)
MSVC_RUNTIME_RECOMMENDED = (14, 51)


class Cancelled(Exception):
    pass


class InstallError(Exception):
    pass


# ---------------------------------------------------------------------------
# Tool set (resolved executable paths)
# ---------------------------------------------------------------------------

def manifest_path(tools_root):
    return os.path.join(tools_root, "installed.json")


def read_manifest(tools_root):
    try:
        with open(manifest_path(tools_root), "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def write_manifest(tools_root, data):
    os.makedirs(tools_root, exist_ok=True)
    tmp = manifest_path(tools_root) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, manifest_path(tools_root))


def resolve(tools_root, overrides=None):
    """Find executables. ``overrides`` may hold colmap / openmvs_dir / ffmpeg paths.

    Order: explicit override, installed copy, system PATH.
    """
    overrides = overrides or {}
    man = read_manifest(tools_root) if tools_root else {}
    ts = {"colmap": None, "colmap_variant": None, "colmap_env": {},
          "openmvs_dir": None, "ffmpeg": None, "ffprobe": None,
          "versions": {}}

    # COLMAP
    colmap = overrides.get("colmap") or ""
    if not colmap and man.get("colmap"):
        cand = os.path.join(tools_root, man["colmap"]["exe"])
        if os.path.isfile(cand):
            colmap = cand
            ts["colmap_variant"] = man["colmap"].get("variant")
            ts["versions"]["colmap"] = man["colmap"].get("version")
    if not colmap:
        colmap = shutil.which("colmap") or ""
    if colmap and os.path.isfile(colmap):
        ts["colmap"] = colmap
        ts["colmap_env"] = colmap_env(colmap)
        # NVIDIA's libraries for COLMAP's AI models on the GPU, when the user downloaded them
        # (ai_backend.install_nvidia): after COLMAP's own folder on its PATH
        nvidia = os.path.join(tools_root, NVIDIA_BIN) if tools_root else ""
        if IS_WINDOWS and nvidia and os.path.isdir(nvidia):
            path = ts["colmap_env"].get("PATH", "")
            ts["colmap_env"]["PATH"] = path + os.pathsep + nvidia if path else nvidia
            ts["nvidia_dir"] = nvidia

    # OpenMVS
    omvs = overrides.get("openmvs_dir") or ""
    if not omvs and man.get("openmvs"):
        cand = os.path.join(tools_root, man["openmvs"]["dir"])
        if os.path.isdir(cand):
            omvs = cand
            ts["versions"]["openmvs"] = man["openmvs"].get("version")
    if not omvs:
        found = shutil.which(exe_name("DensifyPointCloud"))
        if found:
            omvs = os.path.dirname(found)
    if omvs and all(os.path.isfile(os.path.join(omvs, exe_name(a))) for a in OPENMVS_APPS):
        ts["openmvs_dir"] = omvs

    # FFmpeg
    ff = overrides.get("ffmpeg") or ""
    if not ff and man.get("ffmpeg"):
        cand = os.path.join(tools_root, man["ffmpeg"]["exe"])
        if os.path.isfile(cand):
            ff = cand
            ts["versions"]["ffmpeg"] = man["ffmpeg"].get("version")
    if not ff:
        ff = shutil.which("ffmpeg") or ""
    if ff and os.path.isfile(ff):
        ts["ffmpeg"] = ff
        probe = os.path.join(os.path.dirname(ff), exe_name("ffprobe"))
        ts["ffprobe"] = probe if os.path.isfile(probe) else (shutil.which("ffprobe") or None)
    return ts


def colmap_env(colmap_exe):
    """Environment additions so COLMAP finds its DLLs and Qt plugins."""
    env = {}
    bin_dir = os.path.dirname(os.path.abspath(colmap_exe))
    root = os.path.dirname(bin_dir)
    if IS_WINDOWS:
        env["PATH"] = bin_dir
        plugins = os.path.join(root, "plugins")
        if os.path.isdir(plugins):
            env["QT_PLUGIN_PATH"] = plugins
    return env


# COLMAP's ONNX models (AI Features: ALIKED, LightGlue) on the GPU need NVIDIA's CUDA
# libraries, which the COLMAP download does not include: a Windows run of 0.3.6 stopped with
# "onnxruntime_providers_cuda.dll ... depends on cublasLt64_12.dll which is missing" and a
# crash (0xC0000409). The provider imports cuBLAS and the CUDA runtime and loads cuDNN.
ONNX_CUDA_DLLS = ("cublas64_12.dll", "cublasLt64_12.dll", "cudart64_12.dll", "cudnn64_9.dll")
NVIDIA_BIN = os.path.join("nvidia", "bin")    # in the tools folder (ai_backend.install_nvidia)


def onnx_gpu_ready(colmap_exe, env=None):
    """Whether COLMAP's AI models can run on the GPU here: on Windows, NVIDIA's DLLs are next
    to colmap.exe or on PATH; elsewhere this is only known by trying (True)."""
    if not IS_WINDOWS:
        return True
    dirs = [os.path.dirname(os.path.abspath(colmap_exe))]
    dirs += ((env or {}).get("PATH") or "").split(os.pathsep)
    dirs += os.environ.get("PATH", "").split(os.pathsep)
    dirs = [d for d in dirs if d]
    return all(any(os.path.isfile(os.path.join(d, dll)) for d in dirs) for dll in ONNX_CUDA_DLLS)


def openmvs_exe(ts, app):
    return os.path.join(ts["openmvs_dir"], exe_name(app))


def merged_env(extra):
    """os.environ plus ``extra``; PATH entries are prepended."""
    env = dict(os.environ)
    for key, val in (extra or {}).items():
        if key.upper() == "PATH":
            env["PATH"] = val + os.pathsep + env.get("PATH", "")
        else:
            env[key] = val
    return env


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _run_quick(cmd, env=None, timeout=60):
    # OpenMVS apps write a log file into the current folder: use a throwaway one
    # (Blender's own working folder may be Program Files or the user's home).
    scratch = tempfile.mkdtemp(prefix="photogram3d_check_")
    try:
        res = subprocess.run(cmd, capture_output=True, timeout=timeout, cwd=scratch,
                             env=merged_env(env), stdin=subprocess.DEVNULL,
                             creationflags=popen_flags())
    except FileNotFoundError:
        return None, "not found"
    except subprocess.TimeoutExpired:
        return None, "timed out"
    except OSError as exc:
        return None, str(exc)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    out = (res.stdout or b"") + (res.stderr or b"")
    return res.returncode, out.decode("utf-8", "replace")


def verify(ts):
    """Run each tool once. Returns {name: (ok, message)}."""
    report = {}
    if ts.get("colmap"):
        rc, out = _run_quick([ts["colmap"], "version"], ts.get("colmap_env"))
        if rc == 0 and "COLMAP" in out:
            first = next((l for l in out.splitlines() if "COLMAP" in l), "COLMAP")
            report["colmap"] = (True, first.strip())
        elif rc is not None and is_startup_failure(rc):
            report["colmap"] = (False, "COLMAP cannot start: " + describe_returncode(rc))
        else:
            report["colmap"] = (False, f"COLMAP check failed ({describe_returncode(rc) if rc is not None else out})")
    else:
        report["colmap"] = (False, "COLMAP not installed")

    if ts.get("openmvs_dir"):
        rc, out = _run_quick([openmvs_exe(ts, "InterfaceCOLMAP"), "--help"])
        if rc is not None and not is_startup_failure(rc) and "OpenMVS" in out:
            ver = next((l.split("OpenMVS", 1)[1].strip() for l in out.splitlines() if "OpenMVS" in l), "")
            report["openmvs"] = (True, ("OpenMVS " + ver).strip())
        elif rc is not None and is_startup_failure(rc):
            report["openmvs"] = (False, "OpenMVS cannot start: " + describe_returncode(rc))
        else:
            report["openmvs"] = (False, f"OpenMVS check failed ({describe_returncode(rc) if rc is not None else out})")
    else:
        report["openmvs"] = (False, "OpenMVS not installed")

    if ts.get("ffmpeg"):
        rc, out = _run_quick([ts["ffmpeg"], "-hide_banner", "-version"])
        if rc == 0 and "ffmpeg version" in out:
            report["ffmpeg"] = (True, out.splitlines()[0][:60])
        else:
            report["ffmpeg"] = (False, f"FFmpeg check failed ({describe_returncode(rc) if rc is not None else out})")
    else:
        report["ffmpeg"] = (False, "FFmpeg not installed (only needed for video)")
    return report


# ---------------------------------------------------------------------------
# Microsoft Visual C++ runtime (Windows)
# ---------------------------------------------------------------------------

def msvc_runtime_version():
    """File version of the system msvcp140.dll as a 4-tuple, or None."""
    if not IS_WINDOWS:
        return None
    path = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "msvcp140.dll")
    if not os.path.isfile(path):
        return None

    class VS_FIXEDFILEINFO(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint32) for n in (
            "dwSignature", "dwStrucVersion", "dwFileVersionMS", "dwFileVersionLS",
            "dwProductVersionMS", "dwProductVersionLS", "dwFileFlagsMask",
            "dwFileFlags", "dwFileOS", "dwFileType", "dwFileSubtype",
            "dwFileDateMS", "dwFileDateLS")]

    try:
        ver = ctypes.windll.version
        size = ver.GetFileVersionInfoSizeW(path, None)
        if not size:
            return None
        buf = ctypes.create_string_buffer(size)
        if not ver.GetFileVersionInfoW(path, 0, size, buf):
            return None
        ptr = ctypes.c_void_p()
        length = ctypes.c_uint()
        if not ver.VerQueryValueW(buf, "\\", ctypes.byref(ptr), ctypes.byref(length)):
            return None
        info = ctypes.cast(ptr, ctypes.POINTER(VS_FIXEDFILEINFO)).contents
        ms, ls = info.dwFileVersionMS, info.dwFileVersionLS
        return (ms >> 16, ms & 0xFFFF, ls >> 16, ls & 0xFFFF)
    except (AttributeError, OSError):
        return None


def msvc_runtime_status():
    """(state, text): state is 'ok', 'old', 'missing' or 'n/a'."""
    if not IS_WINDOWS:
        return "n/a", "not needed on this OS"
    ver = msvc_runtime_version()
    if ver is None:
        return "missing", "Visual C++ runtime not found"
    text = "Visual C++ runtime {}.{}.{}".format(*ver[:3])
    if ver[:2] < MSVC_RUNTIME_MIN:
        return "missing", text + " (too old)"
    if ver[:2] < MSVC_RUNTIME_RECOMMENDED:
        return "old", text + " (update recommended for COLMAP 4.2)"
    return "ok", text


def launch_vc_redist_installer(download_dir, progress=None):
    """Download the latest Visual C++ v14 redistributable and start it elevated."""
    if not IS_WINDOWS:
        raise InstallError("Only needed on Windows")
    os.makedirs(download_dir, exist_ok=True)
    dest = os.path.join(download_dir, "vc_redist.x64.exe")
    download(VC_REDIST_URL, dest, progress=progress, magic=b"MZ")
    # The installer requires elevation; ShellExecute 'runas' shows the UAC prompt.
    os.startfile(dest, "runas", "/install /passive /norestart")  # noqa: S606
    return dest


# ---------------------------------------------------------------------------
# Download / extract
# ---------------------------------------------------------------------------

def download(url, dest, progress=None, cancel=None, retries=4, magic=b"PK"):
    """Download ``url`` to ``dest`` with validation and retries.

    GitHub occasionally answers a release download with an HTML error page,
    so the first bytes are checked against ``magic``.
    """
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Photogram3D-Blender/0.1"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                total = int(resp.headers.get("Content-Length") or 0)
                done = 0
                with open(tmp, "wb") as f:
                    while True:
                        if cancel is not None and cancel.is_set():
                            raise Cancelled()
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        if progress:
                            progress(done, total)
            if total and done != total:
                raise InstallError(f"incomplete download ({human_bytes(done)} of {human_bytes(total)})")
            with open(tmp, "rb") as f:
                head = f.read(8)
            if magic and not head.startswith(magic):
                raise InstallError("server returned an error page instead of the file")
            os.replace(tmp, dest)
            return dest
        except Cancelled:
            _silent_remove(tmp)
            raise
        except Exception as exc:  # noqa: BLE001 - retry on any network/file error
            last_error = exc
            _silent_remove(tmp)
            if attempt < retries:
                for _ in range(int(3 * attempt * 10)):
                    if cancel is not None and cancel.is_set():
                        raise Cancelled()
                    time.sleep(0.1)
    raise InstallError(f"download failed: {url} ({last_error})")


def _silent_remove(path):
    try:
        os.remove(path)
    except OSError:
        pass


def safe_extract(zip_path, dest_dir, strip_prefix="", skip=None, cancel=None, progress=None):
    """Extract a zip, keeping only members under ``strip_prefix`` (which is removed).

    Guards against absolute paths and '..' components.
    """
    dest_real = os.path.realpath(dest_dir)
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        if strip_prefix:
            members = [m for m in members if m.filename.startswith(strip_prefix)]
        total = sum(m.file_size for m in members) or 1
        done = 0
        for m in members:
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            rel = m.filename[len(strip_prefix):] if strip_prefix else m.filename
            rel = rel.replace("\\", "/").lstrip("/")
            if not rel or (skip and skip(rel)):
                continue
            parts = [p for p in rel.split("/") if p not in ("", ".")]
            if any(p == ".." for p in parts) or ":" in parts[0]:
                raise InstallError(f"unsafe path in archive: {m.filename}")
            target = os.path.realpath(os.path.join(dest_dir, *parts))
            if not (target == dest_real or target.startswith(dest_real + os.sep)):
                raise InstallError(f"unsafe path in archive: {m.filename}")
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with zf.open(m) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            done += m.file_size
            if progress:
                progress(done, total)


def _detect_zip_prefix(zip_path, marker):
    """Return the folder prefix containing ``marker`` (e.g. 'vc17/x64/Release/')."""
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            norm = name.replace("\\", "/")
            if norm.endswith("/" + marker) or norm == marker:
                return norm[: len(norm) - len(marker)]
    raise InstallError(f"{marker} not found in {os.path.basename(zip_path)}")


def _replace_dir(src, dst):
    if os.path.isdir(dst):
        shutil.rmtree(dst, ignore_errors=True)
    os.replace(src, dst)


# ---------------------------------------------------------------------------
# Installer (runs in a background thread)
# ---------------------------------------------------------------------------

class InstallState:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.message = ""
        self.fraction = 0.0
        self.error = ""
        self.done = False
        self.report = {}
        self.cancel = threading.Event()
        self.thread = None

    def set(self, message=None, fraction=None):
        with self.lock:
            if message is not None:
                self.message = message
            if fraction is not None:
                self.fraction = max(0.0, min(1.0, fraction))


def plan_install(want_cuda, include_ffmpeg, components=None):
    """List of (name, url(s), approx_size)."""
    variant = "cuda" if want_cuda else "nocuda"
    plan = []
    if components is None or "colmap" in components:
        plan.append(("colmap", [COLMAP_URL.format(ver=COLMAP_VERSION, variant=variant)],
                     APPROX_SIZES["colmap-" + variant], variant))
    if components is None or "openmvs" in components:
        plan.append(("openmvs", [OPENMVS_URL.format(ver=OPENMVS_VERSION)],
                     APPROX_SIZES["openmvs"], None))
    if include_ffmpeg and (components is None or "ffmpeg" in components):
        plan.append(("ffmpeg", [u.format(br=FFMPEG_BRANCH) for u in FFMPEG_URLS],
                     APPROX_SIZES["ffmpeg"], None))
    return plan


def run_install(state, tools_root, want_cuda, include_ffmpeg, components=None, keep_zips=False):
    """Download, extract and verify. Updates ``state``; never raises."""
    state.running = True
    state.done = False
    state.error = ""
    try:
        if not IS_WINDOWS:
            raise InstallError("Automatic install downloads Windows builds. On this OS set "
                               "the executable paths in the add-on preferences.")
        plan = plan_install(want_cuda, include_ffmpeg, components)
        total_size = sum(p[2] for p in plan) or 1
        dl_dir = os.path.join(tools_root, "_downloads")
        os.makedirs(dl_dir, exist_ok=True)
        manifest = read_manifest(tools_root)
        base = 0.0
        for name, urls, size, variant in plan:
            share = size / total_size
            zip_path = os.path.join(dl_dir, os.path.basename(urls[0]))

            def on_dl(done, total, _base=base, _share=share, _name=name):
                frac = done / total if total else 0.0
                state.set(f"Downloading {_name}: {human_bytes(done)}"
                          + (f" / {human_bytes(total)}" if total else ""),
                          _base + _share * 0.8 * frac)

            last = None
            for url in urls:
                try:
                    zip_path = os.path.join(dl_dir, os.path.basename(url))
                    download(url, zip_path, progress=on_dl, cancel=state.cancel)
                    last = None
                    break
                except InstallError as exc:
                    last = exc
            if last is not None:
                raise last

            def on_ex(done, total, _base=base, _share=share, _name=name):
                state.set(f"Extracting {_name}...", _base + _share * (0.8 + 0.2 * done / total))

            tmp_dir = os.path.join(tools_root, f".tmp_{name}")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            if name == "colmap":
                prefix = _detect_zip_prefix(zip_path, "bin/" + exe_name("colmap"))
                safe_extract(zip_path, tmp_dir, prefix,
                             skip=lambda r: r.endswith("_test.exe") or r.endswith(".pdb")
                             or r.upper().startswith("RUN_TESTS"),
                             cancel=state.cancel, progress=on_ex)
                _replace_dir(tmp_dir, os.path.join(tools_root, "colmap"))
                manifest["colmap"] = {"version": COLMAP_VERSION, "variant": variant,
                                      "exe": os.path.join("colmap", "bin", exe_name("colmap"))}
            elif name == "openmvs":
                prefix = _detect_zip_prefix(zip_path, exe_name("DensifyPointCloud"))
                safe_extract(zip_path, tmp_dir, prefix,
                             skip=lambda r: r.endswith(".lib") or r.endswith(".exp"),
                             cancel=state.cancel, progress=on_ex)
                _replace_dir(tmp_dir, os.path.join(tools_root, "openmvs"))
                manifest["openmvs"] = {"version": OPENMVS_VERSION, "dir": "openmvs"}
            elif name == "ffmpeg":
                prefix = _detect_zip_prefix(zip_path, "bin/" + exe_name("ffmpeg"))
                safe_extract(zip_path, tmp_dir, prefix + "bin/",
                             cancel=state.cancel, progress=on_ex)
                _replace_dir(tmp_dir, os.path.join(tools_root, "ffmpeg"))
                manifest["ffmpeg"] = {"version": FFMPEG_BRANCH,
                                      "exe": os.path.join("ffmpeg", exe_name("ffmpeg"))}
            write_manifest(tools_root, manifest)
            if not keep_zips:
                _silent_remove(zip_path)
            base += share

        state.set("Checking tools...", 0.99)
        ts = resolve(tools_root)
        state.report = verify(ts)
        bad = [v[1] for k, v in state.report.items() if not v[0] and not (k == "ffmpeg" and not include_ffmpeg)]
        if bad:
            state.error = "; ".join(bad)
            state.set("Installed, but a check failed", 1.0)
        else:
            state.set("Tools installed and working", 1.0)
    except Cancelled:
        state.error = "Cancelled"
        state.set("Install cancelled")
    except Exception as exc:  # noqa: BLE001 - report everything to the UI
        state.error = str(exc)
        state.set("Install failed")
    finally:
        state.running = False
        state.done = True


def start_install(state, tools_root, want_cuda, include_ffmpeg, components=None):
    if state.running:
        return False
    state.cancel.clear()
    state.set("Starting...", 0.0)
    state.thread = threading.Thread(
        target=run_install, args=(state, tools_root, want_cuda, include_ffmpeg, components),
        name="Photogram3DInstall", daemon=True)
    state.thread.start()
    return True
