# SPDX-License-Identifier: GPL-3.0-or-later
# Photogram3D - photos or video to a textured mesh with COLMAP + OpenMVS
# Copyright (C) 2026 Kruglov Iurii
# Authors: Kruglov Iurii, Anthropic Claude AI
# License: GNU General Public License (GPL) v3 or later, https://www.blender.org/about/license/
"""Hardware detection (NVIDIA GPU, RAM, CPU) and quality presets."""

import ctypes
import os
import shutil
import subprocess

from .common import IS_WINDOWS, popen_flags


def _nvidia_smi():
    path = shutil.which("nvidia-smi")
    if path:
        return path
    if IS_WINDOWS:
        root = os.environ.get("SystemRoot", r"C:\Windows")
        for cand in (
            os.path.join(root, "System32", "nvidia-smi.exe"),
            r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
        ):
            if os.path.isfile(cand):
                return cand
    return None


def detect_gpus(timeout=10.0):
    """List NVIDIA GPUs as dicts: name, vram_mb, compute_cap (float or None), driver."""
    exe = _nvidia_smi()
    if not exe:
        return []
    queries = (
        ("name", "memory.total", "compute_cap", "driver_version"),
        ("name", "memory.total", "driver_version"),  # old drivers lack compute_cap
    )
    for fields in queries:
        try:
            res = subprocess.run(
                [exe, "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=timeout,
                creationflags=popen_flags(),
            )
        except (OSError, subprocess.SubprocessError):
            return []
        if res.returncode != 0:
            continue
        gpus = []
        for line in res.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != len(fields):
                continue
            info = dict(zip(fields, parts))
            try:
                vram = int(float(info.get("memory.total", "0")))
            except ValueError:
                vram = 0
            cc = None
            if "compute_cap" in info:
                try:
                    cc = float(info["compute_cap"])
                except ValueError:
                    cc = None
            gpus.append({
                "name": info.get("name", "NVIDIA GPU"),
                "vram_mb": vram,
                "compute_cap": cc,
                "driver": info.get("driver_version", ""),
            })
        return gpus
    return []


def total_ram_bytes():
    if IS_WINDOWS:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        try:
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return int(stat.ullTotalPhys)
        except (AttributeError, OSError):
            return None
        return None
    try:
        with open("/proc/meminfo", "r", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def cpu_threads():
    return os.cpu_count() or 4


def detect():
    """Snapshot of the machine, safe to call from the UI (takes up to ~1 s)."""
    gpus = detect_gpus()
    ram = total_ram_bytes()
    return {
        "gpus": gpus,
        "ram_gb": round(ram / (1024 ** 3), 1) if ram else None,
        "threads": cpu_threads(),
    }


def best_gpu(hw):
    gpus = hw.get("gpus") or []
    if not gpus:
        return None
    return max(gpus, key=lambda g: g.get("vram_mb", 0))


def describe(hw):
    gpu = best_gpu(hw)
    parts = []
    if gpu:
        parts.append(f"{gpu['name']} {gpu['vram_mb'] / 1024:.0f} GB")
    else:
        parts.append("no NVIDIA GPU")
    if hw.get("ram_gb"):
        parts.append(f"{hw['ram_gb']:.0f} GB RAM")
    parts.append(f"{hw.get('threads', '?')} threads")
    return " · ".join(parts)


def recommend_quality(hw):
    ram = hw.get("ram_gb") or 16
    if ram < 12:
        return "DRAFT"
    return "NORMAL"


def openmvs_cuda_compatible(hw):
    """The official OpenMVS 2.4.0 CUDA build only contains sm_89 (RTX 40) code."""
    gpu = best_gpu(hw)
    return bool(gpu and gpu.get("compute_cap") and abs(gpu["compute_cap"] - 8.9) < 1e-3)


FULL_SIZE = 1 << 16       # "no limit" for the Full preset's depth maps (larger than any photo)


def preset_params(quality, hw):
    """Numeric settings for a quality level, clamped to the machine.

    Tuned for 16 GB RAM laptops with 4-8 GB NVIDIA GPUs.
    """
    ram = hw.get("ram_gb") or 16
    gpu = best_gpu(hw)
    vram_gb = (gpu["vram_mb"] / 1024.0) if gpu else 0.0

    p = {
        "DRAFT": dict(
            feat_max_image=1600, sift_features=4096, aliked_features=2048,
            guided_matching=False, undist_max_image=2000,
            dense_res_level=1, dense_max_res=1280, dense_views=5,
            target_faces=300_000, refine=False, refine_max_side=1400,
            texture_res_level=1, texture_size=4096,
            colmap_dense_max_image=1600,
        ),
        "NORMAL": dict(
            feat_max_image=2400, sift_features=8192, aliked_features=2048,
            guided_matching=False, undist_max_image=3200,
            dense_res_level=1, dense_max_res=2048, dense_views=8,
            target_faces=1_500_000, refine=True, refine_max_side=2000,
            texture_res_level=0, texture_size=8192,
            colmap_dense_max_image=2000,
        ),
        "HIGH": dict(
            feat_max_image=3200, sift_features=8192, aliked_features=2048,
            guided_matching=True, undist_max_image=0,  # 0 = full resolution
            dense_res_level=0, dense_max_res=2560, dense_views=8,
            target_faces=4_000_000, refine=True, refine_max_side=2600,
            texture_res_level=0, texture_size=8192,
            colmap_dense_max_image=2400,
        ),
        # Ultra: refinement keeps faces up to 16 px instead of 32 (about twice the
        # faces; on the WRX photos at 16 GB that limit, not the depth maps, set the
        # detail), depth maps up to 3200 px. Meant to be used with the crop box.
        "ULTRA": dict(
            feat_max_image=3200, sift_features=8192, aliked_features=2048,
            guided_matching=True, undist_max_image=0,
            dense_res_level=0, dense_max_res=3200, dense_views=8,
            target_faces=6_000_000, refine=True, refine_max_side=3200, refine_face_area=16,
            texture_res_level=0, texture_size=8192,
            colmap_dense_max_image=3200,
        ),
        # Full: Ultra with depth maps at the photos' own size (4624 px for the Pixel 7a,
        # 2.1 times Ultra's pixels, so about twice its depth-map time and disk). COLMAP's
        # GPU-only path stays at 3200 px: its consistency pass keeps the depth and normal maps
        # of 20 neighbouring photos on the GPU, which would not fit in 8 GB at full size.
        "FULL": dict(
            feat_max_image=3200, sift_features=8192, aliked_features=2048,
            guided_matching=True, undist_max_image=0,
            dense_res_level=0, dense_max_res=FULL_SIZE, dense_views=8,
            target_faces=6_000_000, refine=True, refine_max_side=3200, refine_face_area=16,
            texture_res_level=0, texture_size=8192,
            colmap_dense_max_image=3200,
        ),
    }[quality]
    p.setdefault("refine_face_area", 32)
    # AI Features (0.3.7). COLMAP runs ALIKED as one model over the whole photo, whose memory
    # grows with the pixels: 4.0 GB at 1600 px measured, so about 16 GB at 3200 px, where
    # a 16 GB test laptop swapped (45 s per photo, 0.3.6). LightGlue on the CPU runs one
    # model per CPU thread: 1.27 GB each at 4096 points (16 threads: 20 GB, matching stalled),
    # 0.43 GB at COLMAP's own 2048, which also matches 3.7 times faster per pair.
    p["aliked_max_image"] = 1600 if ram >= 12 else 1280

    # Low RAM: dense and texture stages are memory hogs (refinement resolution is
    # chosen later from the actual image sizes, see pipeline.refine_plan).
    if ram < 12:
        p["dense_max_res"] = min(p["dense_max_res"], 1600)
        p["texture_size"] = min(p["texture_size"], 4096)
    # Small GPUs: COLMAP CUDA patch-match memory grows with image size.
    if gpu and vram_gb < 5:
        p["colmap_dense_max_image"] = min(p["colmap_dense_max_image"], 2000)
    return p
