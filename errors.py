# SPDX-License-Identifier: GPL-3.0-or-later
# Photogram3D - photos or video to a textured mesh with COLMAP + OpenMVS
# Copyright (C) 2026 Kruglov Iurii
# Authors: Kruglov Iurii, Anthropic Claude AI
# License: GNU General Public License (GPL) v3 or later, https://www.blender.org/about/license/
"""What a failure means, in plain words (0.3.4).

A failed step leaves an error such as "colmap failed (exit 0xC0000409: stack buffer
overrun / fast-fail (crash))". explain() looks at that text, the end of the step's log
and the run's settings, and returns what most likely happened and what to do about it.
"""

import os
import re

# Steps by key, as the panel names them
STEP_NAMES = {
    "frames": "extracting the video frames", "photos": "checking the photos",
    "features": "finding features",
    "matching": "matching photos", "mapping": "solving the cameras", "orient": "aligning up",
    "undistort": "undistorting the photos", "colmap_dense": "GPU depth maps",
    "even": "evening out the photos", "mvs_import": "preparing the OpenMVS scene", "densify": "computing the dense point cloud",
    "crop_prepare": "preparing the crop box", "mesh": "building the mesh",
    "refine": "refining the mesh", "texture": "texturing", "splats": "training the splats",
    "finish": "finishing",
}

_HEAVY = {"densify", "colmap_dense", "mesh", "refine", "texture"}


def step_log(ctx, key):
    """The step's log file (pipeline._log's naming)."""
    try:
        order = ctx["step_order"]
        idx = order.index(key) + 1 if key in order else 0
        return os.path.join(ctx["paths"]["logs"], f"{idx:02d}_{key}.log")
    except (KeyError, TypeError):
        return ""


def read_tail(path, max_bytes=200_000):
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def _last_run(log_text):
    """The log of the last command only (each starts with "$ ")."""
    i = log_text.rfind("\n$ ")
    return log_text[i + 1:] if i >= 0 else log_text


def _detail_line(text):
    """The most telling error line a tool printed: glog fatal/error lines, OpenMVS
    "error:" lines, FFmpeg's complaints."""
    best = ""
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("$ ") or s.startswith("[exit"):
            continue
        m = re.match(r"^[FE]\d{8} [\d:.]+\s+\d+\s+\S+\]\s*(.+)$", s)      # glog F/E lines
        if m:
            best = m.group(1)
            continue
        if re.search(r"\berror\b|\bfailed\b|Check failed|exception|cannot|could not", s, re.I):
            best = re.sub(r"^\d\d:\d\d:\d\d \[\w+\s*\]\s*", "", s)         # OpenMVS time/tag
    return best[:160]


def _rules():
    """(steps or None, pattern over error + log, explanation). First match wins."""
    any_step = None
    return [
        (any_step, r"No space left on device|not enough space on the disk|ENOSPC|\[Errno 28\]",
         "The output drive is full. Free some space (depth maps take several GB at their peak) "
         "or choose an output folder on another drive."),
        ({"splats"}, r"OutOfMemory|out of device memory|VK_ERROR_OUT_OF_DEVICE_MEMORY|"
                     r"DEVICE_LOST|Device ?lost|DXGI_ERROR_DEVICE_REMOVED",
         "The graphics card ran out of memory or was reset while training the splats. Lower "
         "Splat Photo Size (1280 px, say) and close programs that use the GPU."),
        ({"splats"}, r"No suitable (GPU )?adapter|Failed to (find|create|request) (an )?adapter|"
                     r"RequestAdapterError|No compatible (GPU|graphics)",
         "Brush found no usable graphics card: it needs DirectX 12 or Vulkan. Update the "
         "graphics driver."),
        (any_step, r"(CUDA|cuda|GPU).{0,80}out of memory|cudaErrorMemoryAllocation|"
                   r"CUBLAS_STATUS_ALLOC_FAILED|CUDNN_STATUS_ALLOC_FAILED",
         "The graphics card ran out of memory. Close other programs that use the GPU, choose a "
         "lower Quality, or set Depth Maps to OpenMVS (CPU)."),
        (any_step, r"bad_alloc|Cannot allocate memory|out of memory|Not enough memory|MemoryError|"
                   r"0xC0000017|insufficient memory|failed to allocate",
         "The computer ran out of memory (RAM) while {step}. Close other programs, choose a lower "
         "Quality{crop}."),
        (any_step, r"0xC0000135|0xC0000139|0xC000007B|VCRUNTIME\d*\.dll|MSVCP\d*\.dll|"
                   r"was not found\. Reinstalling the program",
         "A tool could not start: a Windows library it needs is missing. Press Install Visual "
         "C++ Runtime in the Photogram3D preferences, then Check Tools; if it persists, Install / "
         "Repair the tools."),
        (any_step, r"cannot start ([^\s:]+)",
         "The tool's program file could not be started ({m1}): it is missing or blocked, often "
         "by an antivirus. Press Install / Repair in the preferences and check the antivirus "
         "quarantine."),
        (any_step, r"no CUDA-capable device|CUDA driver version is insufficient|cudaError\w*|"
                   r"CUDA error[^\n]{0,80}",
         "COLMAP could not use the NVIDIA GPU ({m0}). Update the NVIDIA driver, or switch off Use "
         "NVIDIA GPU and set Depth Maps to OpenMVS (CPU)."),
        (any_step, r"Permission denied|Access is denied|being used by another process|"
                   r"PermissionError|WinError 32|WinError 5\b|Cannot delete \d+ old depth maps",
         "A file in the output folder is locked or read-only. Pause cloud syncing (Google Drive, "
         "OneDrive, Dropbox) or use a local output folder, and close programs that may have the "
         "files open."),
        ({"frames"}, r"Invalid data found when processing input|moov atom not found|"
                     r"could not find codec parameters|Decoder \(codec [^)]*\) not found|"
                     r"FFmpeg cannot read this video",
         "FFmpeg cannot read this video: it is damaged, still being copied, or uses a codec this "
         "FFmpeg build lacks. Re-export it as MP4 (H.264 or H.265)."),
        ({"features"}, r"Failed to read|cannot identify image|Could not read image|"
                       r"Unsupported (image )?(file )?format|Failed to open image",
         "Some photos could not be read. Remove damaged files from the folder; RAW, HEIC, EXR "
         "and PSD photos are converted first."),
        ({"matching"}, r"0xC0000409|Guided matching not supported",
         "COLMAP's matcher stopped on an internal error{ai}. Switch AI Features off and try again; "
         "if it keeps failing, send the logs folder."),
        ({"mapping"}, r"No good initial image pair|no initial pair|no reconstruction|produced no model|"
                      r"could not register|0 images registered",
         "COLMAP could not place the photos: they share too few matching features. Shoot with more "
         "overlap (each spot in 3+ photos), avoid plain, shiny or dark surfaces, put a textured, "
         "asymmetric surface (a newspaper) under the object, or try AI Features."),
        ({"densify", "mvs_import"},
         r"completed: 0 images|no valid depth-?maps|not enough images|no good images in view|"
         r"produced no point cloud",
         "OpenMVS found no photo pairs to compute depth from: the camera solve is too weak (few "
         "shared 3D points). More overlap between photos or a textured background helps."),
        (any_step, r"0xC0000409",
         "{tool} stopped on an internal error (it does not say which){detail}. Try once more with "
         "Resume off; if it repeats, send the logs folder."),
        (any_step, r"0xC0000005|killed by signal 11|Segmentation fault",
         "{tool} crashed (access violation){detail}. Try once more with Resume off (a fresh run), or "
         "a lower Quality; if it repeats, send the logs folder."),
    ]


def explain(step_key, message, log_text="", cfg=None):
    """Plain-language explanation of a failure, or "" when nothing more can be said."""
    cfg = cfg or {}
    text = (message or "") + "\n" + _last_run(log_text or "")
    tool = "The tool"
    m_tool = re.match(r"\s*([\w.-]+?)(?:\.exe)? failed", message or "")
    if m_tool:
        tool = m_tool.group(1)
    detail = _detail_line(_last_run(log_text or ""))
    for steps, pattern, explanation in _rules():
        if steps is not None and step_key not in steps:
            continue
        m = re.search(pattern, text, re.IGNORECASE)
        if not m:
            continue
        values = {
            "step": STEP_NAMES.get(step_key, step_key or "working"),
            "tool": tool,
            "detail": f" ({detail})" if detail and detail not in (message or "") else "",
            "crop": (", or make the crop box smaller" if step_key in _HEAVY else ""),
            "ai": (" (AI Features: LightGlue)" if cfg.get("ai_features") else ""),
            "m0": m.group(0)[:80],
            "m1": os.path.basename(m.group(1) or "") if m.groups() else "",
        }
        return explanation.format(**values)
    if detail and detail not in (message or ""):
        return f"The log says: {detail}"
    return ""


def notes(step_key, log_text, cfg=None):
    """Side remarks worth knowing even when the step worked (AI Features on the CPU)."""
    out = []
    if re.search(r"AppendExecutionProvider_Cuda: Failed to load shared library", log_text or ""):
        out.append("AI Features ran on the CPU: COLMAP's AI models need NVIDIA's CUDA libraries "
                   "(cuBLAS, cuDNN), which the COLMAP download does not include. For the GPU: "
                   "Preferences › AI Features On › GPU (DirectML), or Download NVIDIA Libraries")
    elif re.search(r"No valid engine configs|HEURISTIC_QUERY_FAILED", log_text or ""):
        # cuDNN 9.27 on a GTX 1650 Ti without NVRTC (0.3.9)
        out.append("AI Features found the features on the CPU: NVIDIA's cuDNN had no kernel for "
                   "ALIKED on this GPU. In Preferences › AI Features: Download NVRTC (cuDNN compiles "
                   "kernels with it), or AI Features On › GPU (DirectML)")
    return out
