# SPDX-License-Identifier: GPL-3.0-or-later
# Photogram3D - photos or video to a textured mesh with COLMAP + OpenMVS
# Copyright (C) 2026 Kruglov Iurii
# Authors: Kruglov Iurii, Anthropic Claude AI
# License: GNU General Public License (GPL) v3 or later, https://www.blender.org/about/license/
"""Shared helpers with no dependency on bpy (usable from worker threads)."""

import hashlib
import json
import os
import subprocess
import sys

IS_WINDOWS = sys.platform == "win32"

# What COLMAP and OpenMVS both read; other photo formats are converted first (formats.py)
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# A run's camera solve and what depends on its coordinates (0.3.5). Each solve of a run
# gets an id (SOLVE_FILE in the colmap/model folder); the alignment the user gave the scan
# (TRANSFORM_FILE, in the run folder) and the scan in Blender carry the id of the solve
# they were made for, so a later solve of the same run is never put on top of them.
SOLVE_FILE = "photogram3d_solve.json"
TRANSFORM_FILE = "photogram3d_transform.json"
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".mts", ".m2ts", ".3gp", ".mxf"}

# Photogram3D was called PhotoMesh up to 0.3.6 (0.3.7 renamed it). What PhotoMesh wrote is
# taken over: files named photomesh_* (current()), project folders named *_photomesh,
# and in Blender (legacy.py) its custom properties, settings, preferences and tools.
NAME = "Photogram3D"
PREFIX = "photogram3d_"
LEGACY_ID = "photomesh"
LEGACY_NAME = "PhotoMesh"
LEGACY_PREFIX = LEGACY_ID + "_"


def legacy_path(path):
    """The name PhotoMesh gave the file ``path`` (photomesh_x for photogram3d_x), or None."""
    folder, name = os.path.split(path)
    if not name.startswith(PREFIX):
        return None
    return os.path.join(folder, LEGACY_PREFIX + name[len(PREFIX):])


def current(path):
    """``path`` of a file Photogram3D keeps (photogram3d_*); the file PhotoMesh wrote there
    under its old name (photomesh_*) is renamed to it first. Where that rename is refused
    (a read-only or locked folder), the old file is used where it is."""
    old = legacy_path(path)
    if old is None or os.path.exists(path) or not os.path.exists(old):
        return path
    try:
        os.replace(old, path)
    except OSError:
        return old
    return path


def legacy_folder(path, suffix):
    """``path`` (a default folder ending in ``suffix``, e.g. Gas_mask_photogram3d), or the
    folder PhotoMesh made for the same input (Gas_mask_photomesh) when only that exists,
    so its runs are continued instead of computed again in a new folder."""
    if os.path.exists(path) or not path.endswith(suffix):
        return path
    old = path[:-len(suffix)] + suffix.replace(PREFIX[:-1], LEGACY_ID)
    return old if os.path.isdir(old) else path


def exe_name(name):
    """Platform executable file name."""
    return name + ".exe" if IS_WINDOWS else name


def has_non_ascii(path):
    return any(ord(c) > 127 for c in str(path))


# Folder names that mark a cloud-synced tree (default and localised names).
_SYNC_FOLDERS = (
    ("Google Drive", ("google drive", "googledrive", "gdrive", "my drive", "мой диск", "google диск",
                      "shared drives", "общие диски")),
    ("OneDrive", ("onedrive",)),
    ("Dropbox", ("dropbox",)),
    ("iCloud Drive", ("icloud drive", "iclouddrive")),
    ("Yandex Disk", ("yandex.disk", "yandexdisk", "yandex disk", "яндекс.диск", "яндекс диск")),
)


def cloud_sync_service(path):
    """Name of the sync service whose folder holds ``path`` ("" if none is recognised).

    A sync client uploads the project's gigabytes of intermediate files while the tools
    write them, and can hold a file open when a tool tries to replace it."""
    parts = [p.strip().lower() for p in os.path.normpath(str(path)).replace("\\", "/").split("/")]
    for service, names in _SYNC_FOLDERS:
        for part in parts:
            if part in names or any(part.startswith(n + " ") or part.startswith(n + "-") for n in names):
                return service
    return ""


def list_images(folder):
    """Image files directly inside ``folder`` (no recursion), sorted by name."""
    names = []
    try:
        entries = list(os.scandir(folder))
    except OSError:
        return names
    for entry in entries:
        if not entry.is_file():
            continue
        ext = os.path.splitext(entry.name)[1].lower()
        if ext in IMAGE_EXTS:
            names.append(entry.name)
    names.sort(key=str.lower)
    return names


def stable_hash(obj):
    data = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(data).hexdigest()[:16]


def popen_flags(below_normal=False):
    """creationflags for subprocess: no console window on Windows."""
    if not IS_WINDOWS:
        return 0
    flags = subprocess.CREATE_NO_WINDOW
    if below_normal:
        flags |= subprocess.BELOW_NORMAL_PRIORITY_CLASS
    return flags


def human_bytes(n):
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024.0 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


def human_time(seconds):
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


# Windows NTSTATUS codes seen as process exit codes when a program cannot start.
NTSTATUS_MESSAGES = {
    0xC0000135: "a required DLL was not found",
    0xC0000139: "a DLL entry point was not found (Visual C++ runtime too old?)",
    0xC000007B: "bad executable image (32/64-bit mismatch or corrupt DLL)",
    0xC0000005: "access violation (crash)",
    0xC00000FD: "stack overflow (crash)",
    0xC0000409: "stack buffer overrun / fast-fail (crash)",
}


def describe_returncode(code):
    if code is None:
        return "not finished"
    if code == 0:
        return "ok"
    unsigned = code & 0xFFFFFFFF
    if unsigned in NTSTATUS_MESSAGES:
        return f"exit 0x{unsigned:08X}: {NTSTATUS_MESSAGES[unsigned]}"
    if code < 0 and not IS_WINDOWS:
        return f"killed by signal {-code}"
    if code > 0x7FFFFFFF:
        code -= 1 << 32  # Windows reports exit(-1) as 4294967295
    return f"exit code {code}"


def is_startup_failure(code):
    """True when the exit code means the program could not even start."""
    if code is None:
        return False
    return (code & 0xFFFFFFFF) in (0xC0000135, 0xC0000139, 0xC000007B)
