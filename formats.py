# SPDX-License-Identifier: GPL-3.0-or-later
# Photogram3D - photos or video to a textured mesh with COLMAP + OpenMVS
# Copyright (C) 2026 Kruglov Iurii
# Authors: Kruglov Iurii, Anthropic Claude AI
# License: GNU General Public License (GPL) v3 or later, https://www.blender.org/about/license/
"""Photo formats beyond JPEG and PNG (0.3.4).

COLMAP and OpenMVS both read JPEG and 8-bit PNG; everything else (camera RAW files,
HEIC/AVIF, EXR, TIFF, TGA, WebP, PSD) is converted to JPEG in the run's photos folder
first, by convert_images.py in a separate Python process with Pillow, pillow-heif, rawpy
(LibRaw) and OpenEXR, which are installed into the tools folder on first use (about 16 MB).

A RAW file and a JPEG of the same shot (phones save both: PXL_..._153323706.RAW-02.ORIGINAL.dng
and PXL_..._153323706.RAW-01.COVER.jpg) count as one photo: the JPEG is used and the RAW
skipped, or the other way round (setting "RAW + JPEG"). The JPEG is the default so that a
folder that already gave a result with 0.3.3 gives the same one.
"""

import json
import os
import shutil
import struct
import sys

from .common import current
from .runner import StepError

NATIVE = {".jpg", ".jpeg", ".png"}
RAW = {".dng", ".cr2", ".cr3", ".crw", ".nef", ".nrw", ".arw", ".sr2", ".srf", ".raf", ".rw2",
       ".orf", ".iiq", ".pef", ".rwl", ".3fr", ".fff", ".dcr", ".kdc", ".srw", ".x3f", ".erf",
       ".mos", ".mrw", ".mef"}
HEIF = {".heic", ".heif", ".hif", ".avif"}
EXR = {".exr"}
PILLOW = {".tif", ".tiff", ".tga", ".webp", ".psd", ".bmp"}
PHOTO_EXTS = NATIVE | RAW | HEIF | EXR | PILLOW

RUNTIME_MARKER = "photogram3d_formats.json"
PACKAGES = (("pillow==13.0.0", "pillow-heif==1.8.1", "rawpy==0.27.1", "openexr==3.5.2"),
            ("pillow", "pillow-heif", "rawpy", "openexr"))
CONVERTED_SUFFIX = ".jpg"


def ext(name):
    return os.path.splitext(name)[1].lower()


def kind(name):
    e = ext(name)
    if e in RAW:
        return "raw"
    if e in HEIF:
        return "heif"
    if e in EXR:
        return "exr"
    if e in PILLOW:
        return "pillow"
    return "native" if e in NATIVE else None


def png_is_plain(path):
    """An 8-bit PNG (what COLMAP and OpenMVS read alike); 16-bit ones are converted."""
    try:
        with open(path, "rb") as f:
            head = f.read(26)
    except OSError:
        return True
    if len(head) < 26 or head[:8] != b"\x89PNG\r\n\x1a\n":
        return True
    return head[24] <= 8


def list_photos(folder):
    """Photo files of any supported format directly inside ``folder``, sorted by name."""
    try:
        entries = [e for e in os.scandir(folder) if e.is_file()]
    except OSError:
        return []
    names = [e.name for e in entries if ext(e.name) in PHOTO_EXTS and not e.name.startswith("._")]
    return sorted(names, key=str.lower)


def shot_key(name):
    """Phones name a RAW and its JPEG alike up to the first dot
    (PXL_..._153323706.RAW-02.ORIGINAL.dng, PXL_..._153323706.RAW-01.COVER.jpg)."""
    return name.split(".", 1)[0].lower()


def _pair_twins(names, key, prefer_raw, use, skipped, twins):
    """Resolve groups of exactly one RAW and one other photo under ``key``; returns the
    names left over. ``twins`` gets {photo used: its twin}."""
    groups = {}
    for n in names:
        groups.setdefault(key(n), []).append(n)
    rest = []
    for members in groups.values():
        raws = [n for n in members if kind(n) == "raw"]
        others = [n for n in members if kind(n) != "raw"]
        if len(raws) == 1 and len(others) == 1:
            keep = raws if prefer_raw else others
            use += keep
            skipped += [n for n in members if n not in keep]
            twins[keep[0]] = others[0] if prefer_raw else raws[0]
        else:
            rest += members
    return rest


def plan(folder, prefer_raw=False):
    """Which photos to use and which of them to convert.

    {"use": names, "convert": names needing conversion, "skipped": partners left out,
     "twins": {name used: its partner}}

    A RAW and a JPEG of one shot share their name up to the extension (IMG_1234.CR2,
    IMG_1234.JPG) or, on phones, up to the first dot. Only groups of exactly one RAW and one
    other photo are twins: scan.001.dng, scan.001.jpg and scan.002.jpg are two shots."""
    names = list_photos(folder)
    use, skipped, twins = [], [], {}
    rest = _pair_twins(names, lambda n: os.path.splitext(n)[0].lower(), prefer_raw, use, skipped, twins)
    use += _pair_twins(rest, shot_key, prefer_raw, use, skipped, twins)
    use.sort(key=str.lower)
    convert = [n for n in use if kind(n) != "native"
               or (ext(n) == ".png" and not png_is_plain(os.path.join(folder, n)))]
    return {"use": use, "convert": convert, "skipped": sorted(skipped, key=str.lower), "twins": twins}


def converted_name(name):
    return name + CONVERTED_SUFFIX


def image_size(path):
    """(width, height) of a JPEG or PNG from its header, as stored (EXIF orientation is
    not applied: COLMAP does not apply it either), or None."""
    try:
        with open(path, "rb") as f:
            head = f.read(2)
            if head == b"\x89P":
                rest = f.read(22)
                return struct.unpack(">II", rest[14:22]) if len(rest) == 22 else None
            if head != b"\xff\xd8":
                return None
            while True:
                byte = f.read(1)
                while byte and byte != b"\xff":
                    byte = f.read(1)
                while byte == b"\xff":
                    byte = f.read(1)
                if not byte:
                    return None
                marker = byte[0]
                if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                    continue
                (length,) = struct.unpack(">H", f.read(2))
                if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                    h, w = struct.unpack(">xHH", f.read(5))
                    return w, h
                f.seek(length - 2, os.SEEK_CUR)
    except (OSError, struct.error):
        return None


def _tiff_orientation(data):
    """Tag 0x0112 of the first IFD of TIFF data (bytes from the "II*" / "MM*" header)."""
    if len(data) < 8 or data[:2] not in (b"II", b"MM"):
        return None
    e = "<" if data[:2] == b"II" else ">"
    (off,) = struct.unpack_from(e + "I", data, 4)
    if off + 2 > len(data):
        return None
    (count,) = struct.unpack_from(e + "H", data, off)
    for i in range(min(count, 512)):
        pos = off + 2 + 12 * i
        if pos + 12 > len(data):
            return None
        tag, typ = struct.unpack_from(e + "HH", data, pos)
        if tag == 0x0112 and typ == 3:
            (value,) = struct.unpack_from(e + "H", data, pos + 8)
            return value if 1 <= value <= 8 else None
    return None


def exif_orientation(path):
    """The EXIF orientation (1-8) of a JPEG or TIFF-based file (TIFF, DNG), or None."""
    try:
        with open(path, "rb") as f:
            head = f.read(4)
            if head[:2] in (b"II", b"MM"):
                f.seek(0)
                return _tiff_orientation(f.read(1 << 20))
            if head[:2] != b"\xff\xd8":
                return None
            f.seek(2)
            while True:
                marker = f.read(2)
                if len(marker) < 2 or marker[0] != 0xFF or marker[1] in (0xD9, 0xDA):
                    return None
                (length,) = struct.unpack(">H", f.read(2))
                if marker[1] == 0xE1:
                    data = f.read(length - 2)
                    if data[:6] == b"Exif\x00\x00":
                        return _tiff_orientation(data[6:])
                else:
                    f.seek(length - 2, os.SEEK_CUR)
    except (OSError, struct.error):
        return None


def paths(tools_root, python_version=None):
    ver = python_version or sys.version_info[:2]
    root = os.path.join(tools_root, "formats")
    return {"root": root, "site": os.path.join(root, f"site-py{ver[0]}{ver[1]}")}


def runtime_info(site):
    path = current(os.path.join(site, RUNTIME_MARKER))
    if not (os.path.isfile(path) and os.path.isdir(os.path.join(site, "rawpy"))):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


VERIFY_CODE = ("import sys; sys.path.insert(0, sys.argv[1]); import PIL, pillow_heif, rawpy, OpenEXR; "
               "from PIL import features; print('FORMATS PIL', PIL.__version__, 'heif', "
               "pillow_heif.__version__, 'LibRaw', '.'.join(map(str, rawpy.libraw_version)), "
               "'avif', features.check('avif'))")


def install_runtime(job, python, site, log):
    """pip-install the image libraries into ``site`` (once)."""
    if job.run([python, "-E", "-m", "pip", "--version"], log, ok_codes=None) != 0:
        job.run([python, "-E", "-m", "ensurepip", "--user"], log, ok_codes=None)
        if job.run([python, "-E", "-m", "pip", "--version"], log, ok_codes=None) != 0:
            raise StepError("Blender's Python has no pip, so the image libraries for RAW/HEIC/EXR "
                            "photos cannot be installed")
    staging = site + ".staging"
    shutil.rmtree(staging, ignore_errors=True)
    installed = None
    for k, pkgs in enumerate(PACKAGES):
        job.set_fraction(0.05 + 0.1 * k, "downloading the image libraries (about 16 MB)")
        code = job.run([python, "-E", "-m", "pip", "install", "--no-deps", "--no-input",
                        "--disable-pip-version-check", "--no-warn-script-location",
                        "--progress-bar", "off", "--target", staging, *pkgs], log, ok_codes=None)
        if code == 0:
            installed = pkgs
            break
        shutil.rmtree(staging, ignore_errors=True)
    if installed is None:
        raise StepError("Could not download the image libraries for RAW/HEIC/EXR photos (pip "
                        "failed: is the internet reachable from Blender?)")
    found = {}

    def grab(text):
        if text.startswith("FORMATS "):
            found["report"] = text[8:]
        return None

    job.run([python, "-I", "-c", VERIFY_CODE, staging], log, parser=grab)
    if os.path.isdir(site):
        shutil.rmtree(site, ignore_errors=True)
    os.replace(staging, site)
    with open(os.path.join(site, RUNTIME_MARKER), "w", encoding="utf-8") as f:
        json.dump({"packages": list(installed), "report": found.get("report", "")}, f)
    job.add_line("Image libraries installed: " + found.get("report", ", ".join(installed)))


def link_or_copy(src, dst):
    try:
        if os.path.exists(dst):
            os.remove(dst)
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _progress_parser(job, total, lo, hi):
    def parse(text):
        if text.startswith("Converted "):
            part = text.split()[1]
            a, _, b = part.partition("/")
            if a.isdigit():
                return lo + (hi - lo) * int(a) / max(1, total), f"converting photo {a}/{total}"
        return None
    return parse


def convert(job, python, site, src_dir, names, out_dir, log, workers=0, max_side=0, twins=None):
    """Convert ``names`` (in src_dir) to JPEG in out_dir, scaled down to ``max_side`` on the
    long side (0 = their own size); returns the names that failed. ``twins`` ({RAW photo:
    its JPEG}) lends a RAW photo the JPEG's 35 mm focal length when it has none."""
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "convert_images.py")
    os.makedirs(out_dir, exist_ok=True)
    jobs = os.path.join(out_dir, "convert_jobs.json")
    items = [{"src": os.path.join(src_dir, n), "dst": os.path.join(out_dir, converted_name(n)),
              "kind": kind(n), "max_side": int(max_side or 0)} for n in names]
    for it, n in zip(items, names):
        if (twins or {}).get(n):
            it["twin"] = os.path.join(src_dir, twins[n])
    with open(jobs, "w", encoding="utf-8") as f:
        json.dump({"items": items}, f)
    failed = []

    def parse(text):
        if text.startswith("Failed "):
            failed.append(text[7:].split(":", 1)[0])
        elif text.startswith(("Exposure for ", "Lens shading maps")):
            job.add_line(text)
        return prog(text)

    prog = _progress_parser(job, len(items), 0.2, 0.98)
    job.run([python, "-I", script, "--site", site, "--jobs", jobs, "--workers", str(workers)],
            log, parser=parse, below_normal=True)
    os.remove(jobs)
    return failed


def is_converted(src, dst):
    """The converted file exists and is newer than its source."""
    try:
        return os.path.getsize(dst) > 0 and os.path.getmtime(dst) >= os.path.getmtime(src)
    except OSError:
        return False

