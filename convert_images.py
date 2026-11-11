# SPDX-License-Identifier: GPL-3.0-or-later
# Photogram3D - photos or video to a textured mesh with COLMAP + OpenMVS
# Copyright (C) 2026 Kruglov Iurii
# Authors: Kruglov Iurii, Anthropic Claude AI
# License: GNU General Public License (GPL) v3 or later, https://www.blender.org/about/license/
"""Convert photos COLMAP and OpenMVS cannot read to JPEG (run by formats.convert in
Blender's Python, outside Blender, with the libraries in --site).

    camera RAW (rawpy / LibRaw)  developed with the camera's white balance, highlights
                                 blended, one exposure for the whole set and a gentle
                                 highlight roll-off; a DNG's lens shading maps (GainMap
                                 opcodes, which phones write and LibRaw does not apply)
                                 are applied to the mosaic first
    EXR (OpenEXR)                the same exposure and roll-off, from linear light
    HEIC / HEIF / AVIF, TIFF, TGA, WebP, PSD, BMP, 16-bit PNG (Pillow, pillow-heif)
                                 converted as they are

The pixels stay the way the file stores them, with the EXIF orientation copied, as COLMAP
treats JPEGs: it does not turn them (it only takes gravity from the tag), and photos of
one camera must all have the same width and height. The JPEGs (quality 95, no chroma
subsampling) keep the camera make, model and focal length, which COLMAP uses for its
first lens estimate. Prints "Converted i/n NAME" and "Failed NAME: reason" lines.
"""

import argparse
import json
import multiprocessing
import os
import struct
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor

WHITE = 6.0            # linear value that the roll-off maps to white
MID_OUT = 0.18         # the set's typical (median) brightness after the roll-off, linear

EXIF_KEEP_IFD0 = (0x010F, 0x0110, 0x0132)                     # Make, Model, DateTime
EXIF_KEEP_EXIF = (0x920A, 0xA405, 0x9003, 0x829A, 0x829D, 0x8827, 0xA434, 0xA433)
# FocalLength, FocalLengthIn35mmFilm, DateTimeOriginal, ExposureTime, FNumber, ISO,
# LensModel, LensMake


def _init(site):
    if site and site not in sys.path:
        sys.path.insert(0, site)
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except ImportError:
        pass


# LibRaw's flip code -> EXIF orientation
_FLIP_TO_EXIF = {0: 1, 3: 3, 5: 8, 6: 6}


def _source_exif(path):
    """The source's EXIF: through Pillow, or for TIFF-based camera RAW files (DNG, CR2,
    NEF, ARW, PEF, ...) that Pillow cannot decode, straight from the TIFF directories."""
    from PIL import Image
    try:
        with Image.open(path) as im:
            ex = im.getexif()
            if len(ex):
                return ex
    except Exception:  # noqa: BLE001 - not an image Pillow can decode
        pass
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] not in (b"II*\x00", b"MM\x00*"):
        return Image.Exif()
    ex = Image.Exif()
    ex.load(b"Exif\x00\x00" + data)
    return ex


def _exif_from(path, orientation=None, twin=None):
    """A fresh EXIF block with the source's camera, lens, focal length and orientation.

    The 35 mm equivalent focal length is what COLMAP turns into a starting focal length in
    pixels; without it COLMAP guesses 1.2 x the image width. Phone DNG files lack it (the
    gas mask DNGs of Test034: 5549 px guessed, about 3340 px true, and the global solver
    warned about it), while the phone's JPEG of the same shot (``twin``) has it: it is
    taken from there."""
    from PIL import Image
    out = Image.Exif()
    out[0x0112] = orientation or 1
    try:
        src = _source_exif(path)
    except Exception:  # noqa: BLE001 - CR3, RAF and the like are not TIFF: no EXIF then
        src = None
    if src is None:
        _focal_from_twin(out, twin)
        return out
    if orientation is None and src.get(0x0112) in range(1, 9):
        out[0x0112] = src[0x0112]
    for tag in EXIF_KEEP_IFD0:
        if tag in src:
            out[tag] = src[tag]
    try:
        sub = src.get_ifd(0x8769)
    except Exception:  # noqa: BLE001 - a broken EXIF directory
        sub = {}
    dst = out.get_ifd(0x8769)
    for tag in EXIF_KEEP_EXIF:
        if tag in sub:
            dst[tag] = sub[tag]
        elif tag in src:                    # DNG may keep them in the main directory
            dst[tag] = src[tag]
    _focal_from_twin(out, twin)
    return out


def _focal_from_twin(out, twin):
    """Fill in the 35 mm equivalent focal length (and the focal length) from the JPEG or
    HEIC twin of a RAW photo when the RAW file has none."""
    dst = out.get_ifd(0x8769)
    if dst.get(0xA405) or not twin:
        return
    try:
        from PIL import Image
        with Image.open(twin) as im:
            sub = im.getexif().get_ifd(0x8769)
    except Exception:  # noqa: BLE001 - an unreadable twin only means no prior focal length
        return
    if sub.get(0xA405):
        dst[0xA405] = sub[0xA405]
        if 0x920A not in dst and 0x920A in sub:
            dst[0x920A] = sub[0x920A]


def _rolloff(x):
    """Highlight roll-off (extended Reinhard, white at WHITE), per channel, linear in/out."""
    return x * (1.0 + x / (WHITE * WHITE)) / (1.0 + x)


def _gain_for(median):
    """Exposure gain that puts a median linear value at MID_OUT after the roll-off."""
    lo, hi = 0.0, 64.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _rolloff(mid) < MID_OUT:
            lo = mid
        else:
            hi = mid
    return ((lo + hi) / 2) / max(median, 1e-6)


def _to_srgb8(linear):
    import numpy as np
    x = np.clip(_rolloff(np.maximum(linear, 0.0)), 0.0, 1.0)
    srgb = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)
    return (srgb * 255.0 + 0.5).astype(np.uint8)


def _luminance_median(rgb):
    import numpy as np
    y = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    return float(np.median(y))


# ---------------------------------------------------------------- DNG lens shading
# Phones store their lens shading correction (vignetting and colour shading, strongest in
# red at the edges) with a DNG as GainMap opcodes in OpcodeList2, to be applied to the
# mosaic before demosaicing. LibRaw does not apply them (checked with rawpy 0.27.1), so
# without them the corners come out darker and tinted.

_TIFF_TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8,
                   13: 4, 16: 8, 17: 8, 18: 8}
OPCODE_LIST2 = 51009
GAIN_MAP = 9
GAIN_RANGE = (0.2, 16.0)       # a map outside this is not a lens shading map: not applied


def _tiff_dirs(data):
    """(byte order, [{tag: (type, count, value bytes)}]) for the directories of a TIFF-based
    file: the IFD chain and the SubIFDs."""
    end = {b"II": "<", b"MM": ">"}.get(data[:2])
    dirs = []
    if end is None or len(data) < 8:
        return end, dirs
    todo, seen = [struct.unpack(end + "I", data[4:8])[0]], set()
    while todo and len(dirs) < 64:
        off = todo.pop(0)
        if off in seen or not 8 <= off <= len(data) - 2:
            continue
        seen.add(off)
        (n,) = struct.unpack(end + "H", data[off:off + 2])
        tags = {}
        for i in range(n):
            e = off + 2 + 12 * i
            if e + 12 > len(data):
                break
            tag, typ, cnt = struct.unpack(end + "HHI", data[e:e + 8])
            size = _TIFF_TYPE_SIZE.get(typ, 1) * cnt
            if size <= 4:
                tags[tag] = (typ, cnt, data[e + 8:e + 8 + size])
            else:
                (at,) = struct.unpack(end + "I", data[e + 8:e + 12])
                tags[tag] = (typ, cnt, data[at:at + size])
        dirs.append(tags)
        e = off + 2 + 12 * n
        if e + 4 <= len(data):
            (nxt,) = struct.unpack(end + "I", data[e:e + 4])
            if nxt:
                todo.append(nxt)
        if 330 in tags:                                       # SubIFDs
            typ, cnt, raw = tags[330]
            if typ in (4, 13) and len(raw) >= 4 * cnt:
                todo += list(struct.unpack(end + f"{cnt}I", raw[:4 * cnt]))
    return end, dirs


def _main_image(end, d):
    """The full-resolution image's directory: NewSubFileType 0 (or not given)."""
    if 254 not in d:
        return True
    typ, _cnt, raw = d[254]
    fmt = {3: "H", 4: "I"}.get(typ)
    return bool(fmt) and len(raw) >= struct.calcsize(fmt) and \
        struct.unpack(end + fmt, raw[:struct.calcsize(fmt)])[0] == 0


def dng_gain_maps(path):
    """The GainMap opcodes (DNG 1.3, opcode 9) of the main image's OpcodeList2, as dicts."""
    import numpy as np
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return []
    end, dirs = _tiff_dirs(data)
    blob = next((d[OPCODE_LIST2][2] for d in dirs if OPCODE_LIST2 in d and _main_image(end, d)),
                None)
    if not blob or len(blob) < 4:
        return []
    maps = []
    (count,) = struct.unpack(">I", blob[:4])                 # opcode lists are big-endian
    pos = 4
    for _ in range(min(count, 256)):
        if pos + 16 > len(blob):
            break
        op, _version, _flags, size = struct.unpack(">I4sII", blob[pos:pos + 16])
        body = blob[pos + 16:pos + 16 + size]
        pos += 16 + size
        if op != GAIN_MAP or len(body) < 76:
            continue
        top, left, bottom, right, plane, planes, rowp, colp, nv, nh = struct.unpack(">10I", body[:40])
        sv, sh, ov, oh = struct.unpack(">4d", body[40:72])
        (mplanes,) = struct.unpack(">I", body[72:76])
        n = nv * nh * mplanes
        if min(nv, nh, mplanes, rowp, colp, planes) < 1 or len(body) < 76 + 4 * n:
            continue
        gains = np.frombuffer(body[76:76 + 4 * n], ">f4").astype(np.float32).reshape(nv, nh, mplanes)
        maps.append({"area": (top, left, bottom, right), "plane": plane, "pitch": (rowp, colp),
                     "spacing": (sv, sh), "origin": (ov, oh), "gains": gains})
    return maps


def _map_axis(size, start, stop, step, spacing, origin, points):
    """Pixel positions along one axis and their map interpolation (index pair, fraction),
    as the DNG SDK computes them: map coordinate (pos + 0.5) / size, clamped to the map."""
    import numpy as np
    pos = np.arange(start, stop, step)
    if points == 1 or spacing <= 0:
        zero = np.zeros(len(pos), int)
        return zero, zero, np.zeros(len(pos))
    f = np.clip(((pos + 0.5) / size - origin) / spacing, 0.0, points - 1)
    i0 = np.floor(f).astype(int)
    return i0, np.minimum(i0 + 1, points - 1), f - i0


def _apply_gain_maps(raw, maps):
    """Apply the maps to the visible mosaic in place (the DNG SDK's stage-2 GainMap: the
    value above black times the interpolated gain, clipped at white). Returns the largest
    gain applied, or None when no map was applied."""
    import numpy as np
    import rawpy
    if not maps or raw.raw_type != rawpy.RawType.Flat:
        return None
    img = raw.raw_image_visible
    h, w = img.shape
    colors = raw.raw_colors_visible
    black = np.asarray(raw.black_level_per_channel, np.float32)
    white = float(raw.white_level)
    top_gain = None
    for m in maps:
        g = m["gains"][..., 0]
        if m["plane"] != 0 or not np.all(np.isfinite(g)) or \
                g.min() < GAIN_RANGE[0] or g.max() > GAIN_RANGE[1]:
            continue
        t, l, b, r = m["area"]
        b, r = min(b, h), min(r, w)
        rp, cp = m["pitch"]
        if t >= b or l >= r:
            continue
        r0, r1, rf = _map_axis(h, t, b, rp, m["spacing"][0], m["origin"][0], g.shape[0])
        c0, c1, cf = _map_axis(w, l, r, cp, m["spacing"][1], m["origin"][1], g.shape[1])
        rf, cf = rf[:, None], cf[None, :]
        gain = ((1 - rf) * ((1 - cf) * g[r0][:, c0] + cf * g[r0][:, c1])
                + rf * ((1 - cf) * g[r1][:, c0] + cf * g[r1][:, c1])).astype(np.float32)
        view = img[t:b:rp, l:r:cp]
        blk = black[colors[t:b:rp, l:r:cp]]
        view[...] = np.clip((view.astype(np.float32) - blk) * gain + blk + 0.5, 0.0, white
                            ).astype(img.dtype)
        top_gain = max(top_gain or 0.0, float(gain.max()))
    return top_gain


def _raw_linear(path, half=False):
    """(linear RGB in [0, 1] as the sensor stores it (not turned), the file's orientation
    as LibRaw's flip code, the largest lens shading gain applied or None)."""
    import numpy as np
    import rawpy
    maps = dng_gain_maps(path) if path.lower().endswith(".dng") else []
    with rawpy.imread(path) as raw:
        flip = int(raw.sizes.flip)
        shading = _apply_gain_maps(raw, maps)
        rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=16,
                              gamma=(1, 1), half_size=half, user_flip=0,
                              highlight_mode=rawpy.HighlightMode.Blend)
    return rgb.astype(np.float32) / 65535.0, flip, shading


def _exr_linear(path, step=1):
    import numpy as np
    import OpenEXR
    with OpenEXR.File(path) as f:
        ch = f.channels()
        names = list(ch.keys())
        if "RGB" in ch:
            rgb = np.asarray(ch["RGB"].pixels, dtype=np.float32)
        elif "RGBA" in ch:
            rgb = np.asarray(ch["RGBA"].pixels, dtype=np.float32)[..., :3]
        elif all(k in ch for k in ("R", "G", "B")):
            rgb = np.stack([np.asarray(ch[k].pixels, dtype=np.float32) for k in ("R", "G", "B")], -1)
        elif "Y" in ch:
            y = np.asarray(ch["Y"].pixels, dtype=np.float32)
            rgb = np.stack([y, y, y], -1)
        else:
            raise ValueError(f"no RGB or Y channels in the EXR (found: {', '.join(names)})")
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)
    return rgb[::step, ::step] if step > 1 else rgb


def measure(item):
    """Median linear brightness of a RAW or EXR photo (for the set's exposure)."""
    _init(item.get("site"))
    try:
        if item["kind"] == "raw":
            return _luminance_median(_raw_linear(item["src"], half=True)[0])
        if item["kind"] == "exr":
            return _luminance_median(_exr_linear(item["src"], step=4))
    except Exception:  # noqa: BLE001 - the conversion reports the error
        return None
    return None


def _fit(img, max_side):
    """Scale down to ``max_side`` on the long side (Lanczos); smaller images stay as they are."""
    from PIL import Image
    if not max_side or max(img.size) <= max_side:
        return img
    k = max_side / max(img.size)
    size = (max(1, round(img.size[0] * k)), max(1, round(img.size[1] * k)))
    return img.resize(size, Image.Resampling.LANCZOS, reducing_gap=3.0)


def _save(img, dst, exif, max_side=0):
    img = _fit(img, max_side)
    tmp = dst + ".tmp.jpg"
    img.save(tmp, "JPEG", quality=95, subsampling=0, exif=exif.tobytes())
    os.replace(tmp, dst)


def convert_one(item):
    """Convert one photo; returns the largest lens shading gain applied (DNG) or None."""
    _init(item.get("site"))
    from PIL import Image
    src, dst, kind = item["src"], item["dst"], item["kind"]
    if kind == "raw":
        rgb, flip, shading = _raw_linear(src)
        rgb *= item["gain"]
        _save(Image.fromarray(_to_srgb8(rgb)), dst,
              _exif_from(src, _FLIP_TO_EXIF.get(flip, 1), item.get("twin")), item.get("max_side", 0))
        return shading
    exif = _exif_from(src, twin=item.get("twin"))
    if kind == "exr":
        rgb = _exr_linear(src) * item["gain"]
        _save(Image.fromarray(_to_srgb8(rgb)), dst, exif, item.get("max_side", 0))
        return None
    with Image.open(src) as im:
        im.load()
        if im.mode in ("I;16", "I;16B", "I;16L", "I"):
            import numpy as np
            a = np.asarray(im, dtype=np.float32)
            top = 65535.0 if im.mode != "I" or a.max() > 255 else 255.0
            im = Image.fromarray(np.clip(a / top * 255.0 + 0.5, 0, 255).astype("uint8"))
        elif im.mode == "F":
            import numpy as np
            a = np.asarray(im, dtype=np.float32)
            im = Image.fromarray(np.clip(a * 255.0 + 0.5, 0, 255).astype("uint8"))
        if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
            im = im.convert("RGBA")
            back = Image.new("RGBA", im.size, (128, 128, 128, 255))   # neutral, featureless
            im = Image.alpha_composite(back, im)
        _save(im.convert("RGB"), dst, exif, item.get("max_side", 0))
    return None


def _run(item):
    try:
        return item["src"], None, convert_one(item)
    except Exception as exc:  # noqa: BLE001 - reported per photo
        return (item["src"], f"{type(exc).__name__}: {exc}".splitlines()[0][:200]
                or traceback.format_exc(limit=1), None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="")
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()
    _init(a.site)
    with open(a.jobs, encoding="utf-8") as f:
        items = json.load(f)["items"]
    for it in items:
        it["site"] = a.site
    workers = a.workers or max(1, min(4, (os.cpu_count() or 2) // 2))
    # LibRaw runs OpenMP threads, which deadlock in forked processes (rawpy warns): spawn
    spawn = multiprocessing.get_context("spawn")
    # one exposure for all RAW (and all EXR) photos, so the set stays consistent
    measured = [it for it in items if it["kind"] in ("raw", "exr")]
    gains = {}
    if measured:
        with ProcessPoolExecutor(workers, mp_context=spawn, initializer=_init,
                                 initargs=(a.site,)) as pool:
            medians = list(pool.map(measure, measured))
        for kind in ("raw", "exr"):
            vals = sorted(m for it, m in zip(measured, medians) if it["kind"] == kind and m)
            if vals:
                gains[kind] = _gain_for(vals[len(vals) // 2])
                print(f"Exposure for the {kind.upper()} photos: x{gains[kind]:.2f} "
                      f"(median brightness {vals[len(vals) // 2]:.4f})", flush=True)
    for it in items:
        it["gain"] = gains.get(it["kind"], 1.0)
    done, shaded = 0, []
    with ProcessPoolExecutor(workers, mp_context=spawn, initializer=_init, initargs=(a.site,)) as pool:
        for src, err, shading in pool.map(_run, items):
            done += 1
            name = os.path.basename(src)
            if err:
                print(f"Failed {name}: {err}", flush=True)
            else:
                if shading is not None:
                    shaded.append(shading)
                print(f"Converted {done}/{len(items)} {name}"
                      + (f" (lens shading map, gain up to x{shading:.2f})" if shading else ""),
                      flush=True)
    dngs = sum(1 for it in items if it["kind"] == "raw" and it["src"].lower().endswith(".dng"))
    if shaded:
        print(f"Lens shading maps from the DNG files applied to {len(shaded)} of {dngs} DNG photos"
              f" (gain up to x{max(shaded):.2f})", flush=True)
    elif dngs:
        print(f"Lens shading maps: none in the DNG files ({dngs} photo{'s' if dngs != 1 else ''}; "
              "phones store them, most cameras do not)", flush=True)


if __name__ == "__main__":
    main()
