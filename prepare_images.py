# SPDX-License-Identifier: GPL-3.0-or-later
# Photogram3D - photos or video to a textured mesh with COLMAP + OpenMVS
# Copyright (C) 2026 Kruglov Iurii
# Authors: Kruglov Iurii, Anthropic Claude AI
# License: GNU General Public License (GPL) v3 or later, https://www.blender.org/about/license/
"""Even out exposure and vignetting across the photos (0.3.5), from the solved scene.

Run by pipeline.step_even in Blender's Python outside Blender, with Pillow from the image
libraries' folder (--site). The same surface point seen in several photos should have the
same colour; what differs is the exposure and white balance of each photo and the
vignetting by the position in the frame. The camera solve tells which pixel of which photo
sees which 3D point, so both are fitted per colour channel c, in linear light:

    log I(p, i, c) = log L(p, c) + log E(i, c) + log V_c(r)
    log V_c(r)     = a1 r^2 + a2 r^4 + a3 r^6      r: distance from the centre / half diagonal

by alternating robust means (Huber weights): each point's radiance from its observations,
each photo's exposure from its points, the vignetting from what is left. Speculars,
occlusions and moving things disagree with the rest and weigh little. A point is seen at
different places in different photos, so exposure and vignetting separate; a subject that
is always in the middle of the frame does not bias the vignetting, as fitting it to the
photos alone would. The corrected photos (exposures to their average, vignetting removed,
optionally sharpened) go to the texture and the splats. Since 0.3.12 they are the undistorted
photos, after the depth maps; the camera solve and the depth maps use the photos as they are.

Input (--jobs, JSON): {"images": [{"src", "dst"}], "obs": npz with per-observation arrays
image, point, x, y (COLMAP pixel coordinates), "sharpen": 0..1}. Prints "Prepared i/n
NAME", "Failed NAME: reason" and summary lines.
"""

import argparse
import json
import math
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor

HUBER = 0.05            # log units (about 5%) beyond which a residual weighs less
ITERATIONS = 600        # the alternating means converge slowly, but each pass is cheap
TOLERANCE = 1e-7
SAMPLE_SIDE = 1600      # photos are sampled at about this size (JPEG DCT scaling)
DARK = 0.01             # linear values below this are noise
BRIGHT = 0.96           # sRGB values above this may be clipped
MIN_OBS = 20            # photos with fewer usable samples keep their exposure
CORNER_MAX = 1.05       # a fitted "vignetting" that brightens corners more is not one
CORNER_MIN = 0.2        # nor one that darkens them below this


def _init(site):
    if site and site not in sys.path:
        sys.path.insert(0, site)


def srgb_to_linear(x):
    import numpy as np
    x = np.asarray(x, np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    import numpy as np
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


# ---------------------------------------------------------------- the fit

def _group_mean(values, groups, weights, n):
    import numpy as np
    num = np.bincount(groups, weights=values * weights, minlength=n)
    den = np.bincount(groups, weights=weights, minlength=n)
    return num / np.maximum(den, 1e-12), den


def fit(obs_point, obs_image, radius2, log_i, n_points, n_images):
    """Exposure and vignetting of one channel. Returns (log exposure per photo with a
    weighted mean of 0, vignetting coefficients (a1, a2, a3), samples per photo)."""
    import numpy as np
    e = np.zeros(n_images)
    a = np.zeros(3)
    w = np.ones(len(log_i))
    basis = np.stack([radius2, radius2 ** 2, radius2 ** 3], 1)
    seen = np.bincount(obs_image, minlength=n_images).astype(float)
    for _ in range(ITERATIONS):
        e_old, a_old = e.copy(), a.copy()
        v = basis @ a
        L, _ = _group_mean(log_i - e[obs_image] - v, obs_point, w, n_points)
        e, seen = _group_mean(log_i - L[obs_point] - v, obs_image, w, n_images)
        e[seen < 1e-9] = 0.0
        e -= np.average(e, weights=np.maximum(seen, 1e-12))
        rest = log_i - L[obs_point] - e[obs_image]
        sw = np.sqrt(w)
        a, *_ = np.linalg.lstsq(basis * sw[:, None], rest * sw, rcond=None)
        r = rest - basis @ a
        w = np.where(np.abs(r) <= HUBER, 1.0, HUBER / np.maximum(np.abs(r), 1e-12))
        if max(np.abs(e - e_old).max(initial=0.0), np.abs(a - a_old).max()) < TOLERANCE:
            break
    counts = np.bincount(obs_image, minlength=n_images)
    return e, a, counts


def vignetting(a, radius2):
    import numpy as np
    r2 = np.asarray(radius2, np.float64)
    return np.exp(a[0] * r2 + a[1] * r2 ** 2 + a[2] * r2 ** 3)


# ---------------------------------------------------------------- photos

def sample(item):
    """Linear RGB at the photo's observations (3 x 3 mean at about SAMPLE_SIDE px) and
    whether each is usable; (index, values N x 3, usable N x 3, width, height)."""
    _init(item.get("site"))
    import numpy as np
    from PIL import Image
    with Image.open(item["src"]) as im:
        w, h = im.size
        scale = min(1.0, SAMPLE_SIDE / max(w, h))
        if im.format == "JPEG" and scale < 0.75:
            im.draft("RGB", (int(w * scale), int(h * scale)))
        small = np.asarray(im.convert("RGB"), np.float32) / 255.0
    sh, sw = small.shape[:2]
    xs = np.clip((np.asarray(item["x"]) * sw / w).astype(int), 1, sw - 2)
    ys = np.clip((np.asarray(item["y"]) * sh / h).astype(int), 1, sh - 2)
    acc = np.zeros((len(xs), 3), np.float32)
    top = np.zeros((len(xs), 3), np.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            v = small[ys + dy, xs + dx]
            acc += v
            top = np.maximum(top, v)
    srgb = acc / 9.0
    lin = srgb_to_linear(srgb)
    usable = (lin > DARK) & (top < BRIGHT)
    return item["index"], lin, usable, w, h


def correct(item):
    """Write the corrected photo; returns (index, error or None)."""
    _init(item.get("site"))
    try:
        import numpy as np
        from PIL import Image, ImageFilter
        with Image.open(item["src"]) as im:
            exif = im.info.get("exif")
            icc = im.info.get("icc_profile")
            rgb = np.asarray(im.convert("RGB"), np.float32) / 255.0
        h, w = rgb.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r2 = ((xx + 0.5 - w / 2) ** 2 + (yy + 0.5 - h / 2) ** 2) / ((w / 2) ** 2 + (h / 2) ** 2)
        lin = srgb_to_linear(rgb)
        for c in range(3):
            gain = 1.0 / (math.exp(item["log_e"][c]) * vignetting(item["a"][c], r2))
            lin[..., c] *= gain.astype(np.float32)
        out = Image.fromarray((linear_to_srgb(lin) * 255.0 + 0.5).astype(np.uint8))
        if item.get("sharpen", 0) > 0:
            out = out.filter(ImageFilter.UnsharpMask(radius=1.2, percent=int(round(item["sharpen"] * 120)),
                                                     threshold=2))
        os.makedirs(os.path.dirname(item["dst"]) or ".", exist_ok=True)
        tmp = item["dst"] + ".tmp.jpg"
        kw = {"quality": 95, "subsampling": 0}
        if exif:
            kw["exif"] = exif
        if icc:
            kw["icc_profile"] = icc
        out.save(tmp, "JPEG", **kw)
        os.replace(tmp, item["dst"])
        return item["index"], None
    except Exception as exc:  # noqa: BLE001 - reported per photo
        return item["index"], f"{type(exc).__name__}: {exc}".splitlines()[0][:200]


def main():
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="")
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()
    _init(a.site)
    with open(a.jobs, encoding="utf-8") as f:
        jobs = json.load(f)
    images = jobs["images"]
    n = len(images)
    with np.load(jobs["obs"]) as d:
        obs_image, obs_point = d["image"].astype(np.int64), d["point"].astype(np.int64)
        obs_x, obs_y = d["x"], d["y"]
    workers = a.workers or max(1, min(4, (os.cpu_count() or 2) // 2))
    spawn = multiprocessing.get_context("spawn")

    # 1. samples
    items = []
    for i, im in enumerate(images):
        sel = obs_image == i
        items.append({"index": i, "src": im["src"], "x": obs_x[sel].tolist(), "y": obs_y[sel].tolist(),
                      "site": a.site})
    values = np.zeros((len(obs_image), 3), np.float32)
    usable = np.zeros((len(obs_image), 3), bool)
    r2 = np.zeros(len(obs_image))
    order = [np.flatnonzero(obs_image == i) for i in range(n)]
    done = 0
    with ProcessPoolExecutor(workers, mp_context=spawn, initializer=_init, initargs=(a.site,)) as pool:
        for i, lin, ok, w, h in pool.map(sample, items):
            idx = order[i]
            values[idx], usable[idx] = lin, ok
            r2[idx] = ((obs_x[idx] - w / 2) ** 2 + (obs_y[idx] - h / 2) ** 2) / ((w / 2) ** 2 + (h / 2) ** 2)
            done += 1
            print(f"Sampled {done}/{n} {os.path.basename(images[i]['src'])}", flush=True)

    # 2. the fit, per channel, on points seen usably at least twice
    n_points = int(obs_point.max()) + 1 if len(obs_point) else 0
    log_e = np.zeros((n, 3))
    coef = np.zeros((3, 3))
    for c in range(3):
        ok = usable[:, c]
        per_point = np.bincount(obs_point[ok], minlength=n_points)
        ok &= per_point[obs_point] >= 2
        if ok.sum() < 200:
            continue
        e, av, counts = fit(obs_point[ok], obs_image[ok], r2[ok], np.log(values[ok, c].astype(np.float64)),
                            n_points, n)
        e[counts < MIN_OBS] = 0.0
        corner = float(vignetting(av, 1.0))
        if not CORNER_MIN <= corner <= CORNER_MAX:
            av = np.zeros(3)
        log_e[:, c], coef[c] = e, av
    stops = log_e.mean(1) / math.log(2)
    print(f"Exposure: {stops.max() - stops.min():.2f} stops between the darkest and brightest photo, "
          "evened out", flush=True)
    corners = [float(vignetting(coef[c], 1.0)) for c in range(3)]
    print("Vignetting at the corners: " + ", ".join(
        f"{ch} {math.log2(v):+.2f} stops" for ch, v in zip("RGB", corners)) + ", removed", flush=True)

    # 3. the corrected photos
    work = [{"index": i, "src": im["src"], "dst": im["dst"], "log_e": log_e[i].tolist(),
             "a": coef.tolist(), "sharpen": float(jobs.get("sharpen") or 0.0), "site": a.site}
            for i, im in enumerate(images)]
    done = 0
    with ProcessPoolExecutor(workers, mp_context=spawn, initializer=_init, initargs=(a.site,)) as pool:
        for i, err in pool.map(correct, work):
            done += 1
            name = os.path.basename(images[i]["src"])
            if err:
                print(f"Failed {name}: {err}", flush=True)
            else:
                print(f"Prepared {done}/{n} {name}", flush=True)
    np.savez(os.path.join(os.path.dirname(a.jobs), "photometric.npz"), log_exposure=log_e, vignetting=coef)


if __name__ == "__main__":
    main()
