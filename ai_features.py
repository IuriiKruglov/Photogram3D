# SPDX-License-Identifier: GPL-3.0-or-later
# Photogram3D - photos or video to a textured mesh with COLMAP + OpenMVS
# Copyright (C) 2026 Kruglov Iurii
# Authors: Kruglov Iurii, Anthropic Claude AI
# License: GNU General Public License (GPL) v3 or later, https://www.blender.org/about/license/
"""COLMAP's AI features (ALIKED keypoints, LightGlue matching) on any GPU. Runs as its own process.

COLMAP 4 runs its ALIKED and LightGlue models with ONNX Runtime, on the GPU only with NVIDIA's
CUDA libraries. This script runs the same model files with ONNX Runtime and DirectML (any
DirectX 12 GPU), or on the CPU, and does exactly what COLMAP does around them, so that COLMAP
continues with the result as if it had computed it itself:

    extract  photos -> one COLMAP feature file per photo (keypoints, for `colmap
             feature_importer`) and the ALIKED descriptors next to it (.npy, float32)
    match    photo pairs -> LightGlue matches written into COLMAP's database (then
             `colmap geometric_verifier` checks them)
    bench    a quick test of the GPU and the CPU on one photo: time and whether it works

Started by Photogram3D with Blender's Python:
    python -I ai_features.py extract --site DIR --model ALIKED.onnx --images DIR --list LIST
                                     --out DIR --max-size 1600 [--max-points 2048] [--device auto|cpu]
    python -I ai_features.py match --site DIR --model LIGHTGLUE.onnx --db DB --pairs PAIRS
                                   [--min-score 0.1] [--device auto|cpu]
        PAIRS: one pair of image names per line, separated by a tab
    python -I ai_features.py bench --site DIR --aliked FILE --lightglue FILE [--image FILE]
                                   [--gpu-sizes 1280,1600,2048,2400] [--cpu-size 1600] [--device auto|cpu]

What is copied from COLMAP 4.2.1 (src/colmap/feature/aliked.cc, onnx_matchers.cc,
controllers/feature_extraction.cc, sensor/bitmap.cc):
- the photo is read as stored (no EXIF rotation), reduced so that its long side is at most
  max_size (each side rounded to the nearest pixel, bilinear with antialiasing), and turned
  upright from its EXIF orientation before ALIKED, the keypoints turned back afterwards;
- ALIKED's input is RGB in [0, 1], padded to a multiple of 32 by repeating the last row and
  column; its keypoints come normalized to [-1, 1] over the padded image (pixel centres) and
  are kept when their score is at least min_score and they lie inside the photo;
- COLMAP's keypoints have the pixel corner as origin (+0.5), and are scaled from the reduced
  photo to its full size;
- LightGlue gets the keypoints with the pixel centre as origin (-0.5), turned upright the
  same way, the image size as (width, height), and a match is kept when its score is at least
  min_score (0.1).
"""

import argparse
import math
import os
import sqlite3
import sys
import time

MAX_IMAGES = 2147483647           # COLMAP's pair id: MAX_IMAGES * smaller id + larger id
ALIKED_N16ROT = 1                 # COLMAP's FeatureExtractorType of the descriptors


# ---------------------------------------------------------------------------- sessions

def make_session(ort, model_path, device, threads=0):
    """An ONNX Runtime session on DirectML (the high-performance GPU), else on the CPU (with
    ``threads`` CPU threads, or ONNX Runtime's default)."""
    available = ort.get_available_providers()
    attempts = []
    if device != "cpu" and "DmlExecutionProvider" in available:
        # high_performance + gpu picks the discrete GPU on laptops with an integrated one;
        # DirectML needs memory patterns off and sequential execution
        attempts.append(([("DmlExecutionProvider", {"performance_preference": "high_performance",
                                                     "device_filter": "gpu"}),
                          "CPUExecutionProvider"], True))
        attempts.append(([("DmlExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"], True))
    if device != "cpu" and "CUDAExecutionProvider" in available:
        attempts.append((["CUDAExecutionProvider", "CPUExecutionProvider"], False))
    attempts.append((["CPUExecutionProvider"], False))
    last = None
    for providers, is_dml in attempts:
        try:
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            if threads:
                opts.intra_op_num_threads = threads
            if is_dml:
                opts.enable_mem_pattern = False
                opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            return ort.InferenceSession(model_path, sess_options=opts, providers=providers)
        except Exception as exc:  # noqa: BLE001 - try the next provider
            last = exc
            name = providers[0] if isinstance(providers[0], str) else providers[0][0]
            print(f"provider {name} unavailable: {exc}", flush=True)
    raise RuntimeError(f"cannot create an inference session: {last}")


class Model:
    """A session that falls back to the CPU when the GPU fails on an input."""

    def __init__(self, ort, path, device, threads=0):
        self.ort, self.path, self.threads = ort, path, threads
        self.sess = make_session(ort, path, device, threads)
        self.provider = self.sess.get_providers()[0]

    def run(self, feed):
        try:
            return self.sess.run(None, feed)
        except Exception as exc:  # noqa: BLE001 - e.g. out of GPU memory
            if self.provider == "CPUExecutionProvider":
                raise
            print(f"Photogram3D AI: {self.provider} failed ({exc}); continuing on the CPU", flush=True)
            self.sess = make_session(self.ort, self.path, "cpu", self.threads)
            self.provider = self.sess.get_providers()[0]
            return self.sess.run(None, feed)


# ---------------------------------------------------------------------------- geometry

def rot90_from_orientation(orientation):
    """COLMAP's turn of a photo to upright (counter-clockwise quarter turns) from its EXIF
    orientation: GravityFromExifOrientation, then ComputeRot90FromGravity."""
    gravity = {1: (0.0, 1.0), 3: (0.0, -1.0), 6: (1.0, 0.0), 8: (-1.0, 0.0)}.get(orientation)
    return rot90_from_gravity(gravity) if gravity else 0


def rot90_from_gravity(gravity):
    if gravity is None:
        return 0
    angle = math.atan2(gravity[1], gravity[0])
    k = int(round_half_away((angle - math.pi / 2) / (math.pi / 2)))
    k = int(math.fmod(k, 4))
    return k + 4 if k < 0 else k


def round_half_away(x):
    """std::round: halves away from zero (Python's round() rounds them to even)."""
    return math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5)


def rotate_points(xy, k, width, height):
    """FeatureKeypoint::Rot90 for an (N, 2) array: k counter-clockwise quarter turns of a
    width x height image."""
    import numpy as np
    k %= 4
    if k == 0:
        return xy
    x, y = xy[:, 0], xy[:, 1]
    if k == 1:
        out = (y, width - x)
    elif k == 2:
        out = (width - x, height - y)
    else:
        out = (height - y, x)
    return np.stack(out, axis=1).astype(np.float32)


def thumbnail_size(width, height, max_size):
    """Bitmap::Thumbnail: the long side at most max_size, each side rounded."""
    if max_size <= 0 or (width <= max_size and height <= max_size):
        return width, height
    scale = max_size / max(width, height)
    return int(round_half_away(width * scale)), int(round_half_away(height * scale))


# ---------------------------------------------------------------------------- ALIKED

def exif_orientation(im):
    try:
        return int(im.getexif().get(0x0112, 1))
    except Exception:  # noqa: BLE001 - unreadable EXIF: as stored
        return 1


def load_photo(Image, np, path, max_size):
    """(RGB uint8 array of the reduced photo, full width, full height, EXIF orientation).

    Decoded at full size and then reduced, as COLMAP does (no JPEG draft decoding, which
    would give other pixels); Pillow's bilinear resize filters over the reduction like
    OpenImageIO's triangle filter that COLMAP uses."""
    with Image.open(path) as im:
        orientation = exif_orientation(im)
        full_w, full_h = im.size
        im = im.convert("RGB")
        size = thumbnail_size(full_w, full_h, max_size)
        if size != im.size:
            im = im.resize(size, Image.Resampling.BILINEAR)
        return np.asarray(im, dtype=np.uint8), full_w, full_h, orientation


def aliked(model, np, rgb, max_points, min_score):
    """ALIKED on an upright RGB image: (keypoints (N, 2) in COLMAP's convention, descriptors
    (N, D) float32, scores (N,))."""
    h, w = rgb.shape[:2]
    ph, pw = -(-h // 32) * 32, -(-w // 32) * 32
    x = rgb.astype(np.float32) / 255.0
    if (ph, pw) != (h, w):
        x = np.pad(x, ((0, ph - h), (0, pw - w), (0, 0)), mode="edge")
    feed = {"image": np.ascontiguousarray(x.transpose(2, 0, 1)[None]),
            "max_keypoints": np.array(max_points, dtype=np.int64),
            "min_score": np.array(min_score, dtype=np.float32)}
    kpts, desc, scores = model.run(feed)
    kpts, desc, scores = kpts[0], desc[0], scores[0]
    px = (kpts[:, 0] + 1.0) * (0.5 * (pw - 1)) + 0.5
    py = (kpts[:, 1] + 1.0) * (0.5 * (ph - 1)) + 0.5
    keep = (scores >= min_score) & (px >= 0) & (px < w) & (py >= 0) & (py < h)
    return (np.stack([px[keep], py[keep]], axis=1).astype(np.float32),
            np.ascontiguousarray(desc[keep], dtype=np.float32), scores[keep])


def features_for(model, Image, np, path, max_size, max_points, min_score):
    """COLMAP's ALIKED extraction of one photo: keypoints in full-size pixels (pixel corner
    origin) and descriptors."""
    rgb, full_w, full_h, orientation = load_photo(Image, np, path, max_size)
    th, tw = rgb.shape[:2]
    k = rot90_from_orientation(orientation)
    if k:
        rgb = np.ascontiguousarray(np.rot90(rgb, k))         # counter-clockwise, as Bitmap::Rot90
    xy, desc, _ = aliked(model, np, rgb, max_points, min_score)
    if k:
        h, w = rgb.shape[:2]
        xy = rotate_points(xy, 4 - k, w, h)
    if (tw, th) != (full_w, full_h):
        xy = xy * np.array([full_w / tw, full_h / th], dtype=np.float32)
    return xy, desc


def write_colmap_features(path, xy):
    """COLMAP's feature text file (LoadSiftFeaturesFromTextFile): keypoints with scale 1 and
    orientation 0; the 128 descriptor values are zeros (the real ones go in afterwards)."""
    zeros = " 0" * 128
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="ascii", newline="\n") as f:
        f.write(f"{len(xy)} 128\n")
        for x, y in xy.tolist():
            f.write(f"{x:.6f} {y:.6f} 1 0{zeros}\n")
    os.replace(tmp, path)


def cmd_extract(args, np, ort, Image):
    with open(args.list, "r", encoding="utf-8") as f:
        names = [ln.strip() for ln in f if ln.strip()]
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    model = Model(ort, args.model, args.device)
    print(f"Photogram3D AI: onnxruntime {ort.__version__}, ALIKED on {model.provider}, "
          f"{len(names)} photos at up to {args.max_size} px, {args.max_points} points", flush=True)
    failures = 0
    for i, name in enumerate(names, start=1):
        t1 = time.time()
        base = os.path.join(args.out, name)
        if os.path.isfile(base + ".txt") and os.path.isfile(base + ".npy"):
            print(f"Features {i}/{len(names)} {name} done before", flush=True)
            continue
        try:
            os.makedirs(os.path.dirname(base), exist_ok=True)
            xy, desc = features_for(model, Image, np, os.path.join(args.images, name), args.max_size,
                                    args.max_points, args.min_score)
            np.save(base + ".npy.tmp.npy", desc)
            os.replace(base + ".npy.tmp.npy", base + ".npy")
            write_colmap_features(base + ".txt", xy)
            print(f"Features {i}/{len(names)} {name} {len(xy)} points {time.time() - t1:.2f} s "
                  f"({model.provider})", flush=True)
        except Exception as exc:  # noqa: BLE001 - COLMAP skips such a photo too
            failures += 1
            print(f"Features {i}/{len(names)} {name} FAILED: {exc}", flush=True)
    print(f"Photogram3D AI: features of {len(names) - failures}/{len(names)} photos in "
          f"{time.time() - t0:.1f} s ({model.provider})", flush=True)
    return 0 if failures < len(names) else 2


# ---------------------------------------------------------------------------- LightGlue

def read_database(np, db):
    """Per image id: (name, keypoints (N, 2), descriptors (N, D), (width, height), rot90)."""
    con = sqlite3.connect(db, timeout=60)
    try:
        cams = {cid: (w, h) for cid, w, h in con.execute("SELECT camera_id, width, height FROM cameras")}
        gravity = {}
        try:
            for data_id, blob in con.execute("SELECT corr_data_id, gravity FROM pose_priors"):
                if blob and len(blob) >= 24:
                    g = np.frombuffer(blob[:24], dtype=np.float64)
                    if np.all(np.isfinite(g)):
                        gravity[data_id] = (float(g[0]), float(g[1]))
        except sqlite3.Error:
            pass
        images = {}
        for iid, name, cid in con.execute("SELECT image_id, name, camera_id FROM images"):
            images[iid] = [name, None, None, cams[cid], rot90_from_gravity(gravity.get(iid))]
        for iid, rows, cols, data in con.execute("SELECT image_id, rows, cols, data FROM keypoints"):
            if iid in images and rows:
                images[iid][1] = np.frombuffer(data, dtype=np.float32).reshape(rows, cols)[:, :2]
        for iid, typ, rows, cols, data in con.execute(
                "SELECT image_id, type, rows, cols, data FROM descriptors"):
            if iid in images and rows:
                if typ != ALIKED_N16ROT or cols % 4:
                    raise SystemExit(f"image {iid}: descriptors are not ALIKED's (type {typ})")
                images[iid][2] = np.frombuffer(data, dtype=np.float32).reshape(rows, cols // 4)
        return {k: tuple(v) for k, v in images.items()}
    finally:
        con.close()


def lightglue_inputs(np, image):
    """kpts (1, N, 2) with the pixel centre as origin, turned upright; desc (1, N, D); size."""
    _, xy, desc, (w, h), k = image
    pts = xy.astype(np.float32)
    if k:
        pts = rotate_points(pts, k, w, h)
        if k % 2:
            w, h = h, w
    return (np.ascontiguousarray((pts - 0.5)[None], dtype=np.float32),
            np.ascontiguousarray(desc[None], dtype=np.float32),
            np.array([[w, h]], dtype=np.float32))


def lightglue(model, np, a, b, min_score):
    k0, d0, s0 = a
    k1, d1, s1 = b
    m, scores = model.run({"kpts0": k0, "kpts1": k1, "desc0": d0, "desc1": d1,
                           "image_size0": s0, "image_size1": s1})
    m = np.asarray(m).reshape(-1).astype(np.int64)
    scores = np.asarray(scores).reshape(-1)
    keep = (m >= 0) & (scores >= min_score)
    idx = np.nonzero(keep)[0]
    return np.stack([idx, m[keep]], axis=1).astype(np.uint32)


def pair_id(id1, id2):
    return MAX_IMAGES * min(id1, id2) + max(id1, id2)


def cmd_match(args, np, ort, Image):
    images = read_database(np, args.db)
    with open(args.pairs, "r", encoding="utf-8") as f:
        by_name = {v[0]: k for k, v in images.items()}
        pairs = []
        for ln in f:
            parts = ln.rstrip("\r\n").split("\t")          # names may hold spaces
            if len(parts) == 2 and parts[0] in by_name and parts[1] in by_name:
                pairs.append((by_name[parts[0]], by_name[parts[1]]))
    t0 = time.time()
    model = Model(ort, args.model, args.device)
    print(f"Photogram3D AI: onnxruntime {ort.__version__}, LightGlue on {model.provider}, "
          f"{len(pairs)} pairs of {len(images)} photos", flush=True)
    con = sqlite3.connect(args.db, timeout=60)
    done = {r[0] for r in con.execute("SELECT pair_id FROM matches")}
    cache = {}

    def inputs(iid):
        if iid not in cache:
            if len(cache) > 256:
                cache.pop(next(iter(cache)))
            cache[iid] = lightglue_inputs(np, images[iid])
        return cache[iid]

    rows, total, t_last = [], 0, 0.0
    try:
        for n, (i1, i2) in enumerate(pairs, start=1):
            pid = pair_id(i1, i2)
            if pid in done:
                continue
            a, b = (i1, i2) if i1 < i2 else (i2, i1)
            if images[a][1] is None or images[b][1] is None:
                found = np.zeros((0, 2), dtype=np.uint32)
            else:
                found = lightglue(model, np, inputs(a), inputs(b), args.min_score)
            rows.append((pid, len(found), 2, found.tobytes()))
            total += len(found)
            done.add(pid)
            now = time.time()
            if len(rows) >= 200 or now - t_last > 2.0 or n == len(pairs):
                con.executemany("INSERT OR REPLACE INTO matches (pair_id, rows, cols, data) "
                                "VALUES (?, ?, ?, ?)", rows)
                con.commit()
                rows = []
                t_last = now
                print(f"Matches {n}/{len(pairs)} {total} matched points ({model.provider})", flush=True)
        if rows:
            con.executemany("INSERT OR REPLACE INTO matches (pair_id, rows, cols, data) VALUES (?, ?, ?, ?)",
                            rows)
            con.commit()
    finally:
        con.close()
    print(f"Photogram3D AI: {len(pairs)} pairs matched in {time.time() - t0:.1f} s ({model.provider})",
          flush=True)
    return 0


# ---------------------------------------------------------------------------- bench

def sample_image(np, size, seed=3):
    """A textured test picture (random shapes) when no photo is given."""
    rng = np.random.default_rng(seed)
    w, h = size
    img = np.full((h, w, 3), 128, dtype=np.uint8)
    for _ in range(w * h // 900):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = int(rng.integers(2, max(3, w // 60)))
        img[max(0, y - r):y + r, max(0, x - r):x + r] = rng.integers(0, 256, 3, dtype=np.uint8)
    return img


def bench_device(args, np, ort, Image, device, sizes):
    """ALIKED at each size (stopping at the first that fails) and LightGlue on one pair, on
    ``device``; prints one BENCH line per measurement."""
    try:
        ali = Model(ort, args.aliked, device)
    except Exception as exc:  # noqa: BLE001
        print(f"BENCH {device} session FAILED {exc}", flush=True)
        return
    provider = ali.provider
    if device == "auto" and provider == "CPUExecutionProvider":
        print("BENCH auto no-gpu", flush=True)
        return
    feats = None
    for size in sizes:
        try:
            if args.image:
                rgb, *_ = load_photo(Image, np, args.image, size)
            else:
                rgb = sample_image(np, (size, size * 3 // 4))
            aliked(ali, np, rgb[:96, :96], 256, 0.2)                     # warm-up
            t = time.time()
            xy, desc, _ = aliked(ali, np, rgb, args.max_points, 0.2)
            dt = time.time() - t
            if ali.provider != provider:
                raise RuntimeError("fell back to the CPU")
            feats = (xy, desc, rgb.shape[1], rgb.shape[0])
            print(f"BENCH {provider} extract {size} {dt:.3f} {len(xy)}", flush=True)
        except Exception as exc:  # noqa: BLE001 - too large for the GPU, say
            print(f"BENCH {provider} extract {size} FAILED {type(exc).__name__}: {str(exc)[:160]}",
                  flush=True)
            break
    if feats is None:
        return
    try:
        # on the CPU COLMAP runs several LightGlue models at once, one thread each
        # (pipeline.ai_matchers): time one on one thread and divide by how many run
        parallel = max(1, args.cpu_matchers) if device == "cpu" else 1
        lg = Model(ort, args.lightglue, device, threads=1 if device == "cpu" else 0)
        xy, desc, w, h = feats
        shifted = xy + np.array([7.0, 3.0], dtype=np.float32)
        ia = lightglue_inputs(np, ("a", xy, desc, (w, h), 0))
        ib = lightglue_inputs(np, ("b", shifted, desc, (w, h), 0))
        lightglue(lg, np, ia, ib, 0.1)                                         # warm-up
        t = time.time()
        found = lightglue(lg, np, ia, ib, 0.1)
        dt = (time.time() - t) / parallel
        if lg.provider != provider:
            raise RuntimeError("fell back to the CPU")
        print(f"BENCH {provider} match {len(xy)} {dt:.3f} {len(found)}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"BENCH {provider} match FAILED {type(exc).__name__}: {str(exc)[:160]}", flush=True)


def cmd_bench(args, np, ort, Image):
    """The GPU (DirectML) at each of --gpu-sizes and the CPU at --cpu-size, each with one
    LightGlue pair. Lines: BENCH <provider> extract <size> <seconds> <points> | FAILED ...,
    BENCH <provider> match <points> <seconds> <matches> | FAILED ..., BENCH done."""
    print(f"BENCH onnxruntime {ort.__version__} {','.join(ort.get_available_providers())}", flush=True)
    if args.device == "auto":
        bench_device(args, np, ort, Image, "auto", [int(s) for s in args.gpu_sizes.split(",") if s.strip()])
    if args.cpu_size > 0:
        bench_device(args, np, ort, Image, "cpu", [args.cpu_size])
    print("BENCH done", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("extract", "match", "bench"):
        p = sub.add_parser(name)
        p.add_argument("--site", required=True, help="folder with onnxruntime and Pillow")
        p.add_argument("--device", default="auto", choices=("auto", "cpu"))
        if name == "extract":
            p.add_argument("--model", required=True)
            p.add_argument("--images", required=True)
            p.add_argument("--list", required=True)
            p.add_argument("--out", required=True)
            p.add_argument("--max-size", type=int, default=1600)
            p.add_argument("--max-points", type=int, default=2048)
            p.add_argument("--min-score", type=float, default=0.2)
        elif name == "match":
            p.add_argument("--model", required=True)
            p.add_argument("--db", required=True)
            p.add_argument("--pairs", required=True)
            p.add_argument("--min-score", type=float, default=0.1)
        else:
            p.add_argument("--aliked", required=True)
            p.add_argument("--lightglue", required=True)
            p.add_argument("--image", default="")
            p.add_argument("--gpu-sizes", default="1280,1600,2048,2400")
            p.add_argument("--cpu-size", type=int, default=1600)
            p.add_argument("--cpu-matchers", type=int, default=1)
            p.add_argument("--max-points", type=int, default=2048)
    args = ap.parse_args(argv)
    # appended, not prepended: Blender's own numpy must win
    if args.site not in sys.path:
        sys.path.append(args.site)
    import numpy as np
    import onnxruntime as ort
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    return {"extract": cmd_extract, "match": cmd_match, "bench": cmd_bench}[args.cmd](args, np, ort, Image)


if __name__ == "__main__":
    sys.exit(main())
