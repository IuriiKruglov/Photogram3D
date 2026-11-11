# SPDX-License-Identifier: GPL-3.0-or-later
# Photogram3D - photos or video to a textured mesh with COLMAP + OpenMVS
# Copyright (C) 2026 Kruglov Iurii
# Authors: Kruglov Iurii, Anthropic Claude AI
# License: GNU General Public License (GPL) v3 or later, https://www.blender.org/about/license/
"""Predict a normal map for each photo with MoGe-2 (ONNX). Runs as its own process.

Started by Photogram3D with Blender's Python:
    python -s -E ai_normals.py --site DIR --model FILE --jobs JOBS.json [--device auto|cpu]

JOBS.json: {"tokens": 2500, "items": [{"image": "...jpg", "out": "...npz"}, ...]}
Each output .npz holds "normal" (H, W, 3, int8, value/127 = OpenCV camera-space
unit normal: x right, y down, z forward) and "mask" (H, W, uint8, 1 = valid).

Running in a separate process keeps onnxruntime and GPU drivers out of
Blender: a driver crash only ends this process, and cancelling just kills it.
"""

import argparse
import json
import math
import os
import sys
import time


def model_input_size(width, height, tokens):
    """Image size MoGe works at internally for ``tokens`` ViT patches (14 px)."""
    aspect = width / float(height)
    rows = max(1, round(math.sqrt(tokens / aspect)))
    cols = max(1, round(math.sqrt(tokens * aspect)))
    return cols * 14, rows * 14


def make_session(ort, model_path, device):
    available = ort.get_available_providers()
    attempts = []
    if device != "cpu" and "DmlExecutionProvider" in available:
        # high_performance + gpu picks the discrete NVIDIA GPU on laptops with
        # an integrated Radeon/Intel GPU; DirectML needs these session options.
        dml = ("DmlExecutionProvider", {"performance_preference": "high_performance",
                                        "device_filter": "gpu"})
        attempts.append(([dml, "CPUExecutionProvider"], True))
        attempts.append(([("DmlExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"], True))
    if device != "cpu" and "CUDAExecutionProvider" in available:
        attempts.append((["CUDAExecutionProvider", "CPUExecutionProvider"], False))
    attempts.append((["CPUExecutionProvider"], False))
    last = None
    for providers, is_dml in attempts:
        try:
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            if is_dml:
                opts.enable_mem_pattern = False
                opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            return ort.InferenceSession(model_path, sess_options=opts, providers=providers)
        except Exception as exc:  # noqa: BLE001 - try the next provider
            last = exc
            print(f"provider {providers[0] if isinstance(providers[0], str) else providers[0][0]} "
                  f"unavailable: {exc}", flush=True)
    raise RuntimeError(f"cannot create an inference session: {last}")


def load_rgb(Image, np, path, size):
    """RGB float32 [0, 1] at ``size``; antialiased here because the ONNX graph is not."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.size != size:
            im = im.resize(size, Image.Resampling.LANCZOS, reducing_gap=3.0)
        return np.asarray(im, dtype=np.float32) / 255.0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True, help="folder with onnxruntime and Pillow")
    ap.add_argument("--model", required=True)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--device", default="auto", choices=("auto", "cpu"))
    args = ap.parse_args(argv)

    # Appended, not prepended: Blender's own numpy must win.
    if args.site not in sys.path:
        sys.path.append(args.site)
    import numpy as np
    import onnxruntime as ort
    from PIL import Image

    with open(args.jobs, "r", encoding="utf-8") as f:
        jobs = json.load(f)
    tokens = int(jobs.get("tokens", 2500))
    items = jobs["items"]

    t0 = time.time()
    sess = make_session(ort, args.model, args.device)
    provider = sess.get_providers()[0]
    inputs = {i.name: i for i in sess.get_inputs()}
    outputs = [o.name for o in sess.get_outputs()]
    if "image" not in inputs or "normal" not in outputs:
        raise SystemExit(f"unexpected model signature: inputs {list(inputs)}, outputs {outputs}")
    want = ["normal"] + (["mask"] if "mask" in outputs else [])
    print(f"Photogram3D AI: onnxruntime {ort.__version__}, provider {provider}, "
          f"{tokens} tokens, model loaded in {time.time() - t0:.1f} s", flush=True)

    def infer(feed):
        nonlocal sess, provider
        try:
            return sess.run(want, feed)
        except Exception as exc:  # noqa: BLE001 - e.g. an operator the GPU path cannot run
            if provider == "CPUExecutionProvider":
                raise
            print(f"Photogram3D AI: {provider} failed ({exc}); continuing on the CPU", flush=True)
            sess = make_session(ort, args.model, "cpu")
            provider = sess.get_providers()[0]
            return sess.run(want, feed)

    failures = 0
    for k, item in enumerate(items, start=1):
        t1 = time.time()
        name = os.path.basename(item["image"])
        try:
            with Image.open(item["image"]) as probe:
                w0, h0 = probe.size
            size = model_input_size(w0, h0, tokens)
            rgb = load_rgb(Image, np, item["image"], size)
            feed = {"image": np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])}
            if "num_tokens" in inputs:
                feed["num_tokens"] = np.array(tokens, dtype=np.int64)
            res = infer(feed)
            normal = np.asarray(res[0], dtype=np.float32)
            if normal.ndim == 4:
                normal = normal[0]
            if normal.shape[0] == 3 and normal.shape[-1] != 3:  # channel-first exports
                normal = normal.transpose(1, 2, 0)
            norm = np.linalg.norm(normal, axis=-1, keepdims=True)
            normal = normal / np.maximum(norm, 1e-6)
            if len(res) > 1:
                mask = np.asarray(res[1])
                mask = mask.reshape(mask.shape[-2:]) > 0.5
            else:
                mask = np.ones(normal.shape[:2], dtype=bool)
            mask &= norm[..., 0] > 0.5
            tmp = item["out"] + ".tmp.npz"
            np.savez_compressed(tmp, normal=np.round(normal * 127.0).astype(np.int8),
                                mask=mask.astype(np.uint8),
                                source_size=np.array([w0, h0], dtype=np.int32))
            os.replace(tmp, item["out"])
            print(f"Normals {k}/{len(items)} {name} {time.time() - t1:.2f} s", flush=True)
        except Exception as exc:  # noqa: BLE001 - skip unreadable photos, keep going
            failures += 1
            print(f"Normals {k}/{len(items)} {name} FAILED: {exc}", flush=True)
    print(f"Photogram3D AI: {len(items) - failures}/{len(items)} normal maps in "
          f"{time.time() - t0:.1f} s ({provider})", flush=True)
    if failures == len(items):
        sys.exit(2)


if __name__ == "__main__":
    main()
