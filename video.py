# SPDX-License-Identifier: GPL-3.0-or-later
# Photogram3D - photos or video to a textured mesh with COLMAP + OpenMVS
# Copyright (C) 2026 Kruglov Iurii
# Authors: Kruglov Iurii, Anthropic Claude AI
# License: GNU General Public License (GPL) v3 or later, https://www.blender.org/about/license/
"""Video -> frames, chosen one of three ways:

* Auto (0.3.8, the default): by the camera's movement. One FFmpeg pass measures the blur of
  every frame (blurdetect) and writes a small grey copy of it; phase correlation between
  frames 0.2 s apart, on nine overlapping parts of the picture (0.3.9), gives how fast the
  picture moves. The frames are placed evenly along
  that movement, one per AUTO_STEP of the frame, at least AUTO_MIN_RATE per second while the
  camera hardly moves, between AUTO_MIN_FRAMES and the Max Frames setting in all. At each
  place the sharpest well-lit frame nearby is taken; when every frame there is blurred, a
  sharp one up to almost the next place is looked for.
* Per second: the sharpest frame of each 1/fps-second window.
* Every frame of a time range.

0.3.3 and earlier wrote the chosen frames with one select expression listing every frame
("eq(n,12)+eq(n,57)+..."). It grows with the frame count: past roughly 2,700 frames it no
longer fits Windows' 32,767-character command line and FFmpeg failed to start. Frames are
now written in groups of FRAMES_PER_RUN, each from its own seek point, so every command
stays short whatever the frame count. (FFmpeg's expressions have no "||" operator: it
rejects "eq(n,1)||eq(n,3)" with "Invalid chars".)

Frame files are named after their frame number in the video (frame_000123.jpg is frame
123, counted from 0), and frames.json next to them records each file's time, which the
tracked camera uses.
"""

import json
import math
import os
import re
import subprocess

from .common import popen_flags
from .runner import StepError

HDR_TRANSFERS = {"arib-std-b67", "smpte2084"}
FRAMES_JSON = "frames.json"
FRAMES_PER_RUN = 100
MIN_FRAMES = 20

# Auto (0.3.8). AUTO_STEP: the picture moves by this share of the frame between two kept
# frames, about 8 degrees of turning for a phone's main camera filming in landscape (88%
# overlap; photo sets work with 70-80%, video frames are softer). Close-ups move the picture
# less per degree, so they get frames more often, which they need.
AUTO_STEP = 0.12
# AUTO_MIN_RATE: frames per second at least, while the camera hardly moves. 0.3.8 had 0.5:
# four of five real walks round vehicles ended at that floor (30-37 frames for 35-72 s) and
# each folded or lost frames in the camera solve; the one with 2 frames per second did not.
AUTO_MIN_RATE = 1.0
AUTO_MIN_FRAMES = 30
AUTO_MAX_FRAMES = 300        # the default of the Max Frames setting
MOTION_SIDE = 128            # long side of the grey copies the movement is measured on
MOTION_GAP = 0.2             # seconds between the two frames of each measurement
BLUR_BAD = 1.3               # blur this much above the sharpest frames within 1 s: blurred
NEAR_WEIGHT = 0.3            # cost of moving a whole spacing away from a frame's place
DARK, BRIGHT = 0.08, 0.92    # mean grey level of a badly exposed frame


def probe(ffprobe, video):
    cmd = [ffprobe, "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration,"
           "start_time,color_transfer,codec_name:stream_side_data=rotation:format=duration,"
           "start_time",
           "-of", "json", video]
    try:
        res = subprocess.run(cmd, capture_output=True, timeout=120, creationflags=popen_flags())
    except (OSError, subprocess.SubprocessError) as exc:
        raise StepError(f"ffprobe failed: {exc}") from exc
    if res.returncode != 0:
        raise StepError("FFmpeg cannot read this video: "
                        + (res.stderr.decode("utf-8", "replace").strip()[-300:] or "unknown format"))
    data = json.loads(res.stdout.decode("utf-8", "replace") or "{}")
    streams = data.get("streams") or []
    if not streams:
        raise StepError("The file has no video stream")
    s = streams[0]
    fmt = data.get("format") or {}
    fps = _rate(s.get("avg_frame_rate")) or _rate(s.get("r_frame_rate")) or 30.0
    # Blender reads a movie at FFmpeg's guess of its frame rate (av_guess_frame_rate: the
    # stream's r_frame_rate, the average one when that is implausibly high) and counts its
    # frames from the video stream's first timestamp
    r_fps, avg = _rate(s.get("r_frame_rate")), _rate(s.get("avg_frame_rate"))
    clip_fps = r_fps or fps
    if r_fps and avg and r_fps > 210 and avg < 70:
        clip_fps = avg
    container_start = _float(fmt.get("start_time")) or 0.0
    stream_start = _float(s.get("start_time"))
    stream_start = max(0.0, stream_start - container_start) if stream_start is not None else 0.0
    duration = _float(s.get("duration")) or _float(fmt.get("duration")) or 0.0
    rotation = 0
    for sd in s.get("side_data_list") or []:
        if "rotation" in sd:
            rotation = int(_float(sd["rotation"]) or 0)
    width, height = int(s.get("width") or 0), int(s.get("height") or 0)
    return {
        "width": width, "height": height,
        # the frames FFmpeg writes are turned upright
        "display_width": height if rotation % 180 else width,
        "display_height": width if rotation % 180 else height,
        "fps": fps, "duration": duration, "nb_frames": int(s.get("nb_frames") or 0),
        # times in this module are FFmpeg's -ss times: from the container's start; the video
        # stream can start later (an audio track first, an edit list)
        "start_time": container_start, "stream_start": stream_start, "clip_fps": clip_fps,
        "transfer": s.get("color_transfer") or "", "codec": s.get("codec_name") or "",
        "rotation": rotation,
    }


def _rate(text):
    if not text or text in ("0/0",):
        return None
    if "/" in text:
        a, b = text.split("/", 1)
        try:
            return float(a) / float(b) if float(b) else None
        except ValueError:
            return None
    return _float(text)


def _float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def has_filter(ffmpeg, name):
    try:
        res = subprocess.run([ffmpeg, "-hide_banner", "-filters"], capture_output=True,
                             timeout=60, creationflags=popen_flags())
    except (OSError, subprocess.SubprocessError):
        return False
    return re.search(rb"\s" + name.encode() + rb"\s", res.stdout) is not None


_FRAME_RE = re.compile(r"frame:\s*(\d+)\s+pts:\s*\S+\s+pts_time:\s*([-\d.eE+]+)")
_BLUR_RE = re.compile(r"lavfi\.blur=([-\d.eE+]+)")
_SHOWINFO_RE = re.compile(r"Parsed_showinfo_\d+ @ [^\]]*\]\s*n:\s*(\d+)\s+pts:\s*(-?\d+)\s+"
                          r"pts_time:\s*([-\d.eE+]+)")


def parse_blur_file(path, keep_missing=False):
    """Parse metadata=print output into [(frame_index, time, blur)] (blur None for a frame
    without a value, kept only with ``keep_missing``, so the list stays one entry per frame)."""
    out = []
    cur = None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _FRAME_RE.search(line)
            if m:
                cur = [int(m.group(1)), float(m.group(2)), None]
                out.append(cur)
                continue
            m = _BLUR_RE.search(line)
            if m and cur is not None:
                cur[2] = float(m.group(1))
    return [tuple(x) for x in out if keep_missing or x[2] is not None]


def media_end(info):
    """When the video's last frame ends, from the container's start: the video stream can
    start later than the container (an audio track first)."""
    duration = info.get("duration") or 0.0
    return (info.get("stream_start") or 0.0) + duration if duration else 0.0


def time_range(info, start, end):
    """(start, end) in seconds from the start of the video; end 0 = to the end."""
    duration = media_end(info)
    start = max(0.0, float(start or 0.0))
    end = float(end or 0.0)
    if end <= 0 or (duration and end > duration):
        end = duration
    if duration and start >= duration:
        raise StepError(f"The start time ({start:.2f} s) is past the end of the video "
                        f"({duration:.2f} s)")
    if end and end <= start:
        raise StepError(f"The end time ({end:.2f} s) must come after the start time ({start:.2f} s)")
    return start, end


def _say(job, log_path, text):
    """A line for the panel that also stays in the frames log (the Auto choice: 0.3.9)."""
    job.add_line(text)
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[Photogram3D] {text}\n")
    except (OSError, TypeError):
        pass


def frame_number(t, fps, stream_start=0.0):
    """The frame number (from 0) of a frame shown at time t (from the container's start):
    counted from the video stream's first frame, as Blender counts a movie clip's frames."""
    return int(math.floor((t - stream_start) * fps + 0.5))


def pick(samples, start, end, keep_fps, all_frames=False, min_frames=MIN_FRAMES):
    """Frames to keep from ``samples`` [(time, blur or None)] (sorted by time).

    All of them, or the sharpest (lowest blur) of each 1/keep_fps-second window of the
    range; the middle one when there is no blur measure. At least min_frames windows."""
    if not samples:
        return []
    if all_frames:
        return list(samples)
    span = max(1e-6, (end or samples[-1][0]) - start)
    count = max(min_frames, int(math.ceil(span * keep_fps - 1e-9)))
    count = min(count, len(samples))
    window = span / count
    buckets = {}
    for t, blur in samples:
        # a frame exactly on a window's start belongs to it (t / window can come out as
        # 41.999999999 for frame 42 of a 250 fps video kept at 250/s)
        k = min(count - 1, max(0, int(math.floor((t - start) / window + 1e-6))))
        cur = buckets.get(k)
        if blur is not None:
            if cur is None or blur < cur[1]:
                buckets[k] = (t, blur)
        else:
            mid = start + (k + 0.5) * window
            if cur is None or abs(t - mid) < abs(cur[0] - mid):
                buckets[k] = (t, blur)
    return [buckets[k] for k in sorted(buckets)]


def select_frames(samples, duration, keep_fps, max_frames=None, min_frames=MIN_FRAMES):
    """0.3.3 interface (frame indices from parse_blur_file samples), for the tests."""
    picked = pick([(t, b) for _, t, b in samples], 0.0, duration, keep_fps, False, min_frames)
    by_time = {t: n for n, t, _ in samples}
    return sorted(by_time[t] for t, _ in picked)


def tonemap_filter(info):
    if info.get("transfer") in HDR_TRANSFERS:
        return ("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
                "tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p")
    return None


# Reading stops at the first packet past the end; with B-frames, packets come in decoding
# order and the last frames of a range can follow it, so ranges are read this much longer.
READ_PAST = 0.5


def _seek_args(start, end):
    args = []
    if start > 0:
        args += ["-ss", f"{start:.6f}"]
    if end and end > start:
        args += ["-t", f"{end - start + READ_PAST:.6f}"]
    return args


BLUR_FILE = "frames_blur.txt"
MOTION_FILE = "frames_motion.gray"


def motion_size(info):
    """(width, height) of the grey copies the movement is measured on: MOTION_SIDE on the
    long side of the upright picture, even sizes."""
    w, h = info.get("display_width") or 16, info.get("display_height") or 9
    if w >= h:
        return MOTION_SIDE, max(8, int(round(MOTION_SIDE * h / w / 2)) * 2)
    return max(8, int(round(MOTION_SIDE * w / h / 2)) * 2), MOTION_SIDE


def scan(job, ffmpeg, video, work_dir, start, end, expected, log_path, info=None, blur=True,
         motion=False, progress=(0.0, 0.6)):
    """One decoding pass over the range. Returns [(time, blur or None, k)] for every frame,
    in time order: its blur (blurdetect on a 640 px copy) and, with ``motion``, the index k
    of its grey copy (motion_size) in work_dir/MOTION_FILE."""
    for name in (BLUR_FILE, MOTION_FILE):
        if os.path.exists(os.path.join(work_dir, name)):
            os.remove(os.path.join(work_dir, name))
    lo, hi = progress
    shown = []
    what = "measuring sharpness and movement" if blur and motion else (
        "measuring sharpness" if blur else "measuring movement")

    def scan_parser(line):
        m = _SHOWINFO_RE.search(line)
        if m:
            shown.append(float(m.group(3)))
            return None
        m = re.search(r"frame=\s*(\d+)", line)
        if m:
            return lo + (hi - lo) * min(1.0, int(m.group(1)) / max(1, expected)), what
        return None

    # relative file names avoid filtergraph escaping of Windows paths
    blur_chain = ("scale=640:-2,format=gray,blurdetect=block_width=32:block_height=32,"
                  f"metadata=mode=print:file={BLUR_FILE}")
    mw, mh = motion_size(info or {})
    motion_chain = f"scale={mw}:{mh}:flags=area,format=gray"
    args = [ffmpeg, "-hide_banner", "-nostdin", "-y"] + _seek_args(start, end) + ["-i", video]
    if blur and motion:
        # both copies come from the same decoded frames, in the same order
        args += ["-filter_complex", f"[0:v:0]split=2[a][b];[a]{blur_chain}[a1];[b]{motion_chain}[b1]",
                 "-map", "[a1]", "-an", "-f", "null", "-",
                 "-map", "[b1]", "-an", "-f", "rawvideo", "-pix_fmt", "gray", MOTION_FILE]
    elif blur:
        args += ["-map", "0:v:0", "-vf", blur_chain, "-an", "-f", "null", "-"]
    else:
        args += ["-map", "0:v:0", "-vf", motion_chain + ",showinfo=checksum=0", "-an",
                 "-f", "rawvideo", "-pix_fmt", "gray", MOTION_FILE]
    job.run(args, log_path, cwd=work_dir, parser=scan_parser)
    if blur:
        rows = [(t, b) for _, t, b in parse_blur_file(os.path.join(work_dir, BLUR_FILE),
                                                     keep_missing=motion)]
    else:
        rows = [(t, None) for t in shown]
    # times from the seek point: make them times in the video
    eps = 1e-4
    out = [(start + t, b, k) for k, (t, b) in enumerate(rows) if not end or start + t < end - eps]
    out.sort(key=lambda r: (r[0], r[2]))
    return out


def scan_blur(job, ffmpeg, video, work_dir, start, end, expected, log_path, progress=(0.0, 0.6)):
    """[(time, blur)] for every frame of the range: blurdetect on a 640 px copy."""
    rows = scan(job, ffmpeg, video, work_dir, start, end, expected, log_path, progress=progress)
    return [(t, b) for t, b, _ in rows if b is not None]


def _phase_shift(fa, fb, w, h):
    """(dx, dy, peak) of picture b against picture a from their spectra (phase correlation,
    subpixel by a parabola through the peak)."""
    import numpy as np
    r = fa * np.conj(fb)
    r /= np.abs(r) + 1e-9
    c = np.fft.irfft2(r, s=(h, w))
    k = int(np.argmax(c))
    py, px = divmod(k, w)
    peak = float(c[py, px])

    def sub(m, z, p):
        den = m - 2.0 * z + p
        return 0.0 if abs(den) < 1e-12 else 0.5 * (m - p) / den
    dy = py + sub(c[(py - 1) % h, px], peak, c[(py + 1) % h, px])
    dx = px + sub(c[py, (px - 1) % w], peak, c[py, (px + 1) % w])
    if dy > h / 2:
        dy -= h
    if dx > w / 2:
        dx -= w
    return dx, dy, peak


def motion_tiles(w, h):
    """(x, y, width, height) of the parts of a grey copy the movement is measured on: half the
    picture each way, at 3 x 3 places a quarter of it apart (0.3.9)."""
    bw, bh = w // 2, h // 2
    return [(x * w // 4, y * h // 4, bw, bh) for y in range(3) for x in range(3)]


def camera_motion(path, size, keys, times, gap=MOTION_GAP):
    """How fast the picture moves at each frame, as a share of the frame per second, and
    each frame's mean grey level (0..1). ``keys``: the frames' indices in the grey copies
    (MOTION_FILE), ``times`` their times, in time order. Each speed comes from the shifts
    between the frame and the one ``gap`` seconds before it, by phase correlation in each of
    the motion_tiles(): the 75th percentile of the tiles the correlation is sure of, or the
    whole picture's shift when fewer than 3 are, and the whole picture's when it is larger (a
    part's shift wraps round past half its size). A whole-picture shift follows what fills
    most of the frame: walking round an object in the middle of the picture it measured about
    two thirds of the turn (0.64 on the rendered walk), the tiles 0.91, and on a pan both are
    exact (0.3.9). A frame the correlation is unsure of takes its neighbours' speed."""
    import numpy as np
    w, h = size
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    count = raw.size // (w * h)
    frames = raw[:count * w * h].reshape(count, h, w)
    n = len(keys)
    t = np.asarray(times, dtype=float)
    speed = np.zeros(n)
    light = np.zeros(n)
    unsure = np.zeros(n, dtype=bool)
    window = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    tiles = motion_tiles(w, h)
    tile_window = np.outer(np.hanning(tiles[0][3]), np.hanning(tiles[0][2])).astype(np.float32)

    def spectra_of(g):
        out = [np.fft.rfft2((g - g.mean()) * window)]
        for x0, y0, bw, bh in tiles:
            a = g[y0:y0 + bh, x0:x0 + bw]
            out.append(np.fft.rfft2((a - a.mean()) * tile_window))
        return out

    spectra = {}
    j = 0
    for i, k in enumerate(keys):
        g = frames[min(k, count - 1)].astype(np.float32)
        light[i] = float(g.mean()) / 255.0
        spectra[i] = spectra_of(g)
        while j + 1 < i and t[i] - t[j + 1] >= gap - 1e-6:
            spectra.pop(j, None)
            j += 1
        if i > j and t[i] - t[j] > 1e-6:
            dt = t[i] - t[j]
            sure_tiles = []
            for (x0, y0, bw, bh), a, b in zip(tiles, spectra[j][1:], spectra[i][1:]):
                dx, dy, peak = _phase_shift(a, b, bw, bh)
                if peak >= 0.03:
                    sure_tiles.append(math.hypot(dx / w, dy / h) / dt)
            dx, dy, peak = _phase_shift(spectra[j][0], spectra[i][0], w, h)
            whole = math.hypot(dx / w, dy / h) / dt
            if len(sure_tiles) >= 3:
                # a part resolves shifts up to half its size (a quarter of the frame per gap):
                # beyond that its shift wraps round and looks small, while the whole picture's
                # does not, so the larger of the two is taken
                speed[i] = max(float(np.percentile(sure_tiles, 75)), whole if peak >= 0.03 else 0.0)
            else:
                speed[i] = whole
                unsure[i] = peak < 0.03
        else:
            unsure[i] = True
    sure = ~unsure
    if sure.any():
        for i in np.nonzero(unsure)[0]:
            near = sure & (np.abs(t - t[i]) <= 1.0)
            speed[i] = float(np.median(speed[near])) if near.any() else float(np.median(speed[sure]))
    del frames, raw
    return speed, light


def blur_quality(times, blur, light):
    """Each frame's quality cost: its blur against the sharpest frames within 1 s (1 = as
    sharp as they are; BLUR_BAD and above = blurred), plus 3 when it is too dark or too
    bright. Without a blur measure only the exposure counts."""
    import numpy as np
    t = np.asarray(times, dtype=float)
    q = np.ones(len(t))
    values = [b for b in blur if b is not None] if blur is not None else []
    if values:
        b = np.asarray([v if v is not None else np.nan for v in blur], dtype=float)
        b = np.where(np.isnan(b), float(np.median(values)), b)
        lo = np.searchsorted(t, t - 1.0, side="left")
        hi = np.searchsorted(t, t + 1.0, side="right")
        ref = np.array([np.percentile(b[a:z], 20) for a, z in zip(lo, hi)])
        q = b / np.maximum(ref, 1e-6)
    light = np.asarray(light, dtype=float)
    return q + np.where((light < DARK) | (light > BRIGHT), 3.0, 0.0)


def pick_auto(times, quality, speed, max_frames=AUTO_MAX_FRAMES, min_frames=AUTO_MIN_FRAMES,
              step=AUTO_STEP, min_rate=AUTO_MIN_RATE):
    """Indices of the frames to keep, by the camera's movement, and what happened.

    Frames are placed at even steps of "progress": one per ``step`` of movement (a share of
    the frame), or one per 1/min_rate seconds when that comes sooner, scaled so that there
    are min_frames to max_frames places. At each place the frame with the lowest quality
    cost (blur_quality) plus NEAR_WEIGHT per spacing away is kept; when that frame is
    blurred, the sharpest frame up to 0.9 of a spacing away is taken instead, if sharper."""
    import numpy as np
    t = np.asarray(times, dtype=float)
    q = np.asarray(quality, dtype=float)
    v = np.asarray(speed, dtype=float)
    n = len(t)
    if n == 0:
        return [], {"count": 0}
    dts = np.diff(t, prepend=t[0])
    progress = np.cumsum(np.maximum(v / step, min_rate) * dts)
    total = float(progress[-1])
    natural = int(round(total))
    count = min(n, max(min(natural, max_frames), min_frames))
    progress = progress * (count / total) if total > 0 else np.linspace(0.0, count, n, endpoint=False)
    order = np.arange(n)
    picks, replaced, blurred = [], 0, 0
    last = -1
    for k in range(count):
        target = k + 0.5
        cand = order[(progress >= target - 0.5) & (progress < target + 0.5) & (order > last)]
        if not len(cand):
            continue
        dist = np.abs(progress[cand] - target)
        best = int(cand[np.argmin(q[cand] + NEAR_WEIGHT * dist)])
        nearest = int(cand[np.argmin(dist)])
        if q[best] > BLUR_BAD:
            wide = order[(progress >= target - 0.9) & (progress < target + 0.9) & (order > last)]
            alt = int(wide[np.argmin(q[wide])])
            if q[alt] < q[best]:
                best = alt
        if q[best] > BLUR_BAD:
            blurred += 1
        elif q[nearest] > BLUR_BAD:
            replaced += 1
        picks.append(best)
        last = best
    # A place with no frame near it is skipped: the camera moved further than a spacing between
    # two frames (a short or low-rate video). Up to min_frames, the frames left out are then
    # added where the movement between kept frames is largest, the best one near the middle.
    filled = 0
    edges = np.concatenate([[0.0], progress, [float(progress[-1])]])
    while len(picks) < min(n, min_frames):
        bounds = [-1] + sorted(picks) + [n]
        gaps = [(edges[b + 1] - edges[a + 1], a, b) for a, b in zip(bounds, bounds[1:]) if b - a >= 2]
        if not gaps:
            break
        _, a, b = max(gaps)
        cand = np.arange(a + 1, b)
        mid = (edges[a + 1] + edges[b + 1]) / 2
        picks.append(int(cand[np.argmin(q[cand] + NEAR_WEIGHT * np.abs(progress[cand] - mid))]))
        filled += 1
    picks.sort()
    moved = float(np.sum(v * dts))
    return picks, {"count": len(picks), "natural": natural, "moved": moved, "span": float(t[-1] - t[0]),
                   "step": moved / max(1, len(picks)), "replaced": replaced, "blurred": blurred,
                   "filled": filled}


def describe_auto(report, max_frames):
    """Log lines for an Auto pick."""
    lines = [f"Auto: the picture moved by about {report['moved']:.1f} frames in {report['span']:.1f} s; "
             f"keeping {report['count']} frames, about one per {100 * report['step']:.0f}% of the frame"]
    if report["natural"] > max_frames:
        lines.append(f"Max Frames ({max_frames}) spreads them further apart than the usual "
                     f"{100 * AUTO_STEP:.0f}% of the frame")
    elif report["natural"] < AUTO_MIN_FRAMES:
        lines.append(f"Little movement: at least {AUTO_MIN_FRAMES} frames are kept")
    if report.get("filled"):
        lines.append(f"{report['filled']} frame(s) added where the picture moved further than a spacing "
                     f"between two frames, to keep at least {AUTO_MIN_FRAMES}")
    if report["replaced"]:
        lines.append(f"{report['replaced']} blurred or badly exposed frame(s) replaced by a sharper one nearby")
    if report["blurred"]:
        lines.append(f"{report['blurred']} place(s) had only blurred or badly exposed frames; the best "
                     "of them were kept")
    return lines


def scan_times(ffprobe, video, info, start, end):
    """[(time, None)] for every frame of the range from the packet timestamps (no decoding)."""
    interval = f"{start:.6f}%" + (f"{end + READ_PAST:.6f}" if end else "")
    cmd = [ffprobe, "-v", "error", "-select_streams", "v:0", "-read_intervals", interval,
           "-show_entries", "packet=pts_time", "-of", "csv=p=0", video]
    times = []
    try:
        res = subprocess.run(cmd, capture_output=True, timeout=600, creationflags=popen_flags())
        for line in res.stdout.decode("ascii", "replace").splitlines():
            t = _float(line.strip().strip(","))
            if t is not None:
                times.append(t - info.get("start_time", 0.0))
    except (OSError, subprocess.SubprocessError):
        times = []
    eps = 0.25 / max(info["fps"], 1.0)
    times = sorted(t for t in set(times) if t >= start - eps and (not end or t < end - eps))
    if not times:
        # no timestamps (raw streams): assume a constant frame rate
        last = end or info.get("duration") or 0.0
        n0 = int(math.ceil(start * info["fps"] - 1e-9))
        n1 = int(math.ceil(last * info["fps"] - 1e-9))
        times = [n / info["fps"] for n in range(n0, n1)]
    return [(t, None) for t in times]


def _min_interval(samples, fps):
    gaps = [b[0] - a[0] for a, b in zip(samples, samples[1:]) if b[0] > a[0]]
    return min(gaps) if gaps else 1.0 / max(fps, 1.0)


def scale_filter(info, max_side):
    """FFmpeg scale filter (Lanczos) to ``max_side`` on the long side, or "" when the video
    is not larger. Sizes are the upright (display) ones: FFmpeg turns phone videos first."""
    w, h = info.get("display_width") or 0, info.get("display_height") or 0
    if not max_side or not w or not h or max(w, h) <= max_side:
        return ""
    k = max_side / max(w, h)
    return f"scale={max(2, round(w * k / 2) * 2)}:{max(2, round(h * k / 2) * 2)}:flags=lanczos"


def write_frames(job, ffmpeg, video, out_dir, picks, info, log_path, every=False,
                 frame_interval=None, progress=(0.6, 1.0), per_run=FRAMES_PER_RUN, max_side=0):
    """Write the frames at times ``picks`` [(time, blur)] as frame_NNNNNN.jpg; ``every``:
    they are all the frames of their time range. ``frame_interval``: the shortest time
    between two frames of the video (the matching tolerance is 40% of it). ``max_side``:
    frames larger than this are scaled down (Lanczos).

    Returns [(file name, frame number, time)] in time order."""
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):
        if ((f.startswith("frame_") or f.startswith("part_")) and f.lower().endswith(".jpg")) \
                or f == FRAMES_JSON:
            os.remove(os.path.join(out_dir, f))
    used = set()
    fps = info["fps"]
    tol = 0.4 * (frame_interval or 1.0 / max(fps, 1.0))
    tm = tonemap_filter(info)
    scale = scale_filter(info, max_side)
    lo, hi = progress
    total = len(picks)
    done = {"n": 0}
    written = []
    times = [t for t, _ in picks]
    # every frame of the range: one run without a select expression
    groups = [times] if every else [times[i:i + per_run] for i in range(0, total, per_run)]
    for group in groups:
        seek = max(0.0, group[0] - 2 * tol)
        stop = group[-1] + max(2 * tol, READ_PAST)
        shown = []

        def parser(line):
            m = _SHOWINFO_RE.search(line)
            if m:
                shown.append(float(m.group(3)))
                return lo + (hi - lo) * min(1.0, (done["n"] + len(shown)) / max(1, total)), "writing frames"
            return None

        chain = []
        if not every:
            terms = "+".join(f"between(t,{t - seek - tol:.6f},{t - seek + tol:.6f})" for t in group)
            chain.append(f"select='{terms}'")
        chain.append("showinfo=checksum=0")
        if scale:
            chain.append(scale)
        if tm:
            chain.append(tm)
        job.run([ffmpeg, "-hide_banner", "-nostdin", "-y", "-ss", f"{seek:.6f}",
                 "-t", f"{stop - seek:.6f}", "-i", video, "-map", "0:v:0",
                 "-vf", ",".join(chain), "-fps_mode", "passthrough", "-q:v", "2", "-an",
                 "-start_number", "1", os.path.join(out_dir, "part_%06d.jpg")],
                log_path, cwd=out_dir, parser=parser)
        # name each file after its frame; a file that is not one of the group's is dropped
        want = list(group)
        for k, t_rel in enumerate(shown, 1):
            part = os.path.join(out_dir, f"part_{k:06d}.jpg")
            if not os.path.isfile(part):
                continue
            t = seek + t_rel
            best = min(range(len(want)), key=lambda j: abs(want[j] - t)) if want else None
            if best is None or abs(want[best] - t) > tol * 1.5:
                os.remove(part)
                continue
            t = want.pop(best)
            n = frame_number(t, info.get("clip_fps") or fps, info.get("stream_start", 0.0))
            while n in used:          # a variable frame rate can round two frames together
                n += 1
            used.add(n)
            name = f"frame_{n:06d}.jpg"
            os.replace(part, os.path.join(out_dir, name))
            written.append((name, n, t))
        for f in os.listdir(out_dir):
            if f.startswith("part_"):
                os.remove(os.path.join(out_dir, f))
        done["n"] += len(group)
        if want:
            _say(job, log_path, f"{len(want)} frame(s) near {want[0]:.2f} s could not be read and are left out")
    written.sort(key=lambda w: w[2])
    return written


def write_range(job, ffmpeg, video, out_dir, info, start, end, log_path, size=None, progress=(0.0, 1.0),
                quality=3):
    """Every frame from ``start`` to ``end`` (seconds from the container's start, as -ss counts;
    end 0: to the end) as frame_NNNNNN.jpg, named like write_frames names them, at ``size``
    (w, h) when that is not the video's own upright size (the same Lanczos filter as the
    reconstruction's frames, so its frames come out again pixel for pixel). For the camera
    track (camera_track.py). Returns [(file name, frame number, time)] in time order."""
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):
        if f.startswith(("frame_", "part_")) and f.lower().endswith(".jpg"):
            os.remove(os.path.join(out_dir, f))
    fps = info.get("clip_fps") or info["fps"]
    stream_start = info.get("stream_start", 0.0)
    end = end or media_end(info)
    expected = max(1, int((end - start) * info["fps"])) if end else 1
    lo, hi = progress
    shown = []

    def parser(line):
        m = _SHOWINFO_RE.search(line)
        if m:
            shown.append(float(m.group(3)))
            return lo + (hi - lo) * min(1.0, len(shown) / expected), f"writing frame {len(shown):,}/{expected:,}"
        return None

    chain = ["showinfo=checksum=0"]
    w, h = info.get("display_width") or 0, info.get("display_height") or 0
    if size and tuple(size) != (w, h):
        chain.append(f"scale={int(size[0])}:{int(size[1])}:flags=lanczos")
    tm = tonemap_filter(info)
    if tm:
        chain.append(tm)
    job.run([ffmpeg, "-hide_banner", "-nostdin", "-y"] + _seek_args(start, end) +
            ["-i", video, "-map", "0:v:0", "-vf", ",".join(chain), "-fps_mode", "passthrough",
             "-q:v", str(quality), "-an", "-start_number", "1", os.path.join(out_dir, "part_%06d.jpg")],
            log_path, cwd=out_dir, parser=parser)
    written, used = [], set()
    eps = 1e-4
    for k, t_rel in enumerate(shown, 1):
        part = os.path.join(out_dir, f"part_{k:06d}.jpg")
        if not os.path.isfile(part):
            continue
        t = start + t_rel
        if t < start - eps or (end and t >= end - eps):
            os.remove(part)
            continue
        n = frame_number(t, fps, stream_start)
        while n in used:
            n += 1
        used.add(n)
        name = f"frame_{n:06d}.jpg"
        os.replace(part, os.path.join(out_dir, name))
        written.append((name, n, t))
    for f in os.listdir(out_dir):
        if f.startswith("part_"):
            os.remove(os.path.join(out_dir, f))
    written.sort(key=lambda w_: w_[2])
    return written


def frames_mode(cfg):
    """AUTO, RATE or ALL; settings from before 0.3.8 have only video_all_frames."""
    mode = cfg.get("video_frames")
    if mode in ("AUTO", "RATE", "ALL"):
        return mode
    return "ALL" if cfg.get("video_all_frames") else "RATE"


def _pick_auto(job, cfg, ffmpeg, ffprobe, video, work_dir, info, start, end, expected, log_path):
    """Auto: (picks [(time, blur)], samples [(time, blur)] of every frame of the range)."""
    max_frames = max(AUTO_MIN_FRAMES, int(cfg.get("video_max_frames") or AUTO_MAX_FRAMES))
    use_blur = bool(cfg.get("video_blur_filter", True)) and has_filter(ffmpeg, "blurdetect")
    rows = scan(job, ffmpeg, video, work_dir, start, end, expected, log_path, info=info,
                blur=use_blur, motion=True)
    path = os.path.join(work_dir, MOTION_FILE)
    try:
        times = [r[0] for r in rows]
        if len(rows) < 2 or not os.path.isfile(path):
            raise ValueError("no frames were measured")
        speed, light = camera_motion(path, motion_size(info), [r[2] for r in rows], times)
    except (OSError, ValueError, MemoryError) as exc:
        samples = [(r[0], r[1]) for r in rows] or scan_times(ffprobe, video, info, start, end)
        if not samples:
            raise StepError("No frames found in the chosen part of the video") from exc
        span = max(1e-6, (end or samples[-1][0]) - start)
        keep_fps = min(2.0, max_frames / span)
        picks = pick(samples, start, end or samples[-1][0], keep_fps, False, AUTO_MIN_FRAMES)
        _say(job, log_path, f"The camera's movement could not be measured ({exc}); {len(picks)} "
             "frames picked evenly instead")
        return picks, samples
    finally:
        if os.path.exists(path):
            os.remove(path)
    quality = blur_quality(times, [r[1] for r in rows] if use_blur else None, light)
    keep, report = pick_auto(times, quality, speed, max_frames)
    for line in describe_auto(report, max_frames):
        _say(job, log_path, line)
    return [(rows[i][0], rows[i][1]) for i in keep], [(r[0], r[1]) for r in rows]


def extract(job, cfg, ffmpeg, ffprobe, video, out_dir, log_path):
    """Probe, measure, pick, write. Returns the number of frames written."""
    work_dir = os.path.dirname(out_dir)
    info = probe(ffprobe, video)
    _say(job, log_path, f"Video: {info['display_width']}x{info['display_height']} @ {info['fps']:.2f} fps, "
         f"{info['duration']:.1f} s, {info['codec']}" + (", HDR" if info["transfer"] in HDR_TRANSFERS else ""))
    start, end = time_range(info, cfg.get("video_start", 0.0), cfg.get("video_end", 0.0))
    mode = frames_mode(cfg)
    all_frames = mode == "ALL"
    keep_fps = float(cfg.get("video_fps", 2.0))
    span = (end or info["duration"]) - start
    expected = max(1, int(span * info["fps"]))
    if start > 0 or (end and end < (info["duration"] or end)):
        _say(job, log_path, f"Using {start:.2f} s to {end:.2f} s of the video")

    if mode == "AUTO":
        picks, samples = _pick_auto(job, cfg, ffmpeg, ffprobe, video, work_dir, info, start, end,
                                    expected, log_path)
    else:
        samples = None
        use_blur = (not all_frames and cfg.get("video_blur_filter", True)
                    and has_filter(ffmpeg, "blurdetect"))
        if use_blur:
            samples = scan_blur(job, ffmpeg, video, work_dir, start, end, expected, log_path)
            if not samples:
                _say(job, log_path, "blurdetect produced no data; picking frames evenly instead")
        if not samples:
            samples = scan_times(ffprobe, video, info, start, end)
        if not samples:
            raise StepError("No frames found in the chosen part of the video")
        picks = pick(samples, start, end or samples[-1][0], keep_fps, all_frames)
        if all_frames:
            _say(job, log_path, f"Keeping every frame: {len(picks)}")
        elif use_blur and samples[0][1] is not None:
            blurs = sorted(b for _, b in samples)
            mean_pick = sum(b for _, b in picks) / max(1, len(picks))
            _say(job, log_path, f"Picked the {len(picks)} sharpest frames of {len(samples)} "
                 f"(median blur {blurs[len(blurs) // 2]:.2f}, picked mean {mean_pick:.2f})")
        else:
            _say(job, log_path, f"Picked {len(picks)} frames evenly")
    if not picks:
        raise StepError("no frames selected from the video")
    max_side = int(cfg.get("photo_max_size") or 0)
    if scale_filter(info, max_side):
        _say(job, log_path, f"Frames scaled down to {max_side} px on the long side")
    written = write_frames(job, ffmpeg, video, out_dir, picks, info, log_path, every=all_frames,
                           frame_interval=_min_interval(samples, info["fps"]), max_side=max_side)
    if len(written) < 3:
        raise StepError(f"only {len(written)} frames were written")
    meta = {"video": video, "fps": info["fps"], "clip_fps": info.get("clip_fps") or info["fps"],
            "stream_start": info.get("stream_start", 0.0), "width": info["display_width"],
            "height": info["display_height"], "rotation": info["rotation"],
            "duration": info["duration"], "start": start, "end": end,
            "frames": [{"file": name, "n": n, "t": round(t, 6)} for name, n, t in written]}
    tmp = os.path.join(out_dir, FRAMES_JSON + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    os.replace(tmp, os.path.join(out_dir, FRAMES_JSON))
    return len(written)


def load_frames_json(frames_dir):
    try:
        with open(os.path.join(frames_dir, FRAMES_JSON), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None
