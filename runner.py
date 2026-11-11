# SPDX-License-Identifier: GPL-3.0-or-later
# Photogram3D - photos or video to a textured mesh with COLMAP + OpenMVS
# Copyright (C) 2026 Kruglov Iurii
# Authors: Kruglov Iurii, Anthropic Claude AI
# License: GNU General Public License (GPL) v3 or later, https://www.blender.org/about/license/
"""Background job runner: executes external tools with live progress and cancel."""

import collections
import os
import re
import shlex
import subprocess
import threading
import time

from .common import describe_returncode, human_time, popen_flags
from .tools import merged_env


_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")  # COLMAP colours its warnings


class StepError(Exception):
    pass


class JobCancelled(Exception):
    pass


class Job:
    """Thread-safe state shared between the worker thread and the UI."""

    def __init__(self, name, steps, project_dir):
        self.name = name
        self.steps = steps            # list of Step
        self.project_dir = project_dir
        self.lock = threading.Lock()
        self.status = "queued"        # queued/running/done/failed/cancelled
        self.step_index = -1
        self.step_fraction = 0.0
        self.step_note = ""
        self.lines = collections.deque(maxlen=400)
        self.error = ""
        self.hint = ""                # what the error means (errors.explain)
        self.failed_step = ""
        self.result = {}
        self.cancel_event = threading.Event()
        self.proc = None
        self.thread = None
        self.t_start = None
        self.t_step = None
        self.skipped = []
        self.timings = {}
        self._mute = None             # the running process's stop event (see run)

    # -- state helpers -------------------------------------------------------
    def add_line(self, text):
        text = text.rstrip()
        if not text:
            return
        with self.lock:
            # Collapse progress spam: replace the previous line if it is the
            # same progress message with a new count.
            if self.lines and _same_progress(self.lines[-1], text):
                self.lines[-1] = text
            else:
                self.lines.append(text)

    def set_fraction(self, frac, note=None):
        with self.lock:
            self.step_fraction = max(0.0, min(1.0, frac))
            if note is not None:
                self.step_note = note

    def snapshot(self):
        with self.lock:
            total_w = sum(s.weight for s in self.steps) or 1.0
            done_w = sum(s.weight for s in self.steps[: max(self.step_index, 0)])
            cur = self.steps[self.step_index] if 0 <= self.step_index < len(self.steps) else None
            if cur is not None:
                done_w += cur.weight * self.step_fraction
            if self.status == "done":
                done_w = total_w
            return {
                "status": self.status,
                "step_index": self.step_index,
                "step_count": len(self.steps),
                "step_label": cur.label if cur else "",
                "step_fraction": self.step_fraction,
                "step_note": self.step_note,
                "overall": done_w / total_w,
                "lines": list(self.lines)[-12:],
                "error": self.error,
                "hint": self.hint,
                "failed_step": self.failed_step,
                "elapsed": (time.time() - self.t_start) if self.t_start else 0.0,
                "result": dict(self.result),
            }

    def cancel(self):
        self.cancel_event.set()
        proc = self.proc
        if proc is not None and proc.poll() is None:
            _terminate(proc)

    def check_cancel(self):
        if self.cancel_event.is_set():
            raise JobCancelled()

    # -- process execution ---------------------------------------------------
    def run(self, args, log_path, cwd=None, env=None, parser=None, below_normal=True,
            ok_codes=(0,), tail=None, stop=None):
        """Run a command, stream output to the log and the UI. Returns the exit code.

        ``tail`` = (folder, prefix) also follows the newest ``prefix-*.log`` file the
        tool writes into ``folder``. Windows builds of OpenMVS send their console
        output to a private console, so their own log file is the only readable source.

        ``stop`` (a threading.Event, set by the parser for example) ends the process
        early without cancelling the job; the exit code is then not checked, and what
        the process prints after it was set goes only to the log file.

        Raises StepError for exit codes not in ``ok_codes`` (unless ok_codes is None).
        """
        self.check_cancel()
        args = [str(a) for a in args]
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        follower = LogFollower(tail[0], tail[1], time.time()) if tail else None
        self._mute = stop
        with open(log_path, "ab") as log:
            log.write(("\n$ " + " ".join(shlex.quote(a) for a in args) + "\n").encode("utf-8"))
            log.flush()
            try:
                proc = subprocess.Popen(
                    args, cwd=cwd, env=merged_env(env), stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, bufsize=0,
                    creationflags=popen_flags(below_normal=below_normal))
            except OSError as exc:
                raise StepError(f"cannot start {os.path.basename(args[0])}: {exc}") from exc
            self.proc = proc
            reader = threading.Thread(target=self._pump, args=(proc, log, parser), daemon=True)
            reader.start()
            stopped = False
            while proc.poll() is None:
                if self.cancel_event.wait(0.2):
                    _terminate(proc)
                if follower is not None:
                    self._follow(follower.read(), log, parser)
                if stop is not None and stop.is_set() and proc.poll() is None:
                    stopped = True
                    _terminate(proc)
            reader.join(timeout=10)
            if follower is not None:
                time.sleep(0.2)
                self._follow(follower.read() + follower.rest(), log, parser)
            self.proc = None
            self._mute = None
            code = proc.returncode
            log.write(f"[exit: {describe_returncode(code)}"
                      f"{', stopped by Photogram3D' if stopped else ''}]\n".encode("utf-8"))
        if self.cancel_event.is_set():
            raise JobCancelled()
        if ok_codes is not None and code not in ok_codes and not stopped:
            tail = " | ".join(list(self.lines)[-3:])
            raise StepError(f"{os.path.basename(args[0])} failed ({describe_returncode(code)}). {tail}")
        return code

    def run_side(self, args, log_path, env=None, on_line=None, stop=None, below_normal=True):
        """Run a second process next to the one ``run`` is running (another thread).

        Its output goes to its own log and to ``on_line`` but not to the panel. It is
        stopped by Cancel or by ``stop`` (a threading.Event). Returns the exit code,
        or None if it was stopped."""
        args = [str(a) for a in args]
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "ab") as log:
            log.write(("\n$ " + " ".join(shlex.quote(a) for a in args) + "\n").encode("utf-8"))
            log.flush()
            try:
                proc = subprocess.Popen(
                    args, env=merged_env(env), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, bufsize=0,
                    creationflags=popen_flags(below_normal=below_normal))
            except OSError as exc:
                log.write(f"[cannot start: {exc}]\n".encode("utf-8"))
                return -1

            def pump():
                buf = b""
                while True:
                    try:
                        chunk = proc.stdout.read(4096)
                    except (OSError, ValueError):
                        break
                    if not chunk:
                        break
                    log.write(chunk)
                    buf += chunk
                    parts = re.split(rb"[\r\n]", buf)
                    buf = parts.pop()
                    for raw in parts:
                        text = _ANSI.sub("", raw.decode("utf-8", "replace")).strip()
                        if text and on_line is not None:
                            try:
                                on_line(text)
                            except Exception:  # noqa: BLE001 - never kill the pump
                                pass

            reader = threading.Thread(target=pump, daemon=True)
            reader.start()
            stopped = False
            while proc.poll() is None:
                if self.cancel_event.wait(0.2) or (stop is not None and stop.is_set()):
                    stopped = True
                    _terminate(proc)
            reader.join(timeout=10)
            code = proc.returncode
            log.write(f"[exit: {describe_returncode(code)}]\n".encode("utf-8"))
        return None if stopped else code

    def _pump(self, proc, log, parser):
        """Read stdout in chunks; split on CR and LF so progress lines update live."""
        buf = b""
        stream = proc.stdout
        while True:
            try:
                chunk = stream.read(4096)
            except (OSError, ValueError):
                break
            if not chunk:
                break
            log.write(chunk)
            buf += chunk
            parts = re.split(rb"[\r\n]", buf)
            buf = parts.pop()
            for raw in parts:
                self._handle_line(raw, parser)
            if len(buf) > 65536:  # pathological line without newline
                self._handle_line(buf, parser)
                buf = b""
        if buf:
            self._handle_line(buf, parser)
        try:
            log.flush()
        except OSError:
            pass

    def _follow(self, lines, log, parser):
        for raw in lines:
            log.write(raw + b"\n")
            self._handle_line(raw, parser)

    def _handle_line(self, raw, parser):
        text = _ANSI.sub("", raw.decode("utf-8", "replace")).strip()
        if not text:
            return
        mute = self._mute
        if mute is not None and mute.is_set():
            return
        self.add_line(text)
        if parser is not None:
            try:
                res = parser(text)
            except Exception:  # noqa: BLE001 - parsers must never kill the pump
                res = None
            if res is not None:
                frac, note = res if isinstance(res, tuple) else (res, None)
                self.set_fraction(frac, note)


class LogFollower:
    """Reads new lines from the ``prefix-*.log`` a tool creates in ``folder``.

    Every OpenMVS run writes a new, uniquely named log, so files that already
    existed when the follower was made (a previous run, or an attempt that just
    crashed) are ignored.
    """

    def __init__(self, folder, prefix, t0):
        self.folder, self.prefix, self.t0 = folder, prefix, t0
        self.path = None
        self.pos = 0
        self.buf = b""
        self.old = set(self._candidates())

    def _candidates(self):
        try:
            return [n for n in os.listdir(self.folder)
                    if n.startswith(self.prefix + "-") and n.endswith(".log")]
        except OSError:
            return []

    def _find(self):
        best = None
        try:
            for name in self._candidates():
                if name in self.old:
                    continue
                full = os.path.join(self.folder, name)
                mtime = os.path.getmtime(full)
                if mtime >= self.t0 - 2.0 and (best is None or mtime > best[0]):
                    best = (mtime, full)
        except OSError:
            return None
        return best[1] if best else None

    def read(self):
        if self.path is None:
            self.path = self._find()
            if self.path is None:
                return []
        try:
            with open(self.path, "rb") as f:
                f.seek(self.pos)
                data = f.read()
        except OSError:
            return []
        self.pos += len(data)
        self.buf += data
        parts = re.split(rb"\r?\n", self.buf)
        self.buf = parts.pop()
        return [p for p in parts if p.strip()]

    def rest(self):
        rest, self.buf = self.buf, b""
        return [rest] if rest.strip() else []


def _same_progress(prev, new):
    """True if two lines are successive updates of one progress counter."""
    p = re.sub(r"[\d.]+", "#", prev)
    n = re.sub(r"[\d.]+", "#", new)
    return p == n and "#" in n and len(n) < 200


def _terminate(proc):
    try:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    except OSError:
        pass


class Step:
    def __init__(self, key, label, func, weight=1.0, optional=False):
        self.key = key
        self.label = label
        self.func = func          # func(job, ctx) -> None
        self.weight = weight
        self.optional = optional


def run_job(job, ctx):
    """Execute all steps (blocking). Use start_job() to run in a thread."""
    job.t_start = time.time()
    with job.lock:
        job.status = "running"
    try:
        for i, step in enumerate(job.steps):
            with job.lock:
                job.step_index = i
                job.step_fraction = 0.0
                job.step_note = ""
            job.t_step = time.time()
            job.add_line(f"=== {step.label}")
            step.func(job, ctx)
            job.timings[step.key] = time.time() - job.t_step
            job.add_line(f"--- {step.label}: done in {human_time(job.timings[step.key])}")
        with job.lock:
            job.status = "done"
            job.step_index = len(job.steps) - 1
            job.step_fraction = 1.0
    except JobCancelled:
        with job.lock:
            job.status = "cancelled"
            job.error = "Cancelled"
    except StepError as exc:
        _failed(job, ctx, str(exc))
    except Exception as exc:  # noqa: BLE001 - surface unexpected errors in the UI
        import traceback
        job.add_line(traceback.format_exc())
        _failed(job, ctx, f"{type(exc).__name__}: {exc}")


def _failed(job, ctx, message):
    """Mark the job failed, with what the failure means in plain words (errors.py)."""
    from . import errors
    step = job.steps[job.step_index] if 0 <= job.step_index < len(job.steps) else None
    hint = ""
    if step is not None:
        try:
            log = errors.step_log(ctx, step.key)
            hint = errors.explain(step.key, message, errors.read_tail(log) if log else "",
                                  (ctx or {}).get("cfg"))
            if hint and log and os.path.isdir(os.path.dirname(log)):
                with open(log, "a", encoding="utf-8") as f:
                    f.write(f"\n[Photogram3D] Failed: {message}\n[Photogram3D] What it means: {hint}\n")
        except Exception:  # noqa: BLE001 - the explanation must never hide the error
            hint = ""
    with job.lock:
        job.status = "failed"
        job.error = message
        job.hint = hint
        job.failed_step = step.label if step is not None else ""


def start_job(job, ctx):
    job.thread = threading.Thread(target=run_job, args=(job, ctx), name="Photogram3DJob", daemon=True)
    job.thread.start()
    return job
