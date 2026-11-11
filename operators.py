# SPDX-License-Identifier: GPL-3.0-or-later
# Photogram3D - photos or video to a textured mesh with COLMAP + OpenMVS
# Copyright (C) 2026 Kruglov Iurii
# Authors: Kruglov Iurii, Anthropic Claude AI
# License: GNU General Public License (GPL) v3 or later, https://www.blender.org/about/license/
"""Operators, the UI timer and the shared session state."""

import os
import re
import sys
import threading
import time

import bpy
import numpy as np
from bpy.props import EnumProperty, FloatProperty, StringProperty
from mathutils import Matrix

from . import ai_backend, colmap_io, hardware, importer, pipeline, polish, polish_ai, tools, video
from .common import cloud_sync_service, has_non_ascii, human_bytes, human_time, list_images
from .props import get_prefs
from .runner import start_job

# One reconstruction, one AI polish and one installer per Blender session.
SESSION = {
    "job": None, "ctx": None, "job_handled": True, "import_error": "",
    "install": tools.InstallState(), "hw": None, "verify": None,
    "vc_state": None, "message": "",
    "ai_job": None, "ai_ctx": None, "ai_handled": True, "ai_message": "",
    # AI Features: the quick test and the NVIDIA download (0.3.9)
    "ai_test_job": None, "ai_test_ctx": None, "nvidia_job": None, "nvidia_ctx": None,
    "ai_test_message": "",
}


def job_running(key="job"):
    job = SESSION.get(key)
    return job is not None and job.snapshot()["status"] in ("running", "queued")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def get_hw(refresh=False):
    if SESSION["hw"] is None or refresh:
        SESSION["hw"] = hardware.detect()
    return SESSION["hw"]


def get_toolset(context=None):
    prefs = get_prefs(context)
    if prefs is None:
        return tools.resolve("")
    return tools.resolve(prefs.resolved_tools_dir(), prefs.overrides())


_VIDEO_INFO = {}


def video_info(context):
    """ffprobe's facts about the chosen video, read once per file version (for the panel)."""
    s = context.scene.photogram3d
    path = bpy.path.abspath(s.video_path) if s.video_path else ""
    try:
        key = (path, os.path.getmtime(path))
    except OSError:
        return None
    if key not in _VIDEO_INFO:
        ffprobe = get_toolset(context).get("ffprobe")
        info = None
        if ffprobe:
            try:
                info = video.probe(ffprobe, path)
            except Exception as exc:  # noqa: BLE001 - shown in the panel
                info = {"error": str(exc)}
        _VIDEO_INFO.clear()
        _VIDEO_INFO[key] = info
    return _VIDEO_INFO[key]


def settings_to_cfg(s):
    """Plain dict snapshot of the scene settings for the worker thread."""
    return {
        "input_mode": s.input_mode,
        "images_dir": bpy.path.abspath(s.images_dir) if s.images_dir else "",
        "video_path": bpy.path.abspath(s.video_path) if s.video_path else "",
        "project_dir": s.resolved_project_dir(),
        "quality": s.quality,
        "ai_features": s.ai_features,
        "mapper": s.mapper,
        "camera_model": s.camera_model,
        "single_camera": s.single_camera,
        "align_up": s.align_up,
        "dense_engine": s.dense_engine,
        "dense_region": s.dense_region,
        "dense_region_size": s.dense_region_size,
        "refine": s.refine,
        "texture": s.texture,
        "texture_size": int(s.texture_size),
        "blend_seams": s.blend_seams,
        "spread_views": s.spread_views,
        "even_photos": s.even_photos,
        "sharpen": s.sharpen,
        "photo_max_size": s.photo_max_size,
        "splats": s.splats,
        "splat_steps": s.splat_steps,
        "splat_resolution": s.splat_resolution,
        "raw_pairs": s.raw_pairs,
        "target_faces": s.target_faces,
        "free_space": s.free_space,
        "use_gpu": s.use_gpu,
        "resume": s.resume,
        "crop_stage": s.crop_stage,
        "video_frames": s.video_frames,
        "video_max_frames": s.video_max_frames,
        "video_fps": s.video_fps,
        "video_all_frames": s.video_frames == "ALL",
        "video_start": s.video_start,
        "video_end": s.video_end,
        "video_blur_filter": s.video_blur_filter,
        "video_camera": s.video_camera,
        "video_track": s.video_track,
    }


def sync_warning(project_dir):
    service = cloud_sync_service(project_dir) if project_dir else ""
    if not service:
        return ""
    return (f"The output folder is inside {service}: it would upload several GB of temporary "
            "files and can lock them while OpenMVS replaces them. Pause syncing or use a local folder")


def path_warnings(cfg, tools_root):
    paths = [cfg.get("project_dir", ""), tools_root]
    paths.append(cfg.get("video_path") if cfg.get("input_mode") == "VIDEO" else cfg.get("images_dir"))
    bad = [p for p in paths if p and has_non_ascii(p)]
    out = []
    if bad:
        out.append("Paths with non-Latin characters can break COLMAP/OpenMVS: "
                   + ", ".join(os.path.basename(os.path.normpath(p)) or p for p in bad))
    sync = sync_warning(cfg.get("project_dir", ""))
    if sync:
        out.append(sync)
    return ". ".join(out)


def redraw_all():
    wm = bpy.context.window_manager
    for win in getattr(wm, "windows", []):
        if win.screen is None:
            continue
        for area in win.screen.areas:
            if area.type in ("VIEW_3D", "PREFERENCES"):
                area.tag_redraw()


def retired_note(retired):
    """What became of the scan of an earlier solve of the run (importer.retire_earlier_solves)."""
    if not retired:
        return ""
    if all(what == "removed" for _, what in retired):
        return "The scan of the run's earlier camera solve was replaced"
    return ("The scan of the run's earlier camera solve was replaced; its mesh was kept, hidden, "
            "in the collection marked (earlier solve)")


def _timer():
    """Polls the worker threads; imports the result on the main thread."""
    job = SESSION["job"]
    install = SESSION["install"]
    busy = install.running
    if job is not None:
        snap = job.snapshot()
        if snap["status"] in ("running", "queued"):
            busy = True
        elif not SESSION["job_handled"]:
            SESSION["job_handled"] = True
            if snap["status"] == "done" and snap["result"].get("stage") == "track":
                SESSION["message"] = _retrack_done(snap)
            elif snap["status"] == "done":
                scene = bpy.context.scene
                crop = snap["result"].get("stage") == "crop"
                SESSION["message"] = ("Dense point cloud ready: place the crop box, then Build Mesh"
                                      if crop else f"Finished in {human_time(snap['elapsed'])}")
                if scene is not None and (crop or scene.photogram3d.auto_import):
                    try:
                        if crop:
                            out = importer.import_crop_stage(
                                snap["result"], name=job.name,
                                import_cameras_flag=scene.photogram3d.import_cameras,
                                camera_photos=scene.photogram3d.camera_photos,
                                tracked_camera=scene.photogram3d.video_camera)
                            if snap["result"].get("joined_from"):     # Complete Joining (0.3.14)
                                from . import tracking
                                hidden = importer.hide_scans(snap["result"]["joined_from"])
                                videos = [o.name for o in tracking.tracked_cameras(out["root"])]
                                SESSION["message"] = ("Joined scan ready: place the crop box, then Build "
                                                      "Mesh" + (f". Hidden: {', '.join(hidden)}" if hidden
                                                                else "")
                                                      + (f". Video cameras: {', '.join(videos)}" if videos
                                                         else ""))
                        else:
                            out = importer.import_result(
                                snap["result"], name=job.name,
                                import_cameras_flag=scene.photogram3d.import_cameras,
                                camera_photos=scene.photogram3d.camera_photos,
                                tracked_camera=scene.photogram3d.video_camera)
                        retired = retired_note(out.get("retired"))
                        if retired:
                            SESSION["message"] += ". " + retired
                    except Exception as exc:  # noqa: BLE001 - report in the panel
                        SESSION["import_error"] = str(exc)
            else:
                SESSION["message"] = snap["error"] or snap["status"]
    if install.done and SESSION.get("install_seen") is not True:
        SESSION["install_seen"] = True
        SESSION["verify"] = install.report or None
    for key in ("ai_test_job", "nvidia_job"):
        other = SESSION.get(key)
        if other is not None:
            snap = other.snapshot()
            if snap["status"] in ("running", "queued"):
                busy = True
            elif not SESSION.get(key + "_handled", True):
                SESSION[key + "_handled"] = True
                if snap["status"] == "done":
                    lines = snap["result"].get("lines") or []
                    SESSION["ai_test_message"] = ("NVIDIA libraries installed" if key == "nvidia_job"
                                                  else "Tested: " + "; ".join(lines[:1]))
                else:
                    SESSION["ai_test_message"] = snap["error"] or snap["status"]
    if SESSION["ai_job"] is not None:
        if job_running("ai_job"):
            busy = True
        elif not SESSION["ai_handled"]:
            SESSION["ai_handled"] = True
            try:
                SESSION["ai_ok"] = finish_ai_job()
            except Exception as exc:  # noqa: BLE001 - show in the panel instead of killing the timer
                SESSION["ai_ok"] = False
                SESSION["ai_message"] = f"AI Polish could not be applied: {exc}"
    redraw_all()
    return 0.5 if busy else None


def _retrack_done(snap):
    """Track Every Frame finished: key the scan's tracked camera on the new track."""
    from . import tracking
    root = bpy.data.objects.get(SESSION.get("track_root") or "")
    cam = (tracking.tracked_camera(root, bpy.data.objects.get(SESSION.get("track_camera") or ""))
           if root is not None else None)
    if cam is None:
        return "Camera track saved; the scan's tracked camera is gone"
    try:
        n = tracking.retrack(cam, {"project": snap["result"].get("project", "")})
    except Exception as exc:  # noqa: BLE001 - report in the panel
        SESSION["import_error"] = str(exc)
        return "Camera tracked, but its keys could not be set"
    total = snap["result"].get("total") or n
    return f"Tracked camera follows the video: {n:,} of {total:,} frames placed"


def ensure_timer():
    if not bpy.app.timers.is_registered(_timer):
        bpy.app.timers.register(_timer, first_interval=0.3, persistent=True)


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------

def ai_settings(prefs, tools_root):
    """Where AI Features run for this run (ai_backend.resolve) and, on a GPU, the largest photo
    size the quick test ran ALIKED at there."""
    test = ai_backend.read_test(tools_root) if tools_root else None
    # Automatic and not tested yet: the run tests this computer first (pipeline.step_ai_test)
    device = ai_backend.resolve(prefs.ai_features_device if prefs else "AUTO", tools_root, test,
                                bool(get_hw().get("gpus")), untested="test" if tools_root else "guess")
    out = {"ai_device": device}
    res = ((test or {}).get("results") or {}).get(device)
    if device in ("DIRECTML", "NVIDIA") and res and res.get("extract"):
        out["ai_max_size"] = ai_backend.max_gpu_size(res)
    return out


def ai_photo_count(s):
    """The photos AI Features would get, for the time estimates: the photo folder's count,
    or Max Frames for a video (Auto)."""
    if s.input_mode == "VIDEO":
        return s.video_max_frames, True
    folder = bpy.path.abspath(s.images_dir) if s.images_dir else ""
    try:
        names = list_images(folder) if folder and os.path.isdir(folder) else []
    except OSError:
        names = []
    return (len(names) or 42), False


class PHOTOGRAM3D_OT_ai_test(bpy.types.Operator):
    bl_idname = "photogram3d.ai_test"
    bl_label = "Test AI Speed"
    bl_description = ("Time AI Features (ALIKED and LightGlue) on this computer's GPU and CPU with "
                      "one of your photos, and suggest the fastest. The first time it downloads ONNX "
                      "Runtime (about 20 MB) and COLMAP's two model files (49 MB)")

    def execute(self, context):
        if job_running("ai_test_job") or job_running("nvidia_job"):
            self.report({"WARNING"}, "A test or download is already running")
            return {"CANCELLED"}
        prefs = get_prefs(context)
        if prefs is None:
            return {"CANCELLED"}
        root = prefs.resolved_tools_dir()
        s = context.scene.photogram3d
        folder = bpy.path.abspath(s.images_dir) if s.images_dir else ""
        sample = ""
        if s.input_mode == "IMAGES" and folder and os.path.isdir(folder):
            jpgs = [n for n in list_images(folder) if n.lower().endswith((".jpg", ".jpeg", ".png"))]
            sample = os.path.join(folder, jpgs[0]) if jpgs else ""
        hw = get_hw()
        ts = get_toolset(context)
        n, is_video = ai_photo_count(s)
        spec = {"tools_root": root, "python": sys.executable, "sample": sample,
                "cpu_size": hardware.preset_params(s.quality, hw)["aliked_max_image"],
                "cpu_matchers": pipeline.ai_matchers(hw, hardware.preset_params(s.quality, hw)["aliked_features"]),
                "colmap": ts.get("colmap"), "colmap_env": ts.get("colmap_env") or {},
                "has_nvidia_gpu": bool(hw.get("gpus")), "hw_text": hardware.describe(hw),
                "n_photos": n, "video": is_video}
        job, ctx = ai_backend.build_test_job(spec)
        SESSION["ai_test_job"], SESSION["ai_test_ctx"] = job, ctx
        SESSION["ai_test_job_handled"] = False
        SESSION["ai_test_message"] = ""
        start_job(job, ctx)
        ensure_timer()
        return {"FINISHED"}


class PHOTOGRAM3D_OT_ai_nvidia(bpy.types.Operator):
    bl_idname = "photogram3d.ai_nvidia"
    bl_label = "Download NVIDIA Libraries"
    bl_description = ("Download NVIDIA's cuBLAS, cuDNN, NVRTC and CUDA runtime (1.3 GB, from NVIDIA's "
                      "packages on PyPI) so COLMAP can run AI Features on an NVIDIA GPU itself. NVIDIA's "
                      "license: free to use, not open source")

    def invoke(self, context, event):
        prefs = get_prefs(context)
        size = ai_backend.nvidia_download_size(prefs.resolved_tools_dir()) if prefs else 0
        return context.window_manager.invoke_confirm(
            self, event, title=f"Download NVIDIA's libraries ({human_bytes(size or ai_backend.NVIDIA_DOWNLOAD)})?",
            message="They are NVIDIA's, under NVIDIA's license: free to use, not open source",
            confirm_text="Download")

    def execute(self, context):
        if job_running("ai_test_job") or job_running("nvidia_job"):
            self.report({"WARNING"}, "A test or download is already running")
            return {"CANCELLED"}
        prefs = get_prefs(context)
        if prefs is None:
            return {"CANCELLED"}
        job, ctx = ai_backend.build_nvidia_job(prefs.resolved_tools_dir())
        SESSION["nvidia_job"], SESSION["nvidia_ctx"] = job, ctx
        SESSION["nvidia_job_handled"] = False
        SESSION["ai_test_message"] = ""
        start_job(job, ctx)
        ensure_timer()
        return {"FINISHED"}


class PHOTOGRAM3D_OT_ai_nvidia_remove(bpy.types.Operator):
    bl_idname = "photogram3d.ai_nvidia_remove"
    bl_label = "Remove NVIDIA Libraries"
    bl_description = "Delete the NVIDIA libraries from the tools folder (2.0 GB on disk)"

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        prefs = get_prefs(context)
        if prefs is None or job_running("nvidia_job"):
            return {"CANCELLED"}
        ai_backend.remove_nvidia(prefs.resolved_tools_dir())
        return {"FINISHED"}


class PHOTOGRAM3D_OT_ai_test_cancel(bpy.types.Operator):
    bl_idname = "photogram3d.ai_test_cancel"
    bl_label = "Cancel"

    def execute(self, context):
        for key in ("ai_test_job", "nvidia_job"):
            job = SESSION.get(key)
            if job is not None and job_running(key):
                job.cancel()
        return {"FINISHED"}


class PHOTOGRAM3D_OT_run(bpy.types.Operator):
    bl_idname = "photogram3d.run"
    bl_label = "Reconstruct"
    bl_description = "Run COLMAP + OpenMVS in the background and import the textured mesh"

    def execute(self, context):
        return start_reconstruction(self, context)


def start_reconstruction(op, context, **extra):
    """Validate the settings and start the pipeline; ``extra`` goes into the cfg.

    Without a project_dir in ``extra`` (the Reconstruct button) the run goes to the newest
    attempt of the output folder with Resume on, else to a new attempt folder; Build Mesh
    passes the folder of the scan it continues."""
    if job_running("job") or job_running("ai_job"):
        op.report({"WARNING"}, "A reconstruction or AI polish is already running")
        return {"CANCELLED"}
    s = context.scene.photogram3d
    cfg = settings_to_cfg(s)
    base = cfg["project_dir"]
    if "project_dir" not in extra and base:
        folder, k = pipeline.choose_attempt(base, cfg["resume"])
        cfg.update(project_dir=folder, attempt=k, base_dir=os.path.normpath(base))
    cfg.update(extra)
    prefs = get_prefs(context)
    cfg.setdefault("tools_root", prefs.resolved_tools_dir() if prefs else "")
    cfg.setdefault("brush_path", bpy.path.abspath(prefs.brush_path) if prefs and prefs.brush_path else "")
    cfg.setdefault("python", sys.executable)
    if cfg.get("ai_features"):
        cfg.update(ai_settings(prefs, cfg["tools_root"]))
    ts = get_toolset(context)
    problems = pipeline.validate(cfg, ts)
    if problems:
        op.report({"ERROR"}, "; ".join(problems))
        return {"CANCELLED"}
    warn = path_warnings(cfg, cfg["tools_root"])
    if warn:
        op.report({"WARNING"}, warn)
    try:
        job, ctx = pipeline.build(cfg, ts, get_hw())
    except OSError as exc:
        op.report({"ERROR"}, f"Cannot create the output folder: {exc}")
        return {"CANCELLED"}
    SESSION.update(job=job, ctx=ctx, job_handled=False, import_error="", message="")
    start_job(job, ctx)
    ensure_timer()
    return {"FINISHED"}


def scan_roots(context):
    """Photogram3D roots in the scene, newest first."""
    roots = [o for o in context.scene.objects if importer.ROOT_PROP in o.keys()]
    return sorted(roots, key=lambda o: o.get("photogram3d_created", 0), reverse=True)


def root_of(obj, context):
    """The root an object belongs to: its parent chain, or the root of a free crop box."""
    root = importer.find_root(obj)
    if root is None and obj is not None and isinstance(obj.get(importer.CROP_BOX_PROP), str):
        rid = obj[importer.CROP_BOX_PROP]
        root = next((r for r in scan_roots(context) if r.get(importer.ROOT_ID_PROP) == rid), None)
    return root


def _of_output(root, base):
    """The root is one of the runs (attempts) of the output folder ``base``."""
    if not base:
        return False
    b = importer._norm(base)
    if importer._norm(root.get("photogram3d_base", "")) == b:
        return True
    proj = importer._norm(root.get(importer.ROOT_PROP, ""))
    return proj == b or re.fullmatch(re.escape(b) + r"_\d+", proj) is not None


def active_root(context):
    """The Photogram3D root to work on: the active object's, else the newest run of the
    output folder in the panel."""
    root = root_of(context.active_object, context) if context.active_object else None
    if root is not None:
        return root
    base = context.scene.photogram3d.resolved_project_dir()
    for prop in (importer.CROP_BOX_PROP, importer.MESH_PROP, importer.CLOUD_PROP):
        for root in scan_roots(context):
            if _of_output(root, base) and importer._children(root, prop):
                return root
    roots = scan_roots(context)
    return roots[0] if len(roots) == 1 else None


def _box_visible(root):
    try:      # visible_get: also not in a hidden collection (Complete Joining hides two scans)
        return any(b.visible_get() for b in importer.crop_boxes(root))
    except RuntimeError:  # object not in the view layer
        return False


def crop_root(context):
    """Root with a visible crop box (the crop stage is on): the active scan's, else the
    newest run of the output folder in the panel.

    The final import hides the box, which ends the crop stage in the panel."""
    root = root_of(context.active_object, context) if context.active_object else None
    if root is not None and _box_visible(root):
        return root
    base = context.scene.photogram3d.resolved_project_dir()
    for root in scan_roots(context):
        if _of_output(root, base) and _box_visible(root):
            return root
    return None


def current_project(context):
    """The run folder the panel's buttons use: the active scan's, else the newest attempt."""
    root = root_of(context.active_object, context) if context.active_object else None
    if root is not None and root.get(importer.ROOT_PROP):
        return root[importer.ROOT_PROP]
    base = context.scene.photogram3d.resolved_project_dir()
    done = pipeline.attempts(base) if base else []
    return done[-1][1] if done else base


class PHOTOGRAM3D_OT_build_mesh(bpy.types.Operator):
    bl_idname = "photogram3d.build_mesh"
    bl_label = "Build Mesh"
    bl_description = ("Mesh, refine and texture only what is inside the crop box "
                      "(finished steps are reused)")

    @classmethod
    def poll(cls, context):
        return crop_root(context) is not None and not job_running("job")

    def execute(self, context):
        root = crop_root(context)
        box = next((b for b in importer.crop_boxes(root) if b.visible_get()), None)
        if box is None:
            self.report({"ERROR"}, "No crop box")
            return {"CANCELLED"}
        importer.attach_crop_box(root, box)     # from now on it moves with the scan
        try:
            obb = importer.crop_box_obb(box)
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        project = root[importer.ROOT_PROP]
        mvs = pipeline.project_paths(project)["mvs"]
        path = os.path.join(mvs, pipeline.CROP_ROI)
        try:
            obb.save_if_changed(path)   # an unmoved box keeps the finished mesh steps
        except OSError as exc:
            self.report({"ERROR"}, f"Cannot save the crop box: {exc}")
            return {"CANCELLED"}
        out = start_reconstruction(self, context, crop_roi=path, **_continue_run(root))
        if out == {"FINISHED"}:
            self.report({"INFO"}, f"Building the mesh of run {int(root.get('photogram3d_attempt') or 1)} "
                                  f"({os.path.basename(os.path.normpath(project))})")
        return out


def _continue_run(root):
    """Settings that make a run continue the scan of ``root`` (its folder, its steps)."""
    project = root[importer.ROOT_PROP]
    out = {"project_dir": project, "resume": True,
           "attempt": int(root.get("photogram3d_attempt") or 1)}
    if root.get("photogram3d_base"):
        out["base_dir"] = root["photogram3d_base"]
    return out


class PHOTOGRAM3D_OT_build_uncropped(bpy.types.Operator):
    bl_idname = "photogram3d.build_uncropped"
    bl_label = "Without Crop"
    bl_description = "Mesh, refine and texture the whole dense point cloud"

    @classmethod
    def poll(cls, context):
        return crop_root(context) is not None and not job_running("job")

    def execute(self, context):
        return start_reconstruction(self, context, crop_skip=True, **_continue_run(crop_root(context)))


JOINED_NO_SOLVE = ("A joined scan has no camera solve of its own: re-solve one of the two scans, then "
                   "Join and Complete Joining again")


def selected_photo_cameras(context):
    """(root, [image names]) of the selected cameras of one scan, or (None, [])."""
    cams = [o for o in context.selected_objects
            if o.type == "CAMERA" and "photogram3d_image" in o.keys()]
    roots = {importer.find_root(o) for o in cams}
    if len(roots) != 1 or None in roots:
        return None, []
    return roots.pop(), sorted({o["photogram3d_image"] for o in cams})


class PHOTOGRAM3D_OT_resolve_without(bpy.types.Operator):
    bl_idname = "photogram3d.resolve_without"
    bl_label = "Re-solve Without Selected"
    bl_description = ("Solve the cameras again without the selected photos (misplaced ones), as a "
                      "new run next to this one. Features and matches are reused; this scan stays "
                      "as it is")

    @classmethod
    def poll(cls, context):
        return not job_running("job") and selected_photo_cameras(context)[0] is not None

    def execute(self, context):
        root, names = selected_photo_cameras(context)
        src = root.get(importer.ROOT_PROP, "")
        if src and pipeline.joined_description(src) is not None:
            self.report({"ERROR"}, JOINED_NO_SOLVE)
            return {"CANCELLED"}
        if not src or not os.path.isfile(pipeline.project_paths(src)["db"]):
            self.report({"ERROR"}, "This scan's folder or its match database is missing")
            return {"CANCELLED"}
        base = root.get("photogram3d_base") or src
        exclude = sorted(set(pipeline.read_excluded(src)) | set(names))
        folder, k = pipeline.fork_target(base, src)
        try:
            pipeline.fork_attempt(src, folder)
        except OSError as exc:
            self.report({"ERROR"}, f"Cannot prepare the new run's folder: {exc}")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Run {k}: solving without {len(exclude)} photo(s)")
        return start_reconstruction(self, context, project_dir=folder, attempt=k,
                                    base_dir=os.path.normpath(base), resume=True,
                                    exclude_images=exclude)


class PHOTOGRAM3D_OT_resolve_from_cameras(bpy.types.Operator):
    bl_idname = "photogram3d.resolve_from_cameras"
    bl_label = "Re-solve From Cameras"
    bl_description = ("Solve the cameras again along the cameras as they stand now, as a new run next "
                      "to this one. First turn the misplaced cameras roughly the way they should face "
                      "(a side laid over the other side: select them, R Z 180); only their direction "
                      "counts. Photo pairs that disagree with the cameras are left out. Features and "
                      "matches are reused; this scan stays as it is")

    @classmethod
    def poll(cls, context):
        return not job_running("job") and _cameras_root(context) is not None

    def execute(self, context):
        root = _cameras_root(context)
        src = root.get(importer.ROOT_PROP, "")
        if src and pipeline.joined_description(src) is not None:
            self.report({"ERROR"}, JOINED_NO_SOLVE)
            return {"CANCELLED"}
        p = pipeline.project_paths(src) if src else None
        if not p or not os.path.isfile(p["db"]) or not os.path.isfile(os.path.join(p["model"], "images.bin")):
            self.report({"ERROR"}, "This scan's folder, its match database or its solved cameras are missing")
            return {"CANCELLED"}
        poses = importer.scene_camera_poses(root)
        solved = {im["name"]: im for im in colmap_io.read_model(p["model"])[1].values()}
        turned = [n for n, (R, _) in poses.items() if n in solved and pipeline.turned_from(R, solved[n])]
        if not turned:
            self.report({"WARNING"}, "No camera was turned: first turn the misplaced cameras the way "
                                     "they should face, then press Re-solve From Cameras")
            return {"CANCELLED"}
        base = root.get("photogram3d_base") or src
        folder, k = pipeline.fork_target(base, src)
        try:
            pipeline.fork_attempt(src, folder)
            n, turned = pipeline.write_guide(src, folder, poses)
        except (OSError, ValueError) as exc:
            self.report({"ERROR"}, f"Cannot prepare the new run's folder: {exc}")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Run {k}: solving along {n} cameras, {len(turned)} of them turned by hand")
        return start_reconstruction(self, context, project_dir=folder, attempt=k,
                                    base_dir=os.path.normpath(base), resume=True,
                                    exclude_images=pipeline.read_excluded(src))


def _cameras_root(context):
    """The scan whose photo cameras Re-solve From Cameras reads: that of the selected cameras,
    else the active scan, when it has photo cameras."""
    root = selected_photo_cameras(context)[0]
    if root is None:
        root = upright_root(context)
    return root if root is not None and importer.photo_cameras(root) else None


class PHOTOGRAM3D_OT_reset_alignment(bpy.types.Operator):
    bl_idname = "photogram3d.reset_alignment"
    bl_label = "Reset"
    bl_description = ("Undo Align Ground and every scale: the scan goes back to where it was "
                      "reconstructed. Nothing is recomputed")
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return active_root(context) is not None

    def execute(self, context):
        importer.reset_alignment(active_root(context))
        return {"FINISHED"}


def upright_root(context):
    """The scan the panel shows: its crop stage, or the finished scan."""
    return crop_root(context) or active_root(context)


class PHOTOGRAM3D_OT_upright_cameras(bpy.types.Operator):
    bl_idname = "photogram3d.upright_cameras"
    bl_label = "Turn Cameras Upright"
    bl_description = ("Turn this scan's photo cameras so that each one shows its photo upright, the "
                      "way the photo's EXIF orientation says (photos taken holding the phone upright). "
                      "A camera only turns about its viewing axis: nothing moves, and the frame becomes "
                      "portrait")
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return upright_root(context) is not None

    def execute(self, context):
        root = upright_root(context)
        n = importer.upright_cameras(root, importer.root_turns(root))
        self.report({"INFO"}, f"{n} camera(s) turned upright" if n else "Every camera is already upright")
        return {"FINISHED"}


class PHOTOGRAM3D_OT_track_camera(bpy.types.Operator):
    bl_idname = "photogram3d.track_camera"
    bl_label = "Track Every Frame"
    bl_description = ("Place this scan's tracked camera on every frame of its video (or every second "
                      "frame, as Camera Track says), so that it follows the footage frame by frame. "
                      "Runs in the background; the scan itself is not computed again")

    @classmethod
    def poll(cls, context):
        from . import tracking
        root = upright_root(context)
        return (not job_running("job") and root is not None
                and tracking.tracked_camera(root) is not None)

    def execute(self, context):
        from . import tracking
        root = upright_root(context)
        cam = tracking.tracked_camera(root, context.active_object)    # a joined scan has several
        project = tracking.track_project(cam, root)
        if tracking.source_solved_again(cam):
            self.report({"ERROR"}, f"{os.path.basename(os.path.normpath(project))} was solved again since the "
                                   "join: its camera track would not fit this scan. Join it again")
            return {"CANCELLED"}
        meta = video.load_frames_json(pipeline.project_paths(project)["frames"]) if project else None
        path = (meta or {}).get("video") or ""
        if not meta or not os.path.isfile(path):
            self.report({"ERROR"}, f"The video of this scan is not at {path or '(unknown)'}")
            return {"CANCELLED"}
        s = context.scene.photogram3d
        every = 2 if s.video_track == "HALF" else 1
        job, ctx = pipeline.build_track_job(project, path, every, get_toolset(context), get_hw(),
                                            use_gpu=s.use_gpu, name=f"{cam.name} track")
        SESSION.update(job=job, ctx=ctx, job_handled=False, import_error="", message="", track_root=root.name,
                       track_camera=cam.name)
        start_job(job, ctx)
        ensure_timer()
        return {"FINISHED"}


class PHOTOGRAM3D_OT_add_video_cameras(bpy.types.Operator):
    bl_idname = "photogram3d.add_video_cameras"
    bl_label = "Add Video Cameras"
    bl_description = ("Add the cameras that follow this scan's videos: for a joined scan one for each "
                      "video it was joined from, where that video's camera was. Nothing else changes")
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        from . import tracking
        root = upright_root(context)
        return root is not None and not tracking.tracked_cameras(root) and importer.scan_video_count(root) > 0

    def execute(self, context):
        root = upright_root(context)
        made, errors = importer.add_video_cameras(root)
        if not made:
            self.report({"ERROR"}, "No video camera added: " + ("; ".join(errors) or "the videos' runs are gone"))
            return {"CANCELLED"}
        msg = f"{len(made)} video camera(s) added: " + ", ".join(o.name for o in made)
        if errors:
            self.report({"WARNING"}, msg + "; not added: " + "; ".join(errors))
        else:
            self.report({"INFO"}, msg)
        return {"FINISHED"}


class PHOTOGRAM3D_OT_reset_crop_box(bpy.types.Operator):
    bl_idname = "photogram3d.reset_crop_box"
    bl_label = "Reset Box"
    bl_description = "Put the crop box back around the whole dense point cloud"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return crop_root(context) is not None

    def execute(self, context):
        box = importer.reset_crop_box(crop_root(context))
        if box is None:
            self.report({"ERROR"}, "No crop box to reset")
            return {"CANCELLED"}
        return {"FINISHED"}


class PHOTOGRAM3D_OT_place_marker(bpy.types.Operator):
    bl_idname = "photogram3d.place_marker"
    bl_label = "Place Marker"
    bl_description = ("Put this marker at the 3D cursor. Shift+right-click on the point cloud or "
                      "the mesh first: the cursor lands on the surface")
    bl_options = {"REGISTER", "UNDO"}

    slot: EnumProperty(items=[(k, v, "") for k, v in importer.MARKER_NAMES.items()])

    @classmethod
    def poll(cls, context):
        return active_root(context) is not None

    def execute(self, context):
        root = active_root(context)
        importer.place_marker(root, self.slot, context.scene.cursor.location)
        return {"FINISHED"}


class PHOTOGRAM3D_OT_align_ground(bpy.types.Operator):
    bl_idname = "photogram3d.align_ground"
    bl_label = "Align Ground"
    bl_description = ("Rotate and move the scan so that the plane through Ground 1-3 is the XY "
                      "plane and Ground 1 to 2 runs along the chosen axis. The origin goes to the "
                      "centre of the rectangle when the three points are corners of one (three "
                      "wheels), else to their centre")
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return active_root(context) is not None

    def execute(self, context):
        axis = context.scene.photogram3d.ground_axis
        try:
            rectangle = importer.align_ground(active_root(context), None if axis == "FREE" else axis)
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, "Ground aligned, origin at the centre of the "
                    + ("rectangle (Ground 1-3 are three of its corners)" if rectangle else "three points"))
        return {"FINISHED"}


class PHOTOGRAM3D_OT_marker_scale(bpy.types.Operator):
    bl_idname = "photogram3d.marker_scale"
    bl_label = "Set Scale"
    bl_description = "Scale the scan so that Anchor A to Anchor B is the Real Distance"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return active_root(context) is not None

    def execute(self, context):
        try:
            factor = importer.apply_real_scale(active_root(context),
                                               context.scene.photogram3d.real_distance)
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Scaled by {factor:.4f}")
        return {"FINISHED"}


def join_target(context):
    """The scan chosen in the panel to join the active scan onto (0.3.13); without a choice
    that names a scan, the one the active scan was joined onto."""
    name = context.scene.photogram3d.join_target
    obj = bpy.data.objects.get(name) if name and name != "NONE" else None
    if obj is None or importer.ROOT_PROP not in obj.keys():
        root = active_root(context)
        rec = importer.join_record(root) if root is not None else None
        obj = bpy.data.objects.get(rec["onto"]) if rec else None
    return obj if obj is not None and importer.ROOT_PROP in obj.keys() else None


def _length(context, value, size):
    """A distance in the scene's units, with its share of the scan's size."""
    try:
        text = bpy.utils.units.to_string(context.scene.unit_settings.system, "LENGTH", value,
                                         precision=3, split_unit=False)
    except Exception:  # noqa: BLE001 - only the display depends on it
        text = f"{value:.4g}"
    return f"{text} ({100.0 * value / size:.2f}% of the scan)" if size else text


class PHOTOGRAM3D_OT_join_scans(bpy.types.Operator):
    bl_idname = "photogram3d.join_scans"
    bl_label = "Join"
    bl_description = ("Move the active scan onto the chosen scan by the points placed on both (at "
                      "least three, the same spots in the same numbers): scale, rotation and "
                      "position. Each point's distance after the join shows below")
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return active_root(context) is not None and join_target(context) is not None

    def execute(self, context):
        moving, fixed = active_root(context), join_target(context)
        try:
            rec = importer.join_scans(moving, fixed)
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        msg = (f"Joined by {len(rec['points'])} points: {_length(context, rec['rms'], rec['size'])} "
               "apart on average")
        if rec["suspect"]:
            msg += "; check " + ", ".join(importer.MARKER_NAMES[s] for s in rec["suspect"])
        if rec.get("narrow"):
            msg += ("; the points are almost on one line, so the turn about it is uncertain: add a point "
                    "away from that line, or Refine")
        self.report({"WARNING"} if rec["suspect"] or rec.get("narrow") else {"INFO"}, msg)
        return {"FINISHED"}


class PHOTOGRAM3D_OT_refine_join(bpy.types.Operator):
    bl_idname = "photogram3d.refine_join"
    bl_label = "Refine"
    bl_description = ("Refine the join where the two point clouds overlap (iterative closest points, "
                      "with scale). Parts only one scan has, or that moved between the shoots, are "
                      "left out; the points placed stay in the fit")
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        root = active_root(context)
        return root is not None and join_target(context) is not None and importer.join_record(root)

    def execute(self, context):
        moving, fixed = active_root(context), join_target(context)
        try:
            rec = importer.refine_join(moving, fixed)
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        r = rec["refined"]
        if not r.get("kept", True):
            self.report({"WARNING"}, "The join by the points is kept: the clouds came no closer "
                                     f"({_length(context, r['median_before'], rec['size'])} -> "
                                     f"{_length(context, r['median_after'], rec['size'])}, median). They "
                                     "overlap too little for it; better placed points improve the join")
            return {"FINISHED"}
        self.report({"INFO"}, f"Refined on {100 * r['overlap']:.0f}% of the points: "
                              f"{_length(context, r['median_before'], rec['size'])} -> "
                              f"{_length(context, r['median_after'], rec['size'])} between the clouds (median)")
        return {"FINISHED"}


def joined_folder(project_a, project_b):
    """The folder of the run joining two scans: next to the first scan's run, named after both."""
    return f"{os.path.normpath(project_a)}_join_{os.path.basename(os.path.normpath(project_b))}"


def scan_title(root):
    """'Gas_mask' for the first run on the Gas_mask photos, 'Gas_mask_2' for the second."""
    k = int(root.get("photogram3d_attempt") or 1)
    coll = root.users_collection[0].name if root.users_collection else root.name
    name = coll.split(" ", 1)[1] if coll.startswith("Photogram3D_") and " " in coll else root.name
    return name if k <= 1 else f"{name}_{k}"


def joined_run(context, moving, fixed):
    """Settings of the run joining the scan ``moving`` onto ``fixed`` (pipeline.build_joined,
    Complete Joining): both runs' folders, the join, and the box the joined scan starts with
    (one holding both scans' boxes). The run stops at the crop stage; Build Mesh on the joined
    scan continues it like any scan's. Saves the box and, for a new join, where the joined
    scan goes (where ``fixed`` is) in the joined run's folder."""
    from . import joinrun
    folders = []
    for root in (fixed, moving):
        project = root.get(importer.ROOT_PROP, "")
        p = pipeline.project_paths(project) if project else None
        if p is None or not os.path.isfile(os.path.join(p["mvs"], "scene_dense.ply")) \
                or not os.path.isfile(os.path.join(p["dense"], "sparse", "images.bin")):
            raise ValueError(f"{root.name}: its run has no dense point cloud. Reconstruct it first "
                             "(it may stop at the crop box)")
        if pipeline.solve_id(p) != root.get(importer.SOLVE_PROP):
            raise ValueError(f"{root.name} shows an earlier camera solve of its run: Import Result again, "
                             "then join it")
        folders.append(project)
    # the scans as Join (and Refine) left them: a scan aligned or moved since then would be
    # joined where it now stands
    rec = importer.join_record(moving) or {}
    slots, P, Q = importer.join_pairs(moving, fixed)
    if len(slots) >= 3 and rec.get("rms") is not None:
        now = float(np.sqrt(np.mean(np.sum((np.array(P) - np.array(Q)) ** 2, axis=1))))
        allowed = max(3.0 * float(rec["rms"]), 0.01 * float(rec.get("size") or 0.0))
        if now > allowed:
            raise ValueError(f"{moving.name} and {fixed.name} moved apart since the join (their points are "
                             f"{_length(context, now, rec.get('size'))} apart): press Join again")
    T = importer.join_transform(moving, fixed)
    solve_a, solve_b = fixed.get(importer.SOLVE_PROP), moving.get(importer.SOLVE_PROP)
    solve = joinrun.join_id(solve_a, solve_b, T.tolist())
    folder = joined_folder(*folders)
    os.makedirs(folder, exist_ok=True)
    box_path = os.path.join(folder, pipeline.CROP_ROI)
    obb = importer.joined_box(fixed, moving, T)
    if obb is not None:
        obb.save_if_changed(box_path)
    elif os.path.exists(box_path):
        os.remove(box_path)
    if not importer.user_transform_saved(folder, solve):
        # a joined scan aligned since (Align Ground) keeps its place
        importer.write_user_transform(folder, fixed.matrix_world, solve)
    base = settings_to_cfg(context.scene.photogram3d)
    keep = ("quality", "refine", "texture", "texture_size", "blend_seams", "spread_views", "even_photos",
            "sharpen", "target_faces", "free_space")
    cfg = {k: base[k] for k in keep}
    prefs = get_prefs(context)
    cfg.update(project_dir=folder, resume=True, input_mode="IMAGES", crop_stage=True,
               scan_name=f"{scan_title(fixed)}+{scan_title(moving)}",
               join_box=box_path if obb is not None else "",
               tools_root=prefs.resolved_tools_dir() if prefs else "", python=sys.executable,
               join={"a": folders[0], "b": folders[1], "matrix": T.tolist(), "solve_a": solve_a,
                     "solve_b": solve_b, "solve": solve})
    return cfg


class PHOTOGRAM3D_OT_complete_join(bpy.types.Operator):
    bl_idname = "photogram3d.complete_join"
    bl_label = "Complete Joining"
    bl_description = ("Make one scan of both, as joined: one point cloud, the cameras of both and a "
                      "crop box holding both crop boxes, in a collection of its own where the chosen "
                      "scan is (no depth maps are computed again). The two scans are hidden. Crop, "
                      "align and Build Mesh on it as on any scan")

    @classmethod
    def poll(cls, context):
        if job_running("job") or job_running("ai_job"):
            return False
        root, target = active_root(context), join_target(context)
        rec = importer.join_record(root) if root is not None else None
        return bool(rec) and target is not None and rec.get("onto") == target.name

    def execute(self, context):
        moving, fixed = active_root(context), join_target(context)
        ts = get_toolset(context)
        if not ts.get("openmvs_dir"):
            self.report({"ERROR"}, "OpenMVS is not installed")
            return {"CANCELLED"}
        try:
            cfg = joined_run(context, moving, fixed)
        except (ValueError, OSError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        warn = path_warnings(cfg, cfg["tools_root"])
        if warn:
            self.report({"WARNING"}, warn)
        try:
            job, ctx = pipeline.build_joined(cfg, ts, get_hw())
        except OSError as exc:
            self.report({"ERROR"}, f"Cannot create the output folder: {exc}")
            return {"CANCELLED"}
        SESSION.update(job=job, ctx=ctx, job_handled=False, import_error="", message="")
        start_job(job, ctx)
        ensure_timer()
        self.report({"INFO"}, f"Joining {scan_title(fixed)} and {scan_title(moving)} into one scan in "
                              f"{os.path.basename(cfg['project_dir'])}")
        return {"FINISHED"}


class PHOTOGRAM3D_OT_cancel(bpy.types.Operator):
    bl_idname = "photogram3d.cancel"
    bl_label = "Cancel"
    bl_description = "Stop the running reconstruction (finished steps are kept for Resume)"

    def execute(self, context):
        job = SESSION["job"]
        if job is not None:
            threading.Thread(target=job.cancel, daemon=True).start()
        return {"FINISHED"}


class PHOTOGRAM3D_OT_import(bpy.types.Operator):
    bl_idname = "photogram3d.import_result"
    bl_label = "Import Result"
    bl_description = "Import the reconstruction from the output folder"

    project_dir: StringProperty(subtype="DIR_PATH", default="")

    def execute(self, context):
        s = context.scene.photogram3d
        folder = self.project_dir or current_project(context)
        try:
            res = importer.load_result_json(folder)
            out = importer.import_result(res, import_cameras_flag=s.import_cameras,
                                         camera_photos=s.camera_photos, tracked_camera=s.video_camera)
        except (OSError, ValueError) as exc:
            self.report({"ERROR"}, f"Import failed: {exc}")
            return {"CANCELLED"}
        for text in ([out["note"]] if out.get("note") else []) + list(res.get("warnings") or []):
            self.report({"WARNING"}, text)
        for text in list(res.get("notes") or []) + ([retired_note(out.get("retired"))]
                                                    if out.get("retired") else []):
            self.report({"INFO"}, text)
        return {"FINISHED"}


class PHOTOGRAM3D_OT_open_folder(bpy.types.Operator):
    bl_idname = "photogram3d.open_folder"
    bl_label = "Open Folder"
    bl_description = "Open the output (or logs) folder in the file browser"

    which: StringProperty(default="project")
    project: StringProperty(default="", options={"SKIP_SAVE"})   # a run folder; else the panel's

    def execute(self, context):
        folder = self.project or current_project(context)
        if self.which == "logs":
            folder = os.path.join(folder, "logs")
        elif self.which == "tools":
            prefs = get_prefs(context)
            folder = prefs.resolved_tools_dir() if prefs else ""
        if not folder or not os.path.isdir(folder):
            self.report({"WARNING"}, "Folder does not exist yet")
            return {"CANCELLED"}
        bpy.ops.wm.path_open(filepath=folder)
        return {"FINISHED"}


class PHOTOGRAM3D_OT_install_tools(bpy.types.Operator):
    bl_idname = "photogram3d.install_tools"
    bl_label = "Install Tools"
    bl_description = "Download COLMAP, OpenMVS (and FFmpeg) from GitHub into the tools folder"

    def execute(self, context):
        prefs = get_prefs(context)
        if prefs is None:
            return {"CANCELLED"}
        hw = get_hw()
        want_cuda = prefs.prefer_cuda and bool(hw.get("gpus"))
        root = prefs.resolved_tools_dir()
        if has_non_ascii(root):
            self.report({"WARNING"}, "The tools folder path has non-Latin characters; "
                                     "set a plain path in the preferences if tools fail")
        SESSION["install_seen"] = False
        if not tools.start_install(SESSION["install"], root, want_cuda, prefs.include_ffmpeg):
            self.report({"WARNING"}, "Installation already running")
            return {"CANCELLED"}
        ensure_timer()
        return {"FINISHED"}


class PHOTOGRAM3D_OT_cancel_install(bpy.types.Operator):
    bl_idname = "photogram3d.cancel_install"
    bl_label = "Cancel Install"

    def execute(self, context):
        SESSION["install"].cancel.set()
        return {"FINISHED"}


class PHOTOGRAM3D_OT_check_tools(bpy.types.Operator):
    bl_idname = "photogram3d.check_tools"
    bl_label = "Check Tools"
    bl_description = "Run each tool once and report whether it starts"

    def execute(self, context):
        SESSION["verify"] = tools.verify(get_toolset(context))
        SESSION["vc_state"] = tools.msvc_runtime_status()
        bad = [v[1] for v in SESSION["verify"].values() if not v[0]]
        self.report({"WARNING"} if bad else {"INFO"},
                    "; ".join(bad) if bad else "All tools start correctly")
        return {"FINISHED"}


class PHOTOGRAM3D_OT_install_vcredist(bpy.types.Operator):
    bl_idname = "photogram3d.install_vcredist"
    bl_label = "Install Visual C++ Runtime"
    bl_description = ("Download Microsoft's latest Visual C++ v14 Redistributable and run it "
                      "(administrator prompt). COLMAP 4.2 needs a 2026 runtime")

    def execute(self, context):
        prefs = get_prefs(context)
        dl = os.path.join(prefs.resolved_tools_dir(), "_downloads")

        def work():
            try:
                SESSION["message"] = "Downloading Visual C++ runtime..."
                tools.launch_vc_redist_installer(dl)
                SESSION["message"] = "Visual C++ installer started; press Check Tools when it finishes"
            except Exception as exc:  # noqa: BLE001
                SESSION["message"] = f"Visual C++ runtime: {exc}"

        threading.Thread(target=work, daemon=True).start()
        return {"FINISHED"}


class PHOTOGRAM3D_OT_detect_hardware(bpy.types.Operator):
    bl_idname = "photogram3d.detect_hardware"
    bl_label = "Detect Hardware"

    def execute(self, context):
        hw = get_hw(refresh=True)
        context.scene.photogram3d.quality = hardware.recommend_quality(hw)
        self.report({"INFO"}, hardware.describe(hw))
        return {"FINISHED"}


def _selected_mesh(context):
    obj = context.active_object
    if obj is None or obj.type != "MESH":
        return None
    return obj


class PHOTOGRAM3D_OT_set_scale(bpy.types.Operator):
    bl_idname = "photogram3d.set_scale"
    bl_label = "Set Real Scale"
    bl_description = ("Select exactly two vertices on the scan (Edit Mode) and enter their real "
                      "distance; the whole reconstruction is scaled")
    bl_options = {"REGISTER", "UNDO"}

    distance: FloatProperty(name="Real Distance", default=1.0, min=1e-6, unit="LENGTH",
                            description="Measured distance between the two vertices")

    @classmethod
    def poll(cls, context):
        obj = _selected_mesh(context)
        return obj is not None and context.mode == "EDIT_MESH"

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        import bmesh
        obj = _selected_mesh(context)
        root = importer.find_root(obj)
        if root is None:
            self.report({"ERROR"}, "The mesh is not a Photogram3D import")
            return {"CANCELLED"}
        bm = bmesh.from_edit_mesh(obj.data)
        sel = [v for v in bm.verts if v.select]
        if len(sel) != 2:
            self.report({"ERROR"}, f"Select exactly 2 vertices (selected: {len(sel)})")
            return {"CANCELLED"}
        a = obj.matrix_world @ sel[0].co
        b = obj.matrix_world @ sel[1].co
        current = (a - b).length
        if current <= 1e-12:
            self.report({"ERROR"}, "The two vertices are at the same position")
            return {"CANCELLED"}
        factor = self.distance / current
        importer.transform_root(root, Matrix.Scale(factor, 4))
        self.report({"INFO"}, f"Scaled by {factor:.4f}")
        return {"FINISHED"}


def mesh_arrays(mesh):
    mesh.calc_loop_triangles()
    nv = len(mesh.vertices)
    verts = np.empty(nv * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", verts)
    tris = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int64)
    mesh.loop_triangles.foreach_get("vertices", tris)
    return verts.reshape(-1, 3), tris.reshape(-1, 3)


class PHOTOGRAM3D_OT_polish(bpy.types.Operator):
    bl_idname = "photogram3d.polish"
    bl_label = "Polish Surface"
    bl_description = ("Remove reconstruction noise while keeping creases "
                      "(feature-preserving normal filtering). Works on the active mesh")
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return _selected_mesh(context) is not None and context.mode == "OBJECT"

    def execute(self, context):
        s = context.scene.photogram3d
        obj = _selected_mesh(context)
        if obj.data.shape_keys is not None:
            self.report({"ERROR"}, "The mesh has shape keys (e.g. an AI Polish result): "
                                   "apply or discard them first")
            return {"CANCELLED"}
        t0 = time.time()
        verts, tris = mesh_arrays(obj.data)
        if len(tris) == 0:
            self.report({"ERROR"}, "Mesh has no faces")
            return {"CANCELLED"}
        wm = context.window_manager
        wm.progress_begin(0, 100)
        try:
            new = polish.polish(verts, tris, iterations=s.polish_iterations,
                                keep_edges=s.polish_keep_edges, strength=s.polish_strength,
                                progress=lambda f: wm.progress_update(int(f * 100)))
        finally:
            wm.progress_end()
        obj.data.vertices.foreach_set("co", new.astype(np.float32).ravel())
        obj.data.update()
        self.report({"INFO"}, f"Polished {len(verts):,} vertices in {time.time() - t0:.1f} s")
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# AI polish
# ---------------------------------------------------------------------------

def scan_sources(obj):
    """(root, project_dir, sparse_model_dir, undistorted_images_dir) of a Photogram3D scan."""
    root = importer.find_root(obj)
    if root is None:
        raise ValueError("The active mesh is not part of a Photogram3D import")
    project = root.get(importer.ROOT_PROP, "")
    paths = pipeline.project_paths(project)
    model_dir = os.path.join(paths["dense"], "sparse")
    images_dir = os.path.join(paths["dense"], "images")
    try:
        res = importer.load_result_json(project)
        if os.path.isdir(res.get("model") or "") and os.path.isdir(res.get("images") or ""):
            model_dir, images_dir = res["model"], res["images"]
    except (OSError, ValueError):
        pass
    if not (os.path.isdir(model_dir) and os.path.isdir(images_dir)):
        raise ValueError(f"Project folder not found or incomplete: {project}")
    return root, project, model_dir, images_dir


def build_ai_spec(context, obj):
    """Everything the AI worker needs, captured on the main thread."""
    root, project, model_dir, images_dir = scan_sources(obj)
    prefs = get_prefs(context)
    s = context.scene.photogram3d
    verts, tris = mesh_arrays(obj.data)
    if len(tris) == 0:
        raise ValueError("The mesh has no faces")
    to_colmap = np.array(root.matrix_world.inverted() @ obj.matrix_world)
    verts_c = verts @ to_colmap[:3, :3].T + to_colmap[:3, 3]
    return {
        "object_name": obj.name, "project": project, "model_dir": model_dir,
        "images_dir": images_dir, "python": sys.executable,
        "tools_root": prefs.resolved_tools_dir() if prefs else "",
        "model_key": prefs.ai_model if prefs else "SMALL",
        "device": "cpu" if prefs and prefs.ai_device == "CPU" else "auto",
        "tokens": polish_ai.DETAIL_TOKENS[s.ai_detail], "max_views": s.ai_views,
        "influence": s.ai_influence, "verts": verts_c, "faces": tris, "to_colmap": to_colmap,
    }


def set_ai_shape_key(obj, coords):
    if obj.data.shape_keys is None:
        obj.shape_key_add(name="Basis", from_mix=False)
    kb = obj.data.shape_keys.key_blocks.get(polish_ai.AI_KEY)
    if kb is None:
        kb = obj.shape_key_add(name=polish_ai.AI_KEY, from_mix=False)
    kb.data.foreach_set("co", np.asarray(coords, dtype=np.float32).ravel())
    kb.value = 1.0
    obj.active_shape_key_index = obj.data.shape_keys.key_blocks.find(polish_ai.AI_KEY)
    obj.data.update()
    return kb


def finish_ai_job():
    """Main thread: turn a finished AI job into a shape key on the scan."""
    job, ctx = SESSION["ai_job"], SESSION["ai_ctx"]
    snap = job.snapshot()
    if snap["status"] != "done":
        SESSION["ai_message"] = snap["error"] or snap["status"]
        return False
    obj = bpy.data.objects.get(ctx["object_name"])
    if obj is None or obj.type != "MESH" or len(obj.data.vertices) != len(ctx["verts"]):
        SESSION["ai_message"] = "The scan changed while AI Polish was running; nothing applied"
        return False
    inv = np.linalg.inv(ctx["to_colmap"])
    local = ctx["new_verts"] @ inv[:3, :3].T + inv[:3, 3]
    set_ai_shape_key(obj, local)
    res = snap["result"]
    size = float(np.linalg.norm(ctx["verts"].max(0) - ctx["verts"].min(0))) or 1.0
    SESSION["ai_message"] = (
        f"Done in {human_time(snap['elapsed'])}: {res.get('valid', 0) * 100:.0f}% of vertices guided "
        f"by {res.get('views_used', 0)} photos; they moved {res.get('moved_median', 0) / size * 100:.3g}% "
        f"of the scan size (median), {res.get('moved_p99', 0) / size * 100:.3g}% at most (99%). "
        f"Compare with the Result Mix slider, then Apply or Discard")
    return True


class PHOTOGRAM3D_OT_ai_polish(bpy.types.Operator):
    bl_idname = "photogram3d.ai_polish"
    bl_label = "AI Polish"
    bl_description = ("Predict surface normals in the photos with a local AI model (MoGe-2), fuse "
                      "them on the scan and move vertices to follow them. Runs in the background; "
                      "the result is added as a shape key. First use downloads the runtime and model")

    @classmethod
    def poll(cls, context):
        obj = _selected_mesh(context)
        return (obj is not None and context.mode == "OBJECT"
                and importer.find_root(obj) is not None)

    def execute(self, context):
        if job_running("job") or job_running("ai_job"):
            self.report({"WARNING"}, "A reconstruction or AI polish is already running")
            return {"CANCELLED"}
        obj = _selected_mesh(context)
        try:
            spec = build_ai_spec(context, obj)
            job, ctx = polish_ai.build_job(spec)
        except (OSError, ValueError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        SESSION.update(ai_job=job, ai_ctx=ctx, ai_handled=False, ai_message="", ai_ok=None)
        start_job(job, ctx)
        ensure_timer()
        return {"FINISHED"}


class PHOTOGRAM3D_OT_ai_cancel(bpy.types.Operator):
    bl_idname = "photogram3d.ai_cancel"
    bl_label = "Cancel"
    bl_description = "Stop AI Polish (cached normal maps are kept)"

    def execute(self, context):
        job = SESSION["ai_job"]
        if job is not None:
            threading.Thread(target=job.cancel, daemon=True).start()
        return {"FINISHED"}


class PHOTOGRAM3D_OT_ai_result(bpy.types.Operator):
    bl_idname = "photogram3d.ai_result"
    bl_label = "AI Polish Result"
    bl_options = {"REGISTER", "UNDO"}

    action: EnumProperty(items=(("APPLY", "Apply", "Bake the AI Polish shape key into the mesh"),
                                ("DISCARD", "Discard", "Remove the AI Polish shape key")))

    @classmethod
    def description(cls, context, props):
        return ("Bake the AI Polish result (at the current slider value) into the mesh"
                if props.action == "APPLY" else "Remove the AI Polish result")

    @classmethod
    def poll(cls, context):
        obj = _selected_mesh(context)
        keys = obj.data.shape_keys if obj is not None else None
        return (keys is not None and context.mode == "OBJECT"
                and keys.key_blocks.get(polish_ai.AI_KEY) is not None)

    def execute(self, context):
        obj = _selected_mesh(context)
        keys = obj.data.shape_keys
        kb = keys.key_blocks[polish_ai.AI_KEY]
        others = [k for k in keys.key_blocks if k != kb and k != keys.reference_key]
        if self.action == "DISCARD":
            obj.shape_key_remove(kb)
            if not others:
                obj.shape_key_clear()
            obj.data.update()
            return {"FINISHED"}
        if others:
            self.report({"ERROR"}, "The mesh has other shape keys; apply the AI key manually")
            return {"CANCELLED"}
        n = len(obj.data.vertices)
        base = np.empty(n * 3, dtype=np.float32)
        tgt = np.empty(n * 3, dtype=np.float32)
        keys.reference_key.data.foreach_get("co", base)
        kb.data.foreach_get("co", tgt)
        mix = base + kb.value * (tgt - base)
        obj.shape_key_clear()
        obj.data.vertices.foreach_set("co", mix)
        obj.data.update()
        return {"FINISHED"}


CLASSES = (
    PHOTOGRAM3D_OT_run, PHOTOGRAM3D_OT_cancel, PHOTOGRAM3D_OT_import, PHOTOGRAM3D_OT_open_folder,
    PHOTOGRAM3D_OT_install_tools, PHOTOGRAM3D_OT_cancel_install, PHOTOGRAM3D_OT_check_tools,
    PHOTOGRAM3D_OT_install_vcredist, PHOTOGRAM3D_OT_detect_hardware, PHOTOGRAM3D_OT_set_scale,
    PHOTOGRAM3D_OT_polish, PHOTOGRAM3D_OT_ai_polish, PHOTOGRAM3D_OT_ai_cancel, PHOTOGRAM3D_OT_ai_result,
    PHOTOGRAM3D_OT_build_mesh, PHOTOGRAM3D_OT_build_uncropped, PHOTOGRAM3D_OT_reset_crop_box,
    PHOTOGRAM3D_OT_place_marker, PHOTOGRAM3D_OT_align_ground, PHOTOGRAM3D_OT_marker_scale,
    PHOTOGRAM3D_OT_join_scans, PHOTOGRAM3D_OT_refine_join, PHOTOGRAM3D_OT_complete_join,
    PHOTOGRAM3D_OT_reset_alignment, PHOTOGRAM3D_OT_resolve_without, PHOTOGRAM3D_OT_resolve_from_cameras,
    PHOTOGRAM3D_OT_upright_cameras,
    PHOTOGRAM3D_OT_track_camera, PHOTOGRAM3D_OT_add_video_cameras,
    PHOTOGRAM3D_OT_ai_test, PHOTOGRAM3D_OT_ai_nvidia, PHOTOGRAM3D_OT_ai_nvidia_remove,
    PHOTOGRAM3D_OT_ai_test_cancel,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for key in ("job", "ai_job", "ai_test_job", "nvidia_job"):
        job = SESSION.get(key)
        if job is not None:
            job.cancel()
    SESSION["install"].cancel.set()
    if bpy.app.timers.is_registered(_timer):
        bpy.app.timers.unregister(_timer)
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
