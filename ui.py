# SPDX-License-Identifier: GPL-3.0-or-later
# Photogram3D - photos or video to a textured mesh with COLMAP + OpenMVS
# Copyright (C) 2026 Kruglov Iurii
# Authors: Kruglov Iurii, Anthropic Claude AI
# License: GNU General Public License (GPL) v3 or later, https://www.blender.org/about/license/
"""Sidebar panels (3D View > N panel > Photogram3D) and preferences UI."""

import os

import bpy

from . import ai_backend, formats, hardware, pipeline, polish_ai, tools, video
from .common import IS_WINDOWS, LEGACY_NAME, cloud_sync_service, human_bytes, human_time, legacy_path
from .operators import (SESSION, active_root, crop_root, current_project, get_hw, get_toolset,
                        selected_photo_cameras, video_info)
from .props import get_prefs


def _tools_line(ts):
    parts = []
    if ts.get("colmap"):
        v = ts["versions"].get("colmap") or ""
        var = ts.get("colmap_variant")
        parts.append(f"COLMAP {v}" + (f" ({var.upper()})" if var else ""))
    if ts.get("openmvs_dir"):
        parts.append("OpenMVS " + (ts["versions"].get("openmvs") or ""))
    if ts.get("ffmpeg"):
        parts.append("FFmpeg")
    return " · ".join(p.strip() for p in parts)


def draw_legacy_tools_warning(layout):
    """The tools are still in PhotoMesh's folder (moving them was refused)."""
    from . import legacy
    if not legacy.tools_left_behind():
        return
    col = layout.column(align=True)
    text = ("The tools are still in PhotoMesh's folder: uninstalling PhotoMesh would delete them. "
            "Close the programs using them and restart Blender before uninstalling PhotoMesh.")
    for i, line in enumerate(_wrap(text, 48)[:4]):
        col.label(text=line, icon="ERROR" if i == 0 else "BLANK1")


def draw_tools_box(layout, context, compact=True):
    ts = get_toolset(context)
    st = SESSION["install"]
    box = layout.box()
    ready = bool(ts.get("colmap") and ts.get("openmvs_dir"))
    if st.running:
        box.label(text="Installing tools...", icon="IMPORT")
        box.progress(factor=st.fraction, type="BAR", text=st.message[:60] or "Working...")
        box.operator("photogram3d.cancel_install", icon="CANCEL")
        return ready
    if ready:
        box.label(text=_tools_line(ts), icon="CHECKMARK")
    else:
        box.label(text="COLMAP / OpenMVS not installed", icon="ERROR")
        row = box.row()
        row.scale_y = 1.3
        row.operator("photogram3d.install_tools", icon="IMPORT")
        if not IS_WINDOWS:
            box.label(text="Linux/macOS: set paths in Preferences", icon="INFO")
    if st.error:
        col = box.column(align=True)
        for i, line in enumerate(_wrap(st.error, 48)[:4]):
            col.label(text=line, icon="ERROR" if i == 0 else "BLANK1")
    draw_legacy_tools_warning(box)
    if IS_WINDOWS and (st.error or not compact):
        state, text = SESSION["vc_state"] or tools.msvc_runtime_status()
        SESSION["vc_state"] = (state, text)
        if state in ("missing", "old"):
            row = box.row()
            row.label(text=text, icon="ERROR" if state == "missing" else "INFO")
            box.operator("photogram3d.install_vcredist", icon="IMPORT")
    if SESSION["message"] and "Visual C++" in SESSION["message"]:
        box.label(text=SESSION["message"][:70], icon="INFO")
    return ready


_PLAN_CACHE = {}


def photo_plan(folder, prefer_raw):
    """formats.plan for the panel, read again only when the folder changes (the panel
    redraws twice a second while a run works)."""
    try:
        key = (folder, os.stat(folder).st_mtime_ns, prefer_raw)
    except OSError:
        return {"use": [], "convert": [], "skipped": []}
    if key not in _PLAN_CACHE:
        _PLAN_CACHE.clear()
        _PLAN_CACHE[key] = formats.plan(folder, prefer_raw)
    return _PLAN_CACHE[key]


def _wrap(text, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines or [""]


class PHOTOGRAM3D_PT_main(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Photogram3D"
    bl_label = "Photogram3D"

    def draw(self, context):
        layout = self.layout
        s = context.scene.photogram3d
        draw_tools_box(layout, context)

        col = layout.column()
        col.row().prop(s, "input_mode", expand=True)
        if s.input_mode == "IMAGES":
            col.prop(s, "images_dir", text="")
            if s.images_dir:
                pl = photo_plan(bpy.path.abspath(s.images_dir), s.raw_pairs == "RAW")
                n = len(pl["use"])
                text = f"{n} photos found"
                if pl["convert"]:
                    text += f", {len(pl['convert'])} converted to JPEG first"
                col.label(text=text, icon="IMAGE_DATA" if n >= 3 else "ERROR")
                if pl["skipped"]:
                    row = col.row()
                    row.label(text=f"RAW + JPEG of one shot: {len(pl['skipped'])} twins skipped",
                              icon="BLANK1")
                    row.prop(s, "raw_pairs", text="")
        else:
            col.prop(s, "video_path", text="")
            draw_video_options(col, context)
        col.separator()
        col.label(text="Output folder (empty = next to input):")
        col.prop(s, "project_dir", text="")
        out = s.resolved_project_dir()
        if out and not s.project_dir:
            col.label(text=out[-48:], icon="FILE_FOLDER")
        if out:
            done = pipeline.attempts(out)
            if done:
                folder, k = pipeline.choose_attempt(out, s.resume)
                text = (f"Reconstruct resumes run {k} ({os.path.basename(folder)})" if s.resume else
                        f"Reconstruct starts run {k} in {os.path.basename(folder)}")
                col.label(text=text, icon="FILE_REFRESH" if s.resume else "ADD")
        service = cloud_sync_service(out) if out else ""
        if service:
            col.label(text=f"Inside {service}: pause syncing while it runs,", icon="ERROR")
            col.label(text="or choose a local output folder", icon="BLANK1")

        col.separator()
        col.row().prop(s, "quality", expand=True)
        hw = get_hw()
        row = col.row(align=True)
        row.label(text=hardware.describe(hw), icon="DESKTOP")
        row.operator("photogram3d.detect_hardware", text="", icon="FILE_REFRESH")


class PHOTOGRAM3D_PT_options(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Photogram3D"
    bl_label = "Options"
    bl_parent_id = "PHOTOGRAM3D_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        s = context.scene.photogram3d
        col = layout.column(heading="COLMAP")
        col.prop(s, "use_gpu")
        col.prop(s, "ai_features", text="AI Features")
        if s.ai_features:
            draw_ai_device_line(col, context)
        col.prop(s, "mapper")
        col.prop(s, "camera_model")
        col.prop(s, "single_camera", text="One Camera")
        col.prop(s, "align_up")
        col = layout.column(heading="Dense")
        col.prop(s, "dense_engine")
        if s.dense_engine in pipeline.DENSE_ENGINE_NOTES:
            why = pipeline.gpu_depth_unavailable({"use_gpu": s.use_gpu}, get_toolset(context), get_hw())
            text = why or pipeline.DENSE_ENGINE_NOTES[s.dense_engine]
            sub = col.column(align=True)
            sub.scale_y = 0.8
            for i, line in enumerate(_wrap(text, 38)[:4]):
                sub.label(text=line, icon=("ERROR" if why else "EXPERIMENTAL") if i == 0 else "BLANK1")
        col.prop(s, "dense_region")
        sub = col.column()
        sub.active = s.dense_region == "AUTO"
        sub.prop(s, "dense_region_size")
        col.prop(s, "refine")
        col.prop(s, "free_space")
        col.prop(s, "target_faces")
        col = layout.column(heading="Texture")
        col.prop(s, "texture", text="Enabled")
        sub = col.column()
        sub.active = s.texture
        sub.prop(s, "texture_size")
        sub.prop(s, "blend_seams")
        col.prop(s, "spread_views")
        col = layout.column(heading="Photos")
        col.prop(s, "even_photos", text="Even Out")
        sub = col.column()
        sub.active = s.even_photos
        sub.prop(s, "sharpen")
        col.prop(s, "photo_max_size")
        col = layout.column(heading="Splats")
        col.prop(s, "splats", text="Gaussian Splats")
        sub = col.column(align=True)
        sub.active = s.splats
        sub.prop(s, "splat_steps", text="Steps")
        sub.prop(s, "splat_resolution", text="Photo Size")
        col = layout.column(heading="Run")
        col.prop(s, "resume")
        col.prop(s, "crop_stage")
        col.prop(s, "auto_import")
        col.prop(s, "import_cameras")
        sub = col.column()
        sub.active = s.import_cameras
        sub.prop(s, "camera_photos")


class PHOTOGRAM3D_PT_run(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Photogram3D"
    bl_label = "Run"
    bl_parent_id = "PHOTOGRAM3D_PT_main"
    bl_options = {"HIDE_HEADER"}

    def draw(self, context):
        layout = self.layout
        job = SESSION["job"]
        snap = job.snapshot() if job is not None else None
        running = snap is not None and snap["status"] in ("running", "queued")
        if running:
            box = layout.box()
            box.label(text=f"Step {snap['step_index'] + 1}/{snap['step_count']}: {snap['step_label']}",
                      icon="TIME")
            box.progress(factor=snap["step_fraction"], type="BAR",
                         text=snap["step_note"][:50] or "working...")
            box.progress(factor=snap["overall"], type="BAR",
                         text=f"Total {snap['overall'] * 100:.0f}% · {human_time(snap['elapsed'])}")
            col = box.column(align=True)
            col.scale_y = 0.75
            for line in snap["lines"][-6:]:
                col.label(text=line[-70:])
            row = box.row()
            row.operator("photogram3d.cancel", icon="CANCEL")
            op = row.operator("photogram3d.open_folder", text="Logs", icon="TEXT")
            op.which, op.project = "logs", job.project_dir      # this run's logs
            return
        if crop_root(context) is not None:
            draw_crop_box(layout, context)
        elif active_root(context) is not None:
            draw_align_box(layout, context)
        row = layout.row()
        row.scale_y = 1.6
        row.operator("photogram3d.run", icon="PLAY")
        if snap is not None:
            box = layout.box()
            status = snap["status"]
            icon = {"done": "CHECKMARK", "failed": "ERROR", "cancelled": "CANCEL"}.get(status, "INFO")
            msg = SESSION["message"] or status
            if status == "failed" and snap.get("failed_step"):
                box.label(text=f"Stopped at: {snap['failed_step']}", icon="ERROR")
                icon = "BLANK1"
            for i, line in enumerate(_wrap(msg, 46)[:5]):
                box.label(text=line, icon=icon if i == 0 else "BLANK1")
            if status == "failed" and snap.get("hint"):
                col = box.column(align=True)
                for i, line in enumerate(_wrap(snap["hint"], 46)[:7]):
                    col.label(text=line, icon="QUESTION" if i == 0 else "BLANK1")
            if status == "done":
                for kind, icon_first in (("warnings", "ERROR"), ("notes", "INFO")):
                    for text in snap["result"].get(kind) or []:
                        for i, line in enumerate(_wrap(text, 46)[:4]):
                            box.label(text=line, icon=icon_first if i == 0 else "BLANK1")
            if SESSION["import_error"]:
                box.label(text="Import: " + SESSION["import_error"][:60], icon="ERROR")
            if status == "done" and snap["result"].get("registered") is not None:
                box.label(text=f"{snap['result'].get('registered')}/{snap['result'].get('num_images')} "
                               "images used", icon="CAMERA_DATA")
            row = box.row()
            for text, which, icon in (("Output", "project", "FILE_FOLDER"), ("Logs", "logs", "TEXT")):
                op = row.operator("photogram3d.open_folder", text=text, icon=icon)
                op.which, op.project = which, job.project_dir   # the run this box reports on
        out = current_project(context)
        result = os.path.join(out, "photogram3d_result.json") if out else ""
        if result and (os.path.isfile(result) or os.path.isfile(legacy_path(result))):
            layout.operator("photogram3d.import_result", icon="IMPORT")


class PHOTOGRAM3D_PT_tools(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Photogram3D"
    bl_label = "Scan Tools"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        s = context.scene.photogram3d
        col = layout.column(align=True)
        col.label(text="Polish (active mesh, Object Mode):")
        col.prop(s, "polish_iterations")
        col.prop(s, "polish_strength", slider=True)
        col.prop(s, "polish_keep_edges", slider=True)
        col.operator("photogram3d.polish", icon="MOD_SMOOTH")
        layout.separator()
        draw_ai_box(layout, context)
        layout.separator()
        col = layout.column(align=True)
        col.label(text="Scale from 2 vertices (Edit Mode):")
        col.operator("photogram3d.set_scale", icon="DRIVER_DISTANCE")


def draw_video_options(col, context):
    import math
    s = context.scene.photogram3d
    info = video_info(context)
    if info and not info.get("error"):
        col.label(text=f"{info['display_width']}x{info['display_height']}, {info['fps']:.4g} fps, "
                       f"{info['duration']:.1f} s, {info['codec']}", icon="FILE_MOVIE")
    elif info:
        for i, line in enumerate(_wrap(info["error"], 46)[:3]):
            col.label(text=line, icon="ERROR" if i == 0 else "BLANK1")
    sub = col.column(align=True)
    row = sub.row(align=True)
    row.prop(s, "video_start")
    row.prop(s, "video_end")
    sub.row(align=True).prop(s, "video_frames", expand=True)
    if s.video_frames != "ALL":
        row = sub.row(align=True)
        if s.video_frames == "AUTO":
            row.prop(s, "video_max_frames")
        else:
            row.prop(s, "video_fps", text="Frames/s")
        row.prop(s, "video_blur_filter", text="Sharpest", toggle=True)
    sub.prop(s, "video_camera")
    row = sub.row(align=True)
    row.active = s.video_camera
    row.prop(s, "video_track", expand=True)
    if info and info.get("duration"):
        end = min(s.video_end or info["duration"], info["duration"])
        span = max(0.0, end - s.video_start)
        total = span * info["fps"]
        if s.video_frames == "AUTO":
            lo, hi = min(total, video.AUTO_MIN_FRAMES), min(total, s.video_max_frames)
            col.label(text=f"{int(lo)} to {int(hi):,} photos from {span:.1f} s of video",
                      icon="ERROR" if hi < 3 else "IMAGE_DATA")
            col.label(text="spaced by the camera's movement", icon="BLANK1")
        else:
            n = total if s.video_frames == "ALL" else min(total, max(20, math.ceil(span * s.video_fps)))
            col.label(text=f"About {int(n):,} photos from {span:.1f} s of video",
                      icon="ERROR" if n > 1500 or n < 3 else "IMAGE_DATA")


def scan_label(root):
    """'Gas_mask, run 2' for the root of the second run on the Gas_mask photos."""
    if root is None:
        return "-"
    k = int(root.get("photogram3d_attempt") or 1)
    coll = root.users_collection[0].name if root.users_collection else root.name
    run = coll.startswith(("Photogram3D_", LEGACY_NAME + "_")) and " " in coll   # PhotoMesh_2 Gas_mask: 0.3.6
    name = coll.split(" ", 1)[1] if run else root.name
    return f"{name}, run {k}"


def draw_crop_box(layout, context):
    root = crop_root(context)
    box = layout.box()
    box.label(text=f"Scan: {scan_label(root)}", icon="OUTLINER_OB_EMPTY")
    box.label(text="1. Align and scale (optional)", icon="ORIENTATION_GLOBAL")
    draw_scene_setup(box, context)
    draw_join(layout, context, root)
    box = layout.box()
    box.label(text="2. Crop box", icon="MESH_CUBE")
    col = box.column(align=True)
    col.scale_y = 0.8
    for line in ("Select the box, then G / R / S to move,", "rotate and scale it around the object.",
                 "Only what is inside is meshed and textured."):
        col.label(text=line)
    row = box.row()
    row.scale_y = 1.4
    row.operator("photogram3d.build_mesh", icon="MOD_REMESH")
    row = box.row(align=True)
    row.operator("photogram3d.build_uncropped", icon="FULLSCREEN_ENTER")
    row.operator("photogram3d.reset_crop_box", icon="LOOP_BACK")
    draw_upright(layout, root)
    draw_track(layout, root)
    draw_resolve(layout, context)


def draw_align_box(layout, context):
    """After the import: align and scale the finished scan again (nothing is recomputed)."""
    root = active_root(context)
    if root is None:
        return
    box = layout.box()
    row = box.row()
    row.label(text=f"Align and scale: {scan_label(root)}", icon="ORIENTATION_GLOBAL")
    row.operator("photogram3d.reset_alignment", text="", icon="LOOP_BACK")
    draw_scene_setup(box, context)
    draw_join(layout, context, root)
    draw_upright(layout, root)
    draw_track(layout, root)
    draw_resolve(layout, context)


def draw_join(layout, context, root):
    """Join the active scan onto another one by control points placed on both (0.3.13)."""
    from . import importer, operators
    if root is None or len(operators.scan_roots(context)) < 2:
        return
    s = context.scene.photogram3d
    box = layout.box()
    box.label(text="Join onto another scan", icon="AUTOMERGE_ON")
    box.prop(s, "join_target")
    target = operators.join_target(context)
    here = {o[importer.MARKER_PROP] for o in importer._children(root, importer.MARKER_PROP)}
    there = ({o[importer.MARKER_PROP] for o in importer._children(target, importer.MARKER_PROP)}
             if target is not None else set())
    col = box.column(align=True)
    col.label(text="The same spots on both scans, 3 or more:", icon="PIVOT_CURSOR")
    for half in (importer.JOIN_SLOTS[:4], importer.JOIN_SLOTS[4:]):
        row = col.row(align=True)
        for slot in half:
            icon = "CHECKMARK" if slot in here and slot in there else ("DOT" if slot in here else "NONE")
            row.operator("photogram3d.place_marker", text=slot[1:], icon=icon).slot = slot
    row = box.row(align=True)
    row.operator("photogram3d.join_scans", icon="AUTOMERGE_ON")
    row.operator("photogram3d.refine_join", icon="MOD_SHRINKWRAP")
    rec = importer.join_record(root)
    if not rec or target is None or rec.get("onto") != target.name:
        return
    col = box.column(align=True)
    col.scale_y = 0.8
    for slot, r in zip(rec["points"], rec["residuals"]):
        col.label(text=f"{importer.MARKER_NAMES[slot]}: {operators._length(context, r, rec['size'])} off",
                  icon="ERROR" if slot in rec["suspect"] else "BLANK1")
    if rec["suspect"]:
        col.label(text="Points marked are far off the others: place them again", icon="INFO")
    if rec.get("narrow"):
        col.label(text="The points are almost on one line: the turn about it is uncertain.", icon="INFO")
        col.label(text="Add one away from that line, or Refine", icon="BLANK1")
    refined = rec.get("refined")
    if refined and refined.get("kept", True):
        col.label(text=f"Refined on {100 * refined['overlap']:.0f}% of the cloud: "
                       f"{operators._length(context, refined['rms'], rec['size'])}", icon="CHECKMARK")
    elif refined:
        col.label(text="Refine: the clouds overlap too little, the points' join is kept", icon="INFO")
    col = box.column(align=True)
    row = col.row()
    row.scale_y = 1.3
    row.operator("photogram3d.complete_join", icon="CHECKMARK")
    sub = col.column(align=True)
    sub.scale_y = 0.8
    sub.label(text="One scan of both: one cloud, both cameras,", icon="INFO")
    sub.label(text="a crop box. Then Build Mesh on it", icon="BLANK1")


def draw_upright(layout, root):
    """Cameras of an earlier import that show portrait photos on their side (before 0.3.9)."""
    from . import importer
    if root is None:
        return
    sideways = importer.sideways_cameras(root, importer.root_turns(root))
    if not sideways:
        return
    box = layout.box()
    box.label(text=f"{len(sideways)} camera(s) show their photo on its side", icon="ORIENTATION_GIMBAL")
    box.operator("photogram3d.upright_cameras", icon="CON_ROTLIKE")


def draw_track(layout, root):
    """The scan's tracked camera: keyed on every frame, or on the solved frames only. A
    joined scan has one for each video (0.3.15): the selected one, else the first."""
    from . import importer, tracking
    if root is None:
        return
    cams = tracking.tracked_cameras(root)
    if not cams:
        # imported without them, or a scan joined before 0.3.15: they can be added
        n = importer.scan_video_count(root)
        if n:
            box = layout.box()
            box.label(text=f"No camera follows the video{'s' if n > 1 else ''} of this scan", icon="VIEW_CAMERA")
            box.operator("photogram3d.add_video_cameras", icon="CAMERA_DATA",
                         text="Add Video Cameras" if n > 1 else "Add Video Camera")
        return
    active = bpy.context.active_object
    cam = tracking.tracked_camera(root, active)
    kind, n = tracking.track_state(root, active)
    box = layout.box()
    who = f"{cam.name}: " if len(cams) > 1 else "Tracked camera "
    if kind == "every":
        box.label(text=f"{who}follows the video, {n:,} frames" if len(cams) > 1
                  else f"Tracked camera follows the video: {n:,} frames", icon="CHECKMARK")
    else:
        box.label(text=f"{who}{n} solved frames, interpolated" if len(cams) > 1
                  else f"Tracked camera: {n} solved frames, interpolated", icon="VIEW_CAMERA")
    if len(cams) > 1:
        sub = box.column(align=True)
        sub.scale_y = 0.8
        sub.label(text=f"{len(cams)} video cameras: select one to see or track it", icon="INFO")
    if kind != "every":
        box.operator("photogram3d.track_camera", icon="TRACKING")


def draw_resolve(layout, context):
    """Misplaced cameras selected: offer to solve again without them, or along the cameras as
    turned by hand."""
    root, names = selected_photo_cameras(context)
    if root is None:
        return
    box = layout.box()
    box.label(text=f"{len(names)} photo camera(s) selected", icon="CAMERA_DATA")
    box.operator("photogram3d.resolve_without", text=f"Re-solve Without {len(names)} Selected",
                 icon="FILE_REFRESH")
    # misplaced cameras turned by hand (R Z 180 for a side laid over the other side): solve again
    # along them (0.3.11)
    box.operator("photogram3d.resolve_from_cameras", icon="ORIENTATION_VIEW")


def draw_scene_setup(layout, context):
    """Ground alignment and real-world scale from markers placed at the 3D cursor."""
    s = context.scene.photogram3d
    col = layout.column(align=True)
    col.label(text="Shift+right-click the surface, then place:", icon="PIVOT_CURSOR")
    row = col.row(align=True)
    for slot, label in (("G1", "Ground 1"), ("G2", "2"), ("G3", "3")):
        row.operator("photogram3d.place_marker", text=label).slot = slot
    row = col.row(align=True)
    row.operator("photogram3d.align_ground", icon="AXIS_TOP")
    row.prop(s, "ground_axis", text="")
    row = col.row(align=True)
    row.operator("photogram3d.place_marker", text="Anchor A").slot = "A"
    row.operator("photogram3d.place_marker", text="Anchor B").slot = "B"
    row = col.row(align=True)
    row.prop(s, "real_distance", text="Distance")
    row.operator("photogram3d.marker_scale", icon="DRIVER_DISTANCE")


def draw_ai_box(layout, context):
    s = context.scene.photogram3d
    box = layout.box()
    box.label(text="AI Polish (normals from the photos)", icon="SHADERFX")
    job = SESSION["ai_job"]
    snap = job.snapshot() if job is not None else None
    if snap is not None and snap["status"] in ("running", "queued"):
        box.label(text=f"Step {snap['step_index'] + 1}/{snap['step_count']}: {snap['step_label']}",
                  icon="TIME")
        box.progress(factor=snap["step_fraction"], type="BAR",
                     text=snap["step_note"][:50] or "working...")
        col = box.column(align=True)
        col.scale_y = 0.75
        for line in snap["lines"][-4:]:
            col.label(text=line[-70:])
        box.operator("photogram3d.ai_cancel", icon="CANCEL")
        return
    col = box.column(align=True)
    col.prop(s, "ai_influence", slider=True)
    col.prop(s, "ai_views")
    col.row().prop(s, "ai_detail", expand=True)
    row = box.row()
    row.scale_y = 1.3
    row.operator("photogram3d.ai_polish", icon="SHADERFX")
    prefs = get_prefs(context)
    if prefs is not None:
        paths = polish_ai.ai_paths(prefs.resolved_tools_dir())
        need = []
        if polish_ai.runtime_info(paths["site"]) is None:
            need.append("AI runtime ~32 MB")
        if polish_ai.model_file(paths["models"], prefs.ai_model) is None:
            need.append(polish_ai.MODELS[prefs.ai_model]["label"].split("(")[-1].rstrip(")"))
        if need:
            box.label(text="First run downloads " + " + ".join(need), icon="INFO")
    obj = context.active_object
    keys = obj.data.shape_keys if obj is not None and obj.type == "MESH" else None
    kb = keys.key_blocks.get(polish_ai.AI_KEY) if keys is not None else None
    if kb is not None:
        box.prop(kb, "value", text="Result Mix", slider=True)
        row = box.row(align=True)
        row.operator("photogram3d.ai_result", text="Apply", icon="CHECKMARK").action = "APPLY"
        row.operator("photogram3d.ai_result", text="Discard", icon="X").action = "DISCARD"
    if SESSION["ai_message"]:
        icon = "CHECKMARK" if SESSION.get("ai_ok") else "ERROR"
        col = box.column(align=True)
        for i, line in enumerate(_wrap(SESSION["ai_message"], 46)[:6]):
            col.label(text=line, icon=icon if i == 0 else "BLANK1")
        if not SESSION.get("ai_ok"):
            box.operator("photogram3d.open_folder", text="Logs", icon="TEXT").which = "logs"


def draw_ai_device_line(layout, context):
    """Under the AI Features option: where they will run, and the estimate from the test."""
    prefs = get_prefs(context)
    if prefs is None:
        return
    root = prefs.resolved_tools_dir()
    test = ai_backend.read_test(root)
    device = ai_backend.resolve(prefs.ai_features_device, root, test, bool(get_hw().get("gpus")),
                                untested="test")
    sub = layout.column(align=True)
    sub.scale_y = 0.8
    if device == "AUTO":
        sub.label(text="on the fastest device here,", icon="BLANK1")
        sub.label(text="found by a quick test on the first run", icon="BLANK1")
        return
    from .operators import ai_photo_count
    n, video = ai_photo_count(context.scene.photogram3d)
    est = None
    res = ((test or {}).get("results") or {}).get(device)
    if res:
        size = ai_backend.max_gpu_size(res) if device != "CPU" else test.get("cpu_size", 1600)
        est = ai_backend.estimate(res, n, video, size)
    text = "on " + ai_backend.LABELS[device] + (f", about {human_time(est)}" if est else "")
    sub.label(text=text, icon="BLANK1")


def draw_ai_features_prefs(prefs, context, layout):
    root = prefs.resolved_tools_dir()
    hw = get_hw()
    has_nvidia = bool(hw.get("gpus"))
    layout.label(text="AI Features (ALIKED + LightGlue)", icon="NODE_COMPOSITING")
    box = layout.box()
    box.prop(prefs, "ai_features_device")
    test = ai_backend.read_test(root)
    device = ai_backend.resolve(prefs.ai_features_device, root, test, has_nvidia, untested="test")
    if device == "AUTO":
        box.label(text="Runs on: the fastest here, tested on the first run with AI Features",
                  icon="CHECKMARK")
    elif prefs.ai_features_device == "AUTO":
        box.label(text="Runs on: " + ai_backend.LABELS[device], icon="CHECKMARK")
    from .operators import SESSION, ai_photo_count, job_running
    running = job_running("ai_test_job") or job_running("nvidia_job")
    row = box.row()
    row.enabled = not running
    row.operator("photogram3d.ai_test", icon="TIME")
    if running:
        job = SESSION.get("ai_test_job") if job_running("ai_test_job") else SESSION.get("nvidia_job")
        snap = job.snapshot()
        r = box.row()
        r.label(text=f"{snap['step_label']}: {snap['step_note'] or ''} {snap['overall'] * 100:.0f}%"[:90],
                icon="SORTTIME")
        r.operator("photogram3d.ai_test_cancel", text="", icon="CANCEL")
    elif SESSION.get("ai_test_message"):
        box.label(text=SESSION["ai_test_message"][:100], icon="INFO")
    if test:
        n, video = ai_photo_count(context.scene.photogram3d)
        best, lines = ai_backend.suggest(test, ai_backend.nvidia_ready(root), n, video)
        col = box.column(align=True)
        col.label(text=f"Tested {test.get('date', '')} with {test.get('sample', '')}", icon="INFO")
        for i, line in enumerate(lines[:4]):
            col.label(text=line + ("  (fastest)" if i == 0 and best else ""), icon="BLANK1")
        for name, res in (test.get("results") or {}).items():
            for why in (res.get("failed") or [])[:1]:
                col.label(text=f"{ai_backend.LABELS.get(name, name)}: {why}"[:100], icon="ERROR")
    if has_nvidia or prefs.ai_features_device == "NVIDIA":
        col = box.column(align=True)
        if ai_backend.nvidia_ready(root):
            r = col.row()
            r.label(text="NVIDIA libraries installed (" + (ai_backend.nvidia_summary(root) or "cuBLAS, cuDNN")
                    + ")", icon="CHECKMARK")
            r.operator("photogram3d.ai_nvidia_remove", text="", icon="TRASH")
            missing = ai_backend.nvidia_missing(root)
            if missing:
                r = col.row()
                r.enabled = not running
                size = human_bytes(sum(w["size"] for w in missing))
                r.operator("photogram3d.ai_nvidia", text=f"Download NVRTC ({size})", icon="IMPORT")
                col.label(text="cuDNN compiles some of its kernels with NVRTC (needed on some GPUs)",
                          icon="BLANK1")
        else:
            r = col.row()
            r.enabled = not running
            r.operator("photogram3d.ai_nvidia", text="Download NVIDIA Libraries ("
                       + human_bytes(ai_backend.NVIDIA_DOWNLOAD) + ")", icon="IMPORT")
            col.label(text="NVIDIA's license: free to use, not open source. Needed only for "
                           "GPU (NVIDIA libraries)", icon="BLANK1")
        col.operator("wm.url_open", text="NVIDIA's license", icon="URL").url = ai_backend.NVIDIA_LICENSE_URL


def draw_preferences(prefs, context):
    layout = prefs.layout
    layout.label(text="Tools", icon="TOOL_SETTINGS")
    box = layout.box()
    box.prop(prefs, "tools_dir")
    box.label(text="Using: " + prefs.resolved_tools_dir(), icon="FILE_FOLDER")
    draw_legacy_tools_warning(box)
    row = box.row()
    row.prop(prefs, "prefer_cuda")
    row.prop(prefs, "include_ffmpeg")
    hw = get_hw()
    box.label(text=hardware.describe(hw), icon="DESKTOP")
    draw_tools_box(box, context, compact=False)
    row = box.row()
    row.operator("photogram3d.install_tools", text="Install / Repair", icon="IMPORT")
    row.operator("photogram3d.check_tools", icon="CHECKMARK")
    row.operator("photogram3d.open_folder", text="", icon="FILE_FOLDER").which = "tools"
    rep = SESSION.get("verify")
    if rep:
        col = box.column(align=True)
        for name, (ok, msg) in rep.items():
            col.label(text=msg[:80], icon="CHECKMARK" if ok else "ERROR")
    layout.label(text="Use existing installs instead (optional)", icon="PREFERENCES")
    box = layout.box()
    box.prop(prefs, "colmap_path")
    box.prop(prefs, "openmvs_dir")
    box.prop(prefs, "ffmpeg_path")
    box.prop(prefs, "brush_path")
    draw_ai_features_prefs(prefs, context, layout)
    layout.label(text="AI Polish", icon="SHADERFX")
    box = layout.box()
    box.prop(prefs, "ai_model")
    box.prop(prefs, "ai_device")
    paths = polish_ai.ai_paths(prefs.resolved_tools_dir())
    info = polish_ai.runtime_info(paths["site"])
    if info:
        box.label(text=(info.get("report") or "AI runtime installed")[:90], icon="CHECKMARK")
    else:
        box.label(text="AI runtime: installed on first use (onnxruntime + Pillow via pip)",
                  icon="INFO")
    for key, meta in polish_ai.MODELS.items():
        have = polish_ai.model_file(paths["models"], key)
        box.label(text=f"{meta['label']}: {'downloaded' if have else 'not downloaded'}",
                  icon="CHECKMARK" if have else "BLANK1")


CLASSES = (PHOTOGRAM3D_PT_main, PHOTOGRAM3D_PT_options, PHOTOGRAM3D_PT_run, PHOTOGRAM3D_PT_tools)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
