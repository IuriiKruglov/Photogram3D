# SPDX-License-Identifier: GPL-3.0-or-later
# Photogram3D - photos or video to a textured mesh with COLMAP + OpenMVS
# Copyright (C) 2026 Kruglov Iurii
# Authors: Kruglov Iurii, Anthropic Claude AI
# License: GNU General Public License (GPL) v3 or later, https://www.blender.org/about/license/
"""Scene settings and add-on preferences."""

import os

import bpy
from bpy.props import (BoolProperty, EnumProperty, FloatProperty, IntProperty,
                       PointerProperty, StringProperty)

from .common import legacy_folder

_JOIN_ITEMS = []     # the items Blender shows: their strings must live as long as it may read them
_JOIN_KEEP = []      # earlier item lists, kept alive for the same reason


def _join_items(context):
    """(identifier, label, description, icon, number) of each other scan in the scene."""
    import zlib
    from . import operators, ui
    try:
        context = context if context is not None else bpy.context
        active = operators.active_root(context)
        roots = operators.scan_roots(context)
    except AttributeError:          # no scene in this context (while registering or loading)
        return []
    return [(r.name, ui.scan_label(r), "", "NONE", zlib.crc32(r.name.encode("utf-8")) & 0x7FFFFFFF or 1)
            for r in roots if r != active]


def _join_targets(context):
    """The other scans in the scene, to join the active scan onto (0.3.13). Blender keeps
    the number of the chosen item, so each scan's number comes from its name: a scan
    imported later (the joined one, say) shifts the list without changing the choice.

    Blender keeps pointers to the strings of the items it was given, so the list is replaced
    only when the scans change, and the lists replaced are kept: 0.3.13 built a new list on
    every call, also from the getter while the panel showed the old one, and Onto came out
    empty (and with it the join's results and the next button). Blender may also call this
    without a context; the scene's context is used then, not an empty list."""
    items = _join_items(context) or [("NONE", "(no other scan)", "", "NONE", 0)]
    if items != _JOIN_ITEMS:
        _JOIN_KEEP.append(list(_JOIN_ITEMS))
        del _JOIN_KEEP[:-64]
        _JOIN_ITEMS[:] = items
    return _JOIN_ITEMS


def _get_join_target(self):
    """The chosen scan's number, or the first scan's (the newest other one) when nothing is
    chosen yet or the chosen scan is gone: with two scans in the scene, the other is chosen."""
    values = [item[4] for item in _join_targets(None)]
    value = self.get("join_target_value", 0)
    return value if value in values else values[0]


def _set_join_target(self, value):
    self["join_target_value"] = value


class Photogram3DSettings(bpy.types.PropertyGroup):
    input_mode: EnumProperty(
        name="Input",
        items=(("IMAGES", "Photos", "A folder of photos", "IMAGE_DATA", 0),
               ("VIDEO", "Video", "A video file; the sharpest frames are extracted", "FILE_MOVIE", 1)),
        default="IMAGES")
    images_dir: StringProperty(
        name="Photos", subtype="DIR_PATH",
        description="Folder with the photos directly inside it: JPG, PNG, TIFF, HEIC, AVIF, "
                    "WebP, TGA, PSD, EXR or camera RAW (DNG, CR2, CR3, NEF, ARW, RAF, RW2, ORF, "
                    "...). Formats other than JPG and PNG are converted to JPEG first")
    raw_pairs: EnumProperty(
        name="RAW + JPEG",
        description="When a shot was saved twice, as RAW and as JPEG (phones do), which one to use",
        items=(("JPEG", "Use JPEG", "Use the camera's JPEG (its own processing; a folder that "
                "gave a result before gives the same one)"),
               ("RAW", "Use RAW", "Develop the RAW file instead: one exposure for the whole set "
                "(more even texture), more highlight detail, the phone's lens shading map applied")),
        default="JPEG")
    video_path: StringProperty(name="Video", subtype="FILE_PATH",
                               description="Video file (MP4, MOV, MKV, WebM, AVI, ...: H.264, "
                                           "H.265, AV1, ProRes, DNxHR)")
    project_dir: StringProperty(
        name="Output", subtype="DIR_PATH",
        description="Where the reconstruction is written. Empty: next to the input")
    quality: EnumProperty(
        name="Quality",
        items=(("DRAFT", "Draft", "Fast preview: reduced resolution, no refinement"),
               ("NORMAL", "Normal", "Balanced for 16 GB RAM laptops"),
               ("HIGH", "High", "Full-resolution images; slow and memory hungry"),
               ("ULTRA", "Ultra", "High with about twice the faces after refinement and depth maps "
                "up to 3200 px. Slow; use it with the crop box"),
               ("FULL", "Full", "Ultra with depth maps at the photos' full size (4624 px for 16 MP "
                "photos): about twice Ultra's depth-map time and disk space. Use it with the crop "
                "box")),
        default="NORMAL")

    # --- options ---
    ai_features: BoolProperty(
        name="AI Features (ALIKED + LightGlue)", default=False,
        description="Learned features and matcher from COLMAP 4. Helps with low texture and "
                    "large viewpoint changes. The models (49 MB) download on first use. Where they "
                    "run (any DirectX 12 GPU, NVIDIA's libraries or the CPU; Automatic tests this "
                    "computer once) is set in the add-on preferences: AI Features On")
    mapper: EnumProperty(
        name="Camera Solver",
        items=(("AUTO", "Auto", "Incremental for photos, global for video. Either way the solved "
                "camera path is checked, and the other solver is tried if photos taken seconds "
                "apart ended up far apart or too few were placed. When the path still folds back "
                "(a symmetric object), the cameras are solved again along the shooting order"),
               ("GLOBAL", "Global first", "Start with the fast global solver (GLOMAP)"),
               ("INCREMENTAL", "Incremental first", "Start with the classic COLMAP mapper"),
               ("WALK", "Shooting Order", "For objects that look alike from opposite sides (a box "
                "printed the same on two sides): always solve along the shooting order as well, "
                "and prefer that solve. Each photo is first linked only to the photos taken just "
                "before and after it; photo pairs that contradict that walk are left out")),
        default="AUTO")
    camera_model: EnumProperty(
        name="Lens Model",
        items=(("SIMPLE_RADIAL", "Simple Radial", "Phones and most cameras"),
               ("OPENCV", "OpenCV", "Stronger distortion (wide lenses)"),
               ("PINHOLE", "Pinhole", "Already undistorted images")),
        default="SIMPLE_RADIAL")
    single_camera: BoolProperty(name="Same Camera for All Images", default=True,
                                description="All images share one lens and zoom")
    align_up: BoolProperty(name="Align Up Axis", default=True,
                           description="Estimate the vertical direction from photo orientation")
    dense_engine: EnumProperty(
        name="Depth Maps",
        items=(("OPENMVS", "OpenMVS (CPU)", "Works on any PC"),
               ("COLMAP", "COLMAP (NVIDIA GPU)", "Depth maps on the GPU with COLMAP CUDA, then "
                "OpenMVS meshing. Experimental"),
               ("HYBRID", "OpenMVS + COLMAP (CPU + GPU)", "The CPU and the NVIDIA GPU share the "
                "first depth-map pass, the slow one; OpenMVS then refines and fuses every depth "
                "map as in the CPU path. Experimental")),
        default="OPENMVS")
    dense_region: EnumProperty(
        name="Dense Region",
        description="How much of what the photos see the depth maps and the dense cloud keep",
        items=(("FULL", "Full", "Everything the photos see, also parts the cameras do not aim at, "
                "such as one at the edge of the frames. The crop box then sets what is meshed"),
               ("AUTO", "Automatic", "OpenMVS guesses the object's region from where the cameras "
                "aim and trims the dense cloud to it, scaled by Region Size: less background, but "
                "parts at the edge of the frames can be cut off. Without a crop box the mesh is "
                "cropped to it as well")),
        default="FULL")
    dense_region_size: FloatProperty(
        name="Region Size", default=1.1, min=0.5, max=10.0, soft_max=3.0, step=10, precision=2,
        description="Automatic: the guessed region is scaled by this factor (OpenMVS's default "
                    "is 1.1). Larger keeps more around the object")
    refine: EnumProperty(
        name="Refine Mesh",
        items=(("AUTO", "Auto", "On for Normal and High when the photos are at least 1000 px "
                 "at the resolution memory allows (phone photos are)"),
               ("ON", "On", "Photometric refinement of fine detail"),
               ("OFF", "Off", "Skip (faster)")),
        default="AUTO")
    texture: BoolProperty(name="Texture", default=True)
    blend_seams: BoolProperty(
        name="Blend Seams", default=True,
        description="Even out brightness and colour where texture patches from different photos "
                    "meet, so photos taken at different exposures leave no light and dark patches. "
                    "The texture as made is kept beside it. Turning it on or off for a finished run "
                    "redoes only this step")
    even_photos: BoolProperty(
        name="Even Out Photos", default=False,
        description="Even out exposure, white balance and vignetting across the photos the texture "
                    "and the splats are made from, from the solved cameras (the same surface seen "
                    "in different photos): no light and dark patches from photos taken at different "
                    "exposures. The camera solve and the depth maps use the photos as they are, so "
                    "turning it on for a finished run redoes only the texture")
    sharpen: FloatProperty(
        name="Sharpen", default=0.0, min=0.0, max=1.0, subtype="FACTOR",
        description="Mild sharpening of the evened-out photos (0 = none). Crisper texture; too "
                    "much adds halos and noise")
    photo_max_size: IntProperty(
        name="Max Photo Size", default=0, min=0, max=16384, subtype="PIXEL",
        description="Scale larger photos and video frames down to this long side first (Lanczos), "
                    "0 = keep their size. Saves time and memory with very large photos")
    splats: BoolProperty(
        name="Gaussian Splats", default=False,
        description="Also train 3D Gaussian splats with Brush after texturing, on the undistorted "
                    "photos, and keep those inside the crop box. Brush (159 MB) downloads on first "
                    "use and runs on the GPU (NVIDIA, AMD or Intel). Blender 5.3 shows splats as "
                    "splats; 5.2 as round points")
    splat_steps: IntProperty(
        name="Splat Steps", default=30000, min=1000, max=200000, soft_max=60000, step=1000,
        description="Training steps (Brush's default is 30,000). Fewer train faster but blurrier")
    splat_resolution: IntProperty(
        name="Splat Photo Size", default=1920, min=480, max=8192, soft_max=4096, subtype="PIXEL",
        description="Long side of the photos the splats are trained on (Brush's default is 1920). "
                    "Larger needs more GPU memory")
    spread_views: BoolProperty(
        name="Spread Photos", default=True,
        description="With many photos (long videos), compute depth maps from photos at least 5° "
                    "apart (above 300 photos) and the texture from photos about 20° apart (above "
                    "60). The camera path and the cameras keep every photo")
    texture_size: EnumProperty(
        name="Texture Size",
        items=(("0", "Auto", "From the quality preset"),
               ("2048", "2K", ""), ("4096", "4K", ""), ("8192", "8K", "")),
        default="0")
    target_faces: IntProperty(
        name="Max Faces", default=0, min=0, soft_max=10_000_000,
        description="Decimate the mesh to this many faces (0 = from the quality preset)")
    free_space: BoolProperty(
        name="Fill Weak Surfaces", default=False,
        description="Free-space support in OpenMVS meshing; can close glossy or "
                    "poorly matched areas, may add bulges")
    use_gpu: BoolProperty(name="Use NVIDIA GPU", default=True,
                          description="COLMAP feature extraction and matching on the GPU")
    resume: BoolProperty(name="Resume", default=True,
                         description="Reuse finished steps when their inputs and settings are unchanged")
    import_cameras: BoolProperty(name="Import Cameras", default=True)
    camera_photos: BoolProperty(
        name="Photos in Cameras", default=True,
        description="Show each undistorted photo as the background of its camera (exact fit). "
                    "Images load only when you look through a camera")
    auto_import: BoolProperty(name="Import When Done", default=True)
    crop_stage: BoolProperty(
        name="Stop to Crop", default=True,
        description="Stop after the dense point cloud and show it with a crop box. Only the "
                    "points inside the box are meshed, refined and textured")
    real_distance: FloatProperty(
        name="Real Distance", default=1.0, min=1e-6, unit="LENGTH",
        description="The real-world distance between Anchor A and Anchor B")
    join_target: EnumProperty(
        name="Onto",
        description="The scan the active scan is joined onto: it stays where it is, and the "
                    "active scan moves onto it by the points placed on both",
        items=lambda self, context: _join_targets(context), get=_get_join_target, set=_set_join_target)
    ground_axis: EnumProperty(
        name="Ground 1 to 2",
        description="Align Ground turns the scan so that the direction from Ground 1 to Ground 2 "
                    "runs along this axis. For a car: Ground 1 on a front wheel, Ground 2 on the "
                    "rear wheel of the same side and +Y make it face -Y, like Blender's front view",
        items=(("POS_Y", "1→2 along +Y", "Ground 1 to 2 points along +Y"),
               ("NEG_Y", "1→2 along -Y", "Ground 1 to 2 points along -Y"),
               ("POS_X", "1→2 along +X", "Ground 1 to 2 points along +X"),
               ("NEG_X", "1→2 along -X", "Ground 1 to 2 points along -X"),
               ("FREE", "Keep Z Rotation", "Only level the ground; the scan keeps its rotation "
                "about Z")),
        default="POS_Y")

    # --- video ---
    video_frames: EnumProperty(
        name="Frames",
        description="How the frames of the video are chosen",
        items=(("AUTO", "Auto", "By the camera's movement: a frame each time the picture has moved "
                "by about 12% of the frame (about 8 degrees of turning), at least one per second, "
                "30 frames up to Max Frames; at each place the sharpest well-lit frame nearby"),
               ("RATE", "Per Second", "A fixed number of frames per second of video: the "
                "sharpest of each 1/fps-second window"),
               ("ALL", "Every Frame", "Every frame between Start and End (no sharpness check). "
                "A 60 fps minute is 3,600 photos: solving takes long")),
        default="AUTO")
    video_max_frames: IntProperty(
        name="Max Frames", default=300, min=30, max=5000, soft_max=1000,
        description="Auto: at most this many frames. With more movement than that, the frames "
                    "are spread further apart")
    video_fps: FloatProperty(
        name="Frames per Second", default=2.0, min=0.1, max=1000.0, soft_max=30.0,
        description="How many frames to keep per second of video: the sharpest of each "
                    "1/fps-second window. Higher values give more photos (and longer runs)")
    video_all_frames: BoolProperty(
        name="Every Frame", default=False,
        description="Up to 0.3.7: keep every frame (now Frames: Every Frame)")
    video_start: FloatProperty(name="Start", default=0.0, min=0.0, unit="TIME_ABSOLUTE",
                               description="Use the video from this time on")
    video_end: FloatProperty(name="End", default=0.0, min=0.0, unit="TIME_ABSOLUTE",
                             description="Use the video up to this time (0 = to the end)")
    video_blur_filter: BoolProperty(name="Skip Blurry Frames", default=True,
                                    description="Keep the sharpest frame of each time window")
    video_camera: BoolProperty(
        name="Tracked Camera", default=True,
        description="Also create one camera that follows the video: animated at the video's "
                    "frame rate, with the video as its background")
    video_track: EnumProperty(
        name="Camera Track",
        description="Where the tracked camera gets its place from: every frame of the video, placed "
                    "after the cameras are solved (it then follows the footage), or only the frames "
                    "the reconstruction used (interpolated in between)",
        items=(("EVERY", "Every Frame", "Place the camera on every frame of the part of the video used: "
                "it follows the footage frame by frame. Adds a few minutes per minute of video"),
               ("HALF", "Every 2nd Frame", "Place the camera on every second frame, interpolated in "
                "between: half the time, for 50-60 fps videos"),
               ("OFF", "Solved Frames", "Only the frames the reconstruction used, interpolated in "
                "between: the scan slides against the footage between them")),
        default="EVERY")

    # --- polish ---
    polish_iterations: IntProperty(name="Iterations", default=8, min=1, max=50)
    polish_strength: FloatProperty(name="Strength", default=0.5, min=0.0, max=1.0)
    polish_keep_edges: FloatProperty(name="Keep Edges", default=0.6, min=0.0, max=1.0,
                                     description="How strongly sharp creases are protected")

    # --- AI polish ---
    ai_influence: FloatProperty(
        name="Influence", default=0.6, min=0.0, max=1.0,
        description="How strongly the AI normals override the scanned surface "
                    "(0 = no change). You can also blend the result later with its shape key")
    ai_views: IntProperty(
        name="Photos", default=60, min=4, soft_max=200,
        description="Maximum number of photos to analyse (spread over the whole set)")
    ai_detail: EnumProperty(
        name="Detail",
        items=(("STANDARD", "Standard", "1600 image patches per photo: faster"),
               ("HIGH", "High", "2500 image patches per photo: finer normals")),
        default="HIGH")

    def resolved_project_dir(self):
        if self.project_dir.strip():
            return bpy.path.abspath(self.project_dir)
        if self.input_mode == "VIDEO" and self.video_path:
            path = bpy.path.abspath(self.video_path)
            stem = os.path.splitext(os.path.basename(path))[0]
            return legacy_folder(os.path.join(os.path.dirname(path), stem + "_photogram3d"), "_photogram3d")
        if self.images_dir:
            folder = os.path.normpath(bpy.path.abspath(self.images_dir))
            return legacy_folder(folder + "_photogram3d", "_photogram3d")
        return ""


def _tools_dir_default():
    from . import legacy
    try:
        path = bpy.utils.extension_path_user(__package__, path="tools", create=True)
    except (ValueError, AttributeError, RuntimeError):
        path = os.path.join(bpy.utils.user_resource("DATAFILES", path="photogram3d", create=True), "tools")
    return legacy.take_over_tools(path)     # the tools PhotoMesh (0.3.6 and older) downloaded


class Photogram3DPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    tools_dir: StringProperty(
        name="Tools Folder", subtype="DIR_PATH",
        description="Where COLMAP, OpenMVS and FFmpeg are installed. Empty: the extension's "
                    "user folder. Prefer a path without non-Latin characters")
    include_ffmpeg: BoolProperty(name="Include FFmpeg (video input)", default=True)
    prefer_cuda: BoolProperty(name="COLMAP CUDA Build", default=True,
                              description="Install the CUDA build of COLMAP when an NVIDIA GPU "
                                          "is present (395 MB instead of 122 MB)")
    colmap_path: StringProperty(name="COLMAP Executable", subtype="FILE_PATH",
                                description="Optional: use your own colmap.exe")
    openmvs_dir: StringProperty(name="OpenMVS Folder", subtype="DIR_PATH",
                                description="Optional: folder with your own OpenMVS executables")
    ffmpeg_path: StringProperty(name="FFmpeg Executable", subtype="FILE_PATH",
                                description="Optional: use your own ffmpeg.exe")
    brush_path: StringProperty(name="Brush Executable", subtype="FILE_PATH",
                               description="Optional: use your own brush_app.exe (Gaussian splats)")
    ai_model: EnumProperty(
        name="AI Model",
        items=(("SMALL", "MoGe-2 Small (141 MB)", "ViT-S: fast, fits 4 GB GPUs easily"),
               ("BASE", "MoGe-2 Base (419 MB)", "ViT-B: more accurate normals, about 3x slower")),
        default="SMALL")
    ai_device: EnumProperty(
        name="AI Device",
        items=(("AUTO", "GPU (DirectML)", "Use the fastest DirectX 12 GPU, fall back to the CPU"),
               ("CPU", "CPU", "Always run on the CPU")),
        default="AUTO")
    ai_features_device: EnumProperty(
        name="AI Features On",
        description="Where AI Features (ALIKED and LightGlue) run",
        items=(("AUTO", "Automatic", "The fastest that works on this computer, from Test AI Speed. "
                "Not tested yet: the GPU through DirectML on Windows, else the CPU"),
               ("DIRECTML", "GPU (DirectML)", "Any DirectX 12 GPU (NVIDIA, AMD, Intel), with Photogram3D's "
                "own runner. ONNX Runtime (about 20 MB, shared with AI Polish) and COLMAP's two model "
                "files (49 MB) are downloaded once"),
               ("NVIDIA", "GPU (NVIDIA libraries)", "COLMAP's own GPU path on NVIDIA GPUs. Needs NVIDIA's "
                "cuBLAS, cuDNN and CUDA runtime: 1.3 GB, NVIDIA's license (free to use, not open source)"),
               ("CPU", "CPU", "COLMAP on the CPU: works on every computer, the slowest")),
        default="AUTO")
    legacy_taken_over: BoolProperty(default=False, options={"HIDDEN"},
                                    description="The PhotoMesh preferences were copied (legacy.py)")

    def resolved_tools_dir(self):
        if self.tools_dir.strip():
            return bpy.path.abspath(self.tools_dir)
        return _tools_dir_default()

    def overrides(self):
        return {"colmap": bpy.path.abspath(self.colmap_path) if self.colmap_path else "",
                "openmvs_dir": bpy.path.abspath(self.openmvs_dir) if self.openmvs_dir else "",
                "ffmpeg": bpy.path.abspath(self.ffmpeg_path) if self.ffmpeg_path else ""}

    def draw(self, context):
        from . import ui
        ui.draw_preferences(self, context)


def get_prefs(context=None):
    context = context or bpy.context
    addon = context.preferences.addons.get(__package__)
    return addon.preferences if addon else None


CLASSES = (Photogram3DSettings, Photogram3DPreferences)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.photogram3d = PointerProperty(type=Photogram3DSettings)


def unregister():
    del bpy.types.Scene.photogram3d
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
