# SPDX-License-Identifier: GPL-3.0-or-later
# Photogram3D - photos or video to a textured mesh with COLMAP + OpenMVS
# Copyright (C) 2026 Kruglov Iurii
# Authors: Kruglov Iurii, Anthropic Claude AI
# License: GNU General Public License (GPL) v3 or later, https://www.blender.org/about/license/
"""Taking over what PhotoMesh (this add-on's name up to 0.3.6) left behind.

* The tools it downloaded (COLMAP, OpenMVS, FFmpeg, Brush, the AI runtime and model) are
  in PhotoMesh's user folder, which Blender deletes when PhotoMesh is uninstalled. They
  are moved into Photogram3D's folder, not copied: on the same drive that is a rename.
* Its preferences are copied once, while PhotoMesh is still installed and enabled.
* In .blend files: the custom properties on imported runs (photomesh_* to photogram3d_*),
  the scene settings and the AI result shape key are renamed when a file is opened.
  Object and collection names (PhotoMesh_2 Gas_mask, PM_Gas_mask_2) are the user's
  scene and stay as they are.
* Files in project folders (common.current) and the default output folders
  (common.legacy_folder) are taken over where they are read.
"""

import os

import bpy
from bpy.app.handlers import persistent

from .common import LEGACY_ID, LEGACY_NAME, LEGACY_PREFIX, NAME, PREFIX

SETTINGS = "photogram3d"                   # bpy.types.Scene.photogram3d
LEGACY_AI_KEY = LEGACY_NAME + " AI"        # polish_ai.AI_KEY in PhotoMesh
_ID_TYPES = ("objects", "cameras", "meshes", "pointclouds", "curves", "collections", "scenes",
             "materials", "images", "node_groups")
_PREFERENCES = ("tools_dir", "include_ffmpeg", "prefer_cuda", "colmap_path", "openmvs_dir",
                "ffmpeg_path", "brush_path", "ai_model", "ai_device")
_tools = {}


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------

def _has_files(folder):
    try:
        with os.scandir(folder) as it:
            return any(True for _ in it)
    except OSError:
        return False


def legacy_tools_dirs(path):
    """Where PhotoMesh kept its tools, for Photogram3D's default tools folder ``path``:
    next to it in the extensions' user folders (.user/<repository>/photomesh/tools), or in
    Blender's data files when PhotoMesh ran as a legacy add-on."""
    found = [os.path.join(os.path.dirname(os.path.dirname(os.path.normpath(path))), LEGACY_ID, "tools")]
    try:
        found.append(os.path.join(bpy.utils.user_resource("DATAFILES", path=LEGACY_ID), "tools"))
    except (ValueError, RuntimeError, AttributeError):
        pass
    own = os.path.normcase(os.path.normpath(path))
    return [p for p in found if os.path.normcase(os.path.normpath(p)) != own]


def take_over_tools(path):
    """Photogram3D's default tools folder ``path``. The first time it is asked for, when it
    holds nothing yet, PhotoMesh's tools are moved into it. Returns the folder to use:
    ``path``, or PhotoMesh's folder when the move is refused (a tool still running)."""
    key = os.path.normcase(os.path.normpath(path))
    if key not in _tools:
        _tools[key] = _take_over_tools(path)
    return _tools[key]


def tools_left_behind():
    """PhotoMesh's tools folder when Photogram3D uses the tools there because moving them was
    refused (a program had a file open), else None. Uninstalling PhotoMesh would delete them."""
    for key, used in _tools.items():
        if os.path.normcase(os.path.normpath(used)) != key:
            return used
    return None


def _take_over_tools(path):
    if _has_files(path):
        return path
    for old in legacy_tools_dirs(path):
        if not _has_files(old):
            continue
        try:
            if os.path.isdir(path):
                os.rmdir(path)                    # the empty folder extension_path_user made
            os.makedirs(os.path.dirname(path), exist_ok=True)
            os.replace(old, path)                 # same drive: a rename, nothing is copied
        except OSError as exc:
            print(f"{NAME}: using the tools {LEGACY_NAME} downloaded where they are ({old}): {exc}")
            return old
        print(f"{NAME}: moved the tools {LEGACY_NAME} downloaded to {path}")
        return path
    return path


# ---------------------------------------------------------------------------
# preferences
# ---------------------------------------------------------------------------

def legacy_preferences():
    """PhotoMesh's add-on preferences while it is installed and enabled, else None."""
    for name, addon in bpy.context.preferences.addons.items():
        if name == LEGACY_ID or name.endswith("." + LEGACY_ID):
            prefs = getattr(addon, "preferences", None)
            if prefs is not None:
                return prefs
    return None


def copy_preferences(prefs):
    """PhotoMesh's preferences into ``prefs`` (Photogram3D's), once. True when copied."""
    if prefs is None or prefs.legacy_taken_over:
        return False
    old = legacy_preferences()
    if old is None:
        return False
    for name in _PREFERENCES:
        if hasattr(old, name):
            try:
                setattr(prefs, name, getattr(old, name))
            except (AttributeError, TypeError, ValueError):
                pass
    prefs.legacy_taken_over = True
    print(f"{NAME}: copied the {LEGACY_NAME} preferences")
    return True


# ---------------------------------------------------------------------------
# .blend data
# ---------------------------------------------------------------------------

def _value(value):
    """An ID property value to store under another name: groups as dicts; arrays are
    assigned as they are (as buffers they keep their float32 / int type)."""
    return value.to_dict() if hasattr(value, "to_dict") else value


def _editable(idb):
    return getattr(idb, "library", None) is None


def migrate_blend():
    """Rename PhotoMesh's data in the open file to Photogram3D's; returns how many items."""
    n = 0
    for attr in _ID_TYPES:
        for idb in getattr(bpy.data, attr, ()):
            if not _editable(idb):
                continue
            for key in [k for k in idb.keys() if k.startswith(LEGACY_PREFIX)]:
                new = PREFIX + key[len(LEGACY_PREFIX):]
                if new not in idb.keys():
                    idb[new] = _value(idb[key])
                del idb[key]
                n += 1
    for scene in bpy.data.scenes:
        if not _editable(scene):
            continue
        # the scene settings (bpy.props) are system properties since Blender 5.0
        stores = [scene]
        if hasattr(scene, "bl_system_properties_get"):
            system = scene.bl_system_properties_get()
            if system is not None:
                stores.insert(0, system)
        for store in stores:
            if LEGACY_ID not in store.keys():
                continue
            old = store[LEGACY_ID]
            # reading scene.photogram3d creates its group empty; only settings someone
            # changed in Photogram3D are kept over PhotoMesh's
            if hasattr(old, "to_dict") and (SETTINGS not in store.keys() or not store[SETTINGS].keys()):
                store[SETTINGS] = old.to_dict()
            del store[LEGACY_ID]
            n += 1
    from .polish_ai import AI_KEY
    for key in bpy.data.shape_keys:
        if not _editable(key):
            continue
        block = key.key_blocks.get(LEGACY_AI_KEY)
        if block is not None and key.key_blocks.get(AI_KEY) is None:
            block.name = AI_KEY
            n += 1
    return n


@persistent
def _on_load(_dummy=None):
    try:
        n = migrate_blend()
    except Exception as exc:             # never stop a file from opening
        print(f"{NAME}: could not take over the {LEGACY_NAME} data in this file: {exc}")
        return
    if n:
        print(f"{NAME}: took over {n} {LEGACY_NAME} items in this file")


def take_over_all(prefs):
    """Preferences, tools and the open file (the add-on was just enabled)."""
    try:
        copy_preferences(prefs)
        if prefs is not None:
            prefs.resolved_tools_dir()   # moves PhotoMesh's tools when they are the ones to use
    except Exception as exc:
        print(f"{NAME}: could not take over the {LEGACY_NAME} preferences or tools: {exc}")
    _on_load()


def _first_run():
    from .props import get_prefs
    take_over_all(get_prefs())
    return None


def register():
    if _on_load not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_load)
    if not bpy.app.timers.is_registered(_first_run):
        # persistent: Blender started with a .blend to open loads it right after the add-ons,
        # and a file load drops the other timers
        bpy.app.timers.register(_first_run, first_interval=0.5, persistent=True)


def unregister():
    if _on_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load)
    if bpy.app.timers.is_registered(_first_run):
        bpy.app.timers.unregister(_first_run)
