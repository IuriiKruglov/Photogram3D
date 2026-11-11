# SPDX-License-Identifier: GPL-3.0-or-later
# Photogram3D - photos or video to a textured mesh with COLMAP + OpenMVS
# Copyright (C) 2026 Kruglov Iurii
# Authors: Kruglov Iurii, Anthropic Claude AI
# License: GNU General Public License (GPL) v3 or later, https://www.blender.org/about/license/
"""Photogram3D: photogrammetry inside Blender.

COLMAP 4 solves the cameras, OpenMVS builds the dense mesh and texture,
and the result is imported with its cameras. Tools install automatically.
"""

from . import importer, legacy, operators, props, ui


def register():
    props.register()
    operators.register()
    ui.register()
    legacy.register()
    importer.register()


def unregister():
    importer.unregister()
    legacy.unregister()
    ui.unregister()
    operators.unregister()
    props.unregister()
