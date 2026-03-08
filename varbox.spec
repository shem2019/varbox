# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import (
    collect_submodules,
    collect_data_files,
    collect_dynamic_libs,
)

# Robust imports for the spec DSL (handles PyInstaller versions)
try:
    from PyInstaller.building.build_main import Analysis, PYZ
except Exception:
    from PyInstaller.building.api import Analysis, PYZ

try:
    from PyInstaller.building.api import EXE, COLLECT
except Exception:
    from PyInstaller.building.build_main import EXE, COLLECT

# -------- Project settings --------
entry = "main.py"
app_name = "VAR Box"

# -------- Third-party binaries & data --------
cv2_bins = collect_dynamic_libs("cv2")

# Only pull the mediapipe bits we actually use (solutions/python), not model_maker
mp_submods = (
    collect_submodules("mediapipe.python")
    + collect_submodules("mediapipe.solutions")
)
# Deduplicate
mp_submods = sorted(set([m for m in mp_submods if not m.startswith("mediapipe.model_maker")]))

mp_data = collect_data_files("mediapipe", include_py_files=True)

# Tk assets (often auto-found; adding for portability)
tk_data = collect_data_files("tkinter")

# Bundle your assets/ (models, etc.)
datas = []
assets_dir = os.path.abspath("assets")
if os.path.isdir(assets_dir):
    datas.append((assets_dir, "assets"))

# -------- Analysis --------
a = Analysis(
    [entry],
    pathex=[],
    binaries=cv2_bins,
    datas=datas + mp_data + tk_data,
    hiddenimports=mp_submods + [
        "mediapipe",
        "mediapipe.python._framework_bindings",
        "mediapipe.python._framework_bindings.image_frame",
        "mediapipe.python._framework_bindings.packet",
        "mediapipe.python._framework_bindings.timestamp",
        "cv2",
        "absl",
        "google.protobuf",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["mediapipe.model_maker"],   # keep noisy warning away
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name=app_name,
    console=False,          # set True if you want a console during debug
    disable_windowed_traceback=False,
    # icon="assets/app.ico", # optional
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=app_name,
)
