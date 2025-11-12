# config.py
import os, sys

# --- Runtime-overridden paths (set by GUI / CLI) ---
INPUT_VIDEO: str | None = None
OUTPUT_VIDEO: str | None = None
SCORECARD_PDF: str | None = None

PUNCH_DISTANCE_THRESHOLD = 50
COOLDOWN_FRAMES = 15

# 0 = auto (use video FPS if available)
FRAME_RATE = int(os.getenv("VARBOX_FPS_OVERRIDE", "0") or "0")

# When packaged, PyInstaller extracts bundled data to sys._MEIPASS
_BASE_DIR = getattr(sys, "_MEIPASS", os.path.dirname(__file__))

# detection backend & model assets
ASSETS_DIR = os.getenv("VARBOX_ASSETS", os.path.join(_BASE_DIR, "assets"))
BACKEND = os.getenv("VARBOX_BACKEND", "opencv")  # "opencv" or "yolov8"

# OpenCV-DNN person detector files (shipped with app)
DNN_PROTO = os.getenv("VARBOX_SSD_PROTO", os.path.join(ASSETS_DIR, "models", "mobilenet_ssd", "deploy.prototxt"))
DNN_MODEL = os.getenv("VARBOX_SSD_MODEL", os.path.join(ASSETS_DIR, "models", "mobilenet_ssd", "deploy.caffemodel"))

# YOLOv8 weights (optional; only if you actually use YOLO)
YOLOV8_WEIGHTS = os.getenv("VARBOX_YOLOV8_WEIGHTS", os.path.join(ASSETS_DIR, "models", "yolov8n.pt"))
