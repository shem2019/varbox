# config.py
import os
PUNCH_DISTANCE_THRESHOLD = 50
COOLDOWN_FRAMES = 15
FRAME_RATE = 30  # fallback in case video info missing

INPUT_VIDEO = "assets/boxing_match.mp4"
OUTPUT_VIDEO = "outputs/boxing_output.mp4"
SCORECARD_PDF = "outputs/boxing_scorecard.pdf"


FRAME_RATE = int(os.getenv("VARBOX_FPS_OVERRIDE","0") or "0")  # 0 = auto

# detection backend & model assets
ASSETS_DIR = os.getenv("VARBOX_ASSETS", os.path.join(os.path.dirname(__file__), "assets"))
BACKEND = os.getenv("VARBOX_BACKEND", "opencv")  # "opencv" or "yolov8"

# OpenCV-DNN (Lite) person detector files (you will ship these with the app)
DNN_PROTO = os.getenv("VARBOX_SSD_PROTO", os.path.join(ASSETS_DIR, "models", "mobilenet_ssd", "deploy.prototxt"))
DNN_MODEL = os.getenv("VARBOX_SSD_MODEL", os.path.join(ASSETS_DIR, "models", "mobilenet_ssd", "deploy.caffemodel"))

# YOLOv8 weights (Pro build)
YOLOV8_WEIGHTS = os.getenv("VARBOX_YOLOV8_WEIGHTS", os.path.join(ASSETS_DIR, "models", "yolov8n.pt"))

