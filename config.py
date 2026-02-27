# config.py
import os

PUNCH_DISTANCE_THRESHOLD = 50
COOLDOWN_FRAMES = 15
FRAME_RATE = int(os.getenv("VARBOX_FPS_OVERRIDE", "0") or "0")  # 0 = auto
BOUT_ROUNDS_COUNT = int(os.getenv("VARBOX_ROUNDS_COUNT", "12") or "12")
BOUT_ROUND_SECONDS = float(os.getenv("VARBOX_ROUND_SECONDS", "180") or "180")
BOUT_REST_SECONDS = float(os.getenv("VARBOX_REST_SECONDS", "60") or "60")
BOUT_WARMUP_SECONDS = float(os.getenv("VARBOX_WARMUP_SECONDS", "0") or "0")
ROUND_START_OFFSET_SECONDS = float(os.getenv("VARBOX_ROUND_START_OFFSET_SECONDS", "0") or "0")
STOP_AT_BOUT_END = int(os.getenv("VARBOX_STOP_AT_BOUT_END", "1") or "1")

_ROOT = os.path.dirname(__file__)
_DEFAULT_OUT_DIR = os.getenv("VARBOX_OUT_DIR", os.path.join(_ROOT, "output", "runs"))
os.makedirs(_DEFAULT_OUT_DIR, exist_ok=True)

INPUT_VIDEO = os.getenv("VARBOX_INPUT", os.path.join(_ROOT, "assets", "boxing_match.mp4"))
OUTPUT_VIDEO = os.getenv("VARBOX_OUTPUT", os.path.join(_DEFAULT_OUT_DIR, "boxing_output.mp4"))
SCORECARD_PDF = os.getenv("VARBOX_PDF", os.path.join(_DEFAULT_OUT_DIR, "boxing_scorecard.pdf"))
PUNCH_EVIDENCE_DIR = os.getenv("VARBOX_EVIDENCE_DIR", os.path.join(_DEFAULT_OUT_DIR, "punch_evidence"))

# detection backend & model assets
ASSETS_DIR = os.getenv("VARBOX_ASSETS", os.path.join(_ROOT, "assets"))
BACKEND = os.getenv("VARBOX_BACKEND", "opencv")  # "opencv" or "yolov8"

# OpenCV-DNN (Lite) person detector files (optional)
DNN_PROTO = os.getenv(
    "VARBOX_SSD_PROTO",
    os.path.join(ASSETS_DIR, "models", "mobilenet_ssd", "deploy.prototxt"),
)
DNN_MODEL = os.getenv(
    "VARBOX_SSD_MODEL",
    os.path.join(ASSETS_DIR, "models", "mobilenet_ssd", "deploy.caffemodel"),
)

# YOLOv8 weights (Pro build)
YOLOV8_WEIGHTS = os.getenv("VARBOX_YOLOV8_WEIGHTS", os.path.join(ASSETS_DIR, "models", "yolov8n.pt"))
