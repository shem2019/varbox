import os

from runtime_profile import detect_runtime_profile

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
RUNTIME_PROFILE = detect_runtime_profile()
IS_APPLE_SILICON = RUNTIME_PROFILE.is_apple_silicon
BACKEND = os.getenv("VARBOX_BACKEND", "auto")  # "auto", "opencv", or "yolov8"
RESOLVED_BACKEND = RUNTIME_PROFILE.preferred_backend
YOLO_DEVICE = os.getenv("VARBOX_YOLO_DEVICE", RUNTIME_PROFILE.yolo_device)
YOLO_IMGSZ = int(os.getenv("VARBOX_YOLO_IMGSZ", str(RUNTIME_PROFILE.yolo_imgsz)) or "640")
YOLO_HALF = int(os.getenv("VARBOX_YOLO_HALF", "1" if RUNTIME_PROFILE.yolo_half else "0") or "0")
YOLO_TRACKER = os.getenv(
    "VARBOX_YOLO_TRACKER",
    "botsort",
).strip().lower()
YOLO_TRACKER_CONFIG = os.getenv("VARBOX_YOLO_TRACKER_CONFIG", "").strip()
YOLO_TRACK_PERSIST = int(os.getenv("VARBOX_YOLO_TRACK_PERSIST", "1") or "1")
YOLO_TRACK_CONF = float(os.getenv("VARBOX_YOLO_TRACK_CONF", "0.15") or "0.15")
YOLO_TRACK_IOU = float(os.getenv("VARBOX_YOLO_TRACK_IOU", "0.45") or "0.45")
REID_MODEL = os.getenv(
    "VARBOX_REID_MODEL",
    "osnet_ain_x1_0",
).strip().lower()
REID_DEVICE = os.getenv(
    "VARBOX_REID_DEVICE",
    "mps" if RUNTIME_PROFILE.has_mps else "cpu",
).strip().lower()
REID_IMGSZ = int(os.getenv("VARBOX_REID_IMGSZ", "256") or "256")
POSE_MODEL_COMPLEXITY = int(
    os.getenv("VARBOX_POSE_MODEL_COMPLEXITY", str(RUNTIME_PROFILE.pose_model_complexity)) or "1"
)
POSE_ENABLE_SEGMENTATION = int(
    os.getenv(
        "VARBOX_POSE_ENABLE_SEGMENTATION",
        "1" if RUNTIME_PROFILE.pose_enable_segmentation else "0",
    )
    or "0"
)

# Strike assessment is independent from the pose/tracking backend above.
STRIKE_BACKEND = os.getenv("VARBOX_STRIKE_BACKEND", "local").strip().lower()
ROBOFLOW_API_URL = os.getenv(
    "VARBOX_ROBOFLOW_API_URL", "https://serverless.roboflow.com"
).strip()
ROBOFLOW_MODEL_ID = os.getenv("VARBOX_ROBOFLOW_MODEL_ID", "boxing-vxhil/1").strip()
ROBOFLOW_CONFIDENCE = float(os.getenv("VARBOX_ROBOFLOW_CONFIDENCE", "0.35") or "0.35")
ROBOFLOW_SAMPLE_FPS = float(os.getenv("VARBOX_ROBOFLOW_SAMPLE_FPS", "5") or "5")

# Output orientation control.
# "portrait": rotate landscape frames to portrait.
# "landscape": rotate portrait frames to landscape.
# "source": keep source orientation.
OUTPUT_ORIENTATION = os.getenv("VARBOX_OUTPUT_ORIENTATION", "source").strip().lower()
# Used when a rotation is required: "clockwise" or "counterclockwise".
OUTPUT_ROTATION_DIRECTION = (
    os.getenv("VARBOX_OUTPUT_ROTATION_DIRECTION", "clockwise").strip().lower()
)

# Identity/corner orientation lock:
# 1 = once RED/BLUE are assigned, keep that orientation and block swaps.
# 0 = allow dynamic re-assignment/swaps.
LOCK_CORNER_ORIENTATION = int(os.getenv("VARBOX_LOCK_CORNER_ORIENTATION", "1") or "1")

# OpenCV-DNN (Lite) person detector files (optional)
DNN_PROTO = os.getenv(
    "VARBOX_SSD_PROTO",
    os.path.join(ASSETS_DIR, "models", "mobilenet_ssd", "deploy.prototxt"),
)
DNN_MODEL = os.getenv(
    "VARBOX_SSD_MODEL",
    os.path.join(ASSETS_DIR, "models", "mobilenet_ssd", "deploy.caffemodel"),
)

# Ultralytics pose weights. `VARBOX_YOLOV8_WEIGHTS` is kept as a legacy alias.
YOLO_POSE_WEIGHTS = os.getenv(
    "VARBOX_YOLO_POSE_WEIGHTS",
    os.getenv("VARBOX_YOLOV8_WEIGHTS", os.path.join(ASSETS_DIR, "models", "yolo11m-pose.pt")),
)
YOLOV8_WEIGHTS = YOLO_POSE_WEIGHTS
POSE_TASK_MODEL = os.getenv(
    "VARBOX_POSE_TASK_MODEL",
    os.path.join(ASSETS_DIR, "models", "mediapipe", "pose_landmarker_lite.task"),
)
