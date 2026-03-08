import datetime
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QAction, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

ROOT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(ROOT_DIR, "src")
if os.path.isdir(SRC_DIR) and SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from boxing_analytics.review import (
    AuditEvent,
    build_audit_chain,
    build_timeline_events,
    build_review_session,
    filter_timeline_events,
    final_audit_hash,
    format_timeline_rows,
    load_review_session,
    preview_scoring_outcome,
    save_review_session,
)
from boxing_analytics.calibration import load_profile
from opencv_guard import install_opencv_circle_guard
from score_tracker import ScoreTracker
from scorecard_generator import generate_scorecard
from video_processor import process_video

APP_TITLE = "VAR Box Pro"

APP_QSS = """
QMainWindow {
  background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #081018, stop:1 #111827);
}
QLabel {
  color: #edf2ff;
  font-family: "Bahnschrift", "Segoe UI", sans-serif;
}
QLabel#muted {
  color: #9fb0c8;
  font-size: 12px;
}
QFrame#card {
  background: rgba(14, 24, 37, 0.88);
  border: 1px solid #27425f;
  border-radius: 16px;
}
QLineEdit, QComboBox, QSpinBox, QTextEdit {
  background: rgba(8, 14, 23, 0.9);
  color: #edf2ff;
  border: 1px solid #2b4768;
  border-radius: 10px;
  padding: 8px 10px;
  font-family: "Segoe UI", sans-serif;
}
QTextEdit {
  font-family: "Consolas", "JetBrains Mono", monospace;
  font-size: 12px;
}
QPushButton {
  background: #0d8ccf;
  color: #f8fbff;
  border: none;
  border-radius: 11px;
  padding: 10px 14px;
  font-family: "Bahnschrift", "Segoe UI", sans-serif;
  font-weight: 600;
}
QPushButton:hover { background: #076ca2; }
QPushButton#ghost {
  background: transparent;
  border: 1px solid #2c4d6f;
}
QPushButton#danger {
  background: #b21f3d;
}
QProgressBar {
  background: rgba(8, 14, 23, 0.9);
  color: #d8e8ff;
  border: 1px solid #2b4768;
  border-radius: 8px;
  text-align: center;
}
QProgressBar::chunk {
  background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0d8ccf, stop:1 #13b67a);
  border-radius: 8px;
}
"""


class RunCancelled(RuntimeError):
    """Signal cancellation requested by operator."""


def ts():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def human_path(p):
    if not p:
        return "-"
    return p if len(p) < 90 else f"...{p[-87:]}"


@dataclass
class RunConfig:
    backend: str = "opencv"
    input_path: Optional[str] = None
    use_camera: bool = False
    camera_index: int = 0
    camera_seconds: int = 60
    fps_override: Optional[int] = None
    red_name: str = "Red Corner"
    blue_name: str = "Blue Corner"
    out_dir: Optional[str] = None
    manual_seeds: Optional[Dict] = None
    rounds_count: int = 12
    round_seconds: float = 180.0
    rest_seconds: float = 60.0
    warmup_seconds: float = 0.0
    round_start_offset_seconds: float = 0.0
    unknown_corners: bool = False
    ref_events: List[Dict] = field(default_factory=list)
    manual_corrections: List[Dict] = field(default_factory=list)
    calibration_profile_path: Optional[str] = None


def _load_annotation_buffer(
    video_path: str, max_frames: int = 500, preview_width: int = 480
) -> List[Dict]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    frames = []
    frame_idx = 0
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        scale = float(preview_width) / max(1, w)
        ph = max(120, int(h * scale))
        preview = cv2.resize(frame, (preview_width, ph), interpolation=cv2.INTER_AREA)
        ts_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
        frames.append(
            {
                "frame_idx": frame_idx,
                "preview": preview,
                "timestamp_s": (ts_msec / 1000.0) if ts_msec > 0 else 0.0,
            }
        )
        frame_idx += 1
    cap.release()
    return frames


def _collect_manual_seed_annotations(video_path: str, max_frames: int = 500) -> Optional[Dict]:
    """
    Loads up to first 500 frames into an annotation canvas and lets user mark
    RED/BLUE fingerprints with mouse ROI.
    """
    frames = _load_annotation_buffer(video_path, max_frames=max_frames, preview_width=480)
    if not frames:
        return None

    window = "VAR Box Fingerprint Canvas"
    current = 0
    seeds: Dict[str, Optional[Dict]] = {"RED": None, "BLUE": None}

    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 980, 680)
        cv2.createTrackbar("Frame", window, 0, len(frames) - 1, lambda *_: None)
    except Exception:
        return None

    while True:
        current = cv2.getTrackbarPos("Frame", window)
        data = frames[current]
        canvas = data["preview"].copy()
        ph, pw = canvas.shape[:2]

        cv2.putText(
            canvas,
            "Loaded first 500 frames | n/p move | r=RED ROI | b=BLUE ROI | Enter=confirm | Esc=cancel",
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            canvas,
            f"Frame {current + 1}/{len(frames)} (source idx {data['frame_idx']})",
            (10, ph - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 220, 255),
            2,
        )

        for role, col, y in (("RED", (0, 0, 255), 54), ("BLUE", (255, 110, 70), 82)):
            seed = seeds[role]
            status = "set" if seed is not None else "unset"
            cv2.putText(
                canvas, f"{role}: {status}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 2
            )
            if seed is not None and seed["buffer_index"] == current:
                x1, y1, x2, y2 = seed["preview_box"]
                cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 2)

        cv2.imshow(window, canvas)
        key = cv2.waitKey(0) & 0xFF

        if key == 27:  # esc
            cv2.destroyWindow(window)
            return None
        if key in (ord("n"), 83):
            current = min(len(frames) - 1, current + 1)
            cv2.setTrackbarPos("Frame", window, current)
            continue
        if key in (ord("p"), 81):
            current = max(0, current - 1)
            cv2.setTrackbarPos("Frame", window, current)
            continue
        if key == ord("r"):
            roi = cv2.selectROI(window, data["preview"], fromCenter=False, showCrosshair=True)
            if roi and roi[2] > 0 and roi[3] > 0:
                x, y, w, h = [int(v) for v in roi]
                seeds["RED"] = {
                    "buffer_index": current,
                    "frame_idx": int(data["frame_idx"]),
                    "preview_box": (x, y, x + w, y + h),
                    "rel_box": [x / pw, y / ph, (x + w) / pw, (y + h) / ph],
                }
            continue
        if key == ord("b"):
            roi = cv2.selectROI(window, data["preview"], fromCenter=False, showCrosshair=True)
            if roi and roi[2] > 0 and roi[3] > 0:
                x, y, w, h = [int(v) for v in roi]
                seeds["BLUE"] = {
                    "buffer_index": current,
                    "frame_idx": int(data["frame_idx"]),
                    "preview_box": (x, y, x + w, y + h),
                    "rel_box": [x / pw, y / ph, (x + w) / pw, (y + h) / ph],
                }
            continue
        if key in (13, 10, 32):  # enter/space
            if seeds["RED"] is None or seeds["BLUE"] is None:
                continue
            break

    cv2.destroyWindow(window)
    return {
        "RED": {"frame_idx": seeds["RED"]["frame_idx"], "rel_box": seeds["RED"]["rel_box"]},
        "BLUE": {"frame_idx": seeds["BLUE"]["frame_idx"], "rel_box": seeds["BLUE"]["rel_box"]},
    }


def _choose_round_start_marker(video_path: str, max_frames: int = 1200) -> Optional[float]:
    frames = _load_annotation_buffer(video_path, max_frames=max_frames, preview_width=720)
    if not frames:
        return None

    window = "VAR Box Round Start Alignment"
    current = 0
    marker_s = 0.0

    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 1100, 760)
        cv2.createTrackbar("Frame", window, 0, len(frames) - 1, lambda *_: None)
    except Exception:
        return None

    while True:
        current = cv2.getTrackbarPos("Frame", window)
        data = frames[current]
        canvas = data["preview"].copy()
        ph, _ = canvas.shape[:2]

        cv2.putText(
            canvas,
            "Round Align | n/p move | m=set marker | Enter=confirm | Esc=cancel",
            (14, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            canvas,
            f"Frame {current + 1}/{len(frames)} | ts {data['timestamp_s']:.2f}s | marker {marker_s:.2f}s",
            (14, ph - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 220, 255),
            2,
        )
        cv2.imshow(window, canvas)
        key = cv2.waitKey(0) & 0xFF

        if key == 27:
            cv2.destroyWindow(window)
            return None
        if key in (ord("n"), 83):
            current = min(len(frames) - 1, current + 1)
            cv2.setTrackbarPos("Frame", window, current)
            continue
        if key in (ord("p"), 81):
            current = max(0, current - 1)
            cv2.setTrackbarPos("Frame", window, current)
            continue
        if key == ord("m"):
            marker_s = float(data["timestamp_s"])
            continue
        if key in (13, 10, 32):
            break

    cv2.destroyWindow(window)
    return marker_s


class PipelineWorker(QThread):
    progress = Signal(str)
    progress_value = Signal(int)
    done = Signal(str, str, str)  # out_dir, video_path, pdf_path
    cancelled = Signal(str, str)  # out_dir, message
    error = Signal(str)

    def __init__(self, cfg: RunConfig, parent=None):
        super().__init__(parent)
        self.cfg = cfg

    def run(self):
        out_dir = ""
        try:
            self.progress.emit("Preparing analysis run...")
            self.progress_value.emit(0)
            if self.isInterruptionRequested():
                raise RunCancelled("Cancelled before run initialization.")
            if self.cfg.use_camera:
                self.progress.emit(
                    f"Recording from camera #{self.cfg.camera_index} ({self.cfg.camera_seconds}s)..."
                )
                tmpdir = tempfile.mkdtemp(prefix="varbox_cam_")
                input_path = os.path.join(tmpdir, f"camera_{ts()}.mp4")
                self._record_camera(self.cfg.camera_index, self.cfg.camera_seconds, input_path)
            else:
                if not self.cfg.input_path or not os.path.isfile(self.cfg.input_path):
                    raise RuntimeError("No valid input video selected.")
                input_path = self.cfg.input_path

            base = os.path.splitext(os.path.basename(input_path))[0]
            out_dir = self.cfg.out_dir or os.path.abspath(
                os.path.join("output", "runs", f"{base}_{ts()}")
            )
            os.makedirs(out_dir, exist_ok=True)
            out_video = os.path.join(out_dir, f"{base}_scored.mp4")
            out_pdf = os.path.join(out_dir, f"{base}_analysis_report.pdf")
            out_meta = os.path.join(out_dir, "analysis_metadata.json")
            evidence_dir = os.path.join(out_dir, "punch_evidence")
            os.makedirs(evidence_dir, exist_ok=True)

            # Runtime env consumed by config/video pipeline
            os.environ["VARBOX_INPUT"] = input_path
            os.environ["VARBOX_OUTPUT"] = out_video
            os.environ["VARBOX_PDF"] = out_pdf
            os.environ["VARBOX_METADATA_PATH"] = out_meta
            os.environ["VARBOX_EVIDENCE_DIR"] = evidence_dir
            os.environ["VARBOX_BACKEND"] = self.cfg.backend
            os.environ["VARBOX_RED_NAME"] = self.cfg.red_name
            os.environ["VARBOX_BLUE_NAME"] = self.cfg.blue_name
            os.environ["VARBOX_OUT_DIR"] = out_dir
            os.environ["VARBOX_ALLOW_MANUAL_BOOTSTRAP"] = "0"
            if self.cfg.manual_seeds:
                os.environ["VARBOX_MANUAL_SEEDS"] = json.dumps(self.cfg.manual_seeds)
            elif "VARBOX_MANUAL_SEEDS" in os.environ:
                del os.environ["VARBOX_MANUAL_SEEDS"]
            if self.cfg.fps_override:
                os.environ["VARBOX_FPS_OVERRIDE"] = str(self.cfg.fps_override)
            elif "VARBOX_FPS_OVERRIDE" in os.environ:
                del os.environ["VARBOX_FPS_OVERRIDE"]
            os.environ["VARBOX_ROUNDS_COUNT"] = str(self.cfg.rounds_count)
            os.environ["VARBOX_ROUND_SECONDS"] = str(self.cfg.round_seconds)
            os.environ["VARBOX_REST_SECONDS"] = str(self.cfg.rest_seconds)
            os.environ["VARBOX_WARMUP_SECONDS"] = str(self.cfg.warmup_seconds)
            os.environ["VARBOX_ROUND_START_OFFSET_SECONDS"] = str(
                self.cfg.round_start_offset_seconds
            )
            os.environ["VARBOX_UNKNOWN_CORNERS"] = "1" if self.cfg.unknown_corners else "0"
            os.environ["VARBOX_REF_EVENTS"] = json.dumps(self.cfg.ref_events)
            os.environ["VARBOX_MANUAL_CORRECTIONS"] = json.dumps(self.cfg.manual_corrections)
            if self.cfg.calibration_profile_path:
                os.environ["VARBOX_CALIBRATION_PROFILE"] = self.cfg.calibration_profile_path
            elif "VARBOX_CALIBRATION_PROFILE" in os.environ:
                del os.environ["VARBOX_CALIBRATION_PROFILE"]

            self.progress.emit("Running object recognition + pose analysis...")
            tracker = ScoreTracker()
            tracker.metadata = {
                "title": "VAR Box Professional Match Analysis",
                "subtitle": f"Source: {os.path.basename(input_path)}",
                "footer": "Automated scoring support tool. Final judging remains under licensed officials.",
            }
            process_video(
                tracker,
                progress_cb=self._on_processing_progress,
                should_stop=self.isInterruptionRequested,
            )

            if int(tracker.metadata.get("cancelled", 0) or 0):
                self.cancelled.emit(out_dir, "Run cancelled by operator.")
                return

            self.progress.emit("Generating professional PDF report...")
            self.progress_value.emit(98)
            if self.isInterruptionRequested():
                raise RunCancelled("Cancelled before report generation.")
            generate_scorecard(tracker, out_pdf)
            self.progress_value.emit(100)
            self.done.emit(out_dir, out_video, out_pdf)
        except RunCancelled as exc:
            self.cancelled.emit(out_dir, str(exc))
        except Exception as exc:
            self.error.emit(str(exc))

    def _on_processing_progress(self, message: str, percent: Optional[int] = None):
        self.progress.emit(message)
        if percent is not None:
            self.progress_value.emit(max(0, min(100, int(percent))))

    def _record_camera(self, index: int, seconds: int, out_path: str):
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {index}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        start = time.time()
        last_tick = -1
        while time.time() - start < seconds:
            if self.isInterruptionRequested():
                cap.release()
                out.release()
                raise RunCancelled("Cancelled during camera recording.")
            ok, frame = cap.read()
            if not ok:
                break
            out.write(frame)
            elapsed = time.time() - start
            tick = int(elapsed)
            if tick != last_tick:
                last_tick = tick
                pct = int(min(95.0, max(0.0, (elapsed / max(1, seconds)) * 20.0)))
                self.progress.emit(f"Recording camera clip... {elapsed:.1f}s/{seconds}s")
                self.progress_value.emit(pct)
        cap.release()
        out.release()


class DropLabel(QLabel):
    fileSelected = Signal(str)

    def __init__(self, parent=None):
        super().__init__("Drop a fight video here\n(MP4 / MOV / AVI / MKV)", parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(140)
        self.setStyleSheet(
            """
            QLabel {
              background: rgba(7, 12, 20, 0.74);
              border: 2px dashed #2f5378;
              border-radius: 14px;
              color: #8db0d6;
              font-size: 14px;
              font-family: "Bahnschrift", "Segoe UI";
            }
            """
        )

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if os.path.isfile(path):
                self.fileSelected.emit(path)
                return


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cfg = RunConfig()
        self.worker: Optional[PipelineWorker] = None
        self.last_out_dir = os.path.abspath("output")
        self.audit_events: List[AuditEvent] = []
        self.last_analysis_payload: Dict = {}
        self.timeline_events: List[Dict] = []
        self.filtered_timeline_events: List[Dict] = []
        self.setWindowTitle(APP_TITLE)
        self.resize(1180, 780)
        self.setStyleSheet(APP_QSS)
        self._build()

    def _build(self):
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        header = QFrame()
        header.setObjectName("card")
        hbox = QHBoxLayout(header)
        hbox.setContentsMargins(18, 14, 18, 14)
        tcol = QVBoxLayout()
        title = QLabel("VAR Box Pro")
        title.setStyleSheet("font-size: 27px; font-weight: 800;")
        subtitle = QLabel(
            "Computer-vision boxing scoring with corner lock, punch trajectories, and evidence report."
        )
        subtitle.setObjectName("muted")
        tcol.addWidget(title)
        tcol.addWidget(subtitle)
        hbox.addLayout(tcol, 1)
        self.run_badge = QLabel("Idle")
        self.run_badge.setStyleSheet(
            "background:#1b2f46; border:1px solid #31577e; border-radius:10px; padding:6px 10px; "
            "font-size:12px; color:#b6d4f2;"
        )
        hbox.addWidget(self.run_badge, 0, Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(header)

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)
        layout.addLayout(grid, 1)

        left = QFrame()
        left.setObjectName("card")
        left_v = QVBoxLayout(left)
        left_v.setContentsMargins(16, 16, 16, 16)
        left_v.setSpacing(12)

        self.drop = DropLabel()
        self.drop.fileSelected.connect(self._set_video)
        left_v.addWidget(self.drop)

        source_row = QHBoxLayout()
        self.btn_browse = QPushButton("Browse Video")
        self.btn_browse.clicked.connect(self._browse)
        self.btn_camera = QPushButton("Use Camera")
        self.btn_camera.setObjectName("ghost")
        self.btn_camera.clicked.connect(self._toggle_cam)
        self.btn_align_round = QPushButton("Align Round Start")
        self.btn_align_round.setObjectName("ghost")
        self.btn_align_round.clicked.connect(self._align_round_start)
        source_row.addWidget(self.btn_browse)
        source_row.addWidget(self.btn_camera)
        source_row.addWidget(self.btn_align_round)
        left_v.addLayout(source_row)
        calib_row = QHBoxLayout()
        self.btn_load_calib = QPushButton("Load Calibration Profile")
        self.btn_load_calib.setObjectName("ghost")
        self.btn_load_calib.clicked.connect(self._select_calibration_profile)
        self.btn_clear_calib = QPushButton("Clear Calibration")
        self.btn_clear_calib.setObjectName("ghost")
        self.btn_clear_calib.clicked.connect(self._clear_calibration_profile)
        calib_row.addWidget(self.btn_load_calib)
        calib_row.addWidget(self.btn_clear_calib)
        left_v.addLayout(calib_row)

        form = QGridLayout()
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(10)
        self.ed_red = QLineEdit("Red Corner")
        self.ed_blue = QLineEdit("Blue Corner")
        self.cb_corner_mode = QComboBox()
        self.cb_corner_mode.addItems(["Named Corners (Red/Blue)", "Unknown Corners (Fighter A/B)"])
        self.cb_corner_mode.currentIndexChanged.connect(self._on_corner_mode_changed)
        self.cb_backend = QComboBox()
        self.cb_backend.addItems(["Lite (OpenCV-DNN)", "Pro (YOLOv8)"])
        self.cb_backend.setCurrentIndex(1)
        self.cb_cam_index = QComboBox()
        self.cb_cam_index.addItems([str(i) for i in range(0, 6)])
        self.cb_cam_index.setEnabled(False)
        self.spn_secs = QSpinBox()
        self.spn_secs.setRange(5, 3600)
        self.spn_secs.setValue(60)
        self.spn_secs.setEnabled(False)
        self.spn_fps = QSpinBox()
        self.spn_fps.setRange(0, 240)
        self.spn_fps.setValue(0)
        self.cb_bout_preset = QComboBox()
        self.cb_bout_preset.addItems(
            [
                "Pro Boxing (12x3 + 1m rest)",
                "Amateur Elite (3x3 + 1m rest)",
                "Amateur Novice (3x2 + 1m rest)",
                "Custom",
            ]
        )
        self.cb_bout_preset.currentIndexChanged.connect(self._apply_bout_preset)
        self.spn_rounds = QSpinBox()
        self.spn_rounds.setRange(1, 15)
        self.spn_rounds.setValue(12)
        self.spn_round_seconds = QDoubleSpinBox()
        self.spn_round_seconds.setRange(30.0, 600.0)
        self.spn_round_seconds.setSingleStep(5.0)
        self.spn_round_seconds.setValue(180.0)
        self.spn_rest_seconds = QDoubleSpinBox()
        self.spn_rest_seconds.setRange(0.0, 300.0)
        self.spn_rest_seconds.setSingleStep(5.0)
        self.spn_rest_seconds.setValue(60.0)
        self.spn_warmup_seconds = QDoubleSpinBox()
        self.spn_warmup_seconds.setRange(0.0, 120.0)
        self.spn_warmup_seconds.setSingleStep(1.0)
        self.spn_warmup_seconds.setValue(0.0)
        self.spn_round_offset = QDoubleSpinBox()
        self.spn_round_offset.setRange(0.0, 600.0)
        self.spn_round_offset.setSingleStep(0.25)
        self.spn_round_offset.setDecimals(2)
        self.spn_round_offset.setValue(0.0)

        form.addWidget(QLabel("Red Corner Name"), 0, 0)
        form.addWidget(self.ed_red, 0, 1)
        form.addWidget(QLabel("Blue Corner Name"), 1, 0)
        form.addWidget(self.ed_blue, 1, 1)
        form.addWidget(QLabel("Corner Label Mode"), 2, 0)
        form.addWidget(self.cb_corner_mode, 2, 1)
        form.addWidget(QLabel("Detection Backend"), 3, 0)
        form.addWidget(self.cb_backend, 3, 1)
        form.addWidget(QLabel("Camera Index"), 4, 0)
        form.addWidget(self.cb_cam_index, 4, 1)
        form.addWidget(QLabel("Record Duration (sec)"), 5, 0)
        form.addWidget(self.spn_secs, 5, 1)
        form.addWidget(QLabel("FPS Override (0=auto)"), 6, 0)
        form.addWidget(self.spn_fps, 6, 1)
        form.addWidget(QLabel("Bout Preset"), 7, 0)
        form.addWidget(self.cb_bout_preset, 7, 1)
        form.addWidget(QLabel("Rounds"), 8, 0)
        form.addWidget(self.spn_rounds, 8, 1)
        form.addWidget(QLabel("Round Seconds"), 9, 0)
        form.addWidget(self.spn_round_seconds, 9, 1)
        form.addWidget(QLabel("Rest Seconds"), 10, 0)
        form.addWidget(self.spn_rest_seconds, 10, 1)
        form.addWidget(QLabel("Warmup Seconds"), 11, 0)
        form.addWidget(self.spn_warmup_seconds, 11, 1)
        form.addWidget(QLabel("Round Start Offset (sec)"), 12, 0)
        form.addWidget(self.spn_round_offset, 12, 1)
        left_v.addLayout(form)

        self.lbl_selected = QLabel("Selected Source: -")
        self.lbl_selected.setObjectName("muted")
        left_v.addWidget(self.lbl_selected)
        self.lbl_disclaimer = QLabel(
            "Decision support only. Human judge/referee confirmation is required. "
            "Unverified without calibration and validation."
        )
        self.lbl_disclaimer.setWordWrap(True)
        self.lbl_disclaimer.setObjectName("muted")
        left_v.addWidget(self.lbl_disclaimer)
        self.lbl_calib_status = QLabel("Calibration: Unverified (no profile loaded)")
        self.lbl_calib_status.setWordWrap(True)
        self.lbl_calib_status.setObjectName("muted")
        left_v.addWidget(self.lbl_calib_status)
        grid.addWidget(left, 0, 0, 2, 1)

        right_top = QFrame()
        right_top.setObjectName("card")
        rt = QVBoxLayout(right_top)
        rt.setContentsMargins(16, 16, 16, 16)
        rt.setSpacing(10)
        rt.addWidget(QLabel("Run"))
        btn_row = QHBoxLayout()
        self.btn_run = QPushButton("Start Analysis")
        self.btn_run.clicked.connect(self._run)
        self.btn_abort = QPushButton("Reset")
        self.btn_abort.setObjectName("danger")
        self.btn_abort.clicked.connect(self._handle_abort_or_reset)
        self.btn_open = QPushButton("Open Output Folder")
        self.btn_open.setObjectName("ghost")
        self.btn_open.clicked.connect(self._open_output)
        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_abort)
        btn_row.addWidget(self.btn_open)
        rt.addLayout(btn_row)
        ref_row = QHBoxLayout()
        self.cb_ref_event = QComboBox()
        self.cb_ref_event.addItems(["Knockdown", "Foul", "Deduction"])
        self.cb_ref_role = QComboBox()
        self.cb_ref_role.addItems(["RED", "BLUE"])
        self.spn_ref_round = QSpinBox()
        self.spn_ref_round.setRange(1, 15)
        self.spn_ref_round.setValue(1)
        self.spn_rounds.valueChanged.connect(lambda v: self.spn_ref_round.setMaximum(int(v)))
        self.spn_ref_time = QDoubleSpinBox()
        self.spn_ref_time.setRange(0.0, 7200.0)
        self.spn_ref_time.setSingleStep(0.25)
        self.spn_ref_time.setDecimals(2)
        self.spn_ref_time.setValue(0.0)
        self.spn_ref_points = QSpinBox()
        self.spn_ref_points.setRange(1, 5)
        self.spn_ref_points.setValue(1)
        self.btn_add_ref = QPushButton("Confirm Ref Event")
        self.btn_add_ref.setObjectName("ghost")
        self.btn_add_ref.clicked.connect(self._add_ref_event)
        ref_row.addWidget(self.cb_ref_event)
        ref_row.addWidget(self.cb_ref_role)
        ref_row.addWidget(self.spn_ref_round)
        ref_row.addWidget(self.spn_ref_time)
        ref_row.addWidget(self.spn_ref_points)
        ref_row.addWidget(self.btn_add_ref)
        rt.addLayout(ref_row)
        corr_row = QHBoxLayout()
        self.spn_corr_time = QDoubleSpinBox()
        self.spn_corr_time.setRange(0.0, 7200.0)
        self.spn_corr_time.setSingleStep(0.25)
        self.spn_corr_time.setDecimals(2)
        self.spn_corr_time.setValue(0.0)
        self.cb_corr_match_role = QComboBox()
        self.cb_corr_match_role.addItems(["ANY", "RED", "BLUE"])
        self.cb_corr_new_role = QComboBox()
        self.cb_corr_new_role.addItems(["Keep Role", "RED", "BLUE"])
        self.cb_corr_new_label = QComboBox()
        self.cb_corr_new_label.addItems(
            [
                "Keep Label",
                "landed_clean",
                "landed_glancing",
                "blocked_guarded",
                "missed",
                "clinch",
            ]
        )
        self.cb_corr_new_zone = QComboBox()
        self.cb_corr_new_zone.addItems(["Keep Zone", "head", "body", "unknown"])
        self.cb_corr_invalidate = QComboBox()
        self.cb_corr_invalidate.addItems(["Keep Event", "Invalidate"])
        self.btn_add_corr = QPushButton("Queue Correction")
        self.btn_add_corr.setObjectName("ghost")
        self.btn_add_corr.clicked.connect(self._add_manual_correction)
        corr_row.addWidget(self.spn_corr_time)
        corr_row.addWidget(self.cb_corr_match_role)
        corr_row.addWidget(self.cb_corr_new_role)
        corr_row.addWidget(self.cb_corr_new_label)
        corr_row.addWidget(self.cb_corr_new_zone)
        corr_row.addWidget(self.cb_corr_invalidate)
        corr_row.addWidget(self.btn_add_corr)
        rt.addLayout(corr_row)
        rt.addWidget(QLabel("Queued Corrections"))
        corr_actions = QHBoxLayout()
        self.spn_corr_remove_index = QSpinBox()
        self.spn_corr_remove_index.setRange(0, 0)
        self.spn_corr_remove_index.setValue(0)
        self.btn_corr_remove = QPushButton("Remove Selected Correction")
        self.btn_corr_remove.setObjectName("ghost")
        self.btn_corr_remove.clicked.connect(self._remove_manual_correction)
        self.btn_corr_clear = QPushButton("Clear Corrections")
        self.btn_corr_clear.setObjectName("ghost")
        self.btn_corr_clear.clicked.connect(self._clear_manual_corrections)
        corr_actions.addWidget(self.spn_corr_remove_index)
        corr_actions.addWidget(self.btn_corr_remove)
        corr_actions.addWidget(self.btn_corr_clear)
        rt.addLayout(corr_actions)
        session_actions = QHBoxLayout()
        self.btn_save_session = QPushButton("Save Edit Session")
        self.btn_save_session.setObjectName("ghost")
        self.btn_save_session.clicked.connect(self._save_edit_session)
        self.btn_load_session = QPushButton("Load Edit Session")
        self.btn_load_session.setObjectName("ghost")
        self.btn_load_session.clicked.connect(self._load_edit_session)
        session_actions.addWidget(self.btn_save_session)
        session_actions.addWidget(self.btn_load_session)
        rt.addLayout(session_actions)
        self.txt_corr_queue = QTextEdit()
        self.txt_corr_queue.setReadOnly(True)
        self.txt_corr_queue.setMaximumHeight(110)
        self.txt_corr_queue.setPlaceholderText("No queued corrections.")
        rt.addWidget(self.txt_corr_queue)
        self.btn_export_audit = QPushButton("Export Audit Log")
        self.btn_export_audit.setObjectName("ghost")
        self.btn_export_audit.clicked.connect(self._export_audit_log)
        rt.addWidget(self.btn_export_audit)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setVisible(False)
        rt.addWidget(self.progress)
        self.lbl_result = QLabel("Output: -")
        self.lbl_result.setObjectName("muted")
        rt.addWidget(self.lbl_result)
        grid.addWidget(right_top, 0, 1)

        right_bottom = QFrame()
        right_bottom.setObjectName("card")
        rb = QVBoxLayout(right_bottom)
        rb.setContentsMargins(16, 16, 16, 16)
        rb.addWidget(QLabel("Processing Log"))
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setPlaceholderText("Pipeline logs appear here...")
        rb.addWidget(self.txt_log, 1)
        rb.addWidget(QLabel("Review Timeline"))
        timeline_filters = QHBoxLayout()
        self.cb_tl_role = QComboBox()
        self.cb_tl_role.addItems(["ALL", "RED", "BLUE"])
        self.cb_tl_label = QComboBox()
        self.cb_tl_label.addItems(
            [
                "ALL",
                "landed_clean",
                "landed_glancing",
                "blocked_guarded",
                "missed",
                "clinch",
                "knockdown",
                "foul",
                "deduction",
            ]
        )
        self.cb_tl_zone = QComboBox()
        self.cb_tl_zone.addItems(["ALL", "head", "body", "unknown"])
        self.spn_tl_round = QSpinBox()
        self.spn_tl_round.setRange(0, 15)
        self.spn_tl_round.setValue(0)
        self.btn_tl_refresh = QPushButton("Refresh")
        self.btn_tl_refresh.setObjectName("ghost")
        self.btn_tl_refresh.clicked.connect(self._refresh_timeline_view)
        timeline_filters.addWidget(self.cb_tl_role)
        timeline_filters.addWidget(self.cb_tl_label)
        timeline_filters.addWidget(self.cb_tl_zone)
        timeline_filters.addWidget(self.spn_tl_round)
        timeline_filters.addWidget(self.btn_tl_refresh)
        rb.addLayout(timeline_filters)
        timeline_actions = QHBoxLayout()
        self.spn_tl_select_index = QSpinBox()
        self.spn_tl_select_index.setRange(0, 0)
        self.btn_tl_use = QPushButton("Load Selected Into Correction")
        self.btn_tl_use.setObjectName("ghost")
        self.btn_tl_use.clicked.connect(self._load_selected_timeline_event)
        self.btn_preview_score = QPushButton("Preview Corrected Scoring")
        self.btn_preview_score.setObjectName("ghost")
        self.btn_preview_score.clicked.connect(self._preview_corrected_scoring)
        timeline_actions.addWidget(self.spn_tl_select_index)
        timeline_actions.addWidget(self.btn_tl_use)
        timeline_actions.addWidget(self.btn_preview_score)
        rb.addLayout(timeline_actions)
        self.txt_timeline = QTextEdit()
        self.txt_timeline.setReadOnly(True)
        self.txt_timeline.setPlaceholderText(
            "Timeline events appear after a run. Use filters then load an event into correction controls."
        )
        rb.addWidget(self.txt_timeline, 1)
        grid.addWidget(right_bottom, 1, 1)

        grid.setColumnStretch(0, 3)
        grid.setColumnStretch(1, 4)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 2)

        self.setCentralWidget(root)

        act_quit = QAction("Quit", self)
        act_quit.triggered.connect(self.close)
        self.menuBar().addAction(act_quit)
        self._apply_bout_preset()
        self._on_corner_mode_changed()
        self._refresh_correction_queue_view()

    def _log(self, msg: str):
        self.txt_log.append(msg)

    def _audit(self, action: str, details: str):
        evt = AuditEvent(timestamp_s=time.time(), actor="operator", action=action, details=details)
        self.audit_events.append(evt)

    def _add_ref_event(self):
        event_type_ui = self.cb_ref_event.currentText().strip().lower()
        event_type = {
            "knockdown": "knockdown",
            "foul": "foul",
            "deduction": "deduction",
        }.get(event_type_ui, "foul")
        role = self.cb_ref_role.currentText().strip().upper()
        round_no = int(self.spn_ref_round.value())
        timestamp_s = float(self.spn_ref_time.value())
        points = int(self.spn_ref_points.value())

        event = {
            "event_type": event_type,
            "role": role,
            "round": round_no,
            "timestamp_s": round(timestamp_s, 3),
            "points": points,
            "count": points if event_type in ("knockdown", "foul") else 1,
            "note": "manual_confirmed",
            "confirmed_at_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        self.cfg.ref_events.append(event)
        self._log(
            f"Ref event confirmed: {event_type.upper()} role={role} round={round_no} "
            f"t={timestamp_s:.2f}s pts={points}"
        )
        self._audit("ref_event_confirmed", json.dumps(event))

    def _add_manual_correction(self):
        match_role = self.cb_corr_match_role.currentText().strip().upper()
        new_role = self.cb_corr_new_role.currentText().strip().upper()
        new_label = self.cb_corr_new_label.currentText().strip()
        new_zone = self.cb_corr_new_zone.currentText().strip().lower()
        invalidate = self.cb_corr_invalidate.currentIndex() == 1
        timestamp_s = float(self.spn_corr_time.value())

        correction = {
            "timestamp_s": round(timestamp_s, 3),
            "match_role": "" if match_role == "ANY" else match_role,
            "new_role": "" if new_role == "KEEP ROLE" else new_role,
            "new_label": "" if new_label == "Keep Label" else new_label,
            "new_target_zone": "" if new_zone == "keep zone" else new_zone,
            "invalidate": 1 if invalidate else 0,
            "note": "operator_review_correction",
        }
        if (
            not correction["new_role"]
            and not correction["new_label"]
            and not correction["new_target_zone"]
            and not correction["invalidate"]
        ):
            QMessageBox.warning(
                self,
                APP_TITLE,
                "Correction ignored: pick a relabel/reassign field or select Invalidate.",
            )
            return

        self.cfg.manual_corrections.append(correction)
        self._refresh_correction_queue_view()
        self._log(
            (
                "Correction queued: "
                f"t={timestamp_s:.2f}s role={correction['match_role'] or 'ANY'} "
                f"new_role={correction['new_role'] or '-'} "
                f"new_label={correction['new_label'] or '-'} "
                f"new_zone={correction['new_target_zone'] or '-'} "
                f"invalidate={correction['invalidate']}"
            )
        )
        self._audit("manual_correction_queued", json.dumps(correction))

    def _refresh_correction_queue_view(self):
        lines = []
        for idx, row in enumerate(self.cfg.manual_corrections):
            timestamp_s = float(row.get("timestamp_s", 0.0) or 0.0)
            role = str(row.get("match_role", "")).strip() or "ANY"
            new_role = str(row.get("new_role", "")).strip() or "-"
            new_label = str(row.get("new_label", "")).strip() or "-"
            new_zone = str(row.get("new_target_zone", "")).strip() or "-"
            inval = int(row.get("invalidate", 0) or 0)
            lines.append(
                (
                    f"[{idx}] t={timestamp_s:.2f}s role={role} "
                    f"new_role={new_role} label={new_label} zone={new_zone} "
                    f"invalidate={inval}"
                )
            )
        if lines:
            self.txt_corr_queue.setPlainText("\n".join(lines))
        else:
            self.txt_corr_queue.setPlainText("No queued corrections.")
        max_idx = max(0, len(self.cfg.manual_corrections) - 1)
        self.spn_corr_remove_index.setMaximum(max_idx)
        if self.spn_corr_remove_index.value() > max_idx:
            self.spn_corr_remove_index.setValue(max_idx)

    def _remove_manual_correction(self):
        if not self.cfg.manual_corrections:
            QMessageBox.warning(self, APP_TITLE, "No manual corrections are queued.")
            return
        idx = int(self.spn_corr_remove_index.value())
        if idx < 0 or idx >= len(self.cfg.manual_corrections):
            QMessageBox.warning(self, APP_TITLE, "Selected correction index is out of range.")
            return
        removed = self.cfg.manual_corrections.pop(idx)
        self._refresh_correction_queue_view()
        self._log(f"Removed correction #{idx}.")
        self._audit("manual_correction_removed", json.dumps(removed))

    def _clear_manual_corrections(self):
        if not self.cfg.manual_corrections:
            self._refresh_correction_queue_view()
            return
        count = len(self.cfg.manual_corrections)
        self.cfg.manual_corrections.clear()
        self._refresh_correction_queue_view()
        self._log(f"Cleared {count} queued manual corrections.")
        self._audit("manual_corrections_cleared", f"count={count}")

    def _save_edit_session(self):
        base = self.last_out_dir if os.path.isdir(self.last_out_dir) else os.path.abspath("output")
        os.makedirs(base, exist_ok=True)
        suggested = os.path.join(base, f"edit_session_{ts()}.json")
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Edit Session",
            suggested,
            "JSON (*.json)",
        )
        if not out_path:
            return

        chain = build_audit_chain(self.audit_events)
        input_video = ""
        output_video = ""
        metadata_path = self._analysis_metadata_path(base)
        if isinstance(self.last_analysis_payload, dict):
            output_video = str(self.last_analysis_payload.get("output_video", "")).strip()
            meta = self.last_analysis_payload.get("metadata", {})
            if isinstance(meta, dict):
                input_video = str(meta.get("input_video", "")).strip()
                metadata_path = str(meta.get("analysis_metadata_path", metadata_path)).strip()
        session = build_review_session(
            input_video=input_video,
            output_video=output_video,
            metadata_path=metadata_path,
            ref_events=[dict(row) for row in self.cfg.ref_events],
            manual_corrections=[dict(row) for row in self.cfg.manual_corrections],
            audit_events=chain,
            final_audit_hash=final_audit_hash(self.audit_events),
        )
        save_review_session(session, out_path)
        self._log(f"Edit session saved: {out_path}")
        self._audit("edit_session_saved", out_path)

    def _load_edit_session(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Edit Session",
            self.last_out_dir if os.path.isdir(self.last_out_dir) else os.path.abspath("output"),
            "JSON (*.json);;All files (*.*)",
        )
        if not path:
            return
        try:
            session = load_review_session(path)
        except Exception as exc:
            QMessageBox.critical(self, APP_TITLE, f"Failed to load edit session:\n{exc}")
            self._audit("edit_session_load_failed", f"path={path} err={exc}")
            return

        self.cfg.ref_events = [dict(row) for row in session.ref_events]
        self.cfg.manual_corrections = [dict(row) for row in session.manual_corrections]
        self._refresh_correction_queue_view()
        self._log(
            (
                f"Loaded edit session: {path} "
                f"(ref_events={len(self.cfg.ref_events)} corrections={len(self.cfg.manual_corrections)})"
            )
        )
        if session.metadata_path and os.path.isfile(session.metadata_path):
            self._load_analysis_payload(os.path.dirname(session.metadata_path))
        self._audit("edit_session_loaded", path)

    def _export_audit_log(self):
        base = self.last_out_dir if os.path.isdir(self.last_out_dir) else os.path.abspath("output")
        os.makedirs(base, exist_ok=True)
        out_path = os.path.join(base, f"audit_log_{ts()}.json")
        chain = build_audit_chain(self.audit_events)
        payload = {
            "generated_at_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "decision_support_notice": (
                "Assistive judging/analytics only. Human judge/referee confirmation required."
            ),
            "ref_events": list(self.cfg.ref_events),
            "manual_corrections": list(self.cfg.manual_corrections),
            "events": chain,
            "final_audit_hash": final_audit_hash(self.audit_events),
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        self._log(f"Audit log exported: {out_path}")

    def _apply_bout_preset(self):
        idx = self.cb_bout_preset.currentIndex()
        presets = {
            0: (12, 180.0, 60.0),
            1: (3, 180.0, 60.0),
            2: (3, 120.0, 60.0),
        }
        custom = idx == 3
        if not custom and idx in presets:
            rounds_count, round_seconds, rest_seconds = presets[idx]
            self.spn_rounds.setValue(rounds_count)
            self.spn_round_seconds.setValue(round_seconds)
            self.spn_rest_seconds.setValue(rest_seconds)

        for widget in (
            self.spn_rounds,
            self.spn_round_seconds,
            self.spn_rest_seconds,
            self.spn_warmup_seconds,
        ):
            widget.setEnabled(custom)
        self.spn_ref_round.setMaximum(int(self.spn_rounds.value()))

    def _on_corner_mode_changed(self):
        unknown = self.cb_corner_mode.currentIndex() == 1
        self.ed_red.setEnabled(not unknown)
        self.ed_blue.setEnabled(not unknown)
        if unknown:
            self.ed_red.setText("Fighter A")
            self.ed_blue.setText("Fighter B")

    def _select_calibration_profile(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Calibration Profile",
            "",
            "Calibration Profile (*.json *.yaml *.yml);;All files (*.*)",
        )
        if not path:
            return
        try:
            profile = load_profile(path)
        except Exception as exc:
            QMessageBox.critical(self, APP_TITLE, f"Failed to load calibration profile:\n{exc}")
            self._audit("calibration_profile_load_failed", f"path={path} err={exc}")
            return
        self.cfg.calibration_profile_path = path
        self.lbl_calib_status.setText(
            (
                f"Calibration: Loaded '{profile.profile_name}' ({profile.image_width}x{profile.image_height}), "
                f"reproj err {profile.reprojection_error:.4f}. "
                "Still requires validation."
            )
        )
        self._log(f"Calibration profile loaded: {path}")
        self._audit("calibration_profile_loaded", path)

    def _clear_calibration_profile(self):
        self.cfg.calibration_profile_path = None
        self.lbl_calib_status.setText("Calibration: Unverified (no profile loaded)")
        self._log("Calibration profile cleared.")
        self._audit("calibration_profile_cleared", "cleared")

    def _analysis_metadata_path(self, out_dir: str) -> str:
        return os.path.join(out_dir, "analysis_metadata.json")

    def _load_analysis_payload(self, out_dir: str):
        meta_path = self._analysis_metadata_path(out_dir)
        if not os.path.isfile(meta_path):
            self.last_analysis_payload = {}
            self.timeline_events = []
            self.filtered_timeline_events = []
            self.txt_timeline.setPlainText("No analysis metadata file found for timeline review.")
            return
        try:
            with open(meta_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            self.last_analysis_payload = {}
            self.timeline_events = []
            self.filtered_timeline_events = []
            self.txt_timeline.setPlainText(f"Failed to load analysis metadata: {exc}")
            return

        self.last_analysis_payload = payload if isinstance(payload, dict) else {}
        metadata = (
            self.last_analysis_payload.get("metadata", {})
            if isinstance(self.last_analysis_payload.get("metadata", {}), dict)
            else {}
        )
        classified = metadata.get("classified_events", [])
        ref_events = metadata.get("confirmed_ref_events", [])
        if not isinstance(classified, list):
            classified = []
        if not isinstance(ref_events, list):
            ref_events = []
        self.timeline_events = build_timeline_events(
            classified_events=classified,
            confirmed_ref_events=ref_events,
        )
        self.spn_tl_select_index.setMaximum(max(0, len(self.timeline_events) - 1))
        self.spn_tl_round.setMaximum(max(15, int(self.spn_rounds.value())))
        self._refresh_timeline_view()

    def _refresh_timeline_view(self):
        self.filtered_timeline_events = filter_timeline_events(
            self.timeline_events,
            role_filter=self.cb_tl_role.currentText(),
            label_filter=self.cb_tl_label.currentText(),
            zone_filter=self.cb_tl_zone.currentText(),
            round_filter=int(self.spn_tl_round.value()),
            include_invalidated=False,
        )
        lines = format_timeline_rows(self.filtered_timeline_events)
        if not lines:
            self.txt_timeline.setPlainText("No timeline events match current filters.")
        else:
            self.txt_timeline.setPlainText("\n".join(lines))
        max_idx = max(0, len(self.filtered_timeline_events) - 1)
        self.spn_tl_select_index.setMaximum(max_idx)
        if self.spn_tl_select_index.value() > max_idx:
            self.spn_tl_select_index.setValue(max_idx)

    def _load_selected_timeline_event(self):
        if not self.filtered_timeline_events:
            QMessageBox.warning(self, APP_TITLE, "No filtered timeline events available.")
            return
        idx = int(self.spn_tl_select_index.value())
        if idx < 0 or idx >= len(self.filtered_timeline_events):
            QMessageBox.warning(self, APP_TITLE, "Selected timeline index is out of range.")
            return
        row = self.filtered_timeline_events[idx]
        self.spn_corr_time.setValue(float(row.get("timestamp_s", 0.0) or 0.0))
        role = str(row.get("role", "")).strip().upper()
        role_idx = self.cb_corr_match_role.findText(role)
        self.cb_corr_match_role.setCurrentIndex(role_idx if role_idx >= 0 else 0)
        self._log(
            (
                "Loaded timeline event for correction: "
                f"idx={int(row.get('event_index', -1))} "
                f"t={float(row.get('timestamp_s', 0.0)):.2f}s role={role or 'ANY'} "
                f"label={row.get('label', '-')}"
            )
        )
        self._audit(
            "timeline_event_loaded",
            (
                f"event_index={int(row.get('event_index', -1))} "
                f"timestamp_s={float(row.get('timestamp_s', 0.0)):.3f}"
            ),
        )

    def _preview_corrected_scoring(self):
        if not self.last_analysis_payload:
            QMessageBox.warning(
                self,
                APP_TITLE,
                "No analysis metadata loaded. Run analysis first to preview corrected scoring.",
            )
            return
        metadata = self.last_analysis_payload.get("metadata", {})
        punch_log = self.last_analysis_payload.get("punch_log", [])
        if not isinstance(metadata, dict) or not isinstance(punch_log, list):
            QMessageBox.warning(self, APP_TITLE, "Analysis metadata is malformed.")
            return
        classified = metadata.get("classified_events", [])
        if not isinstance(classified, list):
            QMessageBox.warning(
                self, APP_TITLE, "Classified events missing from analysis metadata."
            )
            return

        preview = preview_scoring_outcome(
            metadata=metadata,
            classified_events=classified,
            punch_log=punch_log,
            manual_corrections=list(self.cfg.manual_corrections),
            ref_events=list(self.cfg.ref_events),
        )
        can_score = bool(int(preview.get("can_propose_ten_point", 0) or 0))
        proposals = preview.get("proposals", {})
        if can_score and isinstance(proposals, dict) and proposals:
            rows = []
            for round_no in sorted(proposals.keys()):
                red_pts, blue_pts, rationale = proposals[round_no]
                rows.append(f"R{round_no}: RED {red_pts} - BLUE {blue_pts} | {rationale}")
            msg = "Preview scoring proposals:\n" + "\n".join(rows)
        else:
            reasons = preview.get("missing_reasons", [])
            msg = "Preview analytics-only (10-point proposal disabled): " + ",".join(
                reasons if isinstance(reasons, list) else []
            )
        self._log(msg)
        self._audit(
            "preview_corrected_scoring",
            (
                f"can_score={int(can_score)} "
                f"corrections_applied={int(preview.get('manual_corrections_applied_count', 0) or 0)}"
            ),
        )

    def _align_round_start(self):
        if self.cfg.use_camera:
            QMessageBox.warning(
                self,
                APP_TITLE,
                "Round alignment marker is only available for selected video files.",
            )
            return
        if not self.cfg.input_path:
            QMessageBox.warning(self, APP_TITLE, "Select a video before aligning round start.")
            return

        self._log("Opening round start alignment review window...")
        marker_s = _choose_round_start_marker(self.cfg.input_path, max_frames=1200)
        if marker_s is None:
            self._log("Round alignment cancelled. Keeping existing round start offset.")
            self._audit("round_align_cancelled", "cancelled")
            return
        self.spn_round_offset.setValue(float(marker_s))
        self._log(f"Round start marker set to {marker_s:.2f}s.")
        self._audit("round_align_set", f"{marker_s:.3f}s")

    def _set_video(self, path: str):
        self.cfg.input_path = path
        self.cfg.use_camera = False
        self.cb_cam_index.setEnabled(False)
        self.spn_secs.setEnabled(False)
        self.btn_align_round.setEnabled(True)
        self.lbl_selected.setText(f"Selected Source: {human_path(path)}")
        self._log(f"Video selected: {path}")
        self._audit("video_selected", path)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose Fight Video",
            "",
            "Videos (*.mp4 *.mov *.avi *.mkv);;All files (*.*)",
        )
        if path:
            self._set_video(path)

    def _toggle_cam(self):
        self.cfg.use_camera = not self.cfg.use_camera
        on = self.cfg.use_camera
        self.cb_cam_index.setEnabled(on)
        self.spn_secs.setEnabled(on)
        self.btn_align_round.setEnabled(not on)
        if on:
            self.cfg.input_path = None
            self.lbl_selected.setText("Selected Source: Camera")
            self._log("Camera capture enabled.")
        else:
            self.lbl_selected.setText("Selected Source: -")
            self._log("Camera capture disabled.")

    def _run(self):
        if not self.cfg.use_camera and not self.cfg.input_path:
            QMessageBox.warning(self, APP_TITLE, "Pick a video or enable camera capture.")
            return

        self.cfg.unknown_corners = self.cb_corner_mode.currentIndex() == 1
        if self.cfg.unknown_corners:
            self.cfg.red_name = "Fighter A"
            self.cfg.blue_name = "Fighter B"
        else:
            self.cfg.red_name = self.ed_red.text().strip() or "Red Corner"
            self.cfg.blue_name = self.ed_blue.text().strip() or "Blue Corner"
        self.cfg.backend = "opencv" if self.cb_backend.currentIndex() == 0 else "yolov8"
        self.cfg.camera_index = int(self.cb_cam_index.currentText())
        self.cfg.camera_seconds = int(self.spn_secs.value())
        self.cfg.fps_override = int(self.spn_fps.value()) or None
        self.cfg.rounds_count = int(self.spn_rounds.value())
        self.cfg.round_seconds = float(self.spn_round_seconds.value())
        self.cfg.rest_seconds = float(self.spn_rest_seconds.value())
        self.cfg.warmup_seconds = float(self.spn_warmup_seconds.value())
        self.cfg.round_start_offset_seconds = float(self.spn_round_offset.value())
        self.cfg.manual_seeds = None
        self._audit(
            "analysis_start_requested",
            (
                f"backend={self.cfg.backend} rounds={self.cfg.rounds_count} "
                f"round_s={self.cfg.round_seconds} rest_s={self.cfg.rest_seconds} "
                f"ref_events={len(self.cfg.ref_events)} corrections={len(self.cfg.manual_corrections)}"
            ),
        )

        if not self.cfg.use_camera:
            self._log("Loading first 500 frames for manual fingerprint annotation...")
            seeds = _collect_manual_seed_annotations(self.cfg.input_path, max_frames=500)
            if not seeds:
                self._log(
                    "Manual fingerprint annotation skipped. Continuing with automatic identity tracking."
                )
            else:
                self.cfg.manual_seeds = seeds
                self._log("Manual fingerprints captured for RED and BLUE.")

        source_base = (
            "camera_capture"
            if self.cfg.use_camera
            else os.path.splitext(os.path.basename(self.cfg.input_path))[0]
        )
        self.cfg.out_dir = os.path.abspath(os.path.join("output", "runs", f"{source_base}_{ts()}"))

        self.btn_run.setEnabled(False)
        self.btn_abort.setText("Cancel Run")
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.txt_log.clear()
        self.run_badge.setText("Running")
        self.run_badge.setStyleSheet(
            "background:#12422d; border:1px solid #1f6d4a; border-radius:10px; padding:6px 10px; "
            "font-size:12px; color:#bdf5db;"
        )
        self._log("Starting pipeline...")

        self.worker = PipelineWorker(self.cfg)
        self.worker.progress.connect(self._log)
        self.worker.progress_value.connect(self._on_progress_value)
        self.worker.done.connect(self._done)
        self.worker.cancelled.connect(self._cancelled)
        self.worker.error.connect(self._err)
        self.worker.start()

    def _on_progress_value(self, value: int):
        self.progress.setValue(max(0, min(100, int(value))))

    def _cancel_run(self):
        if not self.worker or not self.worker.isRunning():
            return
        self.worker.requestInterruption()
        self.run_badge.setText("Cancelling...")
        self.run_badge.setStyleSheet(
            "background:#4a3a12; border:1px solid #8f6f21; border-radius:10px; padding:6px 10px; "
            "font-size:12px; color:#fbe7a0;"
        )
        self._log("Cancellation requested. Waiting for current processing step to stop...")
        self._audit("analysis_cancel_requested", "operator_requested_interrupt")

    def _handle_abort_or_reset(self):
        if self.worker and self.worker.isRunning():
            self._cancel_run()
            return
        self._reset_view()

    def _done(self, out_dir, video_path, pdf_path):
        self.btn_run.setEnabled(True)
        self.btn_abort.setText("Reset")
        self.progress.setValue(100)
        self.progress.setVisible(False)
        self.last_out_dir = out_dir
        self._load_analysis_payload(out_dir)
        self.run_badge.setText("Complete")
        self.run_badge.setStyleSheet(
            "background:#15314a; border:1px solid #2d597f; border-radius:10px; padding:6px 10px; "
            "font-size:12px; color:#cfe6ff;"
        )
        self.lbl_result.setText(f"Output: {human_path(out_dir)}")
        self._log(f"Analysis complete.\nVideo: {video_path}\nPDF: {pdf_path}")
        self._audit("analysis_complete", f"out_dir={out_dir}")
        QMessageBox.information(self, APP_TITLE, "Processing complete.")

    def _cancelled(self, out_dir: str, message: str):
        self.btn_run.setEnabled(True)
        self.btn_abort.setText("Reset")
        self.progress.setVisible(False)
        self.run_badge.setText("Cancelled")
        self.run_badge.setStyleSheet(
            "background:#4a3a12; border:1px solid #8f6f21; border-radius:10px; padding:6px 10px; "
            "font-size:12px; color:#fbe7a0;"
        )
        if out_dir and os.path.isdir(out_dir):
            self.last_out_dir = out_dir
            self._load_analysis_payload(out_dir)
            self.lbl_result.setText(f"Output: {human_path(out_dir)}")
        self._log(message or "Run cancelled by operator.")
        self._audit("analysis_cancelled", f"out_dir={out_dir or '-'} msg={message}")
        QMessageBox.information(self, APP_TITLE, message or "Run cancelled.")

    def _err(self, err):
        self.btn_run.setEnabled(True)
        self.btn_abort.setText("Reset")
        self.progress.setValue(0)
        self.progress.setVisible(False)
        self.run_badge.setText("Error")
        self.run_badge.setStyleSheet(
            "background:#4a1a27; border:1px solid #8e314a; border-radius:10px; padding:6px 10px; "
            "font-size:12px; color:#ffd2dc;"
        )
        self._log(f"Error: {err}")
        self._audit("analysis_error", err)
        QMessageBox.critical(self, f"{APP_TITLE} - Error", err)

    def _open_output(self):
        target = (
            self.last_out_dir if os.path.isdir(self.last_out_dir) else os.path.abspath("output")
        )
        if sys.platform.startswith("win"):
            os.startfile(target)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", target])
        else:
            subprocess.Popen(["xdg-open", target])

    def _reset_view(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, APP_TITLE, "A run is in progress. Wait until it completes.")
            return
        self.txt_log.clear()
        self.txt_timeline.clear()
        self.progress.setValue(0)
        self.lbl_result.setText("Output: -")
        self.run_badge.setText("Idle")
        self.run_badge.setStyleSheet(
            "background:#1b2f46; border:1px solid #31577e; border-radius:10px; padding:6px 10px; "
            "font-size:12px; color:#b6d4f2;"
        )
        self._log("UI reset.")
        self.cfg.ref_events.clear()
        self.cfg.manual_corrections.clear()
        self._refresh_correction_queue_view()
        self.last_analysis_payload = {}
        self.timeline_events = []
        self.filtered_timeline_events = []
        self.spn_tl_select_index.setMaximum(0)
        self.spn_tl_select_index.setValue(0)
        self._audit("queued_events_cleared", "cleared_ref_events_and_manual_corrections")


def main():
    install_opencv_circle_guard()
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
