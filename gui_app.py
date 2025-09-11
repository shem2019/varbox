# gui_app.py  — VAR Box Desktop (PySide6)
import os, sys, time, tempfile, datetime, cv2
from dataclasses import dataclass
from typing import Optional
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QIcon, QDragEnterEvent, QDropEvent, QAction
from PySide6.QtWidgets import (
    QApplication, QWidget, QFileDialog, QMainWindow, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFrame, QTextEdit, QComboBox, QProgressBar,
    QSpinBox, QMessageBox
)

# your pipeline
from score_tracker import ScoreTracker
from video_processor import process_video
from scorecard_generator import generate_scorecard

APP_TITLE = "VAR Box"
# --- theme colors ---
PRIMARY = "#0ea5e9"; PRIMARY_DARK = "#0369a1"
BG = "#0b1220"; CARD = "#111827"; TEXT = "#e5e7eb"
SUBTEXT = "#9ca3af"; BORDER = "#1f2937"

APP_QSS = f"""
QMainWindow {{ background-color:{BG}; color:{TEXT}; }}
QLabel {{ color:{TEXT}; font-size:14px; }}
QTextEdit {{ background:{CARD}; color:{TEXT}; border:1px solid {BORDER}; border-radius:10px; padding:8px; font-family:Consolas,Menlo,monospace; font-size:12px; }}
QPushButton {{ background:{PRIMARY}; color:white; border:none; border-radius:10px; padding:10px 14px; font-weight:600; }}
QPushButton:hover {{ background:{PRIMARY_DARK}; }}
QPushButton#ghost {{ background:transparent; border:1px solid {BORDER}; }}
QFrame#card {{ background:{CARD}; border:1px solid {BORDER}; border-radius:16px; }}
QComboBox, QSpinBox {{ background:{CARD}; color:{TEXT}; border:1px solid {BORDER}; border-radius:8px; padding:6px 8px; }}
QProgressBar {{ background:{CARD}; color:{TEXT}; border:1px solid {BORDER}; border-radius:8px; text-align:center; }}
QProgressBar::chunk {{ background:{PRIMARY}; border-radius:8px; }}
"""

def ts(): return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
def human_path(p): return p if len(p)<80 else f"...{p[-77:]}"

@dataclass
class RunConfig:
    backend: str = "opencv"       # "opencv" (Lite) or "yolov8" (Pro)
    input_path: Optional[str] = None
    use_camera: bool = False
    camera_index: int = 0
    camera_seconds: int = 60
    fps_override: Optional[int] = None
    output_video: Optional[str] = None
    output_pdf: Optional[str] = None

# -------- Worker (thread) --------
class PipelineWorker(QThread):
    progress = Signal(str)
    done = Signal(str, str)   # video_path, pdf_path
    error = Signal(str)

    def __init__(self, cfg: RunConfig, parent=None):
        super().__init__(parent); self.cfg = cfg

    def run(self):
        try:
            self.progress.emit("Preparing…")

            # Select input (file or camera capture to temp)
            if self.cfg.use_camera:
                self.progress.emit(f"Recording camera #{self.cfg.camera_index} for {self.cfg.camera_seconds}s…")
                tmpdir = tempfile.mkdtemp(prefix="varbox_")
                tmp_input = os.path.join(tmpdir, f"camera_{ts()}.mp4")
                self._record_camera(self.cfg.camera_index, self.cfg.camera_seconds, tmp_input)
                input_path = tmp_input
            else:
                if not self.cfg.input_path or not os.path.isfile(self.cfg.input_path):
                    raise RuntimeError("No valid input video selected.")
                input_path = self.cfg.input_path

            base = os.path.splitext(os.path.basename(input_path))[0]
            out_video = self.cfg.output_video or os.path.abspath(f"{base}_scored_{ts()}.mp4")
            out_pdf   = self.cfg.output_pdf   or os.path.abspath(f"{base}_scorecard_{ts()}.pdf")

            # ---- Configure env for the pipeline & backend ----
            os.environ["VARBOX_INPUT"] = input_path
            os.environ["VARBOX_OUTPUT"] = out_video
            os.environ["VARBOX_PDF"] = out_pdf
            os.environ["VARBOX_BACKEND"] = self.cfg.backend  # "opencv" or "yolov8"
            if self.cfg.fps_override: os.environ["VARBOX_FPS_OVERRIDE"] = str(self.cfg.fps_override)

            # Optional: model paths (must exist in assets when packaged)
            # These are read by config.py or MultiPersonPoseTracker
            assets = os.path.join(os.path.dirname(__file__), "assets")
            os.environ.setdefault("VARBOX_ASSETS", assets)
            os.environ.setdefault("VARBOX_SSD_PROTO", os.path.join(assets, "models", "mobilenet_ssd", "deploy.prototxt"))
            os.environ.setdefault("VARBOX_SSD_MODEL", os.path.join(assets, "models", "mobilenet_ssd", "deploy.caffemodel"))
            os.environ.setdefault("VARBOX_YOLOV8_WEIGHTS", os.path.join(assets, "models", "yolov8n.pt"))

            self.progress.emit(f"Backend: {self.cfg.backend.upper()} | Running model…")
            tracker = ScoreTracker()
            tracker.metadata = {"title":"Boxing Match Scorecard", "subtitle":f"Source: {os.path.basename(input_path)}"}

            process_video(tracker)

            self.progress.emit("Generating scorecard PDF…")
            pdf_path = ""
            try:
                generate_scorecard(tracker, out_pdf)
                pdf_path = out_pdf
            except Exception as e:
                self.progress.emit(f"PDF warning: {e}")

            self.done.emit(out_video, pdf_path)
        except Exception as e:
            self.error.emit(str(e))

    def _record_camera(self, index: int, seconds: int, out_path: str):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened(): raise RuntimeError(f"Cannot open camera index {index}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        t0 = time.time()
        while time.time() - t0 < seconds:
            ok, frame = cap.read()
            if not ok: break
            out.write(frame)
        cap.release(); out.release()

# -------- UI pieces --------
class DropLabel(QLabel):
    fileSelected = Signal(str)
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setAcceptDrops(True); self.setAlignment(Qt.AlignCenter); self.setMinimumHeight(140)
        self.setStyleSheet(f"QLabel {{ background:{CARD}; border:2px dashed {BORDER}; border-radius:14px; color:{SUBTEXT}; font-size:14px; }}")
    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls(): e.acceptProposedAction()
    def dropEvent(self, e: QDropEvent):
        for url in e.mimeData().urls():
            p = url.toLocalFile()
            if os.path.isfile(p):
                self.fileSelected.emit(p); return

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE); self.setWindowIcon(QIcon()); self.resize(980, 720); self.setStyleSheet(APP_QSS)
        self.cfg = RunConfig(); self.worker: Optional[PipelineWorker] = None
        self._build()

    def _build(self):
        header = QFrame(); header.setObjectName("card"); hb = QHBoxLayout(header); hb.setContentsMargins(16,12,16,12)
        title = QLabel("VAR Box"); title.setStyleSheet("font-size:22px; font-weight:800;")
        subtitle = QLabel("AI-assisted boxing judging — load a video or record from camera, then run & export.")
        subtitle.setStyleSheet(f"color:{SUBTEXT}; font-size:12px;")
        left = QVBoxLayout(); left.addWidget(title); left.addWidget(subtitle); hb.addLayout(left); hb.addStretch(1)

        src = QFrame(); src.setObjectName("card")
        sv = QVBoxLayout(src); sv.setContentsMargins(16,16,16,16)
        self.drop = DropLabel("Drop a video here (MP4/MOV/AVI)…"); self.drop.fileSelected.connect(self._set_video); sv.addWidget(self.drop)

        r1 = QHBoxLayout()
        self.btn_browse = QPushButton("Browse Video"); self.btn_browse.clicked.connect(self._browse)
        self.btn_camera = QPushButton("Use Camera"); self.btn_camera.setObjectName("ghost"); self.btn_camera.clicked.connect(self._toggle_cam)
        r1.addWidget(self.btn_browse); r1.addWidget(self.btn_camera); sv.addLayout(r1)

        r2 = QHBoxLayout()
        self.cb_backend = QComboBox(); self.cb_backend.addItems(["Lite (OpenCV-DNN)", "Pro (YOLOv8)"]); self.cb_backend.setCurrentIndex(0)
        self.cb_cam_index = QComboBox(); self.cb_cam_index.addItems([str(i) for i in range(0,6)]); self.cb_cam_index.setEnabled(False)
        self.spn_secs = QSpinBox(); self.spn_secs.setRange(5, 3600); self.spn_secs.setValue(60); self.spn_secs.setEnabled(False)
        r2.addWidget(QLabel("Backend")); r2.addWidget(self.cb_backend)
        r2.addSpacing(16)
        r2.addWidget(QLabel("Camera Index")); r2.addWidget(self.cb_cam_index)
        r2.addSpacing(16)
        r2.addWidget(QLabel("Record (sec)")); r2.addWidget(self.spn_secs)
        r2.addStretch(1)
        sv.addLayout(r2)

        opts = QHBoxLayout()
        self.spn_fps = QSpinBox(); self.spn_fps.setRange(0, 240); self.spn_fps.setValue(0)
        opts.addWidget(QLabel("FPS override (0=auto)")); opts.addWidget(self.spn_fps); opts.addStretch(1)
        sv.addLayout(opts)

        act = QFrame(); act.setObjectName("card")
        av = QVBoxLayout(act); av.setContentsMargins(16,16,16,16)
        self.lbl_sel = QLabel("Selected: —"); self.lbl_sel.setStyleSheet(f"color:{SUBTEXT};")
        self.btn_run = QPushButton("Run Model"); self.btn_run.clicked.connect(self._run)
        self.btn_open = QPushButton("Open Output Folder"); self.btn_open.setObjectName("ghost"); self.btn_open.clicked.connect(self._open_out)
        rr = QHBoxLayout(); rr.addWidget(self.btn_run); rr.addWidget(self.btn_open); rr.addStretch(1)
        self.progress = QProgressBar(); self.progress.setRange(0,0); self.progress.setVisible(False)
        av.addWidget(self.lbl_sel); av.addLayout(rr); av.addWidget(self.progress)

        log = QFrame(); log.setObjectName("card")
        lv = QVBoxLayout(log); lv.setContentsMargins(16,16,16,16)
        self.txt = QTextEdit(readOnly=True); self.txt.setPlaceholderText("Logs will appear here…"); lv.addWidget(self.txt)

        root = QWidget(); rv = QVBoxLayout(root); rv.setContentsMargins(14,14,14,14); rv.setSpacing(12)
        rv.addWidget(header); rv.addWidget(src); rv.addWidget(act); rv.addWidget(log, 1)
        self.setCentralWidget(root)

        act_quit = QAction("Quit", self); act_quit.triggered.connect(self.close)
        self.menuBar().addAction(act_quit)

    # --- helpers ---
    def _log(self, s): self.txt.append(s)
    def _set_video(self, p): self.cfg.input_path=p; self.cfg.use_camera=False; self.cb_cam_index.setEnabled(False); self.spn_secs.setEnabled(False); self.lbl_sel.setText(f"Selected: {human_path(p)}"); self._log(f"Video: {p}")
    def _browse(self):
        p,_=QFileDialog.getOpenFileName(self,"Choose Video","", "Videos (*.mp4 *.mov *.avi *.mkv);;All files (*.*)")
        if p: self._set_video(p)
    def _toggle_cam(self):
        self.cfg.use_camera = not self.cfg.use_camera
        on = self.cfg.use_camera; self.cb_cam_index.setEnabled(on); self.spn_secs.setEnabled(on)
        self.lbl_sel.setText("Selected: Camera" if on else "Selected: —"); self._log("Camera ON" if on else "Camera OFF")

    def _run(self):
        if not self.cfg.use_camera and not self.cfg.input_path:
            QMessageBox.warning(self,"VAR Box","Pick a video or enable camera."); return
        self.cfg.backend = "opencv" if self.cb_backend.currentIndex()==0 else "yolov8"
        self.cfg.camera_index = int(self.cb_cam_index.currentText())
        self.cfg.camera_seconds = int(self.spn_secs.value())
        self.cfg.fps_override = int(self.spn_fps.value()) or None
        self.btn_run.setEnabled(False); self.progress.setVisible(True); self.txt.clear(); self._log("Starting…")
        self.worker = PipelineWorker(self.cfg); self.worker.progress.connect(self._log); self.worker.done.connect(self._done); self.worker.error.connect(self._err); self.worker.start()

    def _done(self, video_path, pdf_path):
        self.progress.setVisible(False); self.btn_run.setEnabled(True)
        self._log(f"✅ Finished.\nVideo: {video_path}\nPDF: {pdf_path or '(not generated)'}")
        QMessageBox.information(self,"VAR Box","Processing complete!")

    def _err(self, err):
        self.progress.setVisible(False); self.btn_run.setEnabled(True)
        self._log(f"❌ Error: {err}"); QMessageBox.critical(self,"VAR Box — Error",err)

    def _open_out(self):
        import re
        text=self.txt.toPlainText(); m=re.findall(r"Video:\s*(.+)",text); target=None
        if m:
            v=m[-1].strip()
            if os.path.isfile(v): target=os.path.dirname(v)
        if not target: target=os.path.expanduser("~")
        if sys.platform.startswith("win"): os.startfile(target)
        elif sys.platform=="darwin": os.system(f'open "{target}"')
        else: os.system(f'xdg-open "{target}"')

def main():
    app = QApplication(sys.argv); w = MainWindow(); w.show(); sys.exit(app.exec())

if __name__=="__main__": main()
