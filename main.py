# main.py
import os
import sys
import threading
import queue
import time
import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

import config as cfg
from score_tracker import ScoreTracker
from video_processor import process_video
from scorecard_generator import generate_scorecard


# =========================
# Utilities
# =========================
def safe_mkdirs(path: str):
    os.makedirs(path, exist_ok=True)

def default_outputs_for(video_path: str, chosen_dir: str | None) -> tuple[str, str, str]:
    """
    Returns (output_dir, output_video_path, scorecard_pdf_path)
    If chosen_dir is None, creates 'outputs' next to the video.
    """
    video_path = os.path.abspath(video_path)
    video_stem = Path(video_path).stem
    if chosen_dir:
        out_dir = os.path.abspath(chosen_dir)
    else:
        out_dir = os.path.join(os.path.dirname(video_path), "outputs")
    safe_mkdirs(out_dir)
    out_vid = os.path.join(out_dir, f"{video_stem}_out.mp4")
    out_pdf = os.path.join(out_dir, f"{video_stem}_scorecard.pdf")
    return out_dir, out_vid, out_pdf

def open_path(p: str):
    if not p or not os.path.exists(p):
        messagebox.showerror("Open", f"Path does not exist:\n{p}")
        return
    if sys.platform.startswith("win"):
        os.startfile(p)  # type: ignore
    elif sys.platform == "darwin":
        subprocess.run(["open", p])
    else:
        subprocess.run(["xdg-open", p])


# =========================
# Tiny Tooltip Helper
# =========================
class Tooltip:
    def __init__(self, widget, text, delay=500):
        self.widget = widget
        self.text = text
        self.delay = delay
        self._id = None
        self.tip = None
        widget.bind("<Enter>", self._schedule)
        widget.bind("<Leave>", self._hide)

    def _schedule(self, _=None):
        self._id = self.widget.after(self.delay, self._show)

    def _show(self):
        if self.tip:
            return
        x = self.widget.winfo_rootx() + 16
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
        self.tip = tw = tk.Toplevel(self.widget)
        tw.overrideredirect(True)
        tw.attributes("-topmost", True)
        lbl = tk.Label(tw, text=self.text, bg="#111827", fg="#e5e7eb",
                       padx=8, pady=4, relief="solid", bd=0,
                       font=("Segoe UI", 9))
        lbl.pack()
        tw.wm_geometry(f"+{x}+{y}")

    def _hide(self, _=None):
        if self._id:
            self.widget.after_cancel(self._id)
            self._id = None
        if self.tip:
            self.tip.destroy()
            self.tip = None


# =========================
# Theming (Light/Dark) for ttk
# =========================
class Theme:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.style = ttk.Style(self.root)
        # use a theme that lets us color widgets
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        self.light = {
            "bg": "#f8fafc", "card": "#ffffff", "fg": "#0f172a",
            "muted": "#475569", "accent": "#4f46e5",
            "accent_hover": "#4338ca", "danger": "#dc2626",
            "danger_hover": "#b91c1c", "border": "#e2e8f0",
            "trough": "#e5e7eb", "progress": "#22c55e",
            "header_bg": "#111827", "header_fg": "#f8fafc",
        }
        self.dark = {
            "bg": "#0b1220", "card": "#0f172a", "fg": "#e5e7eb",
            "muted": "#94a3b8", "accent": "#6366f1",
            "accent_hover": "#4f46e5", "danger": "#ef4444",
            "danger_hover": "#dc2626", "border": "#1f2937",
            "trough": "#111827", "progress": "#22c55e",
            "header_bg": "#0b0f1a", "header_fg": "#e5e7eb",
        }
        self.active = self.light
        self.apply(self.active)

    def toggle(self):
        self.active = self.dark if self.active is self.light else self.light
        self.apply(self.active)

    def apply(self, c: dict):
        self.root.configure(bg=c["bg"])
        s = self.style

        # Frames / Labelframes
        s.configure("Card.TLabelframe", background=c["card"], foreground=c["fg"], bordercolor=c["border"])
        s.configure("Card.TLabelframe.Label", background=c["card"], foreground=c["fg"], font=("Segoe UI", 11, "bold"))
        s.configure("TFrame", background=c["bg"])
        s.configure("Header.TFrame", background=c["header_bg"])

        # Labels
        s.configure("Title.TLabel", background=c["header_bg"], foreground=c["header_fg"], font=("Segoe UI", 18, "bold"))
        s.configure("Subtitle.TLabel", background=c["header_bg"], foreground=c["muted"], font=("Segoe UI", 11))
        s.configure("Body.TLabel", background=c["bg"], foreground=c["fg"], font=("Segoe UI", 10))

        # Entries
        s.configure("TEntry", fieldbackground=c["card"], background=c["card"], foreground=c["fg"])
        s.map("TEntry", highlightcolor=[("focus", c["accent"])])

        # Buttons
        s.configure("Accent.TButton", background=c["accent"], foreground="#ffffff", padding=8, font=("Segoe UI", 10, "bold"))
        s.map("Accent.TButton", background=[("active", c["accent_hover"]), ("pressed", c["accent_hover"])])
        s.configure("Danger.TButton", background=c["danger"], foreground="#ffffff", padding=8, font=("Segoe UI", 10, "bold"))
        s.map("Danger.TButton", background=[("active", c["danger_hover"]), ("pressed", c["danger_hover"])])

        s.configure("TButton", padding=7, font=("Segoe UI", 10))

        # Progressbar
        s.configure("Green.Horizontal.TProgressbar", troughcolor=c["trough"], background=c["progress"],
                    lightcolor=c["progress"], darkcolor=c["progress"], bordercolor=c["trough"])

        # Notebook / Text
        s.configure("TNotebook", background=c["bg"])
        s.configure("TNotebook.Tab", background=c["card"], foreground=c["fg"])

        # ScrolledText outer
        # (inner text widget colored below in GUI builder)


# =========================
# GUI
# =========================
class VarBoxGUI:
    PADX = 12
    PADY = 10

    def __init__(self, root: tk.Tk):
        self.root = root
        self.theme = Theme(root)

        self.root.title("VAR Box — AI Boxing Referee")
        self.root.minsize(900, 600)

        # threading state
        self.progress_queue: "queue.Queue[tuple[int,int]]" = queue.Queue()
        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.worker: threading.Thread | None = None
        self.cancel_evt = threading.Event()

        # inputs
        self.var_input = tk.StringVar()
        self.var_output_dir = tk.StringVar()
        self.var_place_next = tk.BooleanVar(value=True)
        self.var_backend = tk.StringVar(value=cfg.BACKEND)

        # results
        self.last_output_video: str | None = None
        self.last_scorecard_pdf: str | None = None

        self.started_at: float | None = None

        self._build_ui()
        self._poll_queues()

    # ---------- UI ----------
    def _build_ui(self):
        c = self.theme.active

        # Header
        hdr = ttk.Frame(self.root, style="Header.TFrame")
        hdr.pack(fill="x")
        left = ttk.Frame(hdr, style="Header.TFrame")
        left.pack(side="left", padx=self.PADX, pady=(10, 12))
        ttk.Label(left, text="🥊 VAR Box", style="Title.TLabel").pack(anchor="w")
        ttk.Label(left, text="AI-powered boxing referee — OpenPose + OpenCV scoring", style="Subtitle.TLabel").pack(anchor="w", pady=(2, 0))
        # theme toggle
        self.btn_theme = ttk.Button(hdr, text="🌙 Dark", command=self._toggle_theme)
        self.btn_theme.pack(side="right", padx=self.PADX, pady=(14, 12))

        # Files Card
        files = ttk.Labelframe(self.root, text="Files", style="Card.TLabelframe")
        files.pack(fill="x", padx=self.PADX, pady=(self.PADY, self.PADY))

        # Row: Input
        r0 = ttk.Frame(files, style="Card.TLabelframe")
        r0.pack(fill="x", padx=10, pady=8)
        ttk.Label(r0, text="Input Video", style="Body.TLabel", width=16).pack(side="left")
        self.ent_in = ttk.Entry(r0, textvariable=self.var_input)
        self.ent_in.pack(side="left", fill="x", expand=True, padx=8)
        btn_browse = ttk.Button(r0, text="🔍 Browse…", command=self._browse_video)
        btn_browse.pack(side="left")

        # Row: Output folder
        r1 = ttk.Frame(files, style="Card.TLabelframe")
        r1.pack(fill="x", padx=10, pady=8)
        ttk.Label(r1, text="Output Folder", style="Body.TLabel", width=16).pack(side="left")
        self.ent_out = ttk.Entry(r1, textvariable=self.var_output_dir, state="disabled")
        self.ent_out.pack(side="left", fill="x", expand=True, padx=8)
        btn_out = ttk.Button(r1, text="📂 Choose…", command=self._browse_output_dir)
        btn_out.pack(side="left")
        self.chk_place = ttk.Checkbutton(files, text="Place outputs next to video (outputs/)", variable=self.var_place_next,
                                         command=self._toggle_output_dir_state)
        self.chk_place.pack(anchor="w", padx=10)

        # Row: Backend + Controls
        r2 = ttk.Frame(files, style="Card.TLabelframe")
        r2.pack(fill="x", padx=10, pady=(6, 2))
        ttk.Label(r2, text="Detection Backend", style="Body.TLabel", width=16).pack(side="left")
        ttk.OptionMenu(r2, self.var_backend, self.var_backend.get(), "opencv", "yolov8").pack(side="left")
        ttk.Separator(r2, orient="vertical").pack(side="left", fill="y", padx=12)
        self.btn_start = ttk.Button(r2, text="▶ Start", style="Accent.TButton", command=self.start_pipeline)
        self.btn_start.pack(side="left")
        self.btn_cancel = ttk.Button(r2, text="⏹ Cancel", style="Danger.TButton", command=self.cancel_pipeline, state="disabled")
        self.btn_cancel.pack(side="left", padx=8)
        ttk.Separator(r2, orient="vertical").pack(side="left", fill="y", padx=12)
        self.btn_pdf = ttk.Button(r2, text="📄 Open Scorecard", command=lambda: open_path(self.last_scorecard_pdf or ""))
        self.btn_pdf.pack(side="right")
        Tooltip(self.btn_pdf, "Open the generated PDF scorecard")
        self.btn_vid = ttk.Button(r2, text="🎬 Open Output Video", command=lambda: open_path(self.last_output_video or ""))
        self.btn_vid.pack(side="right", padx=(0, 8))
        Tooltip(self.btn_vid, "Open the annotated output video")

        # Progress Card
        prog = ttk.Labelframe(self.root, text="Progress", style="Card.TLabelframe")
        prog.pack(fill="x", padx=self.PADX, pady=(0, self.PADY))
        self.prog_bar = ttk.Progressbar(prog, orient="horizontal", mode="determinate", style="Green.Horizontal.TProgressbar")
        self.prog_bar.pack(fill="x", padx=10, pady=(10, 2))
        self.lbl_prog = ttk.Label(prog, text="Idle.", style="Body.TLabel")
        self.lbl_prog.pack(anchor="w", padx=12, pady=(0, 8))

        # Log Card
        log = ttk.Labelframe(self.root, text="Log", style="Card.TLabelframe")
        log.pack(fill="both", expand=True, padx=self.PADX, pady=(0, self.PADY))
        self.txt_log = ScrolledText(log, height=14, wrap="word", bd=0, relief="flat")
        self.txt_log.pack(fill="both", expand=True, padx=8, pady=8)
        # style inner Text colors
        self._apply_text_colors()

        # Status bar
        self.status = ttk.Label(self.root, text="Ready.", style="Body.TLabel", anchor="w")
        self.status.pack(fill="x", padx=self.PADX, pady=(0, 8))

        # initial
        self._toggle_output_dir_state()

    def _apply_text_colors(self):
        # style ScrolledText inner text according to theme
        c = self.theme.active
        try:
            self.txt_log.configure(bg=c["card"], fg=c["fg"], insertbackground=c["fg"])
        except Exception:
            pass

    def _toggle_theme(self):
        self.theme.toggle()
        self._apply_text_colors()
        if "Dark" in self.btn_theme.cget("text"):
            self.btn_theme.config(text="☀ Light")
        else:
            self.btn_theme.config(text="🌙 Dark")
        self.status.config(text="Theme updated.")

    # ---------- File pickers ----------
    def _browse_video(self):
        path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.mov *.mkv *.avi *.m4v"), ("All files", "*.*")]
        )
        if path:
            self.var_input.set(path)
            if self.var_place_next.get():
                out_dir = os.path.join(os.path.dirname(path), "outputs")
                self.var_output_dir.set(out_dir)

    def _browse_output_dir(self):
        path = filedialog.askdirectory(title="Select Output Folder")
        if path:
            self.var_output_dir.set(path)
            self.var_place_next.set(False)
            self._toggle_output_dir_state()

    def _toggle_output_dir_state(self):
        if self.var_place_next.get():
            self.ent_out.configure(state="disabled")
            # If we have an input, set default outputs/ next to it
            p = self.var_input.get().strip()
            if p:
                self.var_output_dir.set(os.path.join(os.path.dirname(p), "outputs"))
        else:
            self.ent_out.configure(state="normal")

    # ---------- Logging & Progress ----------
    def log(self, msg: str):
        self.log_queue.put(msg)

    def _log_to_ui(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.txt_log.insert("end", f"[{ts}] {msg}\n")
        self.txt_log.see("end")

    def _poll_queues(self):
        # progress
        try:
            while True:
                cur, total = self.progress_queue.get_nowait()
                if total and total > 0:
                    pct = int(cur * 100 / total)
                    self.prog_bar.config(mode="determinate", maximum=total, value=cur)
                    # ETA
                    eta_txt = ""
                    if self.started_at:
                        elapsed = max(0.001, time.time() - self.started_at)
                        fps = cur / elapsed
                        if fps > 0 and total > cur:
                            rem = (total - cur) / fps
                            eta_txt = f" • ETA {int(rem)}s"
                    self.lbl_prog.config(text=f"Processing frames: {cur}/{total} ({pct}%)" + eta_txt)
                else:
                    self.prog_bar.config(mode="indeterminate")
                    self.prog_bar.start(50)
                    self.lbl_prog.config(text=f"Processing frames: {cur} (total unknown)")
        except queue.Empty:
            pass

        # logs
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self._log_to_ui(msg)
        except queue.Empty:
            pass

        self.root.after(100, self._poll_queues)

    # ---------- Pipeline ----------
    def start_pipeline(self):
        if self.worker and self.worker.is_alive():
            return
        input_video = self.var_input.get().strip()
        if not input_video:
            messagebox.showerror("Input Required", "Please select an input video.")
            return
        if not os.path.isfile(input_video):
            messagebox.showerror("Not Found", f"Video not found:\n{input_video}")
            return

        chosen_dir = None if self.var_place_next.get() else (self.var_output_dir.get().strip() or None)
        output_dir, output_video, scorecard_pdf = default_outputs_for(input_video, chosen_dir)
        safe_mkdirs(output_dir)

        # update config for downstream libs
        cfg.INPUT_VIDEO   = input_video
        cfg.OUTPUT_VIDEO  = output_video
        cfg.SCORECARD_PDF = scorecard_pdf
        cfg.BACKEND       = self.var_backend.get() or cfg.BACKEND

        # reset UI
        self.cancel_evt.clear()
        self.btn_start.configure(state="disabled")
        self.btn_cancel.configure(state="normal")
        self.prog_bar.config(value=0, mode="determinate")
        self.lbl_prog.config(text="Queued…")
        self.txt_log.delete("1.0", "end")
        self.last_output_video = None
        self.last_scorecard_pdf = None
        self.status.config(text="Processing…")
        self.started_at = time.time()

        self.log(f"Input: {input_video}")
        self.log(f"Output folder: {output_dir}")
        self.log(f"Backend: {cfg.BACKEND}")

        # launch worker
        self.worker = threading.Thread(
            target=self._run_pipeline_worker,
            args=(input_video, output_video, scorecard_pdf),
            daemon=True
        )
        self.worker.start()

    def cancel_pipeline(self):
        if self.worker and self.worker.is_alive():
            self.cancel_evt.set()
            self.log("Cancel requested…")
            self.status.config(text="Canceling…")

    def _run_pipeline_worker(self, input_video: str, output_video: str, scorecard_pdf: str):
        try:
            tracker = ScoreTracker()

            def _progress(cur: int, total: int):
                self.progress_queue.put((cur, total))

            def _cancel() -> bool:
                return self.cancel_evt.is_set()

            def _log(msg: str):
                self.log_queue.put(msg)

            process_video(
                tracker,
                input_video=input_video,
                output_video=output_video,
                progress_cb=_progress,
                cancel_cb=_cancel,
                log_cb=_log
            )

            if self.cancel_evt.is_set():
                self.log("⚠️ Canceled before completion. Skipping scorecard.")
                self._on_pipeline_done(None, None, canceled=True)
                return

            self.log("🧾 Generating scorecard PDF…")
            path_pdf = generate_scorecard(tracker, output_path=scorecard_pdf)
            self._on_pipeline_done(output_video, path_pdf, canceled=False)

        except Exception as e:
            self._on_pipeline_error(e)

    def _on_pipeline_done(self, out_video: str | None, out_pdf: str | None, canceled: bool):
        self.last_output_video = out_video
        self.last_scorecard_pdf = out_pdf

        self.prog_bar.stop()
        self.btn_start.configure(state="normal")
        self.btn_cancel.configure(state="disabled")
        self.started_at = None

        if canceled:
            self.lbl_prog.config(text="Canceled.")
            self.prog_bar.config(mode="determinate", value=0)
            self.log("Pipeline canceled.")
            self.status.config(text="Canceled.")
        else:
            self.lbl_prog.config(text="✅ Done.")
            self.status.config(text="Complete.")
            if out_video:
                self.log(f"✔ Output video: {out_video}")
            if out_pdf:
                self.log(f"✔ Scorecard PDF: {out_pdf}")
            messagebox.showinfo("VAR Box", f"Processing complete.\n\nVideo:\n{out_video}\n\nScorecard:\n{out_pdf}")

    def _on_pipeline_error(self, e: Exception):
        self.prog_bar.stop()
        self.prog_bar.config(mode="determinate")
        self.lbl_prog.config(text="❌ Error.")
        self.log(f"ERROR: {e}")
        messagebox.showerror("Error", str(e))
        self.btn_start.configure(state="normal")
        self.btn_cancel.configure(state="disabled")
        self.status.config(text="Error.")


def main():
    root = tk.Tk()
    VarBoxGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
