"""VAR Box entrypoint.

If launched with a non-project interpreter, this script will re-exec itself
using the local `.venv` interpreter when available.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from opencv_guard import install_opencv_circle_guard


def run_cli():
    # Import runtime-heavy deps only when CLI mode is actually used.
    from score_tracker import ScoreTracker
    from scorecard_generator import generate_scorecard
    from video_processor import process_video

    tracker = ScoreTracker()
    process_video(tracker)
    generate_scorecard(tracker)


def _venv_python(project_root: Path) -> Path | None:
    candidates = [
        project_root / ".venv" / "bin" / "python",
        project_root / ".venv" / "Scripts" / "python.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _ensure_project_interpreter() -> None:
    if os.environ.get("VARBOX_SKIP_VENV_REEXEC") == "1":
        return

    project_root = Path(__file__).resolve().parent
    venv_root = (project_root / ".venv").resolve()
    venv_python = _venv_python(project_root)
    if venv_python is None:
        return

    current_prefix = Path(sys.prefix).resolve()
    if current_prefix == venv_root:
        return

    os.environ["VARBOX_SKIP_VENV_REEXEC"] = "1"
    os.execv(str(venv_python), [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]])


if __name__ == "__main__":
    _ensure_project_interpreter()
    install_opencv_circle_guard()

    parser = argparse.ArgumentParser(description="VAR Box")
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run without the desktop UI.",
    )
    args = parser.parse_args()

    if args.cli:
        run_cli()
    else:
        from gui_app import main as launch_gui

        launch_gui()
