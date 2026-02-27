# main.py
import argparse

from score_tracker import ScoreTracker
from scorecard_generator import generate_scorecard
from video_processor import process_video


def run_cli():
    tracker = ScoreTracker()
    process_video(tracker)
    generate_scorecard(tracker)


if __name__ == "__main__":
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
