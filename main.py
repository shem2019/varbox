# main.py

from video_processor import process_video
from score_tracker import ScoreTracker
from scorecard_generator import generate_scorecard

if __name__ == "__main__":
    tracker = ScoreTracker()
    process_video(tracker)
    generate_scorecard(tracker.punch_log)
