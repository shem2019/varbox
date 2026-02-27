from __future__ import annotations

from boxing_analytics.video.progress import ProgressSnapshot, progress_message, progress_percent


def test_progress_percent_known_total():
    assert progress_percent(50, 200) == 25
    assert progress_percent(300, 200) == 100


def test_progress_percent_unknown_total():
    assert progress_percent(10, 0) is None


def test_progress_message_formats_with_percent():
    snapshot = ProgressSnapshot(frame_idx=40, total_frames=160, timestamp_s=12.34, events_count=5)
    msg = progress_message(snapshot)
    assert "frame 40/160" in msg
    assert "(25%)" in msg
    assert "landed_events=5" in msg


def test_progress_message_formats_without_total():
    snapshot = ProgressSnapshot(frame_idx=40, total_frames=0, timestamp_s=12.34, events_count=5)
    msg = progress_message(snapshot)
    assert "frame 40 |" in msg
    assert "%" not in msg
