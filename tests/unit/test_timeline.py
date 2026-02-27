import pytest

from boxing_analytics.video.timeline import (
    CanonicalTimeline,
    FrameTimestamp,
    TimelineMarker,
    resolve_frame_timestamp,
)


def test_timeline_keeps_markers_sorted() -> None:
    timeline = CanonicalTimeline()
    timeline.add_marker(TimelineMarker(timestamp_s=4.0, label="late"))
    timeline.add_marker(TimelineMarker(timestamp_s=1.0, label="early"))
    markers = timeline.markers
    assert [m.label for m in markers] == ["early", "late"]


def test_timeline_rejects_non_monotonic_frame_timestamps() -> None:
    timeline = CanonicalTimeline()
    timeline.add_frame_timestamp(FrameTimestamp(frame_index=1, timestamp_s=0.1, source="pts"))
    with pytest.raises(ValueError):
        timeline.add_frame_timestamp(FrameTimestamp(frame_index=2, timestamp_s=0.09, source="pts"))


def test_timestamp_resolution_prefers_pts() -> None:
    ts = resolve_frame_timestamp(
        frame_index=3,
        pts_seconds=1.5,
        fallback_fps=30.0,
        last_timestamp_s=1.4,
    )
    assert ts.timestamp_s == 1.5
    assert ts.source == "pts"


def test_timestamp_resolution_clamps_backward_pts() -> None:
    ts = resolve_frame_timestamp(
        frame_index=5,
        pts_seconds=1.2,
        fallback_fps=30.0,
        last_timestamp_s=1.25,
    )
    assert ts.timestamp_s == 1.25
