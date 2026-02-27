from boxing_analytics.video.rounds import BoutConfig, build_round_segments, round_for_timestamp
from boxing_analytics.video.timeline import CanonicalTimeline, FrameTimestamp, TimelineMarker


def test_timeline_and_rounds_integration() -> None:
    timeline = CanonicalTimeline()
    for idx, timestamp_s in enumerate([0.0, 0.04, 0.08, 0.12, 0.16], start=1):
        timeline.add_frame_timestamp(
            FrameTimestamp(frame_index=idx, timestamp_s=timestamp_s, source="pts")
        )

    timeline.add_marker(TimelineMarker(timestamp_s=0.0, label="round_start_manual"))

    config = BoutConfig(rounds_count=3, round_seconds=2.0, rest_seconds=1.0)
    segments = build_round_segments(start_s=timeline.markers[0].timestamp_s, config=config)

    assert round_for_timestamp(0.5, segments) == 1
    assert round_for_timestamp(2.5, segments) is None
    assert round_for_timestamp(3.5, segments) == 2
