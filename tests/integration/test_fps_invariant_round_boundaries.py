from boxing_analytics.video.rounds import BoutConfig, build_round_segments, round_for_timestamp
from boxing_analytics.video.timeline import resolve_frame_timestamp


def test_round_assignment_is_fps_invariant_when_pts_present() -> None:
    config = BoutConfig(rounds_count=2, round_seconds=3.0, rest_seconds=1.0)
    segments = build_round_segments(start_s=0.0, config=config)

    pts_samples = [0.0, 0.5, 1.0, 2.9, 3.0, 3.5, 4.1, 5.9]

    rounds_30 = []
    last_30 = None
    for frame_idx, pts_s in enumerate(pts_samples, start=1):
        row = resolve_frame_timestamp(
            frame_index=frame_idx,
            pts_seconds=pts_s,
            fallback_fps=30.0,
            last_timestamp_s=last_30,
        )
        last_30 = row.timestamp_s
        rounds_30.append(round_for_timestamp(row.timestamp_s, segments))

    rounds_120 = []
    last_120 = None
    for frame_idx, pts_s in enumerate(pts_samples, start=1):
        row = resolve_frame_timestamp(
            frame_index=frame_idx,
            pts_seconds=pts_s,
            fallback_fps=120.0,
            last_timestamp_s=last_120,
        )
        last_120 = row.timestamp_s
        rounds_120.append(round_for_timestamp(row.timestamp_s, segments))

    assert rounds_30 == rounds_120
