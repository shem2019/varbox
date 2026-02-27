from boxing_analytics.video.rounds import (
    BoutConfig,
    bout_end_timestamp,
    build_round_segments,
    ended_rounds_by_timestamp,
    round_for_timestamp,
)


def test_round_segments_are_timestamp_based() -> None:
    config = BoutConfig(rounds_count=2, round_seconds=180.0, rest_seconds=60.0, warmup_seconds=5.0)
    segments = build_round_segments(start_s=10.0, config=config)

    assert segments[0].start_s == 15.0
    assert segments[0].end_s == 195.0
    assert segments[1].start_s == 255.0
    assert segments[1].end_s == 435.0


def test_round_lookup_returns_none_outside_segments() -> None:
    config = BoutConfig(rounds_count=1, round_seconds=180.0, rest_seconds=60.0)
    segments = build_round_segments(start_s=0.0, config=config)

    assert round_for_timestamp(10.0, segments) == 1
    assert round_for_timestamp(200.0, segments) is None


def test_ended_rounds_by_timestamp_respects_scored_set() -> None:
    config = BoutConfig(rounds_count=3, round_seconds=10.0, rest_seconds=5.0)
    segments = build_round_segments(start_s=0.0, config=config)
    due = ended_rounds_by_timestamp(timestamp_s=26.0, segments=segments, already_scored={1})
    assert due == [2]


def test_bout_end_timestamp_returns_last_round_end() -> None:
    config = BoutConfig(rounds_count=2, round_seconds=10.0, rest_seconds=5.0)
    segments = build_round_segments(start_s=2.0, config=config)
    assert bout_end_timestamp(segments) == 27.0
