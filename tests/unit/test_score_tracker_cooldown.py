from score_tracker import ScoreTracker


def test_score_tracker_uses_time_cooldown_when_event_time_given() -> None:
    tracker = ScoreTracker()

    accepted_1 = tracker.update(
        frame_idx=10,
        fighter_id="RED",
        timestamp="00:01",
        hand="L",
        event_time_s=1.0,
        details={"confidence": 0.8},
    )
    accepted_2 = tracker.update(
        frame_idx=11,
        fighter_id="RED",
        timestamp="00:01",
        hand="L",
        event_time_s=1.1,
        details={"confidence": 0.82},
    )
    accepted_3 = tracker.update(
        frame_idx=20,
        fighter_id="RED",
        timestamp="00:02",
        hand="L",
        event_time_s=1.5,
        details={"confidence": 0.85},
    )

    assert accepted_1
    assert not accepted_2
    assert accepted_3
