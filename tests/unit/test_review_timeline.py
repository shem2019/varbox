from __future__ import annotations

from boxing_analytics.review import (
    build_timeline_events,
    filter_timeline_events,
    format_timeline_rows,
)


def test_build_and_filter_timeline_events():
    classified = [
        {
            "timestamp_s": 1.2,
            "round": 1,
            "role": "RED",
            "label": "landed_clean",
            "target_zone": "Head",
            "confidence": 0.92,
        },
        {
            "timestamp_s": 1.7,
            "round": 1,
            "role": "BLUE",
            "label": "blocked_guarded",
            "target_zone": "unknown",
            "confidence": 0.88,
            "invalidated_by_review": 1,
        },
    ]
    ref_events = [
        {
            "timestamp_s": 2.0,
            "round": 1,
            "role": "BLUE",
            "event_type": "knockdown",
        }
    ]

    events = build_timeline_events(
        classified_events=classified,
        confirmed_ref_events=ref_events,
    )

    assert len(events) == 3
    assert events[0]["event_index"] == 0
    filtered = filter_timeline_events(events, role_filter="RED", include_invalidated=False)
    assert len(filtered) == 1
    assert filtered[0]["label"] == "landed_clean"

    ref_only = filter_timeline_events(events, label_filter="knockdown")
    assert len(ref_only) == 1
    lines = format_timeline_rows(ref_only)
    assert "knockdown" in lines[0]
