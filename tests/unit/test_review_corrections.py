from __future__ import annotations

import json

from boxing_analytics.review.corrections import (
    apply_manual_corrections,
    parse_manual_corrections,
)


def test_parse_manual_corrections_skips_invalid_rows():
    payload = (
        "["
        '{"timestamp_s": 12.3, "match_role": "RED", "new_label": "blocked_guarded"},'
        '{"timestamp_s": 15.0, "match_role": "RED"},'
        '{"timestamp_s": 16.0, "invalidate": 1}'
        "]"
    )
    parsed = parse_manual_corrections(payload)
    assert len(parsed) == 2
    assert parsed[0].new_label == "blocked_guarded"
    assert parsed[1].invalidate is True


def test_apply_manual_corrections_updates_events_and_punch_log():
    classified_events = [
        {
            "timestamp_s": 10.2,
            "role": "RED",
            "label": "landed_clean",
            "target_zone": "Head",
            "round": 1,
        },
        {
            "timestamp_s": 10.9,
            "role": "BLUE",
            "label": "blocked_guarded",
            "target_zone": "Unknown",
            "round": 1,
        },
    ]
    punch_log = [
        {
            "event_time_s": 10.23,
            "role": "RED",
            "classification_label": "landed_clean",
            "target_zone": "Head",
            "evidence_clip": "clip.mp4",
            "evidence_image": "img.jpg",
            "round": 1,
        }
    ]

    corrections = parse_manual_corrections(
        json.dumps(
            [
                {
                    "timestamp_s": 10.2,
                    "match_role": "RED",
                    "new_label": "blocked_guarded",
                    "new_target_zone": "body",
                    "note": "blocked by guard",
                },
                {"timestamp_s": 10.2, "match_role": "RED", "invalidate": 1},
            ]
        )
    )

    corrected_events, corrected_punches, applied = apply_manual_corrections(
        classified_events=classified_events,
        punch_log=punch_log,
        corrections=corrections,
    )

    assert len(applied) == 2
    assert applied[0]["applied"] == 1
    assert corrected_events[0]["label"] == "blocked_guarded"
    assert corrected_events[0]["target_zone"] == "Body"
    assert corrected_events[0]["corrected_by_review"] == 1
    assert corrected_events[0]["invalidated_by_review"] == 1
    assert corrected_punches[0]["classification_label"] == "blocked_guarded"
    assert corrected_punches[0]["evidence_clip"] == ""
    assert corrected_punches[0]["invalidated_by_review"] == 1
