from __future__ import annotations

from boxing_analytics.review import preview_scoring_outcome


def test_preview_scoring_outcome_with_corrections(tmp_path):
    clip_a = tmp_path / "a.mp4"
    clip_b = tmp_path / "b.mp4"
    clip_a.write_bytes(b"a")
    clip_b.write_bytes(b"b")

    metadata = {
        "round_segments": [{"round": 1, "start_s": 0.0, "end_s": 180.0}],
        "timeline_markers": [{"timestamp_s": 0.0, "label": "round_start_manual"}],
        "confirmed_ref_event_flags_present": 1,
        "bout_config": {"rounds_count": 1},
    }
    classified_events = [
        {
            "timestamp_s": 10.0,
            "round": 1,
            "role": "RED",
            "label": "landed_clean",
            "target_zone": "Head",
            "confidence": 0.9,
        },
        {
            "timestamp_s": 10.4,
            "round": 1,
            "role": "BLUE",
            "label": "blocked_guarded",
            "target_zone": "Unknown",
            "confidence": 0.8,
        },
    ]
    punch_log = [
        {
            "event_time_s": 10.0,
            "round": 1,
            "role": "RED",
            "classification_label": "landed_clean",
            "target_zone": "Head",
            "evidence_clip": str(clip_a),
        },
        {
            "event_time_s": 11.0,
            "round": 1,
            "role": "BLUE",
            "classification_label": "landed_glancing",
            "target_zone": "Body",
            "evidence_clip": str(clip_b),
        },
    ]
    corrections = [
        {
            "timestamp_s": 10.0,
            "match_role": "RED",
            "new_label": "blocked_guarded",
        },
        {
            "timestamp_s": 10.4,
            "match_role": "BLUE",
            "new_label": "landed_clean",
        },
    ]
    ref_events = [{"round": 1, "role": "RED", "event_type": "knockdown", "count": 1}]

    preview = preview_scoring_outcome(
        metadata=metadata,
        classified_events=classified_events,
        punch_log=punch_log,
        manual_corrections=corrections,
        ref_events=ref_events,
    )

    assert preview["manual_corrections_applied_count"] == 2
    assert preview["can_propose_ten_point"] == 1
    assert 1 in preview["proposals"]
    red_pts, blue_pts, _ = preview["proposals"][1]
    assert red_pts < 10
    assert blue_pts >= red_pts
