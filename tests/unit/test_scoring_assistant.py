from pathlib import Path

from boxing_analytics.scoring import (
    build_round_criteria,
    evaluate_scoring_gate,
    propose_round_points,
)


def test_build_round_criteria_prefers_clean_work() -> None:
    events = [
        {"round": 1, "role": "RED", "label": "landed_clean"},
        {"round": 1, "role": "RED", "label": "landed_clean"},
        {"round": 1, "role": "RED", "label": "blocked_guarded"},
        {"round": 1, "role": "BLUE", "label": "missed"},
    ]
    out = build_round_criteria(events, total_rounds=1)
    assert out[1]["RED"].clean_punching_score > out[1]["BLUE"].clean_punching_score


def test_scoring_gate_requires_evidence_clips() -> None:
    metadata = {
        "round_segments": [{"round": 1, "start_s": 0.0, "end_s": 180.0}],
        "timeline_markers": [{"label": "round_start_manual", "timestamp_s": 0.0}],
        "confirmed_ref_event_flags_present": 1,
    }
    classified_events = [
        {"label": "landed_clean"},
        {"label": "blocked_guarded"},
    ]
    punch_log = [
        {"classification_label": "landed_clean", "evidence_clip": "/tmp/does_not_exist.mp4"},
    ]
    gate = evaluate_scoring_gate(
        metadata=metadata,
        classified_events=classified_events,
        punch_log=punch_log,
    )
    assert not gate.can_propose_ten_point
    assert "evidence_clips_missing" in gate.missing_reasons()


def test_scoring_gate_allows_proposal_with_requirements(tmp_path: Path) -> None:
    clip_path = tmp_path / "ev.mp4"
    clip_path.write_bytes(b"fake")

    metadata = {
        "round_segments": [{"round": 1, "start_s": 0.0, "end_s": 180.0}],
        "timeline_markers": [{"label": "round_start_manual", "timestamp_s": 0.0}],
        "confirmed_ref_event_flags_present": 1,
    }
    classified_events = [
        {"label": "landed_clean"},
        {"label": "blocked_guarded"},
    ]
    punch_log = [
        {"classification_label": "landed_clean", "evidence_clip": str(clip_path)},
    ]

    gate = evaluate_scoring_gate(
        metadata=metadata,
        classified_events=classified_events,
        punch_log=punch_log,
    )
    assert gate.can_propose_ten_point


def test_propose_round_points_applies_kd_and_deductions() -> None:
    criteria = build_round_criteria(
        [
            {"round": 1, "role": "RED", "label": "landed_clean"},
            {"round": 1, "role": "RED", "label": "landed_clean"},
            {"round": 1, "role": "BLUE", "label": "landed_glancing"},
            {"round": 1, "role": "BLUE", "label": "missed"},
            {"round": 1, "role": "RED", "label": "blocked_guarded"},
            {"round": 1, "role": "RED", "label": "blocked_guarded"},
        ],
        total_rounds=1,
    )
    props = propose_round_points(
        criteria=criteria,
        kd={1: {"RED": 0, "BLUE": 1}},
        deductions={1: {"RED": 0, "BLUE": 1}},
        total_rounds=1,
    )
    red_pts, blue_pts, _ = props[1]
    assert red_pts >= blue_pts
