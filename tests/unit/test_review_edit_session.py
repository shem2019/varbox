from __future__ import annotations

import json

import pytest

from boxing_analytics.review import (
    build_review_session,
    correction_digest,
    load_review_session,
    save_review_session,
)


def test_correction_digest_is_stable():
    corrections = [
        {"timestamp_s": 1.0, "match_role": "RED", "new_label": "missed"},
        {"timestamp_s": 2.0, "match_role": "BLUE", "invalidate": 1},
    ]
    a = correction_digest(corrections)
    b = correction_digest(corrections)
    assert a == b


def test_save_and_load_review_session(tmp_path):
    session = build_review_session(
        input_video="input.mp4",
        output_video="output.mp4",
        metadata_path="analysis_metadata.json",
        ref_events=[{"event_type": "knockdown", "role": "BLUE", "round": 1}],
        manual_corrections=[{"timestamp_s": 1.0, "new_label": "missed"}],
        audit_events=[{"index": 0, "event_hash": "abc"}],
        final_audit_hash="abc",
    )
    out = tmp_path / "session.json"

    save_review_session(session, str(out))
    loaded = load_review_session(str(out))

    assert loaded.input_video == "input.mp4"
    assert loaded.output_video == "output.mp4"
    assert loaded.final_audit_hash == "abc"
    assert loaded.ref_events[0]["event_type"] == "knockdown"


def test_load_review_session_rejects_digest_mismatch(tmp_path):
    payload = {
        "created_utc": "2026-01-01T00:00:00Z",
        "decision_support_notice": "notice",
        "input_video": "in.mp4",
        "output_video": "out.mp4",
        "metadata_path": "meta.json",
        "ref_events": [],
        "manual_corrections": [{"timestamp_s": 1.0, "new_label": "missed"}],
        "correction_digest": "deadbeef",
        "audit_events": [],
        "final_audit_hash": "GENESIS",
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        load_review_session(str(path))
