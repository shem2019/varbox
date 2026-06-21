from boxing_analytics.app.system_check import summarize_guided_system_check


def test_guided_system_check_summary_passes_with_locked_roles() -> None:
    result = summarize_guided_system_check(
        tracker_diagnostics={"enabled": True, "mode": "track", "issue": None},
        lock_status={"RED": 11, "BLUE": 22, "REF": None},
        manual_seeds={
            "RED": {"frame_idx": 12, "rel_box": [0.1, 0.1, 0.2, 0.2]},
            "BLUE": {"frame_idx": 14, "rel_box": [0.6, 0.1, 0.7, 0.2]},
        },
        ring_roi={"normalized_points": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]},
        frames_scanned=24,
        frames_with_track_ids=18,
        frames_with_required_roles=5,
    )

    assert result.passed is True
    assert result.lock_status["RED"] == 11
    assert result.lock_status["BLUE"] == 22
    assert result.summary.startswith("PASS:")


def test_guided_system_check_summary_fails_without_tracker_ids() -> None:
    result = summarize_guided_system_check(
        tracker_diagnostics={
            "enabled": False,
            "mode": "predict_only",
            "issue": "Missing optional dependency 'lap' required by the Ultralytics tracker.",
        },
        lock_status={"RED": None, "BLUE": None, "REF": None},
        manual_seeds={
            "RED": {"frame_idx": 12, "rel_box": [0.1, 0.1, 0.2, 0.2]},
            "BLUE": {"frame_idx": 14, "rel_box": [0.6, 0.1, 0.7, 0.2]},
        },
        ring_roi={"normalized_points": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]},
        frames_scanned=0,
        frames_with_track_ids=0,
        frames_with_required_roles=0,
    )

    assert result.passed is False
    assert "lap" in result.summary.lower()
