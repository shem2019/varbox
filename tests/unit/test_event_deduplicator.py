from boxing_analytics.detection import EventDeduplicator


def test_time_based_attempt_dedup_window() -> None:
    dedup = EventDeduplicator(attempt_window_s=0.3, contact_window_s=0.4, min_travel_px=15.0)

    assert dedup.allow_attempt("RED", "L", 1.00, (10, 10))
    assert not dedup.allow_attempt("RED", "L", 1.10, (12, 12))
    assert dedup.allow_attempt("RED", "L", 1.35, (12, 12))


def test_contact_dedup_allows_label_change() -> None:
    dedup = EventDeduplicator(attempt_window_s=0.3, contact_window_s=0.5, min_travel_px=20.0)

    assert dedup.allow_contact("BLUE", "R", "landed_clean", 2.0, (100, 80))
    assert dedup.allow_contact("BLUE", "R", "blocked_guarded", 2.1, (102, 81))
