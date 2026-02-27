from boxing_analytics.app.positioning import full_disclaimer


def test_disclaimer_contains_required_language() -> None:
    text = full_disclaimer().lower()
    assert "assistive" in text
    assert "decision support" in text
    assert "human judges" in text
    assert "licensed referees" in text
    assert "unverified" in text
    assert "calibration" in text
    assert "validation" in text
