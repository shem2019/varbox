"""Positioning and safety language for product surfaces."""

from __future__ import annotations

ASSISTIVE_NOTICE = "Assistive judging and analytics decision support only."
HUMAN_REQUIRED_NOTICE = "Final decisions require human judges and licensed referees."
UNVERIFIED_NOTICE = "Unverified without calibration and validation."


def full_disclaimer() -> str:
    """Return canonical product disclaimer text."""
    return " ".join([ASSISTIVE_NOTICE, HUMAN_REQUIRED_NOTICE, UNVERIFIED_NOTICE])
