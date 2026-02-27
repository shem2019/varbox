"""Criteria-aligned round signal extraction for assistive judging."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RoleCriteria:
    clean_punching_score: float
    effective_aggressiveness_score: float
    ring_generalship_score: float
    defense_score: float
    landed_clean: int
    landed_glancing: int
    blocked_guarded: int
    missed: int
    clinch: int


LABELS = ("landed_clean", "landed_glancing", "blocked_guarded", "missed", "clinch")


def _blank_counts() -> dict[str, int]:
    return {name: 0 for name in LABELS}


def _attempts(counts: dict[str, int]) -> int:
    return (
        counts["landed_clean"]
        + counts["landed_glancing"]
        + counts["blocked_guarded"]
        + counts["missed"]
    )


def _as_int(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str | bytes | bytearray):
        try:
            return int(value)
        except Exception:
            return 0
    try:
        return int(str(value))
    except Exception:
        return 0


def _role_criteria(
    own: dict[str, int],
    opp: dict[str, int],
) -> RoleCriteria:
    clean_volume = float(own["landed_clean"] + 0.60 * own["landed_glancing"])
    attempts = max(1, _attempts(own))
    opp_attempts = max(1, _attempts(opp))
    opp_clean = float(opp["landed_clean"] + 0.60 * opp["landed_glancing"])

    clean_score = min(10.0, 2.0 * own["landed_clean"] + 1.2 * own["landed_glancing"])
    effective_score = min(10.0, 8.0 * (clean_volume / attempts) + 0.20 * attempts)

    if clean_volume + opp_clean > 0:
        ring_share = clean_volume / (clean_volume + opp_clean)
        ring_score = max(0.0, min(10.0, 10.0 * ring_share))
    else:
        ring_score = 5.0

    defense_block = own["blocked_guarded"] / opp_attempts
    defense_avoid = opp["missed"] / opp_attempts
    defense_score = min(10.0, 6.0 * defense_block + 4.0 * defense_avoid)

    return RoleCriteria(
        clean_punching_score=round(clean_score, 3),
        effective_aggressiveness_score=round(effective_score, 3),
        ring_generalship_score=round(ring_score, 3),
        defense_score=round(defense_score, 3),
        landed_clean=own["landed_clean"],
        landed_glancing=own["landed_glancing"],
        blocked_guarded=own["blocked_guarded"],
        missed=own["missed"],
        clinch=own["clinch"],
    )


def build_round_criteria(
    classified_events: list[dict[str, object]],
    total_rounds: int,
) -> dict[int, dict[str, RoleCriteria]]:
    counts: dict[int, dict[str, dict[str, int]]] = {
        round_no: {"RED": _blank_counts(), "BLUE": _blank_counts()}
        for round_no in range(1, total_rounds + 1)
    }

    for event in classified_events:
        round_no = _as_int(event.get("round", 0) or 0)
        role = str(event.get("role", "")).upper()
        label = str(event.get("label", "")).strip().lower()
        if round_no not in counts:
            continue
        if role not in ("RED", "BLUE"):
            continue
        if label not in LABELS:
            continue
        counts[round_no][role][label] += 1

    out: dict[int, dict[str, RoleCriteria]] = {}
    for round_no in range(1, total_rounds + 1):
        red = _role_criteria(counts[round_no]["RED"], counts[round_no]["BLUE"])
        blue = _role_criteria(counts[round_no]["BLUE"], counts[round_no]["RED"])
        out[round_no] = {"RED": red, "BLUE": blue}
    return out
