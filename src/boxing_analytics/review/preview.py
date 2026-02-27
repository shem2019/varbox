"""In-app scoring preview utilities for corrected event timelines."""

from __future__ import annotations

import json
from typing import Any

from boxing_analytics.review.corrections import apply_manual_corrections, parse_manual_corrections
from boxing_analytics.scoring import (
    build_round_criteria,
    evaluate_scoring_gate,
    propose_round_points,
)


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def _is_truthy_int(value: object) -> bool:
    return _as_int(value, 0) != 0


def _round_penalties(
    *,
    total_rounds: int,
    ref_events: list[dict[str, object]],
) -> tuple[dict[int, dict[str, int]], dict[int, dict[str, int]], dict[int, dict[str, int]]]:
    kd = {r: {"RED": 0, "BLUE": 0} for r in range(1, total_rounds + 1)}
    deductions = {r: {"RED": 0, "BLUE": 0} for r in range(1, total_rounds + 1)}
    fouls = {r: {"RED": 0, "BLUE": 0} for r in range(1, total_rounds + 1)}

    for row in ref_events:
        round_no = _as_int(row.get("round", 0))
        role = str(row.get("role", "")).strip().upper()
        event_type = str(row.get("event_type", "")).strip().lower()
        points = max(1, _as_int(row.get("points", 1), 1))
        count = max(1, _as_int(row.get("count", 1), 1))
        if round_no not in kd or role not in {"RED", "BLUE"}:
            continue
        if event_type == "knockdown":
            kd[round_no][role] += count
        elif event_type == "deduction":
            deductions[round_no][role] += points
        elif event_type == "foul":
            fouls[round_no][role] += count

    return kd, deductions, fouls


def preview_scoring_outcome(
    *,
    metadata: dict[str, object],
    classified_events: list[dict[str, object]],
    punch_log: list[dict[str, object]],
    manual_corrections: list[dict[str, object]],
    ref_events: list[dict[str, object]],
) -> dict[str, Any]:
    corrections = parse_manual_corrections(json.dumps(manual_corrections))
    corrected_events, corrected_punch_log, applied_rows = apply_manual_corrections(
        classified_events=classified_events,
        punch_log=punch_log,
        corrections=corrections,
    )
    scoring_events = [
        row for row in corrected_events if not _is_truthy_int(row.get("invalidated_by_review", 0))
    ]
    scoring_punches = [
        row
        for row in corrected_punch_log
        if not _is_truthy_int(row.get("invalidated_by_review", 0))
    ]

    gate_metadata = dict(metadata)
    gate_metadata["confirmed_ref_event_flags_present"] = (
        1 if ref_events else _as_int(metadata.get("confirmed_ref_event_flags_present", 0), 0)
    )
    gate = evaluate_scoring_gate(
        metadata=gate_metadata,
        classified_events=scoring_events,
        punch_log=scoring_punches,
    )

    bout_cfg = metadata.get("bout_config", {})
    total_rounds = _as_int(
        bout_cfg.get("rounds_count", 12) if isinstance(bout_cfg, dict) else 12,
        12,
    )
    criteria = build_round_criteria(scoring_events, total_rounds)
    kd, deductions, fouls = _round_penalties(total_rounds=total_rounds, ref_events=ref_events)

    proposals: dict[int, tuple[int, int, str]] = {}
    if gate.can_propose_ten_point:
        proposals = propose_round_points(
            criteria=criteria,
            kd=kd,
            deductions=deductions,
            total_rounds=total_rounds,
        )
    return {
        "manual_corrections_applied_count": len(
            [row for row in applied_rows if _is_truthy_int(row.get("applied", 0))]
        ),
        "can_propose_ten_point": int(gate.can_propose_ten_point),
        "missing_reasons": gate.missing_reasons(),
        "proposals": proposals,
        "criteria_by_round": {
            round_no: {
                role: {
                    "clean_punching_score": role_crit.clean_punching_score,
                    "effective_aggressiveness_score": role_crit.effective_aggressiveness_score,
                    "ring_generalship_score": role_crit.ring_generalship_score,
                    "defense_score": role_crit.defense_score,
                }
                for role, role_crit in role_map.items()
            }
            for round_no, role_map in criteria.items()
        },
        "ref_penalties": {
            "kd": kd,
            "deductions": deductions,
            "fouls": fouls,
        },
    }
