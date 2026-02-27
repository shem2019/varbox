"""Scoring assistant gating and provisional round proposals."""

from __future__ import annotations

import os
from dataclasses import dataclass

from boxing_analytics.scoring.criteria import RoleCriteria


@dataclass(frozen=True, slots=True)
class ScoringGate:
    round_alignment_ready: bool
    classification_ready: bool
    evidence_clips_ready: bool
    ref_events_flag_ready: bool

    @property
    def can_propose_ten_point(self) -> bool:
        return (
            self.round_alignment_ready
            and self.classification_ready
            and self.evidence_clips_ready
            and self.ref_events_flag_ready
        )

    def missing_reasons(self) -> list[str]:
        missing = []
        if not self.round_alignment_ready:
            missing.append("round_alignment_missing")
        if not self.classification_ready:
            missing.append("classification_missing")
        if not self.evidence_clips_ready:
            missing.append("evidence_clips_missing")
        if not self.ref_events_flag_ready:
            missing.append("ref_flags_missing")
        return missing


def evaluate_scoring_gate(
    *,
    metadata: dict[str, object],
    classified_events: list[dict[str, object]],
    punch_log: list[dict[str, object]],
) -> ScoringGate:
    round_alignment_ready = bool(metadata.get("round_segments")) and bool(
        metadata.get("timeline_markers")
    )

    labels = {str(ev.get("label", "")).strip().lower() for ev in classified_events}
    classification_ready = (
        "landed_clean" in labels or "landed_glancing" in labels
    ) and "blocked_guarded" in labels

    landed = [
        row
        for row in punch_log
        if str(row.get("classification_label", "")).strip().lower()
        in {"landed_clean", "landed_glancing"}
    ]
    evidence_clips_ready = bool(landed)
    if evidence_clips_ready:
        evidence_clips_ready = all(
            bool(item.get("evidence_clip")) and os.path.isfile(str(item.get("evidence_clip")))
            for item in landed
        )

    ref_events_flag_ready = bool(metadata.get("confirmed_ref_event_flags_present", 0))

    return ScoringGate(
        round_alignment_ready=round_alignment_ready,
        classification_ready=classification_ready,
        evidence_clips_ready=evidence_clips_ready,
        ref_events_flag_ready=ref_events_flag_ready,
    )


def _total_signal(role: RoleCriteria) -> float:
    return (
        0.40 * role.clean_punching_score
        + 0.25 * role.effective_aggressiveness_score
        + 0.20 * role.ring_generalship_score
        + 0.15 * role.defense_score
    )


def propose_round_points(
    *,
    criteria: dict[int, dict[str, RoleCriteria]],
    kd: dict[int, dict[str, int]],
    deductions: dict[int, dict[str, int]],
    total_rounds: int,
) -> dict[int, tuple[int, int, str]]:
    proposals: dict[int, tuple[int, int, str]] = {}

    for round_no in range(1, total_rounds + 1):
        pair = criteria.get(round_no)
        if not pair:
            continue
        red = pair["RED"]
        blue = pair["BLUE"]
        red_total = _total_signal(red)
        blue_total = _total_signal(blue)
        diff = abs(red_total - blue_total)

        if diff < 0.35:
            red_pts, blue_pts = 10, 10
            note = "assistant_even_10_10"
        elif red_total > blue_total:
            red_pts, blue_pts = 10, 9
            note = "assistant_red_10_9"
            if diff >= 2.2:
                blue_pts = 8
                note = "assistant_red_10_8"
        else:
            red_pts, blue_pts = 9, 10
            note = "assistant_blue_10_9"
            if diff >= 2.2:
                red_pts = 8
                note = "assistant_blue_10_8"

        kd_red = int(kd.get(round_no, {}).get("RED", 0))
        kd_blue = int(kd.get(round_no, {}).get("BLUE", 0))
        ded_red = int(deductions.get(round_no, {}).get("RED", 0))
        ded_blue = int(deductions.get(round_no, {}).get("BLUE", 0))

        red_pts -= kd_red + ded_red
        blue_pts -= kd_blue + ded_blue

        red_pts = max(6, min(10, red_pts))
        blue_pts = max(6, min(10, blue_pts))

        rationale = (
            f"{note} | signals red={red_total:.2f} blue={blue_total:.2f} "
            f"| kd(R/B)={kd_red}/{kd_blue} | ded(R/B)={ded_red}/{ded_blue} "
            f"| provisional_assistant_only"
        )
        proposals[round_no] = (red_pts, blue_pts, rationale)

    return proposals
