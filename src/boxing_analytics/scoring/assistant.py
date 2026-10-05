"""Scoring assistant gating and provisional round proposals."""

from __future__ import annotations

import os
from dataclasses import dataclass

from boxing_analytics.scoring.criteria import RoleCriteria


def _flag(value: object, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    try:
        return bool(int(str(value)))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True, slots=True)
class ScoringGate:
    round_alignment_ready: bool
    classification_ready: bool
    evidence_clips_ready: bool
    ref_events_flag_ready: bool
    temporal_model_ready: bool = True

    @property
    def can_propose_ten_point(self) -> bool:
        return (
            self.round_alignment_ready
            and self.classification_ready
            and self.evidence_clips_ready
            and self.ref_events_flag_ready
            and self.temporal_model_ready
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
        if not self.temporal_model_ready:
            missing.append("temporal_model_not_validated")
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
    strike_backend = str(metadata.get("strike_backend", "local")).strip().lower()
    temporal_model_ready = True
    if strike_backend in {"videomae", "hybrid_videomae"}:
        temporal = metadata.get("videomae")
        temporal_model_ready = isinstance(temporal, dict)
        if isinstance(temporal, dict):
            training = temporal.get("training_metadata", {})
            temporal_model_ready = (
                isinstance(training, dict)
                and not _flag(training.get("smoke_trained"), default=True)
                and bool(str(training.get("manifest_digest", "")).strip())
                and bool(str(temporal.get("model_sha256", "")).strip())
            )
            label_mapping = {
                str(value)
                for value in (
                    temporal.get("label_mapping", {}).values()
                    if isinstance(temporal.get("label_mapping"), dict)
                    else []
                )
            }
            temporal_model_ready = temporal_model_ready and {
                "landed_head",
                "landed_body",
                "blocked",
                "missed",
                "no_punch",
            }.issubset(label_mapping)
        uncertain = [
            event
            for event in classified_events
            if _flag(event.get("abstained"))
            or str(event.get("label", "")).strip().lower() == "uncertain"
        ]
        candidate_count = len(classified_events)
        max_abstention_rate = float(
            os.getenv("VARBOX_MAX_TEMPORAL_ABSTENTION_RATE", "0.35") or "0.35"
        )
        if candidate_count and len(uncertain) / candidate_count > max_abstention_rate:
            temporal_model_ready = False
        if _flag(metadata.get("cancelled")):
            temporal_model_ready = False

    return ScoringGate(
        round_alignment_ready=round_alignment_ready,
        classification_ready=classification_ready,
        evidence_clips_ready=evidence_clips_ready,
        ref_events_flag_ready=ref_events_flag_ready,
        temporal_model_ready=temporal_model_ready,
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
