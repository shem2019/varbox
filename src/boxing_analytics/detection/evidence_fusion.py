"""Explainable deterministic fusion of pose and temporal model evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from boxing_analytics.detection.models import ContactClassification, ContactLabel
from boxing_analytics.detection.temporal_classifier import PunchCandidate, StrikeAssessment


@dataclass(frozen=True, slots=True)
class FusionConfig:
    min_transformer_confidence: float = 0.55
    min_identity_confidence: float = 0.55
    min_keypoint_visibility: float = 0.50
    landed_min_extension: float = 0.10
    landed_max_target_distance: float = 170.0
    blocked_min_guard: float = 0.35
    max_overlap_for_landed: float = 0.75
    missed_min_extension: float = 0.10
    missed_min_target_distance: float = 90.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FusionResult:
    classification: ContactClassification | None
    confidence: float
    abstained: bool
    reasons: tuple[str, ...]
    rules: dict[str, int]


def fuse_temporal_and_pose(
    *,
    assessment: StrikeAssessment,
    candidate: PunchCandidate,
    local: ContactClassification,
    config: FusionConfig,
    transformer_only: bool,
) -> FusionResult:
    reasons = list(assessment.abstention_reasons)
    rules = {
        "transformer_confident": int(assessment.confidence >= config.min_transformer_confidence),
        "identity_stable": int(candidate.identity_confidence >= config.min_identity_confidence),
        "keypoints_visible": int(candidate.keypoint_visibility >= config.min_keypoint_visibility),
        "extension_plausible": int(candidate.arm_extension >= config.landed_min_extension),
        "target_near": int(candidate.glove_target_distance <= config.landed_max_target_distance),
        "guard_present": int(candidate.guard_coverage >= config.blocked_min_guard),
        "overlap_acceptable": int(candidate.fighter_overlap <= config.max_overlap_for_landed),
        "miss_geometry_plausible": int(
            candidate.arm_extension >= config.missed_min_extension
            and candidate.glove_target_distance >= config.missed_min_target_distance
        ),
    }
    if not rules["transformer_confident"]:
        reasons.append("fusion_transformer_confidence_low")
    if not rules["identity_stable"]:
        reasons.append("identity_confidence_low")
    if not rules["keypoints_visible"]:
        reasons.append("keypoint_visibility_low")
    outcome = assessment.outcome
    if outcome == "no_punch":
        reasons.append("transformer_no_punch")
    if outcome in {"landed_head", "landed_body"}:
        if not transformer_only:
            if not rules["extension_plausible"]:
                reasons.append("landed_extension_implausible")
            if not rules["target_near"]:
                reasons.append("landed_target_too_far")
            if not rules["overlap_acceptable"]:
                reasons.append("landed_overlap_too_high")
        if reasons:
            return FusionResult(
                None, assessment.confidence, True, tuple(dict.fromkeys(reasons)), rules
            )
        clean = local.label == "landed_clean" or (
            candidate.pose_confidence >= 0.70 and candidate.guard_coverage <= 0.48
        )
        label: ContactLabel = "landed_clean" if clean else "landed_glancing"
        confidence = min(0.99, 0.65 * assessment.confidence + 0.35 * candidate.pose_confidence)
        classification = ContactClassification(
            label=label,
            hand=local.hand,
            confidence=confidence,
            impact_point=local.impact_point,
            target_zone="Head" if outcome == "landed_head" else "Body",
            glove_position=local.glove_position,
            features=dict(local.features),
        )
        return FusionResult(classification, confidence, False, (), rules)
    if outcome == "blocked":
        if not transformer_only and not rules["guard_present"]:
            reasons.append("blocked_guard_evidence_low")
        if reasons:
            return FusionResult(
                None, assessment.confidence, True, tuple(dict.fromkeys(reasons)), rules
            )
        classification = ContactClassification(
            label="blocked_guarded",
            hand=local.hand,
            confidence=assessment.confidence,
            impact_point=local.impact_point,
            target_zone=local.target_zone,
            glove_position=local.glove_position,
            features=dict(local.features),
        )
        return FusionResult(classification, assessment.confidence, False, (), rules)
    if outcome == "missed":
        if not transformer_only and not rules["miss_geometry_plausible"]:
            reasons.append("missed_pose_geometry_conflict")
        if reasons:
            return FusionResult(
                None, assessment.confidence, True, tuple(dict.fromkeys(reasons)), rules
            )
        classification = ContactClassification(
            label="missed",
            hand=local.hand,
            confidence=assessment.confidence,
            impact_point=local.impact_point,
            target_zone=local.target_zone,
            glove_position=local.glove_position,
            features=dict(local.features),
        )
        return FusionResult(classification, assessment.confidence, False, (), rules)
    reasons.append("unsupported_temporal_outcome")
    return FusionResult(None, assessment.confidence, True, tuple(dict.fromkeys(reasons)), rules)
