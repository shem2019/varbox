from boxing_analytics.detection.evidence_fusion import (
    FusionConfig,
    fuse_temporal_and_pose,
)
from boxing_analytics.detection.models import ContactClassification
from boxing_analytics.detection.temporal_classifier import (
    PunchCandidate,
    StrikeAssessment,
)


def _candidate(identity_confidence: float = 0.95) -> PunchCandidate:
    return PunchCandidate(
        candidate_id="RED_1",
        timestamp_s=1.0,
        attacker_role="RED",
        defender_role="BLUE",
        hand="LEFT",
        pose_confidence=0.8,
        wrist_speed=3.0,
        wrist_acceleration=0.0,
        arm_extension=0.6,
        glove_target_distance=25.0,
        guard_coverage=0.1,
        fighter_overlap=0.1,
        clinch_score=0.0,
        identity_confidence=identity_confidence,
        keypoint_visibility=0.9,
    )


def _local() -> ContactClassification:
    return ContactClassification(
        label="landed_clean",
        hand="L",
        confidence=0.8,
        impact_point=(50, 50),
        target_zone="Head",
        glove_position=(50, 50),
        features={"speed": 3.0, "extension": 0.6},
    )


def _assessment() -> StrikeAssessment:
    return StrikeAssessment(
        outcome="landed_head",
        target_zone="Head",
        hand="LEFT",
        confidence=0.9,
        class_probabilities={"landed_head": 0.9},
        model_name="fixture",
        model_version="1",
        device="cpu",
        inference_duration_ms=1.0,
        clip_start_time=0.7,
        clip_end_time=1.3,
        abstained=False,
        abstention_reasons=(),
    )


def test_fusion_accepts_consistent_landed_evidence() -> None:
    result = fuse_temporal_and_pose(
        assessment=_assessment(),
        candidate=_candidate(),
        local=_local(),
        config=FusionConfig(),
        transformer_only=False,
    )
    assert not result.abstained
    assert result.classification is not None
    assert result.classification.label == "landed_clean"


def test_fusion_abstains_when_identity_is_unstable() -> None:
    result = fuse_temporal_and_pose(
        assessment=_assessment(),
        candidate=_candidate(identity_confidence=0.1),
        local=_local(),
        config=FusionConfig(),
        transformer_only=False,
    )
    assert result.abstained
    assert result.classification is None
    assert "identity_confidence_low" in result.reasons


def test_fusion_rejects_miss_prediction_when_pose_is_at_target() -> None:
    landed = _assessment()
    assessment = StrikeAssessment(
        outcome="missed",
        target_zone="Unknown",
        hand=landed.hand,
        confidence=landed.confidence,
        class_probabilities={"missed": landed.confidence},
        model_name=landed.model_name,
        model_version=landed.model_version,
        device=landed.device,
        inference_duration_ms=landed.inference_duration_ms,
        clip_start_time=landed.clip_start_time,
        clip_end_time=landed.clip_end_time,
        abstained=False,
        abstention_reasons=(),
    )
    result = fuse_temporal_and_pose(
        assessment=assessment,
        candidate=_candidate(),
        local=_local(),
        config=FusionConfig(),
        transformer_only=False,
    )
    assert result.abstained
    assert "missed_pose_geometry_conflict" in result.reasons
