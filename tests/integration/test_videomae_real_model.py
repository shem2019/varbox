import os
from pathlib import Path

import numpy as np
import pytest

from boxing_analytics.detection.temporal_classifier import (
    PunchCandidate,
    VideoMAETemporalStrikeClassifier,
)


@pytest.mark.integration
def test_real_local_videomae_checkpoint_opt_in() -> None:
    if os.getenv("VARBOX_RUN_REAL_MODEL_TEST", "0") != "1":
        pytest.skip("Set VARBOX_RUN_REAL_MODEL_TEST=1 for local VideoMAE inference")
    model_dir = Path(
        os.getenv(
            "VARBOX_VIDEOMAE_MODEL_DIR",
            "models/varbox-videomae-development-current/best",
        )
    )
    if not model_dir.is_dir():
        pytest.skip(f"Local VideoMAE model unavailable: {model_dir}")
    classifier = VideoMAETemporalStrikeClassifier(
        model_dir=str(model_dir),
        device="auto",
        confidence_threshold=0.55,
    )
    frames = [
        np.zeros((classifier.image_size, classifier.image_size, 3), dtype=np.uint8)
        for _ in range(classifier.frame_count)
    ]
    candidate = PunchCandidate(
        candidate_id="integration",
        timestamp_s=1.0,
        attacker_role="RED",
        defender_role="BLUE",
        hand="LEFT",
        pose_confidence=0.8,
        wrist_speed=1.0,
        wrist_acceleration=0.0,
        arm_extension=0.5,
        glove_target_distance=50.0,
        guard_coverage=0.2,
        fighter_overlap=0.1,
        clinch_score=0.0,
        identity_confidence=0.9,
        keypoint_visibility=0.9,
    )
    assessment = classifier.classify(
        frames,
        candidate,
        [None] * len(frames),
        [None] * len(frames),
        clip_start_time=0.7,
        clip_end_time=1.3,
    )
    assert set(assessment.class_probabilities) == {
        "landed_head",
        "landed_body",
        "blocked",
        "missed",
        "no_punch",
    }
