"""Strike/contact detection pipeline modules."""

from boxing_analytics.detection.event_deduplicator import EventDeduplicator
from boxing_analytics.detection.models import (
    ContactClassification,
    ContactLabel,
    GloveState,
    GuardState,
)
from boxing_analytics.detection.pipeline import evaluate_strike
from boxing_analytics.detection.roboflow import RoboflowBoxingAssessor, RoboflowFrameResult

__all__ = [
    "ContactClassification",
    "ContactLabel",
    "EventDeduplicator",
    "GloveState",
    "GuardState",
    "RoboflowBoxingAssessor",
    "RoboflowFrameResult",
    "evaluate_strike",
]
