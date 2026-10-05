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
from boxing_analytics.detection.temporal_buffer import TemporalClipBuffer, TemporalFrameSample

__all__ = [
    "ContactClassification",
    "ContactLabel",
    "EventDeduplicator",
    "GloveState",
    "GuardState",
    "RoboflowBoxingAssessor",
    "RoboflowFrameResult",
    "TemporalClipBuffer",
    "TemporalFrameSample",
    "evaluate_strike",
]
