"""Offline dataset preparation and temporal strike-model training."""

from boxing_analytics.training.annotation_schema import (
    EIGHT_CLASS_LABELS,
    FIVE_CLASS_LABELS,
    OlympicPunchAnnotation,
)
from boxing_analytics.training.manifest import ClipManifestRow

__all__ = [
    "EIGHT_CLASS_LABELS",
    "FIVE_CLASS_LABELS",
    "ClipManifestRow",
    "OlympicPunchAnnotation",
]
