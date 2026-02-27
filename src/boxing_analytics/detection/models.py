"""Shared detection pipeline models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ContactLabel = Literal[
    "landed_clean",
    "landed_glancing",
    "blocked_guarded",
    "missed",
    "clinch",
]
TargetZone = Literal["Head", "Body", "Unknown"]


@dataclass(frozen=True, slots=True)
class GloveState:
    hand: Literal["L", "R"]
    position: tuple[int, int] | None
    speed: float
    extension: float
    visibility: float


@dataclass(frozen=True, slots=True)
class GuardState:
    head_guard: float
    body_guard: float
    clinch_score: float


@dataclass(frozen=True, slots=True)
class ContactClassification:
    label: ContactLabel
    hand: Literal["L", "R", "ANY"]
    confidence: float
    impact_point: tuple[int, int] | None
    target_zone: TargetZone
    glove_position: tuple[int, int] | None
    features: dict[str, float]
