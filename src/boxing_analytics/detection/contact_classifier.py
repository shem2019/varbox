"""Contact event classifier from glove and guard features."""

from __future__ import annotations

import math

from boxing_analytics.detection.models import (
    ContactClassification,
    GloveState,
    GuardState,
    TargetZone,
)


def _distance(a: tuple[int, int] | None, b: tuple[int, int] | None) -> float:
    if a is None or b is None:
        return 9_999.0
    return math.hypot(float(a[0] - b[0]), float(a[1] - b[1]))


def _point_to_box_distance(point: tuple[int, int] | None, box: tuple[int, int, int, int]) -> float:
    if point is None:
        return 9_999.0
    x, y = point
    x1, y1, x2, y2 = box
    dx = max(float(x1 - x), 0.0, float(x - x2))
    dy = max(float(y1 - y), 0.0, float(y - y2))
    return math.hypot(dx, dy)


def classify_contact(
    gloves: list[GloveState],
    guard: GuardState,
    head_target: tuple[int, int] | None,
    body_target: tuple[int, int] | None,
    defender_box: tuple[int, int, int, int],
    target_scale: float,
    overlap_ratio: float,
) -> ContactClassification:
    if not gloves:
        return ContactClassification(
            label="missed",
            hand="ANY",
            confidence=0.0,
            impact_point=None,
            target_zone="Unknown",
            glove_position=None,
            features={"overlap": overlap_ratio},
        )

    def glove_score(glove: GloveState) -> float:
        return (0.45 * glove.speed) + (28.0 * glove.extension) + (15.0 * glove.visibility)

    glove = max(gloves, key=glove_score)
    head_dist = _distance(glove.position, head_target)
    body_dist = _distance(glove.position, body_target)
    box_dist = _point_to_box_distance(glove.position, defender_box)
    target_zone: TargetZone = "Head" if head_dist <= body_dist else "Body"
    target_dist = min(head_dist, body_dist, box_dist)

    guard_cover = guard.head_guard if target_zone == "Head" else guard.body_guard
    target_scale = max(36.0, float(target_scale))
    hit_radius = max(42.0, 0.72 * target_scale)
    glance_radius = max(hit_radius * 1.45, target_scale * 1.05)
    inside_defender = box_dist <= 1.0

    features = {
        "speed": float(glove.speed),
        "extension": float(glove.extension),
        "target_dist": float(target_dist),
        "box_dist": float(box_dist),
        "target_scale": float(target_scale),
        "guard_cover": float(guard_cover),
        "clinch_score": float(guard.clinch_score),
        "overlap": float(overlap_ratio),
    }

    if guard.clinch_score >= 0.92 and target_dist > hit_radius * 0.55:
        return ContactClassification(
            label="clinch",
            hand=glove.hand,
            confidence=min(0.97, 0.55 + 0.4 * guard.clinch_score),
            impact_point=glove.position,
            target_zone=target_zone,
            glove_position=glove.position,
            features=features,
        )

    if (
        (target_dist <= hit_radius or (inside_defender and glove.speed >= 0.8))
        and glove.speed >= 0.8
        and (glove.extension >= 0.16 or inside_defender)
        and guard_cover <= 0.48
    ):
        confidence = min(0.99, 0.40 + 0.12 * glove.speed + 0.24 * glove.extension)
        return ContactClassification(
            label="landed_clean",
            hand=glove.hand,
            confidence=confidence,
            impact_point=glove.position,
            target_zone=target_zone,
            glove_position=glove.position,
            features=features,
        )

    if (
        target_dist <= glance_radius
        and glove.speed >= 0.45
        and (glove.extension >= 0.08 or box_dist <= target_scale * 0.18)
        and guard_cover <= 0.72
    ):
        confidence = min(0.93, 0.32 + 0.10 * glove.speed + 0.16 * glove.extension)
        return ContactClassification(
            label="landed_glancing",
            hand=glove.hand,
            confidence=confidence,
            impact_point=glove.position,
            target_zone=target_zone,
            glove_position=glove.position,
            features=features,
        )

    if target_dist <= hit_radius * 1.20 and guard_cover > 0.42:
        confidence = min(0.94, 0.32 + 0.35 * guard_cover)
        return ContactClassification(
            label="blocked_guarded",
            hand=glove.hand,
            confidence=confidence,
            impact_point=glove.position,
            target_zone=target_zone,
            glove_position=glove.position,
            features=features,
        )

    confidence = min(0.90, 0.28 + 0.07 * glove.extension + 0.03 * glove.speed)
    return ContactClassification(
        label="missed",
        hand=glove.hand,
        confidence=confidence,
        impact_point=glove.position,
        target_zone=target_zone,
        glove_position=glove.position,
        features=features,
    )
