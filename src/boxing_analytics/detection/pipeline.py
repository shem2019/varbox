"""Multi-stage strike event pipeline."""

from __future__ import annotations

from collections.abc import Mapping

from boxing_analytics.detection.contact_classifier import classify_contact
from boxing_analytics.detection.glove_state import (
    LEFT_SHOULDER,
    NOSE,
    RIGHT_SHOULDER,
    extract_glove_states,
)
from boxing_analytics.detection.guard_state import infer_guard_state
from boxing_analytics.detection.models import ContactClassification

LEFT_HIP = 23
RIGHT_HIP = 24


def _as_point(value: object) -> tuple[int, int] | None:
    if isinstance(value, tuple | list) and len(value) >= 2:
        return int(value[0]), int(value[1])
    return None


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter = float((inter_x2 - inter_x1) * (inter_y2 - inter_y1))
    area_a = float(max(1, (ax2 - ax1) * (ay2 - ay1)))
    area_b = float(max(1, (bx2 - bx1) * (by2 - by1)))
    return inter / max(1.0, min(area_a, area_b))


def _targets(
    defender_keypoints: Mapping[int, object],
) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    nose = _as_point(defender_keypoints.get(NOSE))
    ls = _as_point(defender_keypoints.get(LEFT_SHOULDER))
    rs = _as_point(defender_keypoints.get(RIGHT_SHOULDER))
    lh = _as_point(defender_keypoints.get(LEFT_HIP))
    rh = _as_point(defender_keypoints.get(RIGHT_HIP))

    head = nose
    if head is None and ls is not None and rs is not None:
        head = (int((ls[0] + rs[0]) * 0.5), int((ls[1] + rs[1]) * 0.5) - 30)

    body = None
    if ls is not None and rs is not None and lh is not None and rh is not None:
        body = (
            int((ls[0] + rs[0] + lh[0] + rh[0]) * 0.25),
            int((ls[1] + rs[1] + lh[1] + rh[1]) * 0.25),
        )
    return head, body


def evaluate_strike(
    attacker_keypoints: Mapping[int, object],
    defender_keypoints: Mapping[int, object],
    prev_wrists: Mapping[str, tuple[int, int] | None],
    attacker_box: tuple[int, int, int, int],
    defender_box: tuple[int, int, int, int],
) -> ContactClassification:
    gloves = extract_glove_states(attacker_keypoints, prev_wrists)
    guard = infer_guard_state(
        defender_keypoints,
        attacker_box=attacker_box,
        defender_box=defender_box,
    )
    head_target, body_target = _targets(defender_keypoints)
    overlap = _bbox_overlap(attacker_box, defender_box)
    return classify_contact(
        gloves=gloves,
        guard=guard,
        head_target=head_target,
        body_target=body_target,
        overlap_ratio=overlap,
    )
