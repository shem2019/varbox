"""Guard and clinch inference helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping

from boxing_analytics.detection.models import GuardState

NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24


def _as_point(value: object) -> tuple[int, int] | None:
    if isinstance(value, tuple | list) and len(value) >= 2:
        return int(value[0]), int(value[1])
    return None


def _distance(a: tuple[int, int] | None, b: tuple[int, int] | None) -> float:
    if a is None or b is None:
        return 9_999.0
    return math.hypot(float(a[0] - b[0]), float(a[1] - b[1]))


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


def infer_guard_state(
    defender_keypoints: Mapping[int, object],
    attacker_box: tuple[int, int, int, int],
    defender_box: tuple[int, int, int, int],
) -> GuardState:
    nose = _as_point(defender_keypoints.get(NOSE))
    ls = _as_point(defender_keypoints.get(LEFT_SHOULDER))
    rs = _as_point(defender_keypoints.get(RIGHT_SHOULDER))
    lh = _as_point(defender_keypoints.get(LEFT_HIP))
    rh = _as_point(defender_keypoints.get(RIGHT_HIP))
    lw = _as_point(defender_keypoints.get(LEFT_WRIST))
    rw = _as_point(defender_keypoints.get(RIGHT_WRIST))

    shoulder_span = _distance(ls, rs)
    if shoulder_span >= 9_000:
        shoulder_span = 60.0

    head_target = nose
    if head_target is None and ls is not None and rs is not None:
        head_target = (int((ls[0] + rs[0]) * 0.5), int((ls[1] + rs[1]) * 0.5) - 30)

    body_target: tuple[int, int] | None = None
    if ls is not None and rs is not None and lh is not None and rh is not None:
        body_target = (
            int((ls[0] + rs[0] + lh[0] + rh[0]) * 0.25),
            int((ls[1] + rs[1] + lh[1] + rh[1]) * 0.25),
        )

    min_head_dist = min(_distance(lw, head_target), _distance(rw, head_target))
    min_body_dist = min(_distance(lw, body_target), _distance(rw, body_target))

    head_guard = max(0.0, min(1.0, 1.0 - (min_head_dist / max(20.0, shoulder_span * 0.9))))
    body_guard = max(0.0, min(1.0, 1.0 - (min_body_dist / max(26.0, shoulder_span * 1.1))))

    overlap = _bbox_overlap(attacker_box, defender_box)
    clinch_score = max(0.0, min(1.0, overlap / 0.30))

    return GuardState(head_guard=head_guard, body_guard=body_guard, clinch_score=clinch_score)
