"""Glove/keypoint derived state extraction."""

from __future__ import annotations

import math
from collections.abc import Mapping

from boxing_analytics.detection.models import GloveState

NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16


def _as_point(value: object) -> tuple[int, int] | None:
    if isinstance(value, tuple | list) and len(value) >= 2:
        return int(value[0]), int(value[1])
    return None


def _distance(a: tuple[int, int] | None, b: tuple[int, int] | None) -> float:
    if a is None or b is None:
        return 0.0
    return math.hypot(float(a[0] - b[0]), float(a[1] - b[1]))


def _extension(
    wrist: tuple[int, int] | None,
    elbow: tuple[int, int] | None,
    shoulder: tuple[int, int] | None,
) -> float:
    if wrist is None or elbow is None or shoulder is None:
        return 0.0
    upper = _distance(elbow, shoulder)
    forearm = _distance(wrist, elbow)
    if upper <= 1e-6:
        return 0.0
    ratio = forearm / upper
    return max(0.0, min(1.0, (ratio - 0.8) / 0.9))


def _visibility_score(point: tuple[int, int] | None) -> float:
    return 1.0 if point is not None else 0.0


def extract_glove_states(
    keypoints: Mapping[int, object],
    prev_wrists: Mapping[str, tuple[int, int] | None],
) -> list[GloveState]:
    lw = _as_point(keypoints.get(LEFT_WRIST))
    rw = _as_point(keypoints.get(RIGHT_WRIST))
    le = _as_point(keypoints.get(LEFT_ELBOW))
    re = _as_point(keypoints.get(RIGHT_ELBOW))
    ls = _as_point(keypoints.get(LEFT_SHOULDER))
    rs = _as_point(keypoints.get(RIGHT_SHOULDER))

    l_speed = _distance(lw, prev_wrists.get("L"))
    r_speed = _distance(rw, prev_wrists.get("R"))

    return [
        GloveState(
            hand="L",
            position=lw,
            speed=l_speed,
            extension=_extension(lw, le, ls),
            visibility=_visibility_score(lw),
        ),
        GloveState(
            hand="R",
            position=rw,
            speed=r_speed,
            extension=_extension(rw, re, rs),
            visibility=_visibility_score(rw),
        ),
    ]
