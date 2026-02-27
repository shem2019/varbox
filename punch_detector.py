# punch_detector.py

from typing import Dict, Optional, Tuple

import numpy as np

from config import PUNCH_DISTANCE_THRESHOLD


def calculate_distance(p1, p2):
    return float(np.linalg.norm(np.array(p1, dtype=np.float32) - np.array(p2, dtype=np.float32)))


def _unit(v):
    n = np.linalg.norm(v)
    return (v / n) if n > 1e-6 else np.zeros_like(v)


def _clip01(x):
    return float(max(0.0, min(1.0, x)))


def _extension_score(wrist, elbow, shoulder):
    if wrist is None or elbow is None or shoulder is None:
        return 0.0
    upper = calculate_distance(elbow, shoulder)
    forearm = calculate_distance(wrist, elbow)
    if upper < 1e-6:
        return 0.0
    ratio = forearm / upper
    return _clip01((ratio - 0.85) / 0.95)


def _dynamic_hit_radius(opponent_head, opponent_shoulders):
    if opponent_head is None:
        return float(PUNCH_DISTANCE_THRESHOLD)
    if opponent_shoulders[0] is not None and opponent_shoulders[1] is not None:
        shoulder_span = calculate_distance(opponent_shoulders[0], opponent_shoulders[1])
        return max(float(PUNCH_DISTANCE_THRESHOLD) * 0.8, shoulder_span * 0.42)
    return float(PUNCH_DISTANCE_THRESHOLD)


def _score_hand(
    wrist: Optional[Tuple[int, int]],
    prev_wrist: Optional[Tuple[int, int]],
    elbow: Optional[Tuple[int, int]],
    shoulder: Optional[Tuple[int, int]],
    opponent_head: Optional[Tuple[int, int]],
    hit_radius: float,
):
    if wrist is None or opponent_head is None:
        return {
            "landed": False,
            "confidence": 0.0,
            "distance": 9999.0,
            "speed": 0.0,
            "impact_point": opponent_head or (0, 0),
        }

    w = np.array(wrist, dtype=np.float32)
    head = np.array(opponent_head, dtype=np.float32)
    dist = float(np.linalg.norm(w - head))

    vel = np.zeros(2, dtype=np.float32)
    if prev_wrist is not None:
        vel = w - np.array(prev_wrist, dtype=np.float32)
    speed = float(np.linalg.norm(vel))

    to_target = head - w
    toward = float(np.dot(_unit(vel), _unit(to_target))) if speed > 1e-6 else 0.0

    distance_score = _clip01(1.0 - (dist / max(hit_radius * 1.5, 1.0)))
    speed_score = _clip01((speed - 1.2) / 12.0)
    approach_score = _clip01((toward + 0.1) / 1.1)
    extension_score = _extension_score(wrist, elbow, shoulder)

    confidence = (
        0.48 * distance_score
        + 0.24 * approach_score
        + 0.18 * speed_score
        + 0.10 * extension_score
    )

    landed = (
        dist <= hit_radius * 1.1
        and confidence >= 0.57
        and (speed >= 1.8 or distance_score >= 0.90)
    )
    return {
        "landed": bool(landed),
        "confidence": float(confidence),
        "distance": dist,
        "speed": speed,
        "impact_point": (int(opponent_head[0]), int(opponent_head[1])),
    }


def detect_punch(fighter: Dict, opponent_head):
    """
    Confidence-based punch detector.

    fighter keys:
      - left_wrist, right_wrist
      - prev_left_wrist, prev_right_wrist (optional)
      - left_elbow, right_elbow (optional)
      - left_shoulder, right_shoulder (optional)
      - opp_left_shoulder, opp_right_shoulder (optional)
    """
    hit_radius = _dynamic_hit_radius(
        opponent_head,
        (fighter.get("opp_left_shoulder"), fighter.get("opp_right_shoulder")),
    )

    left = _score_hand(
        wrist=fighter.get("left_wrist"),
        prev_wrist=fighter.get("prev_left_wrist"),
        elbow=fighter.get("left_elbow"),
        shoulder=fighter.get("left_shoulder"),
        opponent_head=opponent_head,
        hit_radius=hit_radius,
    )
    right = _score_hand(
        wrist=fighter.get("right_wrist"),
        prev_wrist=fighter.get("prev_right_wrist"),
        elbow=fighter.get("right_elbow"),
        shoulder=fighter.get("right_shoulder"),
        opponent_head=opponent_head,
        hit_radius=hit_radius,
    )

    if right["confidence"] > left["confidence"]:
        chosen, hand = right, "R"
    else:
        chosen, hand = left, "L"

    return {
        "landed": chosen["landed"],
        "hand": hand if chosen["landed"] else "ANY",
        "confidence": round(float(chosen["confidence"]), 3),
        "distance": round(float(chosen["distance"]), 2),
        "speed": round(float(chosen["speed"]), 2),
        "impact_point": chosen["impact_point"],
        "hit_radius": round(float(hit_radius), 2),
    }
