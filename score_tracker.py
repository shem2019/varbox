# score_tracker.py

import os
from typing import Dict, List, Optional, Tuple

from config import COOLDOWN_FRAMES


class ScoreTracker:
    """
    Role-locked scorer for exactly two participants: 'RED' and 'BLUE'.
    - update(frame_num, fighter_id, timestamp, hand=None) -> bool
    - get_score(role) -> int
    - add_round_points(round_no, red_pts, blue_pts, rationale) -> None
    Exposes:
      - punch_log: List[dict(frame, time, role, hand, score_after)]
      - round_points: Dict[int, Tuple[int, int, str]]
      - ten_point_totals: {"RED": int, "BLUE": int}
    """

    def __init__(self):
        self.scores: Dict[str, int] = {"RED": 0, "BLUE": 0}
        self.cooldown_seconds = float(os.getenv("VARBOX_EVENT_COOLDOWN_SECONDS", "0.32") or "0.32")
        self.last_punch_frame: Dict[str, Dict[str, int]] = {
            "RED":  {"L": -COOLDOWN_FRAMES, "R": -COOLDOWN_FRAMES, "ANY": -COOLDOWN_FRAMES},
            "BLUE": {"L": -COOLDOWN_FRAMES, "R": -COOLDOWN_FRAMES, "ANY": -COOLDOWN_FRAMES},
        }
        self.last_punch_time_s: Dict[str, Dict[str, float]] = {
            "RED": {"L": -9999.0, "R": -9999.0, "ANY": -9999.0},
            "BLUE": {"L": -9999.0, "R": -9999.0, "ANY": -9999.0},
        }
        self.punch_log: List[Dict] = []

        # 10-point must bookkeeping
        self.ten_point_totals: Dict[str, int] = {"RED": 0, "BLUE": 0}
        self.round_points: Dict[int, Tuple[int, int, str]] = {}

        # optional metadata used by scorecard (safe to ignore)
        self.metadata: Dict[str, str] = {}
        self.attempts: Dict[str, int] = {"RED": 0, "BLUE": 0}

    def register_attempt(self, fighter_id: str) -> None:
        role = self._normalize_role(fighter_id)
        if role in self.attempts:
            self.attempts[role] += 1

    def update(
        self,
        frame_num: Optional[int] = None,
        fighter_id: str = "",
        timestamp: str = "",
        hand: Optional[str] = None,
        frame_idx: Optional[int] = None,
        event_time_s: Optional[float] = None,
        details: Optional[Dict] = None,
    ) -> bool:
        if frame_num is None:
            frame_num = frame_idx
        if frame_num is None:
            return False

        role = self._normalize_role(fighter_id)
        if role not in ("RED", "BLUE"):
            return False

        hand_key = self._normalize_hand(hand)
        if event_time_s is not None:
            last_time = self.last_punch_time_s[role][hand_key]
            if event_time_s - last_time <= self.cooldown_seconds:
                return False
            self.last_punch_time_s[role][hand_key] = event_time_s
            if hand_key != "ANY":
                self.last_punch_time_s[role]["ANY"] = event_time_s
        else:
            last_frame = self.last_punch_frame[role][hand_key]
            if frame_num - last_frame <= COOLDOWN_FRAMES:
                return False
            self.last_punch_frame[role][hand_key] = frame_num
            if hand_key != "ANY":
                self.last_punch_frame[role]["ANY"] = frame_num

        # commit score
        self.scores[role] += 1

        event = {
            "frame": frame_num,
            "time": timestamp,
            "role": role,
            "hand": hand_key,
            "score_after": self.scores[role],
            "event_time_s": event_time_s,
        }
        if isinstance(details, dict):
            event.update(details)
        self.punch_log.append(event)
        return True

    def get_score(self, role: str) -> int:
        return self.scores.get(self._normalize_role(role), 0)

    def add_round_points(self, round_no: int, red_pts: int, blue_pts: int, rationale: str) -> None:
        self.round_points[round_no] = (red_pts, blue_pts, rationale)
        self.ten_point_totals["RED"] += red_pts
        self.ten_point_totals["BLUE"] += blue_pts

    @staticmethod
    def _normalize_role(role: str) -> str:
        if not isinstance(role, str): return ""
        r = role.strip().upper()
        if r in ("RED", "R"): return "RED"
        if r in ("BLUE", "B"): return "BLUE"
        return ""

    @staticmethod
    def _normalize_hand(hand: Optional[str]) -> str:
        if not hand: return "ANY"
        h = str(hand).strip().upper()
        if h in ("L", "LEFT", "LEFT_HAND", "LEFT_WRIST"): return "L"
        if h in ("R", "RIGHT", "RIGHT_HAND", "RIGHT_WRIST"): return "R"
        return "ANY"
