# score_tracker.py

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
        self.last_punch_frame: Dict[str, Dict[str, int]] = {
            "RED":  {"L": -COOLDOWN_FRAMES, "R": -COOLDOWN_FRAMES, "ANY": -COOLDOWN_FRAMES},
            "BLUE": {"L": -COOLDOWN_FRAMES, "R": -COOLDOWN_FRAMES, "ANY": -COOLDOWN_FRAMES},
        }
        self.punch_log: List[Dict] = []

        # 10-point must bookkeeping
        self.ten_point_totals: Dict[str, int] = {"RED": 0, "BLUE": 0}
        self.round_points: Dict[int, Tuple[int, int, str]] = {}

        # optional metadata used by scorecard (safe to ignore)
        self.metadata: Dict[str, str] = {}

    def update(self, frame_num: int, fighter_id: str, timestamp: str, hand: Optional[str] = None) -> bool:
        role = self._normalize_role(fighter_id)
        if role not in ("RED", "BLUE"):
            return False

        hand_key = self._normalize_hand(hand)
        last_frame = self.last_punch_frame[role][hand_key]
        if frame_num - last_frame <= COOLDOWN_FRAMES:
            return False

        # commit score
        self.scores[role] += 1
        self.last_punch_frame[role][hand_key] = frame_num
        if hand_key != "ANY":
            self.last_punch_frame[role]["ANY"] = frame_num

        self.punch_log.append({
            "frame": frame_num,
            "time": timestamp,
            "role": role,
            "hand": hand_key,
            "score_after": self.scores[role],
        })
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
