# judge_10point.py
from typing import Dict, Tuple

def _dominance_adjust(red_landed: int, blue_landed: int) -> int:
    """
    Return extra point(s) to reduce the loser's score when there is dominance with NO knockdowns.
    Heuristic derived from WBO/ABC guidance about 10-8 without KD (sustained aggression + clean punching).
    - 10-8: clear, one-sided dominance (≈ 2.5x ratio AND diff >= 10)
    - 10-7: extremely rare, overwhelming dominance (≈ 4x ratio AND diff >= 18)
    """
    r, b = red_landed, blue_landed
    w, l = (r, b) if r >= b else (b, r)
    total = max(r + b, 1)
    ratio = w / max(l, 1)

    if ratio >= 4.0 and (w - l) >= 18:
        return 2  # 10-7 dominance (extra -2 to loser)
    if ratio >= 2.5 and (w - l) >= 10:
        return 1  # 10-8 dominance (extra -1 to loser)
    return 0

def judge_round(stats: Dict, kd_red: int = 0, kd_blue: int = 0,
                ded_red: int = 0, ded_blue: int = 0) -> Tuple[int, int, str]:
    """
    Professional 10-Point Must judging (per ABC unified rules and WBO guidance):
      • Base: 10-9 to round winner; 10-10 if truly even. :contentReference[oaicite:1]{index=1}
      • Knockdowns: each KD = -1 to the knocked-down boxer (e.g., one KD → 10-8 typical). :contentReference[oaicite:2]{index=2}
      • 10-8 (no KD) allowed for clear dominance; 10-7 extremely rare dominance. :contentReference[oaicite:3]{index=3}
      • Fouls/deductions applied AFTER scoring. :contentReference[oaicite:4]{index=4}
    Returns (red_points, blue_points, rationale).
    """
    r = int(stats.get("RED", {}).get("landed", 0))
    b = int(stats.get("BLUE", {}).get("landed", 0))
    rationale = []

    # Base score
    if r == b:
        red_pts, blue_pts = 10, 10
        rationale.append("even 10-10")
    elif r > b:
        red_pts, blue_pts = 10, 9
        rationale.append("RED 10-9 (more effective)")
    else:
        red_pts, blue_pts = 9, 10
        rationale.append("BLUE 10-9 (more effective)")

    # Knockdowns (each KD reduces the knocked-down boxer's points by 1)
    if kd_red > 0:
        red_pts -= kd_red
        rationale.append(f"RED knocked down x{kd_red}")
    if kd_blue > 0:
        blue_pts -= kd_blue
        rationale.append(f"BLUE knocked down x{kd_blue}")

    # Dominance adjustments ONLY when no KDs
    if kd_red == 0 and kd_blue == 0:
        bonus = _dominance_adjust(r, b)
        if bonus == 1:
            if r > b:
                blue_pts -= 1; rationale.append("10-8 dominance (RED)")
            else:
                red_pts  -= 1; rationale.append("10-8 dominance (BLUE)")
        elif bonus == 2:
            if r > b:
                blue_pts -= 2; rationale.append("10-7 extreme dominance (RED)")
            else:
                red_pts  -= 2; rationale.append("10-7 extreme dominance (BLUE)")

    # Deductions AFTER scoring
    if ded_red:
        red_pts -= ded_red; rationale.append(f"RED deduction -{ded_red}")
    if ded_blue:
        blue_pts -= ded_blue; rationale.append(f"BLUE deduction -{ded_blue}")

    # Clamp within reasonable bounds
    red_pts  = max(6, min(10, red_pts))
    blue_pts = max(6, min(10, blue_pts))
    return red_pts, blue_pts, " | ".join(rationale)
