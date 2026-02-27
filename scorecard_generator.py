import os
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from fpdf import FPDF

try:
    from config import SCORECARD_PDF as _DEFAULT_PDF
except Exception:
    _DEFAULT_PDF = "boxing_scorecard.pdf"

_SAFE_MAP = {
    "\u2014": "-",
    "\u2013": "-",
    "\u2212": "-",
    "\u2026": "...",
    "\u2018": "'",
    "\u2019": "'",
    "\u201C": '"',
    "\u201D": '"',
    "\u2022": "*",
    "\u00D7": "x",
    "\u00A0": " ",
    "\u200B": "",
}


def _safe_text(x: Any) -> str:
    s = "" if x is None else str(x)
    for u, a in _SAFE_MAP.items():
        s = s.replace(u, a)
    return s.encode("latin-1", "ignore").decode("latin-1")


def _extract_logs(punch_log: Iterable[Any]) -> List[Dict]:
    rows: List[Dict] = []
    for item in punch_log or []:
        if isinstance(item, dict):
            if int(item.get("invalidated_by_review", 0) or 0):
                continue
            role = str(item.get("role", "")).upper()
            hand = str(item.get("hand", "ANY")).upper()
            if hand not in ("L", "R", "ANY"):
                hand = "ANY"
            rows.append(
                {
                    "frame": item.get("frame", ""),
                    "time": item.get("time", ""),
                    "role": role,
                    "hand": hand,
                    "score": item.get("score_after", ""),
                    "confidence": item.get("confidence", ""),
                    "target_zone": item.get("target_zone", ""),
                    "fighter_name": item.get("fighter_name", ""),
                    "opponent_name": item.get("opponent_name", ""),
                    "round": item.get("round", ""),
                    "evidence_image": item.get("evidence_image", ""),
                    "evidence_clip": item.get("evidence_clip", ""),
                    "corrected_by_review": int(item.get("corrected_by_review", 0) or 0),
                }
            )
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            fighter, time_s, score = item[0], item[1], item[2]
            rows.append(
                {
                    "frame": "",
                    "time": time_s,
                    "role": str(fighter).upper(),
                    "hand": "ANY",
                    "score": score,
                    "confidence": "",
                    "target_zone": "",
                    "fighter_name": "",
                    "opponent_name": "",
                    "round": "",
                    "evidence_image": "",
                    "evidence_clip": "",
                }
            )
    return rows


def _totals(rows: List[Dict]) -> Dict[str, int]:
    out = {"RED": 0, "BLUE": 0}
    for r in rows:
        role = r.get("role")
        if role in out:
            out[role] += 1
    return out


def _hand_totals(rows: List[Dict]) -> Dict[str, Dict[str, int]]:
    out = {"RED": {"L": 0, "R": 0}, "BLUE": {"L": 0, "R": 0}}
    for r in rows:
        role = r.get("role")
        hand = r.get("hand")
        if role in out and hand in ("L", "R"):
            out[role][hand] += 1
    return out


def _extract_round_artifacts(tracker: Any):
    round_points = getattr(tracker, "round_points", {}) if tracker else {}
    ten_totals = (
        getattr(tracker, "ten_point_totals", {"RED": 0, "BLUE": 0})
        if tracker
        else {"RED": 0, "BLUE": 0}
    )

    meta = getattr(tracker, "metadata", {}) if tracker and hasattr(tracker, "metadata") else {}
    round_stats = getattr(tracker, "round_stats", None) or meta.get("round_stats")
    kd = getattr(tracker, "kd", None) or meta.get("kd")
    deductions = getattr(tracker, "deductions", None) or meta.get("deductions")
    fouls = getattr(tracker, "fouls", None) or meta.get("fouls")

    return round_points or {}, ten_totals, round_stats, kd, deductions, fouls


class ProScorecardPDF(FPDF):
    def __init__(self, title="VAR Box Match Report"):
        super().__init__()
        self.report_title = _safe_text(title)

    def header(self):
        self.set_fill_color(19, 24, 34)
        self.rect(0, 0, self.w, 16, "F")
        self.set_text_color(245, 248, 255)
        self.set_font("Arial", "B", 11)
        self.set_xy(10, 4)
        self.cell(0, 6, self.report_title, 0, 0, "L")
        self.set_text_color(0, 0, 0)
        self.ln(12)

    def footer(self):
        self.set_y(-10)
        self.set_text_color(95, 102, 118)
        self.set_font("Arial", "", 8)
        self.cell(0, 6, f"Page {self.page_no()}", 0, 0, "C")
        self.set_text_color(0, 0, 0)


def _write_key_value(pdf: FPDF, k: str, v: Any):
    pdf.set_font("Arial", "B", 10)
    pdf.cell(42, 6, _safe_text(f"{k}:"))
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 6, _safe_text(v), ln=True)


def _table_header(pdf: FPDF, headers, widths):
    pdf.set_fill_color(238, 242, 249)
    pdf.set_font("Arial", "B", 9)
    for h, w in zip(headers, widths):
        pdf.cell(w, 7, _safe_text(h), border=1, align="C", fill=True)
    pdf.ln(7)
    pdf.set_font("Arial", "", 9)


def _add_log_table(pdf: FPDF, rows: List[Dict]):
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Punch Log", ln=True)
    headers = ["Rd", "Time", "Role", "Hand", "Zone", "Conf", "Score"]
    widths = [12, 20, 18, 15, 30, 18, 18]
    _table_header(pdf, headers, widths)
    for r in rows:
        vals = [
            r.get("round", ""),
            r.get("time", ""),
            r.get("role", ""),
            r.get("hand", ""),
            r.get("target_zone", ""),
            r.get("confidence", ""),
            r.get("score", ""),
        ]
        for v, w in zip(vals, widths):
            pdf.cell(w, 7, _safe_text(v), border=1)
        pdf.ln(7)


def _add_round_tables(pdf: FPDF, round_stats, round_points, kd, deductions, fouls, ten_totals):
    if isinstance(round_stats, dict) and round_stats:
        pdf.ln(3)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 7, "Per-Round Landed Punches", ln=True)
        headers = ["Round", "RED Landed", "BLUE Landed"]
        widths = [20, 32, 32]
        _table_header(pdf, headers, widths)
        for rnd in sorted(round_stats.keys()):
            rs = round_stats[rnd]
            pdf.cell(widths[0], 7, _safe_text(rnd), border=1)
            pdf.cell(widths[1], 7, _safe_text(rs.get("RED", {}).get("landed", 0)), border=1)
            pdf.cell(widths[2], 7, _safe_text(rs.get("BLUE", {}).get("landed", 0)), border=1)
            pdf.ln(7)

    if isinstance(round_points, dict) and round_points:
        pdf.ln(3)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 7, "10-Point Must Summary", ln=True)
        headers = ["Round", "RED", "BLUE", "KD(R/B)", "Ded(R/B)", "Foul(R/B)", "Rationale"]
        page_w = pdf.w - 2 * pdf.l_margin
        widths = [12, 12, 12, 20, 22, 24, max(40, page_w - 102)]
        _table_header(pdf, headers, widths)
        for rnd in sorted(round_points.keys()):
            red_pts, blue_pts, rationale = round_points[rnd]
            kd_r = kd.get(rnd, {}).get("RED", 0) if isinstance(kd, dict) else 0
            kd_b = kd.get(rnd, {}).get("BLUE", 0) if isinstance(kd, dict) else 0
            dd_r = deductions.get(rnd, {}).get("RED", 0) if isinstance(deductions, dict) else 0
            dd_b = deductions.get(rnd, {}).get("BLUE", 0) if isinstance(deductions, dict) else 0
            foul_r = fouls.get(rnd, {}).get("RED", 0) if isinstance(fouls, dict) else 0
            foul_b = fouls.get(rnd, {}).get("BLUE", 0) if isinstance(fouls, dict) else 0
            row = [
                rnd,
                red_pts,
                blue_pts,
                f"{kd_r}/{kd_b}",
                f"{dd_r}/{dd_b}",
                f"{foul_r}/{foul_b}",
                rationale,
            ]
            for i, (v, w) in enumerate(zip(row, widths)):
                align = "L" if i == len(row) - 1 else "C"
                pdf.cell(w, 7, _safe_text(v), border=1, align=align)
            pdf.ln(7)

        pdf.set_font("Arial", "B", 11)
        pdf.ln(2)
        pdf.cell(
            0,
            7,
            _safe_text(
                f"Cumulative 10-Point Totals  |  RED: {ten_totals.get('RED', 0)}  BLUE: {ten_totals.get('BLUE', 0)}"
            ),
            ln=True,
        )


def _add_criteria_table(pdf: FPDF, criteria_by_round):
    if not isinstance(criteria_by_round, dict) or not criteria_by_round:
        return
    pdf.ln(3)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 7, "Criteria Signals (Assistant)", ln=True)
    headers = ["Round", "Role", "Clean", "Agg", "Ring", "Def"]
    widths = [16, 16, 20, 20, 20, 20]
    _table_header(pdf, headers, widths)
    for rnd in sorted(criteria_by_round.keys()):
        pair = criteria_by_round[rnd]
        if not isinstance(pair, dict):
            continue
        for role in ("RED", "BLUE"):
            sig = pair.get(role, {})
            row = [
                rnd,
                role,
                sig.get("clean_punching_score", 0),
                sig.get("effective_aggressiveness_score", 0),
                sig.get("ring_generalship_score", 0),
                sig.get("defense_score", 0),
            ]
            for value, width in zip(row, widths):
                pdf.cell(width, 7, _safe_text(value), border=1, align="C")
            pdf.ln(7)


def _add_evidence_pages(pdf: FPDF, rows: List[Dict], max_items: int = 80):
    evidence_rows = [
        r for r in rows if r.get("evidence_image") and os.path.isfile(r.get("evidence_image"))
    ]
    if not evidence_rows:
        return
    evidence_rows = evidence_rows[:max_items]

    slot = 0
    for r in evidence_rows:
        if slot == 0:
            pdf.add_page()
            pdf.set_font("Arial", "B", 13)
            pdf.cell(0, 8, "Punch Evidence", ln=True)
            pdf.ln(1)
            y = 26
        else:
            y = 150

        x = 10
        card_h = 116
        card_w = pdf.w - 20
        pdf.set_draw_color(75, 83, 100)
        pdf.rect(x, y, card_w, card_h)

        pdf.set_xy(x + 3, y + 3)
        pdf.set_font("Arial", "B", 10)
        header = (
            f"R{r.get('round', '-')}: {r.get('fighter_name', r.get('role'))} -> "
            f"{r.get('opponent_name', r.get('opponent_role', ''))} | "
            f"{r.get('target_zone', 'Unknown')} | conf {r.get('confidence', '-')}"
        )
        pdf.cell(card_w - 6, 6, _safe_text(header), ln=True)

        img_path = r.get("evidence_image")
        try:
            pdf.image(img_path, x=x + 5, y=y + 12, w=card_w - 10)
        except Exception:
            pdf.set_xy(x + 6, y + 16)
            pdf.set_font("Arial", "", 9)
            pdf.cell(0, 6, _safe_text(f"Unable to render image: {img_path}"), ln=True)

        slot = (slot + 1) % 2


def generate_scorecard(data: Any, output_path: Optional[str] = None) -> str:
    output_path = output_path or _DEFAULT_PDF
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if hasattr(data, "punch_log"):
        tracker = data
        punch_log = getattr(tracker, "punch_log", [])
        meta: Dict[str, Any] = (
            getattr(tracker, "metadata", {}) if hasattr(tracker, "metadata") else {}
        )
    else:
        tracker = None
        punch_log = data
        meta = {}

    rows = _extract_logs(punch_log)
    totals = _totals(rows)
    hand_totals = _hand_totals(rows)
    round_points, ten_totals, round_stats, kd, deductions, fouls = _extract_round_artifacts(tracker)

    attempts = (meta.get("attempts") or getattr(tracker, "attempts", {})) if tracker else {}
    accuracy = meta.get("accuracy", {})
    red_name = meta.get("red_name", "Red Corner")
    blue_name = meta.get("blue_name", "Blue Corner")
    lock_state = meta.get("corner_lock", {"RED": None, "BLUE": None})
    tracking_stats = meta.get("tracking_stats", {})
    fingerprint_status = meta.get("fingerprint_status", {})
    backend = meta.get("backend", "unknown")
    calibration = meta.get("calibration", {})
    scoring_mode = meta.get("scoring_mode", "analytics_only")
    no_official_score = int(meta.get("no_official_score", 0))
    no_official_reason = meta.get("no_official_score_reason", "")
    criteria_by_round = meta.get("criteria_by_round", {})
    manual_corrections_count = int(meta.get("manual_corrections_count", 0) or 0)
    manual_corrections_hash = str(meta.get("manual_corrections_hash", "")).strip()

    title = meta.get("title", "VAR Box Match Analysis")
    subtitle = meta.get("subtitle", "")

    pdf = ProScorecardPDF(title=title)
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()

    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 9, _safe_text(title), ln=True, align="C")
    pdf.set_font("Arial", "", 11)
    if subtitle:
        pdf.cell(0, 6, _safe_text(subtitle), ln=True, align="C")
    pdf.cell(
        0,
        6,
        _safe_text(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"),
        ln=True,
        align="C",
    )
    pdf.ln(4)

    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Executive Summary", ln=True)
    _write_key_value(pdf, "Red Corner", red_name)
    _write_key_value(pdf, "Blue Corner", blue_name)
    _write_key_value(pdf, "Detection Backend", backend)
    if isinstance(calibration, dict):
        calib_loaded = bool(int(calibration.get("loaded", 0) or 0))
        if calib_loaded:
            _write_key_value(
                pdf,
                "Calibration",
                (
                    f"loaded ({calibration.get('profile_name', '-')}) "
                    f"err={calibration.get('reprojection_error', '-')}"
                ),
            )
        else:
            _write_key_value(
                pdf, "Calibration", f"unverified ({calibration.get('status', 'no_profile')})"
            )
    _write_key_value(pdf, "Scoring Mode", scoring_mode)
    _write_key_value(pdf, "Manual Corrections", manual_corrections_count)
    if manual_corrections_hash:
        _write_key_value(pdf, "Correction Hash", manual_corrections_hash[:16])
    if no_official_score:
        _write_key_value(pdf, "Official Score", f"disabled ({no_official_reason})")
    _write_key_value(
        pdf, "Corner Lock", f"RED={lock_state.get('RED')}  BLUE={lock_state.get('BLUE')}"
    )
    if isinstance(tracking_stats, dict) and tracking_stats:
        red_reacq = tracking_stats.get("RED", {}).get("reacquired", 0)
        blue_reacq = tracking_stats.get("BLUE", {}).get("reacquired", 0)
        _write_key_value(pdf, "Re-Identified", f"RED={red_reacq}  BLUE={blue_reacq}")
    if isinstance(fingerprint_status, dict) and fingerprint_status:
        red_fp = fingerprint_status.get("RED", {})
        blue_fp = fingerprint_status.get("BLUE", {})
        _write_key_value(
            pdf,
            "Fingerprint Build",
            (
                f"RED frames={red_fp.get('frames', 0)} ready={red_fp.get('ready', 0)} "
                f"| BLUE frames={blue_fp.get('frames', 0)} ready={blue_fp.get('ready', 0)}"
            ),
        )
    _write_key_value(
        pdf, "Total Landed", f"RED={totals.get('RED', 0)}  BLUE={totals.get('BLUE', 0)}"
    )
    _write_key_value(
        pdf,
        "Attempts",
        f"RED={attempts.get('RED', 0)}  BLUE={attempts.get('BLUE', 0)}",
    )
    _write_key_value(
        pdf,
        "Accuracy",
        f"RED={accuracy.get('RED', 0)}%  BLUE={accuracy.get('BLUE', 0)}%",
    )
    _write_key_value(
        pdf,
        "Hand Breakdown",
        f"RED L/R={hand_totals['RED']['L']}/{hand_totals['RED']['R']}  "
        f"BLUE L/R={hand_totals['BLUE']['L']}/{hand_totals['BLUE']['R']}",
    )

    pdf.ln(4)
    _add_log_table(pdf, rows)
    _add_criteria_table(pdf, criteria_by_round)
    _add_round_tables(pdf, round_stats, round_points, kd, deductions, fouls, ten_totals)
    _add_evidence_pages(pdf, rows, max_items=80)

    footer = meta.get("footer")
    if footer:
        pdf.add_page()
        pdf.set_font("Arial", "I", 10)
        pdf.multi_cell(0, 6, _safe_text(footer))

    pdf.output(output_path)
    print(f"Scorecard saved to {output_path}")
    return output_path
