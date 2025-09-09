# scorecard_generator.py
from typing import Any, Dict, Iterable, List, Optional
from fpdf import FPDF

# Optional default path from config; safely falls back
try:
    from config import SCORECARD_PDF as _DEFAULT_PDF
except Exception:
    _DEFAULT_PDF = "boxing_scorecard.pdf"

# ---------------- Text sanitation (FPDF is Latin-1 only by default) ----------------
_SAFE_MAP = {
    "\u2014": "-",   # em dash —
    "\u2013": "-",   # en dash –
    "\u2212": "-",   # minus sign −
    "\u2026": "...", # ellipsis …
    "\u2018": "'",   # left single ‘
    "\u2019": "'",   # right single ’
    "\u201C": '"',   # left double “
    "\u201D": '"',   # right double ”
    "\u2022": "*",   # bullet •
    "\u00D7": "x",   # multiply ×
    "\u00A0": " ",   # nbsp
    "\u200B": "",    # zero-width space
}

def _safe_text(x: Any) -> str:
    s = "" if x is None else str(x)
    for u, a in _SAFE_MAP.items():
        s = s.replace(u, a)
    # Strip anything still outside Latin-1
    return s.encode("latin-1", "ignore").decode("latin-1")

def _cell(pdf: FPDF, w, h, txt, **kw):
    pdf.cell(w, h, _safe_text(txt), **kw)

def _multi_cell(pdf: FPDF, w, h, txt, **kw):
    pdf.multi_cell(w, h, _safe_text(txt), **kw)

# ---------------- PDF helpers ----------------
def _add_table_header(pdf: FPDF, headers, widths, height=8):
    pdf.set_font("Arial", "B", 11)
    for h, w in zip(headers, widths):
        _cell(pdf, w, height, h, border=1, align="C")
    pdf.ln(height)
    pdf.set_font("Arial", "", 11)

def _cell_row(pdf: FPDF, row_vals, widths, height=8, align="L"):
    for v, w in zip(row_vals, widths):
        _cell(pdf, w, height, v, border=1, align=align)
    pdf.ln(height)

# ---------------- Punch log normalization ----------------
def _extract_logs(punch_log: Iterable[Any]) -> List[Dict]:
    """
    Supports:
      - dict entries: {'frame','time','role','hand','score_after',...}
      - legacy tuples: (fighter, time, score)
    Returns normalized list of dict rows.
    """
    rows: List[Dict] = []
    for item in (punch_log or []):
        if isinstance(item, dict):
            role = str(item.get("role", "")).upper()
            hand = item.get("hand", "ANY")
            hand = "ANY" if hand is None else str(hand).upper()
            if hand not in ("L", "R", "ANY"):
                hand = "ANY"
            rows.append({
                "frame": item.get("frame", ""),
                "time": item.get("time", ""),
                "role": role,
                "hand": hand,
                "score": item.get("score_after", ""),
            })
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            fighter, time_s, score = item[0], item[1], item[2]
            rows.append({
                "frame": "",
                "time": time_s,
                "role": str(fighter).upper(),
                "hand": "ANY",
                "score": score,
            })
    return rows

def _totals_from_rows(rows: List[Dict]) -> Dict[str, int]:
    totals = {"RED": 0, "BLUE": 0}
    for r in rows:
        role = r.get("role")
        if role in totals:
            totals[role] += 1
    return totals

def _hand_totals(rows: List[Dict]) -> Dict[str, Dict[str, int]]:
    res = {"RED": {"L": 0, "R": 0, "ANY": 0}, "BLUE": {"L": 0, "R": 0, "ANY": 0}}
    for r in rows:
        role = r.get("role")
        hand = r.get("hand", "ANY")
        if role in res:
            if hand in ("L", "R"):
                res[role][hand] += 1
            res[role]["ANY"] += 1
    return res

# ---------------- Round artifacts (optional) ----------------
def _extract_round_artifacts(tracker: Any):
    """
    Returns:
      round_points: {round -> (red_pts, blue_pts, rationale)}
      ten_totals: {"RED": int, "BLUE": int}
      round_stats: {round -> {"RED":{"landed":int}, "BLUE":{"landed":int}}} | None
      kd: {round -> {"RED": int, "BLUE": int}} | None
      deductions: {round -> {"RED": int, "BLUE": int}} | None
    """
    round_points = getattr(tracker, "round_points", {}) if tracker else {}
    ten_totals = getattr(tracker, "ten_point_totals", {"RED": 0, "BLUE": 0}) if tracker else {"RED": 0, "BLUE": 0}

    round_stats = getattr(tracker, "round_stats", None)
    kd = getattr(tracker, "kd", None)
    deductions = getattr(tracker, "deductions", None)

    meta = getattr(tracker, "metadata", {}) if tracker and hasattr(tracker, "metadata") else {}
    if round_stats is None:
        round_stats = meta.get("round_stats")
    if kd is None:
        kd = meta.get("kd")
    if deductions is None:
        deductions = meta.get("deductions")

    return round_points or {}, ten_totals or {"RED": 0, "BLUE": 0}, round_stats, kd, deductions

# ---------------- Public API ----------------
def generate_scorecard(data: Any, output_path: Optional[str] = None) -> str:
    """
    Use:
      generate_scorecard(tracker)              # preferred
      generate_scorecard(tracker.punch_log)    # legacy (no round summaries)
    """
    output_path = output_path or _DEFAULT_PDF

    # Detect tracker vs raw log
    if hasattr(data, "punch_log"):
        tracker = data
        punch_log = getattr(tracker, "punch_log", [])
        meta: Dict[str, Any] = getattr(tracker, "metadata", {}) if hasattr(tracker, "metadata") else {}
    else:
        tracker = None
        punch_log = data
        meta = {}

    rows = _extract_logs(punch_log)
    totals = _totals_from_rows(rows)
    hand_totals = _hand_totals(rows)
    round_points, ten_totals, round_stats, kd, deductions = _extract_round_artifacts(tracker)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()

    # ---- Title / Meta ----
    title = meta.get("title", "Boxing Match Scorecard")
    subtitle = meta.get("subtitle", "")
    pdf.set_font("Arial", "B", 16)
    _cell(pdf, 0, 10, title, ln=True, align="C")
    if subtitle:
        pdf.set_font("Arial", "", 12)
        _cell(pdf, 0, 8, subtitle, ln=True, align="C")
    pdf.ln(2)

    # ---- Summary totals ----
    pdf.set_font("Arial", "B", 13)
    _cell(pdf, 0, 8, "Summary", ln=True)
    pdf.set_font("Arial", "", 11)
    _cell(pdf, 0, 6, f"Total Landed (RED):  {totals.get('RED', 0)}", ln=True)
    _cell(pdf, 0, 6, f"Total Landed (BLUE): {totals.get('BLUE', 0)}", ln=True)

    # Hand breakdown (if present)
    if any(r.get("hand") in ("L", "R") for r in rows):
        pdf.ln(1)
        pdf.set_font("Arial", "I", 10)
        _cell(
            pdf, 0, 6,
            f"Hands - RED: L {hand_totals['RED']['L']} | R {hand_totals['RED']['R']}    "
            f"BLUE: L {hand_totals['BLUE']['L']} | R {hand_totals['BLUE']['R']}",
            ln=True
        )
    pdf.ln(2)

    # ---- Punch Log table ----
    pdf.set_font("Arial", "B", 14)
    _cell(pdf, 0, 8, "Punch Log", ln=True)
    pdf.set_font("Arial", "", 11)

    headers = ["Frame", "Time", "Role", "Hand", "Score After"]
    widths  = [22, 25, 25, 20, 28]
    total_w = sum(widths)
    page_w = pdf.w - 2 * pdf.l_margin
    scale = page_w / total_w if total_w > 0 else 1.0
    widths = [max(14, int(w * scale)) for w in widths]

    _add_table_header(pdf, headers, widths, height=8)
    for r in rows:
        _cell_row(pdf, [r["frame"], r["time"], r["role"], r["hand"], r["score"]], widths, height=8)

    # ---- Optional: Per-round landed punches ----
    if isinstance(round_stats, dict) and round_stats:
        pdf.ln(4)
        pdf.set_font("Arial", "B", 14)
        _cell(pdf, 0, 8, "Per-Round Landed Punches", ln=True)
        pdf.set_font("Arial", "", 11)

        headers = ["Round", "RED Landed", "BLUE Landed"]
        widths  = [25, 35, 35]
        total_w = sum(widths)
        page_w = pdf.w - 2 * pdf.l_margin
        scale = page_w / total_w if total_w > 0 else 1.0
        widths = [max(16, int(w * scale)) for w in widths]

        _add_table_header(pdf, headers, widths, height=8)
        for rnd in sorted(round_stats.keys()):
            rs = round_stats[rnd]
            r_l = rs.get("RED", {}).get("landed", 0)
            b_l = rs.get("BLUE", {}).get("landed", 0)
            _cell_row(pdf, [rnd, r_l, b_l], widths, height=8)

    # ---- 10-Point Must Round Summary ----
    if isinstance(round_points, dict) and round_points:
        pdf.ln(4)
        pdf.set_font("Arial", "B", 14)
        _cell(pdf, 0, 8, "Round Summary (10-Point Must)", ln=True)
        pdf.set_font("Arial", "", 11)

        headers = ["Round", "RED", "BLUE", "Rationale", "KD (R/B)", "Ded (R/B)"]
        widths  = [20, 18, 18, 0, 24, 26]  # rationale expands
        page_w = pdf.w - 2 * pdf.l_margin
        fixed = sum(w for w in widths if w > 0)
        rem = max(40, page_w - fixed)
        widths = [widths[0], widths[1], widths[2], rem, widths[4], widths[5]]

        _add_table_header(pdf, headers, widths, height=8)
        for rnd in sorted(round_points.keys()):
            red_pts, blue_pts, rationale = round_points[rnd]
            # Optional KD/deductions if present
            kd_r = kd.get(rnd, {}).get("RED", 0) if isinstance(kd, dict) else 0
            kd_b = kd.get(rnd, {}).get("BLUE", 0) if isinstance(kd, dict) else 0
            dd_r = deductions.get(rnd, {}).get("RED", 0) if isinstance(deductions, dict) else 0
            dd_b = deductions.get(rnd, {}).get("BLUE", 0) if isinstance(deductions, dict) else 0
            _cell_row(pdf, [rnd, red_pts, blue_pts, rationale, f"{kd_r}/{kd_b}", f"{dd_r}/{dd_b}"], widths, height=8)

        pdf.ln(2)
        pdf.set_font("Arial", "B", 12)
        _cell(pdf, 0, 8,
              f"Cumulative 10-Point Totals - RED: {ten_totals.get('RED', 0)}   "
              f"BLUE: {ten_totals.get('BLUE', 0)}",
              ln=True)

    # Footer meta (optional)
    footer = meta.get("footer")
    if footer:
        pdf.ln(2)
        pdf.set_font("Arial", "I", 9)
        _multi_cell(pdf, 0, 5, footer)

    pdf.output(output_path)
    print(f"✅ Scorecard saved to {output_path}")
    return output_path
