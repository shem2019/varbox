"""Manual correction parsing and application for expert review flows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

VALID_ROLES = {"RED", "BLUE"}
VALID_LABELS = {"landed_clean", "landed_glancing", "blocked_guarded", "missed", "clinch"}
VALID_ZONES = {"head", "body", "unknown"}


def _normalize_role(value: object) -> str | None:
    role = str(value).strip().upper()
    if role in VALID_ROLES:
        return role
    return None


def _normalize_label(value: object) -> str | None:
    label = str(value).strip().lower()
    if label in VALID_LABELS:
        return label
    return None


def _normalize_zone(value: object) -> str | None:
    zone = str(value).strip().lower()
    if zone in VALID_ZONES:
        return zone
    return None


def _render_zone(zone: str) -> str:
    return zone.capitalize()


def _coerce_float(value: object, default: float = 0.0) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True, slots=True)
class ManualCorrection:
    timestamp_s: float
    match_role: str | None = None
    new_role: str | None = None
    new_label: str | None = None
    new_target_zone: str | None = None
    invalidate: bool = False
    note: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ManualCorrection:
        timestamp_s = float(payload.get("timestamp_s", 0.0) or 0.0)
        match_role = _normalize_role(payload.get("match_role", ""))
        new_role = _normalize_role(payload.get("new_role", ""))
        new_label = _normalize_label(payload.get("new_label", ""))
        new_target_zone = _normalize_zone(payload.get("new_target_zone", ""))
        invalidate = bool(int(payload.get("invalidate", 0) or 0))
        note = str(payload.get("note", "")).strip()

        if not invalidate and not (new_role or new_label or new_target_zone):
            raise ValueError("manual correction requires a modification or invalidate=1")
        return cls(
            timestamp_s=timestamp_s,
            match_role=match_role,
            new_role=new_role,
            new_label=new_label,
            new_target_zone=new_target_zone,
            invalidate=invalidate,
            note=note,
        )


def parse_manual_corrections(payload: str) -> list[ManualCorrection]:
    if not payload.strip():
        return []
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError:
        return []
    if not isinstance(raw, list):
        return []
    out: list[ManualCorrection] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            out.append(ManualCorrection.from_dict(item))
        except ValueError:
            continue
    return out


def _find_match_index(
    rows: list[dict[str, object]],
    *,
    timestamp_s: float,
    role: str | None,
    ts_key: str,
    role_key: str,
    window_s: float,
) -> int | None:
    best_idx: int | None = None
    best_delta = window_s + 1.0
    for idx, row in enumerate(rows):
        if bool(row.get("invalidated_by_review", 0)):
            continue
        row_role = _normalize_role(row.get(role_key, ""))
        if role is not None and row_role != role:
            continue
        row_ts = _coerce_float(row.get(ts_key, 0.0), 0.0)
        delta = abs(row_ts - timestamp_s)
        if delta <= window_s and delta < best_delta:
            best_idx = idx
            best_delta = delta
    return best_idx


def _apply_event_update(
    row: dict[str, object], correction: ManualCorrection, *, is_punch: bool
) -> None:
    if correction.new_role is not None:
        row["role"] = correction.new_role
        if is_punch and correction.new_role in VALID_ROLES:
            opponent = "BLUE" if correction.new_role == "RED" else "RED"
            row["opponent_role"] = opponent
    if correction.new_label is not None:
        key = "classification_label" if is_punch else "label"
        row[key] = correction.new_label
        if is_punch and correction.new_label not in {"landed_clean", "landed_glancing"}:
            row["evidence_clip"] = ""
            row["evidence_image"] = ""
    if correction.new_target_zone is not None:
        row["target_zone"] = _render_zone(correction.new_target_zone)
    if correction.invalidate:
        row["invalidated_by_review"] = 1
    if correction.note:
        row["review_note"] = correction.note
    row["corrected_by_review"] = 1


def apply_manual_corrections(
    *,
    classified_events: list[dict[str, object]],
    punch_log: list[dict[str, object]],
    corrections: list[ManualCorrection],
    match_window_s: float = 0.75,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    event_rows = [dict(row) for row in classified_events]
    punch_rows = [dict(row) for row in punch_log]
    applied: list[dict[str, object]] = []

    for idx, correction in enumerate(corrections):
        event_idx = _find_match_index(
            event_rows,
            timestamp_s=correction.timestamp_s,
            role=correction.match_role,
            ts_key="timestamp_s",
            role_key="role",
            window_s=match_window_s,
        )
        punch_idx = _find_match_index(
            punch_rows,
            timestamp_s=correction.timestamp_s,
            role=correction.match_role,
            ts_key="event_time_s",
            role_key="role",
            window_s=match_window_s,
        )

        if event_idx is not None:
            _apply_event_update(event_rows[event_idx], correction, is_punch=False)
        if punch_idx is not None:
            _apply_event_update(punch_rows[punch_idx], correction, is_punch=True)

        applied.append(
            {
                "correction_index": idx,
                "timestamp_s": round(correction.timestamp_s, 3),
                "match_role": correction.match_role or "",
                "event_match_index": event_idx if event_idx is not None else -1,
                "punch_match_index": punch_idx if punch_idx is not None else -1,
                "applied": int(event_idx is not None or punch_idx is not None),
                "invalidate": int(correction.invalidate),
                "new_role": correction.new_role or "",
                "new_label": correction.new_label or "",
                "new_target_zone": correction.new_target_zone or "",
                "note": correction.note,
            }
        )

    return event_rows, punch_rows, applied
