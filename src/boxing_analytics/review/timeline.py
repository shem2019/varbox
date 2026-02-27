"""Timeline event normalization/filtering for expert review UI."""

from __future__ import annotations


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def build_timeline_events(
    *,
    classified_events: list[dict[str, object]],
    confirmed_ref_events: list[dict[str, object]],
) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for row in classified_events:
        events.append(
            {
                "timestamp_s": round(_as_float(row.get("timestamp_s", 0.0)), 3),
                "round": _as_int(row.get("round", 0)),
                "role": str(row.get("role", "")).strip().upper(),
                "label": str(row.get("label", "")).strip().lower(),
                "target_zone": str(row.get("target_zone", "")).strip().lower() or "unknown",
                "confidence": round(_as_float(row.get("confidence", 0.0)), 3),
                "source": "model_strike",
                "invalidated": _as_int(row.get("invalidated_by_review", 0)),
                "corrected": _as_int(row.get("corrected_by_review", 0)),
            }
        )

    for row in confirmed_ref_events:
        events.append(
            {
                "timestamp_s": round(_as_float(row.get("timestamp_s", 0.0)), 3),
                "round": _as_int(row.get("round", 0)),
                "role": str(row.get("role", "")).strip().upper(),
                "label": str(row.get("event_type", "")).strip().lower(),
                "target_zone": "unknown",
                "confidence": 1.0,
                "source": "ref_confirmed",
                "invalidated": 0,
                "corrected": 0,
            }
        )

    events = sorted(
        events,
        key=lambda row: (_as_float(row.get("timestamp_s", 0.0)), str(row.get("source", ""))),
    )
    for idx, row in enumerate(events):
        row["event_index"] = idx
    return events


def filter_timeline_events(
    events: list[dict[str, object]],
    *,
    role_filter: str = "ALL",
    label_filter: str = "ALL",
    zone_filter: str = "ALL",
    round_filter: int = 0,
    include_invalidated: bool = False,
) -> list[dict[str, object]]:
    role_want = role_filter.strip().upper()
    label_want = label_filter.strip().lower()
    zone_want = zone_filter.strip().lower()

    out: list[dict[str, object]] = []
    for row in events:
        if not include_invalidated and _as_int(row.get("invalidated", 0)):
            continue
        if round_filter > 0 and _as_int(row.get("round", 0)) != round_filter:
            continue
        if role_want != "ALL" and str(row.get("role", "")).upper() != role_want:
            continue
        if label_want != "all" and str(row.get("label", "")).lower() != label_want:
            continue
        if zone_want != "all" and str(row.get("target_zone", "")).lower() != zone_want:
            continue
        out.append(dict(row))
    return out


def format_timeline_rows(events: list[dict[str, object]]) -> list[str]:
    lines: list[str] = []
    for row in events:
        lines.append(
            f"[{_as_int(row.get('event_index', -1), -1)}] "
            f"t={_as_float(row.get('timestamp_s', 0.0)):.2f}s "
            f"r={_as_int(row.get('round', 0), 0)} "
            f"{str(row.get('role', '-'))} "
            f"{str(row.get('label', '-'))} "
            f"zone={str(row.get('target_zone', '-'))} "
            f"conf={_as_float(row.get('confidence', 0.0)):.2f} "
            f"src={str(row.get('source', '-'))}"
        )
    return lines
