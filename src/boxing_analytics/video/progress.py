"""Progress formatting utilities for long-running video processing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProgressSnapshot:
    frame_idx: int
    total_frames: int
    timestamp_s: float
    events_count: int


def progress_percent(frame_idx: int, total_frames: int) -> int | None:
    if total_frames <= 0:
        return None
    raw = int(round((max(0, frame_idx) / max(1, total_frames)) * 100.0))
    return max(0, min(100, raw))


def progress_message(snapshot: ProgressSnapshot) -> str:
    frame_idx = max(0, int(snapshot.frame_idx))
    total_frames = int(snapshot.total_frames)
    timestamp_s = max(0.0, float(snapshot.timestamp_s))
    events_count = max(0, int(snapshot.events_count))
    if total_frames > 0:
        pct = progress_percent(frame_idx, total_frames)
        pct_txt = f"{pct}%" if pct is not None else "n/a"
        return (
            f"Processing frame {frame_idx}/{total_frames} "
            f"({pct_txt}) | t={timestamp_s:.1f}s | landed_events={events_count}"
        )
    return f"Processing frame {frame_idx} | t={timestamp_s:.1f}s " f"| landed_events={events_count}"
