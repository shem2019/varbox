"""Timestamp-driven canonical timeline representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TimestampSource = Literal["pts", "monotonic_fallback"]


@dataclass(frozen=True, slots=True)
class TimelineMarker:
    """Discrete timeline marker created from UI or detector events."""

    timestamp_s: float
    label: str


@dataclass(frozen=True, slots=True)
class FrameTimestamp:
    """Resolved frame timestamp and its originating source."""

    frame_index: int
    timestamp_s: float
    source: TimestampSource


class CanonicalTimeline:
    """Monotonic timestamp container for frames and explicit markers."""

    def __init__(self) -> None:
        self._frame_timestamps: list[FrameTimestamp] = []
        self._markers: list[TimelineMarker] = []

    def add_frame_timestamp(self, timestamp: FrameTimestamp) -> None:
        """Add frame timestamp; enforces monotonic non-decreasing order."""
        if timestamp.timestamp_s < 0:
            msg = "timestamp must be non-negative"
            raise ValueError(msg)
        if (
            self._frame_timestamps
            and timestamp.timestamp_s < self._frame_timestamps[-1].timestamp_s
        ):
            msg = "frame timestamps must be monotonic"
            raise ValueError(msg)
        self._frame_timestamps.append(timestamp)

    def add_marker(self, marker: TimelineMarker) -> None:
        """Add timeline marker and keep sorted by timestamp."""
        if marker.timestamp_s < 0:
            msg = "marker timestamp must be non-negative"
            raise ValueError(msg)
        self._markers.append(marker)
        self._markers.sort(key=lambda item: item.timestamp_s)

    @property
    def frame_timestamps(self) -> list[float]:
        return [f.timestamp_s for f in self._frame_timestamps]

    @property
    def frame_timestamp_rows(self) -> list[FrameTimestamp]:
        return list(self._frame_timestamps)

    @property
    def markers(self) -> list[TimelineMarker]:
        return list(self._markers)


def resolve_frame_timestamp(
    *,
    frame_index: int,
    pts_seconds: float | None,
    fallback_fps: float,
    last_timestamp_s: float | None,
) -> FrameTimestamp:
    """
    Resolve a monotonic frame timestamp from decoder PTS when available,
    otherwise fall back to frame-index/fps.
    """
    source: TimestampSource
    if pts_seconds is not None and pts_seconds > 0:
        timestamp_s = float(pts_seconds)
        source = "pts"
    else:
        timestamp_s = float(frame_index) / max(float(fallback_fps), 1.0)
        source = "monotonic_fallback"

    if last_timestamp_s is not None and timestamp_s < last_timestamp_s:
        timestamp_s = last_timestamp_s
    return FrameTimestamp(frame_index=frame_index, timestamp_s=timestamp_s, source=source)
