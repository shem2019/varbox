"""Round segmentation from absolute timestamps."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BoutConfig:
    rounds_count: int = 12
    round_seconds: float = 180.0
    rest_seconds: float = 60.0
    warmup_seconds: float = 0.0

    def validate(self) -> None:
        if self.rounds_count <= 0:
            msg = "rounds_count must be positive"
            raise ValueError(msg)
        if self.round_seconds <= 0:
            msg = "round_seconds must be positive"
            raise ValueError(msg)
        if self.rest_seconds < 0:
            msg = "rest_seconds must be non-negative"
            raise ValueError(msg)
        if self.warmup_seconds < 0:
            msg = "warmup_seconds must be non-negative"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class RoundSegment:
    round_index: int
    start_s: float
    end_s: float

    def contains(self, timestamp_s: float) -> bool:
        return self.start_s <= timestamp_s < self.end_s


def build_round_segments(start_s: float, config: BoutConfig) -> list[RoundSegment]:
    """Create timestamp-based round windows from a configured bout."""
    config.validate()
    if start_s < 0:
        msg = "start_s must be non-negative"
        raise ValueError(msg)

    cursor = start_s + config.warmup_seconds
    segments: list[RoundSegment] = []

    for round_index in range(1, config.rounds_count + 1):
        segment = RoundSegment(
            round_index=round_index,
            start_s=cursor,
            end_s=cursor + config.round_seconds,
        )
        segments.append(segment)
        cursor = segment.end_s + config.rest_seconds

    return segments


def round_for_timestamp(timestamp_s: float, segments: list[RoundSegment]) -> int | None:
    for segment in segments:
        if segment.contains(timestamp_s):
            return segment.round_index
    return None


def ended_rounds_by_timestamp(
    timestamp_s: float,
    segments: list[RoundSegment],
    already_scored: set[int],
) -> list[int]:
    """
    Return round indexes whose end timestamp has passed and are not yet scored.
    """
    due = []
    for segment in segments:
        if segment.round_index in already_scored:
            continue
        if timestamp_s >= segment.end_s:
            due.append(segment.round_index)
    return due


def bout_end_timestamp(segments: list[RoundSegment]) -> float:
    if not segments:
        return 0.0
    return max(seg.end_s for seg in segments)
