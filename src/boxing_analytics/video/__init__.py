"""Video timebase, decoding, and round segmentation primitives."""

from boxing_analytics.video.progress import ProgressSnapshot, progress_message, progress_percent
from boxing_analytics.video.rounds import (
    BoutConfig,
    RoundSegment,
    bout_end_timestamp,
    build_round_segments,
    ended_rounds_by_timestamp,
    round_for_timestamp,
)
from boxing_analytics.video.timeline import (
    CanonicalTimeline,
    FrameTimestamp,
    TimelineMarker,
    resolve_frame_timestamp,
)

__all__ = [
    "BoutConfig",
    "FrameTimestamp",
    "CanonicalTimeline",
    "RoundSegment",
    "TimelineMarker",
    "bout_end_timestamp",
    "build_round_segments",
    "ended_rounds_by_timestamp",
    "ProgressSnapshot",
    "progress_message",
    "progress_percent",
    "resolve_frame_timestamp",
    "round_for_timestamp",
]
