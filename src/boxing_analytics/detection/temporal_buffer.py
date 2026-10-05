"""Bounded temporal evidence buffer for delayed live-video classification."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

Frame: TypeAlias = NDArray[np.uint8]
Box = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class TemporalFrameSample:
    frame: Frame
    timestamp_s: float
    red_box: Box | None
    blue_box: Box | None
    fighter_identities: dict[str, int | None]
    pose_keypoints: dict[str, dict[str, Any]]
    identity_confidence: float
    ring_roi: tuple[tuple[int, int], ...] | None


class TemporalClipBuffer:
    """Keep recent evidence and release windows only after post-event frames arrive."""

    def __init__(self, *, retention_seconds: float = 2.0) -> None:
        if retention_seconds <= 0:
            raise ValueError("retention_seconds must be positive")
        self.retention_seconds = float(retention_seconds)
        self._samples: deque[TemporalFrameSample] = deque()

    def append(self, sample: TemporalFrameSample) -> None:
        if self._samples and sample.timestamp_s < self._samples[-1].timestamp_s:
            raise ValueError("Temporal samples must be appended in timestamp order")
        self._samples.append(
            replace(
                sample,
                frame=sample.frame.copy(),
                fighter_identities=dict(sample.fighter_identities),
                pose_keypoints={
                    role: dict(keypoints) for role, keypoints in sample.pose_keypoints.items()
                },
            )
        )
        cutoff = sample.timestamp_s - self.retention_seconds
        while self._samples and self._samples[0].timestamp_s < cutoff:
            self._samples.popleft()

    def is_ready(self, event_time_s: float, *, post_event_s: float) -> bool:
        return bool(
            self._samples and self._samples[-1].timestamp_s >= event_time_s + max(0.0, post_event_s)
        )

    def window(
        self,
        event_time_s: float,
        *,
        pre_event_s: float,
        post_event_s: float,
    ) -> tuple[TemporalFrameSample, ...]:
        start = event_time_s - max(0.0, pre_event_s)
        end = event_time_s + max(0.0, post_event_s)
        return tuple(sample for sample in self._samples if start <= sample.timestamp_s <= end)

    def __len__(self) -> int:
        return len(self._samples)
