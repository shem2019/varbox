"""Time-based deduplication for attempts and contact events."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(slots=True)
class _EventState:
    timestamp_s: float
    position: tuple[int, int] | None
    label: str


class EventDeduplicator:
    """Deduplicate events by elapsed time and trajectory movement."""

    def __init__(
        self,
        attempt_window_s: float = 0.24,
        contact_window_s: float = 0.34,
        min_travel_px: float = 18.0,
    ) -> None:
        self.attempt_window_s = attempt_window_s
        self.contact_window_s = contact_window_s
        self.min_travel_px = min_travel_px
        self._attempt_state: dict[tuple[str, str], _EventState] = {}
        self._contact_state: dict[tuple[str, str], _EventState] = {}

    @staticmethod
    def _travel(a: tuple[int, int] | None, b: tuple[int, int] | None) -> float:
        if a is None or b is None:
            return 0.0
        return math.hypot(float(a[0] - b[0]), float(a[1] - b[1]))

    def allow_attempt(
        self,
        role: str,
        hand: str,
        timestamp_s: float,
        position: tuple[int, int] | None,
    ) -> bool:
        key = (role.upper(), hand.upper())
        prev = self._attempt_state.get(key)
        if prev is None:
            self._attempt_state[key] = _EventState(
                timestamp_s=timestamp_s,
                position=position,
                label="attempt",
            )
            return True

        elapsed = timestamp_s - prev.timestamp_s
        travel = self._travel(position, prev.position)
        accept = elapsed >= self.attempt_window_s or travel >= self.min_travel_px
        if accept:
            self._attempt_state[key] = _EventState(
                timestamp_s=timestamp_s,
                position=position,
                label="attempt",
            )
        return accept

    def allow_contact(
        self,
        role: str,
        hand: str,
        label: str,
        timestamp_s: float,
        position: tuple[int, int] | None,
    ) -> bool:
        key = (role.upper(), hand.upper())
        prev = self._contact_state.get(key)
        if prev is None:
            self._contact_state[key] = _EventState(
                timestamp_s=timestamp_s,
                position=position,
                label=label,
            )
            return True

        elapsed = timestamp_s - prev.timestamp_s
        travel = self._travel(position, prev.position)
        # Allow quicker repeats if classification changed.
        switched_label = label != prev.label
        accept = elapsed >= self.contact_window_s or travel >= self.min_travel_px or switched_label
        if accept:
            self._contact_state[key] = _EventState(
                timestamp_s=timestamp_s,
                position=position,
                label=label,
            )
        return accept
