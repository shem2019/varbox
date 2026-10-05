"""Time alignment between two cameras from their audio tracks (a clap, the bell, crowd noise)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

NDArray = np.ndarray[Any, Any]


@dataclass(frozen=True)
class SyncResult:
    offset_s: float  # time in B = time in A + offset_s
    confidence: float  # peak-to-sidelobe ratio, > ~6 is trustworthy
    method: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _onset_envelope(audio: NDArray, sample_rate: int, hop: int) -> NDArray:
    """Rectified energy flux; robust to the two phones having different gain."""
    n = audio.shape[0] // hop
    if n < 4:
        return np.zeros(0)
    frames = audio[: n * hop].reshape(n, hop)
    energy = np.log1p(100.0 * np.sqrt((frames**2).mean(axis=1)))
    flux = np.maximum(np.diff(energy, prepend=energy[0]), 0.0)
    flux -= flux.mean()
    std = flux.std()
    return flux / std if std > 0 else flux


def audio_offset(
    audio_a: NDArray,
    audio_b: NDArray,
    sample_rate: int,
    *,
    max_offset_s: float = 600.0,
    hop: int = 40,
) -> SyncResult:
    """Offset such that an event at time t in A appears at t + offset in B.

    Cross-correlates onset envelopes (hop/sample_rate resolution, 5 ms at 8 kHz and hop 40),
    then refines the peak with a parabolic fit.
    """
    env_a = _onset_envelope(audio_a, sample_rate, hop)
    env_b = _onset_envelope(audio_b, sample_rate, hop)
    if env_a.size == 0 or env_b.size == 0:
        return SyncResult(0.0, 0.0, "audio-empty")
    size = 1
    while size < env_a.size + env_b.size:
        size *= 2
    spec = np.fft.rfft(env_b, size) * np.conj(np.fft.rfft(env_a, size))
    xcorr = np.fft.irfft(spec, size)
    # Lags: index k means B is shifted by k envelope frames relative to A.
    lags = np.arange(size)
    lags[lags > size // 2] -= size
    env_rate = sample_rate / hop
    max_lag = int(max_offset_s * env_rate)
    allowed = np.abs(lags) <= max_lag
    scores = np.where(allowed, xcorr, -np.inf)
    k = int(np.argmax(scores))
    peak = xcorr[k]
    refined = float(lags[k])
    if 0 < k < size - 1:
        y0, y1, y2 = xcorr[k - 1], xcorr[k], xcorr[k + 1]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-12:
            refined += 0.5 * (y0 - y2) / denom
    exclude = np.abs(lags - lags[k]) > int(0.25 * env_rate)
    side = xcorr[allowed & exclude]
    sidelobe = float(np.std(side)) if side.size else 1.0
    confidence = float((peak - float(np.mean(side) if side.size else 0.0)) / max(sidelobe, 1e-9))
    return SyncResult(refined / env_rate, confidence, "audio-onset-xcorr")


def map_frame(frame_a: int, fps_a: float, fps_b: float, offset_s: float) -> int:
    """Frame index in B showing the same instant as frame_a in A."""
    return int(round((frame_a / fps_a + offset_s) * fps_b))
