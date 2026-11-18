"""Time alignment between two cameras.

Two cameras can be lined up two ways. When the phones recorded with the VAR Box Camera rig, each
leaves a sidecar of per-frame capture times on one shared clock, and the offset between them is
exact. Older footage, or a phone whose sidecar is missing, is lined up from the audio instead (a
clap, the bell, crowd noise). When both are available the pipeline uses the sidecar and keeps the
audio answer as a cross-check.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
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


def _tonal_track(audio: NDArray, sample_rate: int, hop: int, n: int = 1024) -> NDArray:
    """Whistle-ness over time: the strongest narrow peak between 1.8 and 5 kHz against the frame's
    median spectrum. Whistles stand out here while crowd noise and punches stay flat."""
    if audio.shape[0] < n + hop:
        return np.zeros(0)
    frames = np.lib.stride_tricks.sliding_window_view(audio, n)[::hop] * np.hanning(n)
    spectrum = np.abs(np.fft.rfft(frames, axis=1))
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    band = (freqs > 1800) & (freqs < 5000)
    ratio = np.log1p(spectrum[:, band].max(axis=1) / (np.median(spectrum, axis=1) + 1e-9))
    ratio = np.maximum(ratio - np.median(ratio), 0.0)
    std = ratio.std()
    return (ratio - ratio.mean()) / std if std > 0 else ratio


def tonal_offset(
    audio_a: NDArray,
    audio_b: NDArray,
    sample_rate: int,
    *,
    max_offset_s: float = 600.0,
    hop: int = 400,
) -> SyncResult:
    """Offset from whistles (or any sharp tone) heard by both cameras; B time = A time + offset."""
    env_a = _tonal_track(audio_a, sample_rate, hop)
    env_b = _tonal_track(audio_b, sample_rate, hop)
    if env_a.size == 0 or env_b.size == 0:
        return SyncResult(0.0, 0.0, "tonal-empty")
    size = 1
    while size < env_a.size + env_b.size:
        size *= 2
    xcorr = np.fft.irfft(np.fft.rfft(env_b, size) * np.conj(np.fft.rfft(env_a, size)), size)
    lags = np.arange(size)
    lags[lags > size // 2] -= size
    rate = sample_rate / hop
    allowed = np.abs(lags) <= int(max_offset_s * rate)
    k = int(np.argmax(np.where(allowed, xcorr, -np.inf)))
    side = xcorr[allowed & (np.abs(lags - lags[k]) > int(rate))]
    confidence = (
        float((xcorr[k] - side.mean()) / max(float(side.std()), 1e-9)) if side.size else 0.0
    )
    return SyncResult(float(lags[k]) / rate, confidence, "whistle-xcorr")


def best_offset(audio_a: NDArray, audio_b: NDArray, sample_rate: int) -> SyncResult:
    """Clap, planks or whistle: run both detectors and keep the more confident answer."""
    candidates = [
        audio_offset(audio_a, audio_b, sample_rate),
        tonal_offset(audio_a, audio_b, sample_rate),
    ]
    return max(candidates, key=lambda r: r.confidence)


def map_frame(frame_a: int, fps_a: float, fps_b: float, offset_s: float) -> int:
    """Frame index in B showing the same instant as frame_a in A."""
    return int(round((frame_a / fps_a + offset_s) * fps_b))


def load_sidecar(path: str | Path | None) -> dict[str, Any] | None:
    """Read a camera's sidecar JSON, or None when it is missing or unreadable."""
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _frame_times_ms(sidecar: dict[str, Any]) -> list[float]:
    """Per-frame capture times (shared-clock milliseconds) from a sidecar."""
    frames = sidecar.get("frames")
    if not isinstance(frames, list):
        return []
    return [float(t) for t in frames]


def _video_start_ms(sidecar: dict[str, Any]) -> float | None:
    """Shared-clock time of the first frame in the video file.

    The app records this explicitly as `video_first_frame_ms`, since the frame-time series can
    start a little before or after the file's first frame depending on how it was captured. Falls
    back to the first frame time when the field is absent.
    """
    value = sidecar.get("video_first_frame_ms")
    if isinstance(value, (int | float)) and value:
        return float(value)
    frames = _frame_times_ms(sidecar)
    return frames[0] if frames else None


def sidecar_offset(
    sidecar_a: dict[str, Any], sidecar_b: dict[str, Any], *, fps: float = 30.0
) -> SyncResult:
    """Offset from the sidecars' shared-clock frame times: time in B = time in A + offset.

    The sidecars time each camera's first file frame on one shared clock (SNTP), so that instant is
    absolute; the offset is the gap between the two. Confidence is how small the combined clock
    uncertainty is next to a frame — above ~6 the two frame clocks agree inside half a frame.
    """
    start_a = _video_start_ms(sidecar_a)
    start_b = _video_start_ms(sidecar_b)
    if start_a is None or start_b is None:
        return SyncResult(0.0, 0.0, "sidecar-empty")
    offset_s = (start_a - start_b) / 1000.0
    unc_a = max(float(sidecar_a.get("clock_uncertainty_ms", 2.0)), 1e-3)
    unc_b = max(float(sidecar_b.get("clock_uncertainty_ms", 2.0)), 1e-3)
    combined_ms = math.hypot(unc_a, unc_b)
    frame_ms = 1000.0 / max(fps, 1.0)
    confidence = min(1000.0, frame_ms / combined_ms)
    return SyncResult(offset_s, confidence, "sidecar-timestamps")


def frames_apart(a: SyncResult, b: SyncResult, fps: float) -> float:
    """How many frames separate two offset estimates, for flagging disagreement."""
    return abs(a.offset_s - b.offset_s) * max(fps, 1.0)


@dataclass(frozen=True)
class Reconciliation:
    """The offset the pipeline will use, with both estimates kept for the log and sync.json."""

    chosen: SyncResult
    sidecar: SyncResult | None
    audio: SyncResult | None
    frames_apart: float | None
    agree: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.chosen.to_dict(),
            "sidecar": self.sidecar.to_dict() if self.sidecar else None,
            "audio": self.audio.to_dict() if self.audio else None,
            "frames_apart": self.frames_apart,
            "agree": self.agree,
        }


def reconcile(
    sidecar: SyncResult | None, audio: SyncResult | None, *, fps: float
) -> Reconciliation:
    """Pick the offset to use and compare the two methods.

    The sidecar wins when it has real timestamps and the audio agrees with it. But the audio offset
    is measured on the video files themselves, so when the two disagree by more than one frame the
    audio is used instead and `agree` is false, leaving a flag for the caller to warn. With only one
    method available that method is used.
    """
    use_sidecar = sidecar is not None and sidecar.confidence > 0
    apart = frames_apart(sidecar, audio, fps) if sidecar is not None and audio is not None else None
    agree = apart is None or apart <= 1.0
    chosen: SyncResult | None
    if use_sidecar and (audio is None or agree):
        chosen = sidecar
    elif audio is not None:
        chosen = audio
    else:
        chosen = sidecar
    if chosen is None:
        chosen = SyncResult(0.0, 0.0, "none")
    return Reconciliation(chosen, sidecar, audio, apart, agree)
