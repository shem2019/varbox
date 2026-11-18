"""Sidecar timestamp alignment and its reconciliation with the audio offset."""

from __future__ import annotations

import json
from pathlib import Path

from boxing_analytics.mesh4d.sync import (
    SyncResult,
    frames_apart,
    load_sidecar,
    reconcile,
    sidecar_offset,
)


def _sidecar(start_ms: float, *, n: int = 30, fps: float = 30.0, unc: float = 2.0) -> dict:
    step = 1000.0 / fps
    return {
        "clock_uncertainty_ms": unc,
        "frames": [start_ms + i * step for i in range(n)],
    }


def test_sidecar_offset_is_the_gap_between_frame_zero_times() -> None:
    # B's frame zero lands 40 ms later on the shared clock, so an instant in A shows up 40 ms
    # earlier in B's own timeline: B = A - 0.040 s.
    result = sidecar_offset(_sidecar(1000.0), _sidecar(1040.0), fps=30.0)
    assert result.method == "sidecar-timestamps"
    assert abs(result.offset_s - (-0.040)) < 1e-9


def test_sidecar_offset_confidence_rises_as_clock_uncertainty_falls() -> None:
    tight = sidecar_offset(_sidecar(0.0, unc=1.0), _sidecar(0.0, unc=1.0), fps=30.0)
    loose = sidecar_offset(_sidecar(0.0, unc=20.0), _sidecar(0.0, unc=20.0), fps=30.0)
    assert tight.confidence > loose.confidence
    # A couple of milliseconds of uncertainty against a 33 ms frame is trustworthy.
    assert tight.confidence > 6


def test_sidecar_offset_without_frames_is_unusable() -> None:
    result = sidecar_offset({"frames": []}, _sidecar(0.0), fps=30.0)
    assert result.confidence == 0.0
    assert result.method == "sidecar-empty"


def test_reconcile_prefers_the_sidecar_and_flags_disagreement() -> None:
    sidecar = SyncResult(0.040, 10.0, "sidecar-timestamps")
    audio = SyncResult(0.200, 8.0, "audio-onset-xcorr")  # 0.16 s apart at 30 fps = 4.8 frames
    rec = reconcile(sidecar, audio, fps=30.0)
    assert rec.chosen is sidecar
    assert rec.frames_apart is not None and rec.frames_apart > 1.0
    assert rec.agree is False
    assert rec.to_dict()["method"] == "sidecar-timestamps"


def test_reconcile_falls_back_to_audio_without_a_sidecar() -> None:
    audio = SyncResult(0.2, 8.0, "whistle-xcorr")
    rec = reconcile(None, audio, fps=30.0)
    assert rec.chosen is audio
    assert rec.frames_apart is None
    assert rec.agree is True


def test_frames_apart_counts_offset_difference_in_frames() -> None:
    a = SyncResult(0.0, 1.0, "x")
    b = SyncResult(1.0 / 30.0, 1.0, "y")
    assert abs(frames_apart(a, b, 30.0) - 1.0) < 1e-9


def test_load_sidecar_reads_json_and_tolerates_missing(tmp_path: Path) -> None:
    p = tmp_path / "camera_A.sidecar.json"
    p.write_text(json.dumps(_sidecar(0.0)), encoding="utf-8")
    assert load_sidecar(p) is not None
    assert load_sidecar(tmp_path / "missing.json") is None
    assert load_sidecar(None) is None
