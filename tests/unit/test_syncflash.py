"""The sync test reads the flash code from video and recovers a known clock error."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from boxing_analytics.mesh4d.syncflash import (
    TICK_MS,
    Camera,
    Decoder,
    decode_video,
    gray,
    measure,
    render_code,
    ungray,
    unwrap,
)


def test_gray_code_round_trips() -> None:
    for n in [0, 1, 2, 3, 255, 4096, (1 << 24) - 1, 123456]:
        assert ungray(gray(n)) == n
    # Consecutive ticks differ in one bit, so a frame caught mid-change reads as one or the other.
    assert bin(gray(1000) ^ gray(1001)).count("1") == 1


def test_unwrap_picks_the_tick_nearest_the_stamp() -> None:
    full = 105_000_000_000  # a tick count around today's shared-clock time
    stamp = (full + 0.5) * TICK_MS + 40.0  # the stamp is 40 ms off; still the same tick
    assert unwrap(full % (1 << 24), stamp) == full


def test_decoder_reads_a_code_seen_at_an_angle() -> None:
    img = render_code(9_876_543, 640, 400)
    h, w = img.shape
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[60, 30], [w - 20, 0], [w - 50, h - 10], [10, h - 40]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, matrix, (w, h), borderValue=255)
    assert Decoder().decode(warped) == 9_876_543 % (1 << 24)
    assert Decoder().decode(np.full((400, 640), 255, np.uint8)) is None


def _film(path: Path, ticks: list[int]) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 400))
    for k in ticks:
        writer.write(cv2.cvtColor(render_code(k), cv2.COLOR_GRAY2BGR))
    writer.release()


def test_measure_recovers_an_injected_clock_error(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    base = 105_000_000_000
    ticks = [base + 2 * i for i in range(120)]
    # Each camera captures at a random moment inside the tick the screen shows.
    true_a = [(k + rng.uniform()) * TICK_MS for k in ticks]
    true_b = [(k + rng.uniform()) * TICK_MS for k in ticks]
    error_ms = 7.0  # camera B's clock runs 7 ms ahead
    _film(tmp_path / "a.mp4", ticks)
    _film(tmp_path / "b.mp4", ticks)
    src = "web-videoframe-capture"
    a = Camera("A", decode_video(tmp_path / "a.mp4"), true_a, 30.0, 2.0, src)
    b = Camera(
        "B", decode_video(tmp_path / "b.mp4"), [t + error_ms for t in true_b], 30.0, 2.0, src
    )

    assert sum(t is not None for t in a.ticks) == len(ticks)
    report = measure(a, b)
    assert report["ticks_compared"] == len(ticks)
    assert abs(report["median_ms"] - (-error_ms)) < 3.0
    assert report["passed"] is True

    # Three frames off fails.
    b_late = Camera("B", b.ticks, [t + 100.0 for t in true_b], 30.0, 2.0, src)
    assert measure(a, b_late)["passed"] is False
