"""Sync test: how closely two phones' clock-stamped frames agree, measured with light, not sound.

A laptop shows `/sync-flash`, which draws the shared clock as a Gray-coded tick (one tick per
1/60 s) between four ArUco markers. Two phones film it with the web camera. Every decoded video
frame says which tick the screen showed; its sidecar says when that phone captured the frame on the
shared clock. Frames of both phones that caught the same tick were captured within one display
tick of each other, so the median difference of their stamps is the error between the two phones'
clocks, and arena noise plays no part in it.

    python -m boxing_analytics.mesh4d.syncflash \
        --a a.mp4 --a-sidecar a.json --b b.mp4 --b-sidecar b.json
    python -m boxing_analytics.mesh4d.syncflash \
        --server https://varbox.guestpassvms.com --take sync-20261119

The second form downloads the take's two cameras with the worker token (VARBOX_WORKER_TOKEN).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

NDArray = np.ndarray[Any, Any]

# The layout of web/public/assets/camera/sync-flash.js, in units of a 16 x 10 grid.
TICK_MS = 1000.0 / 60.0
BITS = 24
MARKER_AT = [(0.5, 0.5), (12.5, 0.5), (12.5, 6.5), (0.5, 6.5)]
MARKER_SIZE = 3.0
BIT_ORIGIN = (4.0, 3.5)
BIT_COLS = 8
BLACK_REF = (0.75, 2.0)
WHITE_REF = (2.0, 5.0)


def gray(n: int) -> int:
    return n ^ (n >> 1)


def ungray(g: int) -> int:
    n = g
    shift = 1
    while (g >> shift) > 0:
        n ^= g >> shift
        shift += 1
    return n


def _dictionary() -> Any:
    return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def render_code(tick: int, width: int = 640, height: int = 400) -> NDArray:
    """The flash page's image for one tick, as a grayscale array (used by the tests)."""
    img = np.full((height, width), 255, np.uint8)
    unit = min(width // 16, height // 10)
    x0 = (width - 16 * unit) // 2
    y0 = (height - 10 * unit) // 2

    def px(x: float, y: float) -> tuple[int, int]:
        return x0 + round(x * unit), y0 + round(y * unit)

    for marker_id, (mx, my) in enumerate(MARKER_AT):
        side = round(MARKER_SIZE * unit)
        marker = cv2.aruco.generateImageMarker(_dictionary(), marker_id, side)
        x, y = px(mx, my)
        img[y : y + side, x : x + side] = marker
    code = gray(tick % (1 << BITS))
    for i in range(BITS):
        if (code >> (BITS - 1 - i)) & 1:
            ax, ay = px(BIT_ORIGIN[0] + i % BIT_COLS, BIT_ORIGIN[1] + i // BIT_COLS)
            bx, by = px(BIT_ORIGIN[0] + i % BIT_COLS + 1, BIT_ORIGIN[1] + i // BIT_COLS + 1)
            img[ay:by, ax:bx] = 0
    return img


class Decoder:
    """Reads the tick from one camera frame, or None when the code is not in view."""

    def __init__(self) -> None:
        self.detector = cv2.aruco.ArucoDetector(_dictionary(), cv2.aruco.DetectorParameters())

    def decode(self, frame: NDArray) -> int | None:
        grey = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected: Any = self.detector.detectMarkers(grey)
        corners, ids = detected[0], detected[1]
        if ids is None:
            return None
        src: list[tuple[float, float]] = []
        dst: list[tuple[float, float]] = []
        for quad, marker_id in zip(corners, ids.flatten(), strict=True):
            if not 0 <= int(marker_id) < 4:
                continue
            mx, my = MARKER_AT[int(marker_id)]
            end_x, end_y = mx + MARKER_SIZE, my + MARKER_SIZE
            layout = [(mx, my), (end_x, my), (end_x, end_y), (mx, end_y)]
            src.extend(layout)
            dst.extend((float(p[0]), float(p[1])) for p in quad.reshape(4, 2))
        if len(src) < 8:
            return None
        found: Any = cv2.findHomography(np.array(src, np.float32), np.array(dst, np.float32))
        homography = found[0]
        if homography is None:
            return None

        def sample(x: float, y: float) -> float:
            steps = (-0.2, 0.0, 0.2)
            pts = np.array([[[x + dx, y + dy]] for dx in steps for dy in steps], np.float32)
            mapped = cv2.perspectiveTransform(pts, homography).reshape(-1, 2)
            h, w = grey.shape[:2]
            vals = [
                float(grey[int(round(py)), int(round(px))])
                for px, py in mapped
                if 0 <= round(px) < w and 0 <= round(py) < h
            ]
            return float(np.mean(vals)) if vals else math.nan

        black = sample(*BLACK_REF)
        white = sample(*WHITE_REF)
        if math.isnan(black) or math.isnan(white) or white - black < 40:
            return None
        threshold = (black + white) / 2
        code = 0
        for i in range(BITS):
            value = sample(BIT_ORIGIN[0] + i % BIT_COLS + 0.5, BIT_ORIGIN[1] + i // BIT_COLS + 0.5)
            if math.isnan(value):
                return None
            code = (code << 1) | (1 if value < threshold else 0)
        return ungray(code)


def decode_video(path: str | Path) -> list[int | None]:
    """The tick each frame of a video shows, in file order."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    decoder = Decoder()
    ticks: list[int | None] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ticks.append(decoder.decode(frame))
    cap.release()
    return ticks


def unwrap(tick_mod: int, stamped_ms: float) -> int:
    """The full tick count congruent to `tick_mod` nearest the frame's own stamp."""
    period = 1 << BITS
    approx = stamped_ms / TICK_MS
    k = round((approx - tick_mod) / period)
    return tick_mod + k * period


@dataclass(frozen=True)
class Camera:
    name: str
    ticks: list[int | None]
    stamps_ms: list[float]
    fps: float
    uncertainty_ms: float | None
    source: str


def _per_tick(cam: Camera) -> dict[int, float]:
    """First stamp of each full tick a camera saw, with misreads (far from the median) dropped."""
    n = min(len(cam.ticks), len(cam.stamps_ms))
    pairs = [
        (unwrap(t, cam.stamps_ms[i]), cam.stamps_ms[i])
        for i, t in enumerate(cam.ticks[:n])
        if t is not None
    ]
    if not pairs:
        return {}
    residual = np.array([s - (k + 0.5) * TICK_MS for k, s in pairs])
    med = float(np.median(residual))
    mad = float(np.median(np.abs(residual - med)))
    keep = np.abs(residual - med) <= 5 * mad + TICK_MS
    out: dict[int, float] = {}
    for (k, s), ok in zip(pairs, keep, strict=True):
        if ok and k not in out:
            out[k] = s
    return out


def measure(a: Camera, b: Camera) -> dict[str, Any]:
    """Compare two cameras over every display tick both caught."""
    ta = _per_tick(a)
    tb = _per_tick(b)
    common = sorted(set(ta) & set(tb))
    frame_ms = 1000.0 / max(a.fps, b.fps, 1.0)
    report: dict[str, Any] = {
        "frames_decoded": {c.name: sum(t is not None for t in c.ticks) for c in (a, b)},
        "frames_in_file": {a.name: len(a.ticks), b.name: len(b.ticks)},
        "frames_in_sidecar": {a.name: len(a.stamps_ms), b.name: len(b.stamps_ms)},
        "frame_times_source": {a.name: a.source, b.name: b.source},
        "clock_uncertainty_ms": {a.name: a.uncertainty_ms, b.name: b.uncertainty_ms},
        "ticks_compared": len(common),
        "frame_ms": frame_ms,
    }
    if len(common) < 10:
        report.update(passed=False, reason="Fewer than 10 display ticks were seen by both cameras.")
        return report
    diff = np.array([ta[k] - tb[k] for k in common])
    median = float(np.median(diff))
    p95 = float(np.percentile(np.abs(diff), 95))
    rng = np.random.default_rng(0)
    boot = [float(np.median(rng.choice(diff, diff.size))) for _ in range(500)]
    report.update(
        median_ms=median,
        median_frames=median / frame_ms,
        p95_abs_ms=p95,
        p95_abs_frames=p95 / frame_ms,
        median_ci95_ms=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        passed=abs(median) <= frame_ms and p95 <= 2 * frame_ms,
    )
    return report


def camera_from_files(name: str, video: str | Path, sidecar: str | Path | dict[str, Any]) -> Camera:
    data = sidecar if isinstance(sidecar, dict) else json.loads(Path(sidecar).read_text())
    return Camera(
        name=name,
        ticks=decode_video(video),
        stamps_ms=[float(x) for x in data.get("frames") or []],
        fps=float(data.get("fps") or 30),
        uncertainty_ms=data.get("clock_uncertainty_ms"),
        source=str(data.get("frame_times_source") or "unknown"),
    )


Downloaded = list[tuple[str, Path, dict[str, Any]]]


def _download(server: str, take: str, token: str, into: Path) -> Downloaded:
    def get(path: str) -> bytes:
        url = server.rstrip("/") + "/api/worker/" + path
        req = urllib.request.Request(url, headers={"Authorization": "Bearer " + token})
        with urllib.request.urlopen(req, timeout=300) as res:  # noqa: S310 (the VAR Box server)
            return bytes(res.read())

    info = json.loads(get("recordings/" + take))
    out = []
    for cam in info["cameras"]:
        if not cam.get("sidecar"):
            continue
        path = into / f"{take}_{cam['letter']}.mp4"
        path.write_bytes(get(f"uploads/{cam['upload_id']}"))
        out.append((str(cam["letter"]), path, cam["sidecar"]))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--a")
    ap.add_argument("--a-sidecar")
    ap.add_argument("--b")
    ap.add_argument("--b-sidecar")
    ap.add_argument("--server")
    ap.add_argument("--take")
    ap.add_argument("--out", default="sync_test.json")
    args = ap.parse_args(argv)

    if args.server and args.take:
        token = os.environ.get("VARBOX_WORKER_TOKEN", "")
        if not token:
            print("Set VARBOX_WORKER_TOKEN to the site's worker token.", file=sys.stderr)
            return 2
        tmp = Path(tempfile.mkdtemp(prefix="varbox-sync-"))
        cams = _download(args.server, args.take, token, tmp)
        if len(cams) < 2:
            found_n = len(cams)
            print(
                f"The take has {found_n} camera(s) with a sidecar; two are needed.", file=sys.stderr
            )
            return 2
        a = camera_from_files(cams[0][0], cams[0][1], cams[0][2])
        b = camera_from_files(cams[1][0], cams[1][1], cams[1][2])
    elif args.a and args.a_sidecar and args.b and args.b_sidecar:
        a = camera_from_files("A", args.a, args.a_sidecar)
        b = camera_from_files("B", args.b, args.b_sidecar)
    else:
        ap.error("give --a/--a-sidecar/--b/--b-sidecar, or --server and --take")

    report = measure(a, b)
    Path(args.out).write_text(json.dumps(report, indent=2))
    if "median_ms" in report:
        verdict = "PASS" if report["passed"] else "FAIL"
        print(
            f"{verdict}: cameras agree to {report['median_ms']:+.1f} ms median "
            f"({report['median_frames']:+.2f} frames), p95 {report['p95_abs_ms']:.1f} ms, "
            f"over {report['ticks_compared']} ticks. Stamps: {a.source} / {b.source}. "
            f"Written to {args.out}"
        )
    else:
        print(f"FAIL: {report['reason']} Written to {args.out}")
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    sys.exit(main())
