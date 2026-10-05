"""Video probing, windowed frame reading and audio extraction."""

from __future__ import annotations

import queue
import shutil
import subprocess
import threading
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

NDArray = np.ndarray[Any, Any]


@dataclass(frozen=True)
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["duration_s"] = self.duration_s
        return payload


def probe(path: str | Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    info = VideoInfo(
        path=str(path),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    cap.release()
    if info.fps <= 0 or info.frame_count <= 0:
        raise ValueError(f"invalid video metadata for {path}: {info}")
    return info


def iter_frames(
    path: str | Path, start: int, stop: int, step: int = 1
) -> Iterator[tuple[int, NDArray]]:
    """Yield (frame_index, BGR frame) for start <= index < stop, decoding sequentially."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    try:
        if start > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        index = start
        while index < stop:
            ok, frame = cap.read()
            if not ok:
                break
            if (index - start) % step == 0:
                yield index, frame
            index += 1
    finally:
        cap.release()


def prefetch(
    items: Iterator[Any], depth: int = 24, transform: Callable[[Any], Any] | None = None
) -> Iterator[Any]:
    """Run an iterator (and an optional per-item transform) in a background thread.

    Decoding video and preparing frames happen while the GPU works on earlier frames, so the
    model never waits on the video file. Exceptions surface in the consuming thread.
    """
    q: queue.Queue[Any] = queue.Queue(maxsize=depth)
    done = object()
    failure: list[BaseException] = []

    def work() -> None:
        try:
            for item in items:
                q.put(transform(item) if transform else item)
        except BaseException as exc:  # handed to the consumer below
            failure.append(exc)
        finally:
            q.put(done)

    threading.Thread(target=work, daemon=True).start()
    while True:
        item = q.get()
        if item is done:
            if failure:
                raise failure[0]
            return
        yield item


def read_frame(path: str | Path, index: int) -> NDArray:
    for _, frame in iter_frames(path, index, index + 1):
        return frame
    raise IndexError(f"frame {index} not readable in {path}")


def extract_audio(path: str | Path, sample_rate: int = 8000) -> NDArray | None:
    """Mono float32 audio via ffmpeg, or None when the file has no audio or ffmpeg is missing."""
    if shutil.which("ffmpeg") is None:
        return None
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0 or not proc.stdout:
        return None
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def transcode_h264(src: str | Path, dst: str | Path, crf: int = 20) -> bool:
    """Re-encode an OpenCV mp4v file to browser-playable H.264. Returns False without ffmpeg."""
    if shutil.which("ffmpeg") is None:
        return False
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-i",
        str(src),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(crf),
        "-movflags",
        "+faststart",
        str(dst),
    ]
    return subprocess.run(cmd, check=False).returncode == 0
