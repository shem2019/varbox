"""Evidence clip extraction helpers."""

from __future__ import annotations

import os
import re

import cv2


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return slug.strip("_") or "event"


def export_evidence_clip(
    input_video: str,
    timestamp_s: float,
    out_dir: str,
    tag: str,
    pre_s: float = 0.5,
    post_s: float = 0.5,
) -> str | None:
    if not os.path.isfile(input_video):
        return None

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        return None

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        return None

    start_s = max(0.0, float(timestamp_s) - max(0.0, pre_s))
    end_s = max(start_s, float(timestamp_s) + max(0.0, post_s))

    start_frame = int(start_s * fps)
    end_frame = int(end_s * fps)
    if end_frame <= start_frame:
        end_frame = start_frame + 1

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{_safe_slug(tag)}_{int(timestamp_s * 1000):08d}.mp4")
    fourcc = int(cv2.VideoWriter.fourcc(*"mp4v"))

    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames_written = 0
    frame_no = start_frame
    while frame_no <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        frames_written += 1
        frame_no += 1

    cap.release()
    writer.release()

    if frames_written == 0:
        return None
    return out_path
