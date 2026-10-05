"""Decode an interaction clip around a runtime punch candidate."""

from __future__ import annotations

from typing import TypeAlias, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from boxing_analytics.training.interaction_crop import crop_and_pad

Frame: TypeAlias = NDArray[np.uint8]
Box = tuple[int, int, int, int]


def _union_box(
    attacker_box: Box,
    defender_box: Box,
    frame_width: int,
    frame_height: int,
    *,
    margin: float = 0.18,
) -> Box:
    x1 = min(attacker_box[0], defender_box[0])
    y1 = min(attacker_box[1], defender_box[1])
    x2 = max(attacker_box[2], defender_box[2])
    y2 = max(attacker_box[3], defender_box[3])
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    return (
        max(0, int(round(x1 - width * margin))),
        max(0, int(round(y1 - height * margin))),
        min(frame_width, int(round(x2 + width * margin))),
        min(frame_height, int(round(y2 + height * margin))),
    )


def decode_temporal_interaction_clip(
    *,
    video_path: str,
    event_time_s: float,
    duration_s: float,
    frame_count: int,
    image_size: int,
    attacker_box: Box,
    defender_box: Box,
    orientation_mode: str = "source",
    rotation_direction: str = "clockwise",
) -> tuple[list[Frame], float, float, Box]:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open temporal source video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if fps <= 0 or total_frames <= 0:
        capture.release()
        raise ValueError("Temporal source video has invalid frame metadata")
    video_duration_s = total_frames / fps
    start = max(0.0, event_time_s - duration_s * 0.5)
    end = min(video_duration_s, start + duration_s)
    start = max(0.0, end - duration_s)
    mode = orientation_mode.strip().lower()
    rotate = (mode == "portrait" and width > height) or (mode == "landscape" and height > width)
    rotation_code = (
        cv2.ROTATE_90_COUNTERCLOCKWISE
        if rotation_direction.strip().lower() == "counterclockwise"
        else cv2.ROTATE_90_CLOCKWISE
    )
    if rotate:
        width, height = height, width
    crop_box = _union_box(attacker_box, defender_box, width, height)
    times = np.linspace(start, end, num=frame_count, endpoint=True)
    requested_indices = [
        max(0, min(total_frames - 1, int(round(float(timestamp_s) * fps)))) for timestamp_s in times
    ]
    wanted = set(requested_indices)
    decoded_frames: dict[int, Frame] = {}
    first_requested = min(requested_indices)
    last_requested = max(requested_indices)
    capture.set(cv2.CAP_PROP_POS_FRAMES, first_requested)
    decoder_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES) or first_requested)
    while decoder_index <= last_requested:
        ok, raw_frame = capture.read()
        if not ok or raw_frame is None:
            break
        if decoder_index in wanted:
            frame = cast(Frame, raw_frame)
            if rotate:
                frame = cast(Frame, cv2.rotate(frame, rotation_code))
            decoded_frames[decoder_index] = frame
        decoder_index += 1
    frames: list[Frame] = []
    for timestamp_s, frame_index in zip(times, requested_indices, strict=True):
        decoded_frame: Frame | None = decoded_frames.get(frame_index)
        if decoded_frame is None:
            capture.release()
            raise ValueError(f"Could not decode temporal frame {frame_index} at {timestamp_s:.3f}s")
        frames.append(crop_and_pad(decoded_frame, crop_box, output_size=image_size))
    capture.release()
    return frames, start, end, crop_box
