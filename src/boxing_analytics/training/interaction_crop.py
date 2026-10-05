"""Deterministic interaction-crop helpers."""

from __future__ import annotations

from typing import TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray

Frame: TypeAlias = NDArray[np.uint8]
Box = tuple[float, float, float, float]


def expanded_box(
    box: Box,
    frame_width: int,
    frame_height: int,
    *,
    margin: float = 1.5,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    width = max(32.0, (x2 - x1) * (1.0 + 2.0 * margin))
    height = max(32.0, (y2 - y1) * (1.0 + 2.0 * margin))
    return (
        max(0, int(round(cx - width * 0.5))),
        max(0, int(round(cy - height * 0.5))),
        min(frame_width, int(round(cx + width * 0.5))),
        min(frame_height, int(round(cy + height * 0.5))),
    )


def crop_and_pad(
    frame: Frame,
    box: tuple[int, int, int, int] | None,
    *,
    output_size: int,
) -> Frame:
    height, width = frame.shape[:2]
    if box is None:
        crop = frame
    else:
        x1, y1, x2, y2 = box
        crop = frame[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)]
        if crop.size == 0:
            crop = frame
    crop_h, crop_w = crop.shape[:2]
    scale = min(output_size / max(1, crop_w), output_size / max(1, crop_h))
    resized_w = max(1, int(round(crop_w * scale)))
    resized_h = max(1, int(round(crop_h * scale)))
    resized = cv2.resize(crop, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((output_size, output_size, 3), dtype=np.uint8)
    x_off = (output_size - resized_w) // 2
    y_off = (output_size - resized_h) // 2
    canvas[y_off : y_off + resized_h, x_off : x_off + resized_w] = resized
    return canvas
