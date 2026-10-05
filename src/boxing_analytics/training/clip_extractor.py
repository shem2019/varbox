"""Variable-FPS temporal decoding and cached evidence-clip creation."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from boxing_analytics.training.interaction_crop import crop_and_pad, expanded_box
from boxing_analytics.training.manifest import ClipManifestRow

Frame: TypeAlias = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class DecodedClip:
    frames_rgb: tuple[Frame, ...]
    timestamps_s: tuple[float, ...]
    source_frame_indices: tuple[int, ...]
    crop_box: tuple[int, int, int, int] | None
    duplicated_frames: int


def uniform_frame_times(start_time_s: float, end_time_s: float, frame_count: int) -> list[float]:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if end_time_s <= start_time_s:
        raise ValueError("end_time_s must be greater than start_time_s")
    if frame_count == 1:
        return [(start_time_s + end_time_s) * 0.5]
    return [
        start_time_s + (end_time_s - start_time_s) * index / float(frame_count - 1)
        for index in range(frame_count)
    ]


def decode_manifest_clip(
    row: ClipManifestRow,
    *,
    frame_count: int,
    output_size: int,
    allow_padding: bool = False,
    crop_margin: float = 1.5,
) -> DecodedClip:
    capture = cv2.VideoCapture(row.source_video)
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open source video: {row.source_video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or row.source_fps or 0.0)
    if fps <= 0:
        capture.release()
        raise ValueError(f"Source video has invalid FPS: {row.source_video}")
    crop_box = (
        expanded_box(row.annotation_box, width, height, margin=crop_margin)
        if row.annotation_box is not None
        else None
    )
    frames: list[Frame] = []
    timestamps: list[float] = []
    indices: list[int] = []
    duplicated = 0
    last_frame: Frame | None = None
    last_index: int | None = None
    prepared: Frame
    requested_times = uniform_frame_times(row.start_time_s, row.end_time_s, frame_count)
    requested_indices = [
        max(0, min(row.source_frame_count - 1, int(round(timestamp_s * fps))))
        for timestamp_s in requested_times
    ]
    wanted = set(requested_indices)
    decoded_frames: dict[int, Frame] = {}
    first_requested = min(requested_indices)
    last_requested = max(requested_indices)
    capture.set(cv2.CAP_PROP_POS_FRAMES, first_requested)
    decoder_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES) or first_requested)
    while decoder_index <= last_requested:
        ok, frame_bgr = capture.read()
        if not ok or frame_bgr is None:
            break
        if decoder_index in wanted:
            decoded_frames[decoder_index] = cast(Frame, frame_bgr)
        decoder_index += 1
    for timestamp_s, frame_index in zip(
        requested_times,
        requested_indices,
        strict=True,
    ):
        decoded_frame: Frame | None = decoded_frames.get(frame_index)
        if decoded_frame is None:
            if not allow_padding or last_frame is None:
                capture.release()
                raise ValueError(
                    f"Insufficient valid frames for {row.clip_id} at {timestamp_s:.3f}s"
                )
            prepared = last_frame.copy()
            duplicated += 1
            frame_index = int(last_index or 0)
        else:
            prepared = crop_and_pad(
                decoded_frame,
                crop_box,
                output_size=output_size,
            )
            prepared = cast(Frame, cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB))
            last_frame = prepared
            last_index = frame_index
        frames.append(prepared)
        timestamps.append(timestamp_s)
        indices.append(frame_index)
    capture.release()
    return DecodedClip(
        frames_rgb=tuple(frames),
        timestamps_s=tuple(timestamps),
        source_frame_indices=tuple(indices),
        crop_box=crop_box,
        duplicated_frames=duplicated,
    )


def decode_manifest_clip_cached(
    row: ClipManifestRow,
    *,
    frame_count: int,
    output_size: int,
    allow_padding: bool = False,
    crop_margin: float = 1.5,
    cache_dir: str | Path,
) -> DecodedClip:
    """Decode once and cache lossless frames plus crop/timing metadata."""
    source = Path(row.source_video)
    try:
        stat = source.stat()
    except OSError:
        return decode_manifest_clip(
            row,
            frame_count=frame_count,
            output_size=output_size,
            allow_padding=allow_padding,
            crop_margin=crop_margin,
        )
    signature_payload = {
        "clip": row.to_dict(),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "frame_count": frame_count,
        "output_size": output_size,
        "allow_padding": allow_padding,
        "crop_margin": crop_margin,
        "schema_version": 1,
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    root = Path(cache_dir).expanduser().resolve()
    cache_path = root / signature[:2] / f"{signature}.npz"
    if cache_path.is_file():
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                frames_array = cached["frames"]
                timestamps_array = cached["timestamps"]
                indices_array = cached["indices"]
                crop_array = cached["crop_box"]
                duplicated = int(cached["duplicated"][0])
            return DecodedClip(
                frames_rgb=tuple(cast(Frame, frame) for frame in frames_array),
                timestamps_s=tuple(float(value) for value in timestamps_array),
                source_frame_indices=tuple(int(value) for value in indices_array),
                crop_box=(
                    None
                    if crop_array.size == 0
                    else (
                        int(crop_array[0]),
                        int(crop_array[1]),
                        int(crop_array[2]),
                        int(crop_array[3]),
                    )
                ),
                duplicated_frames=duplicated,
            )
        except (KeyError, OSError, ValueError):
            pass
    decoded = decode_manifest_clip(
        row,
        frame_count=frame_count,
        output_size=output_size,
        allow_padding=allow_padding,
        crop_margin=crop_margin,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez_compressed(
        temporary_path,
        frames=np.stack(decoded.frames_rgb),
        timestamps=np.asarray(decoded.timestamps_s, dtype=np.float64),
        indices=np.asarray(decoded.source_frame_indices, dtype=np.int64),
        crop_box=np.asarray(decoded.crop_box or (), dtype=np.int32),
        duplicated=np.asarray([decoded.duplicated_frames], dtype=np.int32),
    )
    temporary_path.replace(cache_path)
    return decoded


def export_preview_clip(
    row: ClipManifestRow,
    output_path: str | Path,
    *,
    frame_count: int = 16,
    output_size: int = 224,
) -> str:
    decoded = decode_manifest_clip(
        row,
        frame_count=frame_count,
        output_size=output_size,
        allow_padding=False,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fps = frame_count / max(0.1, row.end_time_s - row.start_time_s)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore[attr-defined]
        fps,
        (output_size, output_size),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create preview clip: {path}")
    for frame_rgb in decoded.frames_rgb:
        writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    writer.release()
    return str(path)
