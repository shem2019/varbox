from pathlib import Path

import cv2
import numpy as np
import pytest

from boxing_analytics.training import clip_extractor
from boxing_analytics.training.clip_extractor import (
    decode_manifest_clip_cached,
    uniform_frame_times,
)
from boxing_analytics.training.manifest import ClipManifestRow


def test_uniform_frame_times_cover_configured_interval() -> None:
    times = uniform_frame_times(4.0, 4.6, 4)
    assert times == pytest.approx([4.0, 4.2, 4.4, 4.6])


def test_uniform_frame_times_reject_invalid_interval() -> None:
    with pytest.raises(ValueError):
        uniform_frame_times(2.0, 2.0, 16)


def test_clip_cache_reuses_lossless_frames_and_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(
        str(video),
        cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore[attr-defined]
        10.0,
        (64, 48),
    )
    for index in range(12):
        writer.write(np.full((48, 64, 3), index * 12, dtype=np.uint8))
    writer.release()
    row = ClipManifestRow(
        clip_id="clip",
        source_video=str(video),
        source_group="bout",
        task_name="task",
        camera="cam",
        original_label="AUTO_DERIVED_NO_PUNCH",
        label_five="no_punch",
        label_eight="no_punch",
        hand="UNKNOWN",
        split="train",
        start_time_s=0.1,
        end_time_s=0.7,
        event_time_s=0.4,
        source_fps=10.0,
        source_frame_count=12,
        annotation_box=None,
        derived_label=True,
        nearest_punch_distance_s=2.0,
    )
    first = decode_manifest_clip_cached(
        row,
        frame_count=4,
        output_size=32,
        cache_dir=tmp_path / "cache",
    )
    monkeypatch.setattr(
        clip_extractor,
        "decode_manifest_clip",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cache miss")),
    )
    second = decode_manifest_clip_cached(
        row,
        frame_count=4,
        output_size=32,
        cache_dir=tmp_path / "cache",
    )
    assert second.timestamps_s == first.timestamps_s
    assert second.source_frame_indices == first.source_frame_indices
    assert np.array_equal(np.stack(second.frames_rgb), np.stack(first.frames_rgb))
