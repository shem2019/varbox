"""PyTorch dataset backed by the deterministic clip manifest."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from torch.utils.data import Dataset

from boxing_analytics.training.clip_extractor import (
    decode_manifest_clip,
    decode_manifest_clip_cached,
)
from boxing_analytics.training.manifest import ClipManifestRow


def select_rows(
    rows: list[ClipManifestRow],
    *,
    split: str,
    max_samples: int | None,
    seed: int,
) -> list[ClipManifestRow]:
    selected = [row for row in rows if row.split == split]
    if max_samples is None or max_samples <= 0 or len(selected) <= max_samples:
        return selected
    by_label: dict[str, list[ClipManifestRow]] = defaultdict(list)
    for row in selected:
        by_label[row.label_five].append(row)
    rng = random.Random(seed)
    for values in by_label.values():
        rng.shuffle(values)
    output: list[ClipManifestRow] = []
    labels = sorted(by_label)
    while len(output) < max_samples and labels:
        next_labels: list[str] = []
        for label in labels:
            values = by_label[label]
            if values and len(output) < max_samples:
                output.append(values.pop())
            if values:
                next_labels.append(label)
        labels = next_labels
    return sorted(output, key=lambda row: row.clip_id)


class VideoMAEManifestDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        *,
        rows: list[ClipManifestRow],
        processor: Any,
        label2id: dict[str, int],
        frame_count: int,
        image_size: int,
        allow_padding: bool,
        cache_dir: str = "",
    ) -> None:
        self.rows = rows
        self.processor = processor
        self.label2id = label2id
        self.frame_count = frame_count
        self.image_size = image_size
        self.allow_padding = allow_padding
        self.cache_dir = cache_dir

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        row = self.rows[index]
        decoded = (
            decode_manifest_clip_cached(
                row,
                frame_count=self.frame_count,
                output_size=self.image_size,
                allow_padding=self.allow_padding,
                cache_dir=self.cache_dir,
            )
            if self.cache_dir
            else decode_manifest_clip(
                row,
                frame_count=self.frame_count,
                output_size=self.image_size,
                allow_padding=self.allow_padding,
            )
        )
        processed = self.processor(
            list(decoded.frames_rgb),
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
        )
        pixel_values = processed["pixel_values"].squeeze(0)
        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(self.label2id[row.label_five], dtype=torch.long),
            "clip_id": row.clip_id,
            "source_group": row.source_group,
        }
