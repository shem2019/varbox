"""Training-manifest serialization and reproducibility digests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ClipManifestRow:
    clip_id: str
    source_video: str
    source_group: str
    task_name: str
    camera: str
    original_label: str
    label_five: str
    label_eight: str
    hand: str
    split: str
    start_time_s: float
    end_time_s: float
    event_time_s: float
    source_fps: float
    source_frame_count: int
    annotation_box: tuple[float, float, float, float] | None
    derived_label: bool
    nearest_punch_distance_s: float | None
    padded_frames: int = 0

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        if self.annotation_box is not None:
            payload["annotation_box"] = list(self.annotation_box)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ClipManifestRow:
        box_raw = payload.get("annotation_box")
        box = None
        if isinstance(box_raw, list | tuple) and len(box_raw) == 4:
            box = (
                float(box_raw[0]),
                float(box_raw[1]),
                float(box_raw[2]),
                float(box_raw[3]),
            )
        return cls(
            clip_id=str(payload["clip_id"]),
            source_video=str(payload["source_video"]),
            source_group=str(payload["source_group"]),
            task_name=str(payload["task_name"]),
            camera=str(payload.get("camera", "unknown")),
            original_label=str(payload["original_label"]),
            label_five=str(payload["label_five"]),
            label_eight=str(payload["label_eight"]),
            hand=str(payload.get("hand", "UNKNOWN")),
            split=str(payload.get("split", "unassigned")),
            start_time_s=float(payload["start_time_s"]),
            end_time_s=float(payload["end_time_s"]),
            event_time_s=float(payload["event_time_s"]),
            source_fps=float(payload["source_fps"]),
            source_frame_count=int(payload["source_frame_count"]),
            annotation_box=box,
            derived_label=bool(payload.get("derived_label", False)),
            nearest_punch_distance_s=(
                None
                if payload.get("nearest_punch_distance_s") is None
                else float(payload["nearest_punch_distance_s"])
            ),
            padded_frames=int(payload.get("padded_frames", 0)),
        )


def write_manifest(rows: list[ClipManifestRow], output_path: str | Path) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), sort_keys=True, ensure_ascii=False) + "\n")
    return str(path)


def load_manifest(path: str | Path) -> list[ClipManifestRow]:
    rows: list[ClipManifestRow] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Manifest line {line_no} must be an object")
            rows.append(ClipManifestRow.from_dict(payload))
    return rows


def manifest_digest(rows: list[ClipManifestRow]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item.clip_id):
        payload = json.dumps(row.to_dict(), sort_keys=True, separators=(",", ":"))
        digest.update(payload.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
