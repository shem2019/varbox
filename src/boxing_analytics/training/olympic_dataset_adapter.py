"""Adapter for the real CVAT JSON layout in the Olympic boxing dataset."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2

from boxing_analytics.training.annotation_schema import (
    ORIGINAL_LABEL_MAPPING,
    AnnotationBox,
    OlympicPunchAnnotation,
)
from boxing_analytics.training.manifest import ClipManifestRow
from boxing_analytics.training.split_dataset import grouped_split


@dataclass(frozen=True, slots=True)
class SourceVideo:
    task_name: str
    path: str
    source_group: str
    camera: str
    fps: float
    frame_count: int
    width: int
    height: int
    negative_sampling_safe: bool = True

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0


def _dataset_path(dataset_dir: str | Path) -> Path:
    path = Path(dataset_dir).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(
            f"Olympic boxing dataset not found at {path}. "
            "Pass --dataset-dir or set VARBOX_OLYMPIC_DATASET_DIR."
        )
    return path


def _group_and_camera(task_name: str, video_stem: str) -> tuple[str, str]:
    camera_match = re.search(r"task_(kam\d+)_", task_name.lower())
    camera = camera_match.group(1) if camera_match else "unknown"
    segment_match = re.match(r"GH(\d{2})", video_stem.upper())
    segment = segment_match.group(1) if segment_match else video_stem
    return f"bout_segment_{segment}", camera


def discover_sources(dataset_dir: str | Path) -> tuple[list[SourceVideo], list[dict[str, str]]]:
    root = _dataset_path(dataset_dir)
    sources: list[SourceVideo] = []
    invalid: list[dict[str, str]] = []
    for task_dir in sorted(root.glob("task_*")):
        annotations_path = task_dir / "annotations.json"
        videos = sorted((task_dir / "data").glob("*.mp4"))
        if not annotations_path.is_file():
            invalid.append({"task": task_dir.name, "reason": "missing_annotations"})
            continue
        if len(videos) != 1:
            invalid.append(
                {"task": task_dir.name, "reason": f"expected_one_video_found_{len(videos)}"}
            )
            continue
        video = videos[0]
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            invalid.append({"task": task_dir.name, "reason": "unreadable_video"})
            continue
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        capture.release()
        if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
            invalid.append({"task": task_dir.name, "reason": "invalid_video_metadata"})
            continue
        source_group, camera = _group_and_camera(task_dir.name, video.stem)
        sources.append(
            SourceVideo(
                task_name=task_dir.name,
                path=str(video.resolve()),
                source_group=source_group,
                camera=camera,
                fps=fps,
                frame_count=frame_count,
                width=width,
                height=height,
            )
        )
    return sources, invalid


def _load_cvat_item(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list) and len(payload) == 1 and isinstance(payload[0], dict):
        return payload[0]
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"{path} must contain a CVAT object or one-element object array")


def _parse_box(shape: dict[str, Any]) -> AnnotationBox | None:
    points = shape.get("points")
    if not isinstance(points, list) or len(points) != 4:
        return None
    try:
        point_tuple = (
            float(points[0]),
            float(points[1]),
            float(points[2]),
            float(points[3]),
        )
        return AnnotationBox(
            frame_index=int(shape["frame"]),
            points=point_tuple,
            outside=bool(shape.get("outside", False)),
            occluded=bool(shape.get("occluded", False)),
        )
    except (KeyError, TypeError, ValueError):
        return None


def load_annotations(
    dataset_dir: str | Path,
) -> tuple[list[OlympicPunchAnnotation], list[SourceVideo], list[dict[str, object]]]:
    sources, discovery_invalid = discover_sources(dataset_dir)
    invalid: list[dict[str, object]] = [dict(row) for row in discovery_invalid]
    annotations: list[OlympicPunchAnnotation] = []
    unsafe_negative_tasks: set[str] = set()
    seen_annotations: set[tuple[object, ...]] = set()
    for source in sources:
        annotations_path = Path(source.path).parent.parent / "annotations.json"
        try:
            item = _load_cvat_item(annotations_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            invalid.append({"task": source.task_name, "reason": f"invalid_json:{exc}"})
            unsafe_negative_tasks.add(source.task_name)
            continue
        tracks = item.get("tracks", [])
        if not isinstance(tracks, list):
            invalid.append({"task": source.task_name, "reason": "tracks_not_array"})
            unsafe_negative_tasks.add(source.task_name)
            continue
        for track_index, track_raw in enumerate(tracks):
            if not isinstance(track_raw, dict):
                unsafe_negative_tasks.add(source.task_name)
                invalid.append(
                    {
                        "task": source.task_name,
                        "track_index": track_index,
                        "reason": "track_not_object",
                    }
                )
                continue
            original_label = str(track_raw.get("label", "")).strip()
            mapping = ORIGINAL_LABEL_MAPPING.get(original_label)
            if mapping is None:
                unsafe_negative_tasks.add(source.task_name)
                invalid.append(
                    {
                        "task": source.task_name,
                        "track_index": track_index,
                        "reason": "unknown_label",
                        "label": original_label,
                    }
                )
                continue
            shapes_raw = track_raw.get("shapes", [])
            boxes = tuple(
                box
                for shape in shapes_raw
                if isinstance(shape, dict) and (box := _parse_box(shape)) is not None
            )
            visible = tuple(box for box in boxes if not box.outside)
            if not visible:
                unsafe_negative_tasks.add(source.task_name)
                invalid.append(
                    {
                        "task": source.task_name,
                        "track_index": track_index,
                        "reason": "no_visible_shapes",
                    }
                )
                continue
            start_frame = min(box.frame_index for box in visible)
            end_frame = max(box.frame_index for box in visible)
            if start_frame < 0 or end_frame >= source.frame_count:
                unsafe_negative_tasks.add(source.task_name)
                invalid.append(
                    {
                        "task": source.task_name,
                        "track_index": track_index,
                        "reason": "out_of_range_frames",
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                    }
                )
                continue
            peak_frame = visible[len(visible) // 2].frame_index
            five_label, eight_label, hand = mapping
            duplicate_key = (
                source.path,
                original_label,
                start_frame,
                end_frame,
                tuple((box.frame_index, box.points, box.outside) for box in boxes),
            )
            if duplicate_key in seen_annotations:
                unsafe_negative_tasks.add(source.task_name)
                invalid.append(
                    {
                        "task": source.task_name,
                        "track_index": track_index,
                        "reason": "duplicate_annotation",
                    }
                )
                continue
            seen_annotations.add(duplicate_key)
            annotation_id = f"{source.task_name}:{track_index:06d}"
            annotations.append(
                OlympicPunchAnnotation(
                    annotation_id=annotation_id,
                    task_name=source.task_name,
                    source_video=source.path,
                    source_group=source.source_group,
                    camera=source.camera,
                    original_label=original_label,
                    five_class_label=five_label,
                    eight_class_label=eight_label,
                    hand=hand,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    peak_frame=peak_frame,
                    start_time_s=start_frame / source.fps,
                    end_time_s=end_frame / source.fps,
                    peak_time_s=peak_frame / source.fps,
                    boxes=boxes,
                )
            )
    sources = [
        replace(source, negative_sampling_safe=source.task_name not in unsafe_negative_tasks)
        for source in sources
    ]
    return annotations, sources, invalid


def _representative_box(
    annotation: OlympicPunchAnnotation,
) -> tuple[float, float, float, float] | None:
    visible = [box for box in annotation.boxes if not box.outside]
    if not visible:
        return None
    selected = min(visible, key=lambda box: abs(box.frame_index - annotation.peak_frame))
    return selected.points


def _negative_centers(
    source: SourceVideo,
    events: list[OlympicPunchAnnotation],
    *,
    safety_margin_s: float,
    clip_duration_s: float,
    limit: int,
    seed: int,
) -> list[tuple[float, float]]:
    if limit <= 0 or source.duration_s <= clip_duration_s:
        return []
    forbidden: list[tuple[float, float]] = []
    for event in events:
        forbidden.append(
            (
                max(0.0, event.start_time_s - safety_margin_s - clip_duration_s * 0.5),
                min(
                    source.duration_s,
                    event.end_time_s + safety_margin_s + clip_duration_s * 0.5,
                ),
            )
        )
    candidates: list[tuple[float, float]] = []
    step = max(clip_duration_s, source.duration_s / max(8, limit * 5))
    cursor = clip_duration_s * 0.5
    while cursor <= source.duration_s - clip_duration_s * 0.5:
        if not any(start <= cursor <= end for start, end in forbidden):
            nearest = min(
                (
                    min(abs(cursor - event.start_time_s), abs(cursor - event.end_time_s))
                    for event in events
                ),
                default=source.duration_s,
            )
            candidates.append((cursor, nearest))
        cursor += step
    stable_seed = int.from_bytes(
        hashlib.sha256(f"{seed}:{source.task_name}".encode()).digest()[:8], "big"
    )
    rng = random.Random(stable_seed)
    rng.shuffle(candidates)
    selected = sorted(candidates[:limit], key=lambda row: row[0])
    return selected


def build_manifest_rows(
    annotations: list[OlympicPunchAnnotation],
    sources: list[SourceVideo],
    *,
    clip_duration_s: float = 0.60,
    safety_margin_s: float = 1.0,
    negatives_per_video: int = 24,
    seed: int = 42,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> list[ClipManifestRow]:
    if clip_duration_s <= 0:
        raise ValueError("clip_duration_s must be positive")
    by_task: dict[str, list[OlympicPunchAnnotation]] = defaultdict(list)
    for annotation in annotations:
        by_task[annotation.task_name].append(annotation)
    assignments = grouped_split(
        [source.source_group for source in sources],
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    source_map = {source.task_name: source for source in sources}
    rows: list[ClipManifestRow] = []
    half = clip_duration_s * 0.5
    for annotation in annotations:
        source = source_map[annotation.task_name]
        start = max(0.0, annotation.peak_time_s - half)
        end = min(source.duration_s, start + clip_duration_s)
        start = max(0.0, end - clip_duration_s)
        rows.append(
            ClipManifestRow(
                clip_id=annotation.annotation_id.replace(":", "_"),
                source_video=annotation.source_video,
                source_group=annotation.source_group,
                task_name=annotation.task_name,
                camera=annotation.camera,
                original_label=annotation.original_label,
                label_five=annotation.five_class_label,
                label_eight=annotation.eight_class_label,
                hand=annotation.hand,
                split=assignments[annotation.source_group],
                start_time_s=round(start, 6),
                end_time_s=round(end, 6),
                event_time_s=round(annotation.peak_time_s, 6),
                source_fps=source.fps,
                source_frame_count=source.frame_count,
                annotation_box=_representative_box(annotation),
                derived_label=False,
                nearest_punch_distance_s=None,
            )
        )
    for source in sources:
        if not source.negative_sampling_safe or not by_task[source.task_name]:
            continue
        negatives = _negative_centers(
            source,
            by_task[source.task_name],
            safety_margin_s=max(1.0, safety_margin_s),
            clip_duration_s=clip_duration_s,
            limit=negatives_per_video,
            seed=seed,
        )
        for index, (center, nearest) in enumerate(negatives):
            start = max(0.0, center - half)
            end = min(source.duration_s, start + clip_duration_s)
            start = max(0.0, end - clip_duration_s)
            rows.append(
                ClipManifestRow(
                    clip_id=f"{source.task_name}_negative_{index:04d}",
                    source_video=source.path,
                    source_group=source.source_group,
                    task_name=source.task_name,
                    camera=source.camera,
                    original_label="AUTO_DERIVED_NO_PUNCH",
                    label_five="no_punch",
                    label_eight="no_punch",
                    hand="UNKNOWN",
                    split=assignments[source.source_group],
                    start_time_s=round(start, 6),
                    end_time_s=round(end, 6),
                    event_time_s=round(center, 6),
                    source_fps=source.fps,
                    source_frame_count=source.frame_count,
                    annotation_box=None,
                    derived_label=True,
                    nearest_punch_distance_s=round(nearest, 6),
                )
            )
    return sorted(rows, key=lambda row: (row.source_video, row.event_time_s, row.clip_id))
