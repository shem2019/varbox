"""Typed dataset schema definitions for evaluation harness."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

ContactLabel = Literal["landed_clean", "landed_glancing", "blocked_guarded", "missed", "clinch"]
TargetZone = Literal["head", "body", "unknown"]

CONTACT_LABELS: tuple[ContactLabel, ...] = (
    "landed_clean",
    "landed_glancing",
    "blocked_guarded",
    "missed",
    "clinch",
)
TARGET_ZONES: tuple[TargetZone, ...] = ("head", "body", "unknown")


def _parse_contact_label(value: object) -> ContactLabel:
    label = str(value).strip().lower()
    if label not in CONTACT_LABELS:
        raise ValueError(f"Unsupported contact label: {value}")
    return label  # type: ignore[return-value]


def _parse_target_zone(value: object) -> TargetZone:
    zone = str(value).strip().lower()
    if zone not in TARGET_ZONES:
        raise ValueError(f"Unsupported target zone: {value}")
    return zone  # type: ignore[return-value]


def _parse_int(value: object, *, field_name: str, min_value: int = 0) -> int:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer-like value") from exc
    if parsed < min_value:
        raise ValueError(f"{field_name} must be >= {min_value}")
    return parsed


@dataclass(frozen=True, slots=True)
class RoundMarker:
    round_no: int
    timestamp_s: float

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RoundMarker:
        return cls(
            round_no=_parse_int(payload.get("round", 0), field_name="round", min_value=1),
            timestamp_s=float(payload.get("timestamp_s", 0.0)),
        )


@dataclass(frozen=True, slots=True)
class RefereeEvent:
    timestamp_s: float
    event_type: str
    fighter_id: str
    points: int

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RefereeEvent:
        event_type = str(payload.get("event_type", "")).strip().lower()
        if event_type not in {"knockdown", "foul", "deduction"}:
            raise ValueError(f"Unsupported referee event: {event_type}")
        fighter_id = str(payload.get("fighter_id", "")).strip()
        if not fighter_id:
            raise ValueError("ref event requires fighter_id")
        return cls(
            timestamp_s=float(payload.get("timestamp_s", 0.0)),
            event_type=event_type,
            fighter_id=fighter_id,
            points=_parse_int(payload.get("points", 1), field_name="points", min_value=1),
        )


@dataclass(frozen=True, slots=True)
class EventSample:
    timestamp_s: float
    fighter_id: str
    glove_box: list[float] | None
    glove_keypoints: list[list[float]] | None
    ground_truth_label: ContactLabel
    predicted_label: ContactLabel
    ground_truth_zone: TargetZone
    predicted_zone: TargetZone
    ground_truth_track_id: str
    predicted_track_id: str
    evidence_clip: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EventSample:
        fighter_id = str(payload.get("fighter_id", "")).strip()
        if not fighter_id:
            raise ValueError("event sample requires fighter_id")

        glove_box_raw = payload.get("glove_box")
        glove_box: list[float] | None = None
        if isinstance(glove_box_raw, list):
            if len(glove_box_raw) != 4:
                raise ValueError("glove_box must contain four values")
            glove_box = [float(v) for v in glove_box_raw]

        keypoints_raw = payload.get("glove_keypoints")
        glove_keypoints: list[list[float]] | None = None
        if isinstance(keypoints_raw, list):
            parsed_points: list[list[float]] = []
            for row in keypoints_raw:
                if not isinstance(row, list) or len(row) != 2:
                    raise ValueError("glove_keypoints rows must be [x, y]")
                parsed_points.append([float(row[0]), float(row[1])])
            glove_keypoints = parsed_points

        gt_track = str(payload.get("ground_truth_track_id", "")).strip()
        pred_track = str(payload.get("predicted_track_id", "")).strip()
        if not gt_track or not pred_track:
            raise ValueError("track ids are required for ID switch metrics")

        return cls(
            timestamp_s=float(payload.get("timestamp_s", 0.0)),
            fighter_id=fighter_id,
            glove_box=glove_box,
            glove_keypoints=glove_keypoints,
            ground_truth_label=_parse_contact_label(payload.get("ground_truth_label")),
            predicted_label=_parse_contact_label(payload.get("predicted_label")),
            ground_truth_zone=_parse_target_zone(payload.get("ground_truth_zone", "unknown")),
            predicted_zone=_parse_target_zone(payload.get("predicted_zone", "unknown")),
            ground_truth_track_id=gt_track,
            predicted_track_id=pred_track,
            evidence_clip=str(payload.get("evidence_clip", "")).strip(),
        )


@dataclass(frozen=True, slots=True)
class VideoEvalRecord:
    video_id: str
    samples: list[EventSample]
    round_markers_gt: list[RoundMarker]
    round_markers_pred: list[RoundMarker]
    ref_events: list[RefereeEvent]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> VideoEvalRecord:
        video_id = str(payload.get("video_id", "")).strip()
        if not video_id:
            raise ValueError("video_id is required")

        samples_raw = payload.get("samples")
        if not isinstance(samples_raw, list) or not samples_raw:
            raise ValueError("samples must be a non-empty list")
        samples = [EventSample.from_dict(row) for row in samples_raw if isinstance(row, dict)]
        if not samples:
            raise ValueError("samples must contain valid event sample objects")

        gt_raw = payload.get("round_markers_gt", [])
        pred_raw = payload.get("round_markers_pred", [])
        ref_raw = payload.get("ref_events", [])

        round_markers_gt = [RoundMarker.from_dict(row) for row in gt_raw if isinstance(row, dict)]
        round_markers_pred = [
            RoundMarker.from_dict(row) for row in pred_raw if isinstance(row, dict)
        ]
        ref_events = [RefereeEvent.from_dict(row) for row in ref_raw if isinstance(row, dict)]

        return cls(
            video_id=video_id,
            samples=sorted(samples, key=lambda row: row.timestamp_s),
            round_markers_gt=sorted(round_markers_gt, key=lambda row: row.round_no),
            round_markers_pred=sorted(round_markers_pred, key=lambda row: row.round_no),
            ref_events=sorted(ref_events, key=lambda row: row.timestamp_s),
        )


@dataclass(frozen=True, slots=True)
class EvalDataset:
    dataset_name: str
    videos: list[VideoEvalRecord]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EvalDataset:
        dataset_name = str(payload.get("dataset_name", "")).strip() or "unnamed_dataset"
        videos_raw = payload.get("videos")
        if not isinstance(videos_raw, list) or not videos_raw:
            raise ValueError("videos must be a non-empty list")
        videos = [VideoEvalRecord.from_dict(row) for row in videos_raw if isinstance(row, dict)]
        if not videos:
            raise ValueError("videos must contain valid objects")
        return cls(dataset_name=dataset_name, videos=videos)


def load_dataset(dataset_path: str) -> EvalDataset:
    path = Path(dataset_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("dataset root must be a JSON object")
    return EvalDataset.from_dict(payload)
