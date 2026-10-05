"""Typed representation of the Olympic boxing CVAT annotations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

FiveClassLabel = Literal["landed_head", "landed_body", "blocked", "missed", "no_punch"]
HandLabel = Literal["LEFT", "RIGHT", "UNKNOWN"]

FIVE_CLASS_LABELS = ("landed_head", "landed_body", "blocked", "missed", "no_punch")
EIGHT_CLASS_LABELS = (
    "head_left",
    "head_right",
    "body_left",
    "body_right",
    "blocked_left",
    "blocked_right",
    "missed_left",
    "missed_right",
)

ORIGINAL_LABEL_MAPPING: dict[str, tuple[str, str, HandLabel]] = {
    "Głowa lewą ręką": ("landed_head", "head_left", "LEFT"),
    "Głowa prawą ręką": ("landed_head", "head_right", "RIGHT"),
    "Korpus lewą ręką": ("landed_body", "body_left", "LEFT"),
    "Korpus prawą ręką": ("landed_body", "body_right", "RIGHT"),
    "Blok lewą ręką": ("blocked", "blocked_left", "LEFT"),
    "Blok prawą ręką": ("blocked", "blocked_right", "RIGHT"),
    "Chybienie lewą ręką": ("missed", "missed_left", "LEFT"),
    "Chybienie prawą ręką": ("missed", "missed_right", "RIGHT"),
}


@dataclass(frozen=True, slots=True)
class AnnotationBox:
    frame_index: int
    points: tuple[float, float, float, float]
    outside: bool
    occluded: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class OlympicPunchAnnotation:
    annotation_id: str
    task_name: str
    source_video: str
    source_group: str
    camera: str
    original_label: str
    five_class_label: str
    eight_class_label: str
    hand: HandLabel
    start_frame: int
    end_frame: int
    peak_frame: int
    start_time_s: float
    end_time_s: float
    peak_time_s: float
    boxes: tuple[AnnotationBox, ...]
    derived_label: bool = False

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["boxes"] = [box.to_dict() for box in self.boxes]
        return payload
