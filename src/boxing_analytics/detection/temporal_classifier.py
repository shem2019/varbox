"""Offline temporal strike-classifier contracts and VideoMAE implementation."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray

Frame: TypeAlias = NDArray[np.uint8]
BoundingBox = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class PunchCandidate:
    candidate_id: str
    timestamp_s: float
    attacker_role: str
    defender_role: str
    hand: str
    pose_confidence: float
    wrist_speed: float
    wrist_acceleration: float
    arm_extension: float
    glove_target_distance: float
    guard_coverage: float
    fighter_overlap: float
    clinch_score: float
    identity_confidence: float
    keypoint_visibility: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StrikeAssessment:
    outcome: str
    target_zone: str
    hand: str
    confidence: float
    class_probabilities: dict[str, float]
    model_name: str
    model_version: str
    device: str
    inference_duration_ms: float
    clip_start_time: float
    clip_end_time: float
    abstained: bool
    abstention_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class TemporalStrikeClassifier(Protocol):
    def classify(
        self,
        frames: Sequence[Frame],
        candidate: PunchCandidate,
        attacker_boxes: Sequence[BoundingBox | None],
        defender_boxes: Sequence[BoundingBox | None],
        *,
        clip_start_time: float,
        clip_end_time: float,
    ) -> StrikeAssessment: ...


def resolve_inference_device(requested: str) -> str:
    import torch

    value = (requested or "auto").strip().lower()
    if value == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        return "cpu"
    if value == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    if value not in {"cuda", "mps", "cpu"}:
        return "cpu"
    return value


def _model_digest(model_dir: Path) -> str:
    digest = hashlib.sha256()
    candidates = sorted(model_dir.glob("*.safetensors"))
    if not candidates:
        candidates = sorted(model_dir.glob("pytorch_model*.bin"))
    for path in candidates:
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


class VideoMAETemporalStrikeClassifier:
    """Lazy, local-files-only VideoMAE classifier."""

    def __init__(
        self,
        *,
        model_dir: str,
        device: str = "auto",
        confidence_threshold: float = 0.55,
    ) -> None:
        local_dir = Path(model_dir).expanduser().resolve()
        if not local_dir.is_dir():
            raise FileNotFoundError(f"Local VideoMAE model directory not found: {local_dir}")
        self.model_dir = local_dir
        self.device = resolve_inference_device(device)
        self.confidence_threshold = max(0.0, min(1.0, confidence_threshold))
        self._processor: Any = None
        self._model: Any = None
        self._config: Any = None
        metadata_path = local_dir / "varbox_model_metadata.json"
        self.training_metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
        )
        self.model_sha256 = _model_digest(local_dir)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoConfig, AutoImageProcessor, AutoModelForVideoClassification

        self._processor = AutoImageProcessor.from_pretrained(
            self.model_dir,
            local_files_only=True,
            use_fast=False,
        )
        self._config = AutoConfig.from_pretrained(self.model_dir, local_files_only=True)
        self._model = AutoModelForVideoClassification.from_pretrained(
            self.model_dir,
            local_files_only=True,
        ).to(self.device)
        self._model.eval()

    @property
    def frame_count(self) -> int:
        self._ensure_loaded()
        return int(getattr(self._config, "num_frames", 16) or 16)

    @property
    def image_size(self) -> int:
        self._ensure_loaded()
        return int(getattr(self._config, "image_size", 224) or 224)

    def diagnostics(self) -> dict[str, object]:
        self._ensure_loaded()
        processor_config: dict[str, Any] = (
            self._processor.to_dict()
            if self._processor is not None and hasattr(self._processor, "to_dict")
            else {}
        )
        model_config: dict[str, Any] = (
            self._config.to_dict()
            if self._config is not None and hasattr(self._config, "to_dict")
            else {}
        )
        return {
            "model_dir": str(self.model_dir),
            "model_sha256": self.model_sha256,
            "model_type": str(getattr(self._config, "model_type", "")),
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "frame_count": self.frame_count,
            "image_size": self.image_size,
            "processor": type(self._processor).__name__,
            "processor_config": processor_config,
            "model_config": model_config,
            "label_mapping": dict(getattr(self._config, "id2label", {})),
            "training_metadata": self.training_metadata,
            "offline_local_files_only": 1,
        }

    def classify(
        self,
        frames: Sequence[Frame],
        candidate: PunchCandidate,
        attacker_boxes: Sequence[BoundingBox | None],
        defender_boxes: Sequence[BoundingBox | None],
        *,
        clip_start_time: float,
        clip_end_time: float,
    ) -> StrikeAssessment:
        del attacker_boxes, defender_boxes
        import torch

        self._ensure_loaded()
        if len(frames) != self.frame_count:
            return StrikeAssessment(
                outcome="uncertain",
                target_zone="Unknown",
                hand=candidate.hand,
                confidence=0.0,
                class_probabilities={},
                model_name=str(self.model_dir.name),
                model_version=self.model_sha256[:16],
                device=self.device,
                inference_duration_ms=0.0,
                clip_start_time=clip_start_time,
                clip_end_time=clip_end_time,
                abstained=True,
                abstention_reasons=("invalid_frame_count",),
            )
        rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
        started = time.perf_counter()
        inputs = self._processor(
            rgb_frames,
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
        )
        pixel_values = inputs["pixel_values"].to(self.device)
        with torch.inference_mode():
            logits = self._model(pixel_values=pixel_values).logits
            probabilities_tensor = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        id2label = {int(key): str(value) for key, value in dict(self._config.id2label).items()}
        probabilities = {
            id2label[index]: round(float(probabilities_tensor[index]), 6)
            for index in range(len(probabilities_tensor))
        }
        predicted_index = int(probabilities_tensor.argmax().item())
        outcome = id2label[predicted_index]
        confidence = float(probabilities_tensor[predicted_index])
        reasons: list[str] = []
        if confidence < self.confidence_threshold:
            reasons.append("transformer_confidence_below_threshold")
        if outcome not in {"landed_head", "landed_body", "blocked", "missed", "no_punch"}:
            reasons.append("incompatible_label_mapping")
        return StrikeAssessment(
            outcome=outcome,
            target_zone=(
                "Head"
                if outcome == "landed_head"
                else "Body" if outcome == "landed_body" else "Unknown"
            ),
            hand=candidate.hand,
            confidence=round(confidence, 6),
            class_probabilities=probabilities,
            model_name=str(self.model_dir.name),
            model_version=self.model_sha256[:16],
            device=self.device,
            inference_duration_ms=round(elapsed_ms, 3),
            clip_start_time=clip_start_time,
            clip_end_time=clip_end_time,
            abstained=bool(reasons),
            abstention_reasons=tuple(reasons),
        )
