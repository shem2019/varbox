"""Typed calibration model storage."""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return (
        _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )


def _to_float_matrix(values: list[list[float]]) -> list[list[float]]:
    return [[float(v) for v in row] for row in values]


def _to_float_vector(values: list[float]) -> list[float]:
    return [float(v) for v in values]


@dataclass(slots=True)
class CalibrationProfile:
    """Serialized intrinsics/extrinsics profile."""

    profile_name: str
    camera_matrix: list[list[float]]
    dist_coeffs: list[float]
    image_width: int
    image_height: int
    reprojection_error: float
    created_utc: str = field(default_factory=_utc_now_iso)
    source_images: list[str] = field(default_factory=list)
    ring_homography: list[list[float]] | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CalibrationProfile:
        profile_name = str(payload.get("profile_name", "")).strip()
        if not profile_name:
            raise ValueError("calibration profile is missing profile_name")

        matrix_raw = payload.get("camera_matrix")
        if not isinstance(matrix_raw, list) or len(matrix_raw) != 3:
            raise ValueError("camera_matrix must be a 3x3 matrix")
        camera_matrix = _to_float_matrix(
            [[float(v) for v in row] for row in matrix_raw if isinstance(row, list)]
        )
        if len(camera_matrix) != 3 or any(len(row) != 3 for row in camera_matrix):
            raise ValueError("camera_matrix must be a 3x3 matrix")

        dist_raw = payload.get("dist_coeffs")
        if not isinstance(dist_raw, list) or not dist_raw:
            raise ValueError("dist_coeffs must be a non-empty list")
        dist_coeffs = _to_float_vector([float(v) for v in dist_raw])

        image_width = int(payload.get("image_width", 0) or 0)
        image_height = int(payload.get("image_height", 0) or 0)
        if image_width <= 0 or image_height <= 0:
            raise ValueError("image_width and image_height must be positive")

        reprojection_error = float(payload.get("reprojection_error", 0.0) or 0.0)
        created_utc = str(payload.get("created_utc", "")).strip() or _utc_now_iso()
        source_images_raw = payload.get("source_images")
        source_images = (
            [str(item) for item in source_images_raw] if isinstance(source_images_raw, list) else []
        )

        ring_h_raw = payload.get("ring_homography")
        ring_homography: list[list[float]] | None = None
        if isinstance(ring_h_raw, list):
            ring_homography = _to_float_matrix(
                [[float(v) for v in row] for row in ring_h_raw if isinstance(row, list)]
            )
            if len(ring_homography) != 3 or any(len(row) != 3 for row in ring_homography):
                raise ValueError("ring_homography must be a 3x3 matrix when provided")

        return cls(
            profile_name=profile_name,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            image_width=image_width,
            image_height=image_height,
            reprojection_error=reprojection_error,
            created_utc=created_utc,
            source_images=source_images,
            ring_homography=ring_homography,
        )


def save_profile(profile: CalibrationProfile, output_path: str) -> str:
    """Save calibration profile to JSON/YAML path.

    YAML output is intentionally JSON-compatible (valid YAML 1.2) to avoid
    extra runtime dependencies.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(profile.to_dict(), handle, indent=2)
    return str(path)


def load_profile(profile_path: str) -> CalibrationProfile:
    """Load calibration profile from JSON/YAML file."""

    path = Path(profile_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = handle.read()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Unsupported calibration file format for {profile_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Calibration payload must be a JSON object")
    return CalibrationProfile.from_dict(payload)
