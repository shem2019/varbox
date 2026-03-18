from __future__ import annotations

import os
import platform
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeProfile:
    system: str
    machine: str
    is_apple_silicon: bool
    has_mps: bool
    preferred_backend: str
    yolo_device: str
    yolo_imgsz: int
    yolo_half: bool
    pose_model_complexity: int
    pose_enable_segmentation: bool


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _torch_mps_available() -> bool:
    try:
        import torch

        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def resolve_backend(requested_backend: str | None, *, is_apple_silicon: bool) -> str:
    requested = (requested_backend or "auto").strip().lower()
    if requested in {"", "auto"}:
        return "yolov8" if is_apple_silicon else "opencv"
    if requested in {"opencv", "yolov8"}:
        return requested
    return "yolov8" if is_apple_silicon else "opencv"


def detect_runtime_profile() -> RuntimeProfile:
    system = platform.system().lower()
    machine = platform.machine().lower()
    is_apple_silicon = system == "darwin" and machine == "arm64"
    has_mps = is_apple_silicon and _torch_mps_available()
    preferred_backend = resolve_backend(
        os.getenv("VARBOX_BACKEND", "auto"),
        is_apple_silicon=is_apple_silicon,
    )
    yolo_device = os.getenv("VARBOX_YOLO_DEVICE", "mps" if has_mps else "cpu").strip().lower()
    yolo_imgsz = int(os.getenv("VARBOX_YOLO_IMGSZ", "960" if is_apple_silicon else "640") or "640")
    yolo_half = _env_flag("VARBOX_YOLO_HALF", yolo_device == "mps")
    pose_model_complexity = int(
        os.getenv("VARBOX_POSE_MODEL_COMPLEXITY", "1" if is_apple_silicon else "2") or "1"
    )
    pose_enable_segmentation = _env_flag(
        "VARBOX_POSE_ENABLE_SEGMENTATION",
        not is_apple_silicon,
    )
    return RuntimeProfile(
        system=system,
        machine=machine,
        is_apple_silicon=is_apple_silicon,
        has_mps=has_mps,
        preferred_backend=preferred_backend,
        yolo_device=yolo_device,
        yolo_imgsz=yolo_imgsz,
        yolo_half=yolo_half,
        pose_model_complexity=pose_model_complexity,
        pose_enable_segmentation=pose_enable_segmentation,
    )


def runtime_summary(profile: RuntimeProfile | None = None) -> str:
    current = profile or detect_runtime_profile()
    return (
        f"platform={current.system}/{current.machine} "
        f"apple_silicon={int(current.is_apple_silicon)} "
        f"mps={int(current.has_mps)} "
        f"backend={current.preferred_backend} "
        f"yolo_device={current.yolo_device} "
        f"yolo_imgsz={current.yolo_imgsz} "
        f"yolo_half={int(current.yolo_half)} "
        f"pose_model_complexity={current.pose_model_complexity} "
        f"pose_segmentation={int(current.pose_enable_segmentation)}"
    )
