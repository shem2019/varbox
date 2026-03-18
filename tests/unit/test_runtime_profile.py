from __future__ import annotations

import runtime_profile


def test_resolve_backend_prefers_yolov8_on_apple_silicon() -> None:
    assert runtime_profile.resolve_backend("auto", is_apple_silicon=True) == "yolov8"


def test_resolve_backend_prefers_opencv_off_apple_silicon() -> None:
    assert runtime_profile.resolve_backend("auto", is_apple_silicon=False) == "opencv"


def test_detect_runtime_profile_apple_defaults(monkeypatch) -> None:
    monkeypatch.setattr(runtime_profile.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(runtime_profile.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(runtime_profile, "_torch_mps_available", lambda: True)
    monkeypatch.delenv("VARBOX_BACKEND", raising=False)
    monkeypatch.delenv("VARBOX_YOLO_DEVICE", raising=False)
    monkeypatch.delenv("VARBOX_YOLO_IMGSZ", raising=False)
    monkeypatch.delenv("VARBOX_YOLO_HALF", raising=False)
    monkeypatch.delenv("VARBOX_POSE_MODEL_COMPLEXITY", raising=False)
    monkeypatch.delenv("VARBOX_POSE_ENABLE_SEGMENTATION", raising=False)

    profile = runtime_profile.detect_runtime_profile()

    assert profile.is_apple_silicon is True
    assert profile.has_mps is True
    assert profile.preferred_backend == "yolov8"
    assert profile.yolo_device == "mps"
    assert profile.yolo_imgsz == 960
    assert profile.yolo_half is True
    assert profile.pose_model_complexity == 1
    assert profile.pose_enable_segmentation is False


def test_runtime_summary_includes_backend_and_device() -> None:
    profile = runtime_profile.RuntimeProfile(
        system="darwin",
        machine="arm64",
        is_apple_silicon=True,
        has_mps=True,
        preferred_backend="yolov8",
        yolo_device="mps",
        yolo_imgsz=960,
        yolo_half=True,
        pose_model_complexity=1,
        pose_enable_segmentation=False,
    )

    summary = runtime_profile.runtime_summary(profile)

    assert "backend=yolov8" in summary
    assert "yolo_device=mps" in summary
