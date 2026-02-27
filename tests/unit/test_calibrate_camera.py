from __future__ import annotations

import numpy as np
import pytest

from boxing_analytics.calibration.calibrate_camera import (
    CalibrationInputs,
    calibrate_camera_profile,
    collect_calibration_inputs,
)


def test_collect_calibration_inputs_missing_images():
    with pytest.raises(FileNotFoundError):
        collect_calibration_inputs(
            images_glob="/definitely/not/found/*.png",
            board_cols=7,
            board_rows=6,
            square_size_mm=25.0,
        )


def test_calibrate_camera_profile_uses_cv_outputs(monkeypatch, tmp_path):
    dummy_inputs = CalibrationInputs(
        image_paths=["a.jpg", "b.jpg", "c.jpg"],
        image_size=(1920, 1080),
        object_points=[np.zeros((42, 3), dtype=np.float32) for _ in range(3)],
        image_points=[np.zeros((42, 1, 2), dtype=np.float32) for _ in range(3)],
    )

    def fake_collect(**_kwargs):
        return dummy_inputs

    monkeypatch.setattr(
        "boxing_analytics.calibration.calibrate_camera.collect_calibration_inputs",
        fake_collect,
    )

    camera_matrix = np.array(
        [[900.0, 0.0, 960.0], [0.0, 905.0, 540.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist = np.array([[0.1, -0.02, 0.0, 0.0, 0.005]], dtype=np.float64)
    rvecs = [np.zeros((3, 1), dtype=np.float64) for _ in range(3)]
    tvecs = [np.zeros((3, 1), dtype=np.float64) for _ in range(3)]

    monkeypatch.setattr(
        "boxing_analytics.calibration.calibrate_camera.cv2.calibrateCamera",
        lambda *_args, **_kwargs: (0.12, camera_matrix, dist, rvecs, tvecs),
    )
    monkeypatch.setattr(
        "boxing_analytics.calibration.calibrate_camera.cv2.projectPoints",
        lambda obj, *_args, **_kwargs: (np.zeros((len(obj), 1, 2), dtype=np.float32), None),
    )
    monkeypatch.setattr(
        "boxing_analytics.calibration.calibrate_camera.cv2.norm",
        lambda *_args, **_kwargs: 0.0,
    )

    ring_json = tmp_path / "ring_points.json"
    ring_json.write_text(
        (
            '{"ring_points":[[0,0],[1,0],[1,1],[0,1]],'
            '"image_points":[[10,10],[110,10],[112,108],[8,104]]}'
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "boxing_analytics.calibration.calibrate_camera.cv2.findHomography",
        lambda *_args, **_kwargs: (np.eye(3, dtype=np.float64), None),
    )

    profile = calibrate_camera_profile(
        profile_name="ring_a",
        images_glob="unused/*.jpg",
        board_cols=7,
        board_rows=6,
        square_size_mm=25.0,
        ring_points_path=str(ring_json),
    )

    assert profile.profile_name == "ring_a"
    assert profile.image_width == 1920
    assert profile.image_height == 1080
    assert profile.camera_matrix[0][0] == pytest.approx(900.0)
    assert profile.ring_homography is not None
    assert profile.source_images == ["a.jpg", "b.jpg", "c.jpg"]
