from __future__ import annotations

import json

import pytest

from boxing_analytics.calibration.models import CalibrationProfile, load_profile, save_profile


def _sample_profile() -> CalibrationProfile:
    return CalibrationProfile(
        profile_name="ringside_cam_a",
        camera_matrix=[[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]],
        dist_coeffs=[0.1, -0.01, 0.0, 0.0, 0.001],
        image_width=1280,
        image_height=720,
        reprojection_error=0.42,
        ring_homography=[[1.0, 0.0, 12.0], [0.0, 1.0, -7.0], [0.0, 0.0, 1.0]],
    )


def test_save_and_load_profile_json(tmp_path):
    profile = _sample_profile()
    out_path = tmp_path / "calibration_profile.json"

    save_profile(profile, str(out_path))
    loaded = load_profile(str(out_path))

    assert loaded.profile_name == "ringside_cam_a"
    assert loaded.image_width == 1280
    assert loaded.ring_homography is not None
    assert loaded.camera_matrix[0][0] == pytest.approx(1000.0)


def test_save_and_load_profile_yaml_extension(tmp_path):
    profile = _sample_profile()
    out_path = tmp_path / "calibration_profile.yaml"

    save_profile(profile, str(out_path))
    loaded = load_profile(str(out_path))

    assert loaded.profile_name == profile.profile_name
    assert loaded.dist_coeffs[-1] == pytest.approx(0.001)


def test_load_profile_rejects_invalid_payload(tmp_path):
    bad_path = tmp_path / "bad_profile.json"
    bad_path.write_text(json.dumps({"profile_name": "x"}), encoding="utf-8")

    with pytest.raises(ValueError):
        load_profile(str(bad_path))
