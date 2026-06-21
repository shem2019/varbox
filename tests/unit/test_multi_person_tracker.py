import numpy as np

from multi_person_tracker import MultiPersonPoseTracker


def _tracker_with_manual_ring_roi() -> MultiPersonPoseTracker:
    tracker = MultiPersonPoseTracker.__new__(MultiPersonPoseTracker)
    tracker.manual_ring_roi = {
        "normalized_points": [
            [0.20, 0.20],
            [0.80, 0.20],
            [0.80, 0.85],
            [0.20, 0.85],
        ]
    }
    tracker.max_ring_candidates = 5
    tracker.ring_min_box_inside_ratio = 0.55
    tracker.ring_min_pose_points_inside = 2
    return tracker


def test_ring_gate_boxes_excludes_people_outside_manual_ring_roi() -> None:
    tracker = _tracker_with_manual_ring_roi()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    inside = (25, 25, 45, 75, 0.85)
    outside = (0, 5, 18, 75, 0.98)

    kept = tracker._ring_gate_boxes(frame, [inside, outside])

    assert kept == [inside]


def test_ring_gate_boxes_rejects_box_with_small_polygon_overlap() -> None:
    tracker = _tracker_with_manual_ring_roi()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    mostly_outside = (0, 0, 24, 60, 0.95)

    kept = tracker._ring_gate_boxes(frame, [mostly_outside])

    assert kept == []


def test_ring_gate_boxes_preserves_dict_entries() -> None:
    tracker = _tracker_with_manual_ring_roi()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    inside = {"x1": 25, "y1": 25, "x2": 45, "y2": 75, "conf": 0.85, "track_id": 7}
    outside = {"x1": 0, "y1": 5, "x2": 18, "y2": 75, "conf": 0.98, "track_id": 9}

    kept = tracker._ring_gate_boxes(frame, [inside, outside])

    assert kept == [inside]


def test_tracker_config_resolution_uses_repo_fixed_camera_presets() -> None:
    path = MultiPersonPoseTracker._resolve_tracker_config_path("", "bytetrack")

    assert path is not None
    assert path.endswith("assets/trackers/bytetrack_fixedcam.yaml")


def test_expand_box_adds_margin_without_leaving_frame() -> None:
    expanded = MultiPersonPoseTracker._expand_box((20, 30, 60, 90), (100, 120, 3))

    assert expanded[0] < 20
    assert expanded[1] < 30
    assert expanded[2] > 60
    assert expanded[3] > 90
    assert expanded[0] >= 0
    assert expanded[1] >= 0
    assert expanded[2] <= 120
    assert expanded[3] <= 100


def test_remap_pose_keypoints_maps_coco_pose_to_mediapipe_slots() -> None:
    keypoints_xy = np.zeros((17, 2), dtype=np.float32)
    keypoints_conf = np.zeros((17,), dtype=np.float32)
    keypoints_xy[0] = (10.4, 20.6)
    keypoints_xy[5] = (30.0, 40.0)
    keypoints_xy[6] = (50.0, 60.0)
    keypoints_xy[9] = (70.0, 80.0)
    keypoints_xy[10] = (90.0, 100.0)
    keypoints_conf[[0, 5, 6, 9, 10]] = [0.9, 0.8, 0.75, 0.7, 0.65]

    mapped = MultiPersonPoseTracker._remap_pose_keypoints(keypoints_xy, keypoints_conf)

    assert mapped[0][:2] == [10, 21]
    assert mapped[11][:2] == [30, 40]
    assert mapped[12][:2] == [50, 60]
    assert mapped[15][:2] == [70, 80]
    assert mapped[16][:2] == [90, 100]
    assert np.isclose(mapped[0][2], 0.9)
    assert np.isclose(mapped[11][2], 0.8)
    assert np.isclose(mapped[12][2], 0.75)
    assert np.isclose(mapped[15][2], 0.7)
    assert np.isclose(mapped[16][2], 0.65)


def test_tracking_diagnostics_fail_without_lap(monkeypatch) -> None:
    monkeypatch.setattr(MultiPersonPoseTracker, "_lap_available", staticmethod(lambda: False))
    tracker = MultiPersonPoseTracker.__new__(MultiPersonPoseTracker)
    tracker.person_model_kind = "ultralytics_pose"
    tracker.yolo_tracker = "botsort"
    tracker.yolo_tracker_config = "assets/trackers/botsort_fixedcam.yaml"
    tracker._yolo_track_issue = None

    diag = tracker.tracking_diagnostics()

    assert diag["enabled"] is False
    assert diag["mode"] == "predict_only"
    assert "lap" in str(diag["issue"]).lower()


def test_tracking_diagnostics_fail_after_tracker_exception(monkeypatch) -> None:
    monkeypatch.setattr(MultiPersonPoseTracker, "_lap_available", staticmethod(lambda: True))
    tracker = MultiPersonPoseTracker.__new__(MultiPersonPoseTracker)
    tracker.person_model_kind = "ultralytics_pose"
    tracker.yolo_tracker = "botsort"
    tracker.yolo_tracker_config = "assets/trackers/botsort_fixedcam.yaml"
    tracker._yolo_track_issue = "RuntimeError: tracker init failed"

    diag = tracker.tracking_diagnostics()

    assert diag["enabled"] is False
    assert diag["mode"] == "predict_only"
    assert "tracker init failed" in str(diag["issue"])
