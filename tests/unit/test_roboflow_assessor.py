import json

import numpy as np

from boxing_analytics.detection.roboflow import (
    RoboflowBoxingAssessor,
    RoboflowHTTPClient,
    parse_detections,
)


class FakeClient:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls = 0

    def infer(self, image, *, model_id: str) -> dict:
        self.calls += 1
        assert model_id == "boxing-vxhil/1"
        return self.payload


def _pose(left_wrist: tuple[int, int], right_wrist: tuple[int, int]) -> dict[int, object]:
    return {15: left_wrist, 16: right_wrist}


def test_parse_detections_converts_center_boxes() -> None:
    detections = parse_detections(
        {
            "predictions": [
                {
                    "class": "Punch",
                    "confidence": 0.8,
                    "x": 100,
                    "y": 80,
                    "width": 20,
                    "height": 40,
                }
            ]
        }
    )

    assert detections[0].label == "punch"
    assert detections[0].box == (90, 60, 110, 100)


def test_assessor_assigns_punch_to_nearest_fighter_and_target() -> None:
    client = FakeClient(
        {
            "predictions": [
                {"class": "punch", "confidence": 0.9, "x": 145, "y": 75, "width": 30, "height": 20},
                {"class": "head", "confidence": 0.8, "x": 160, "y": 75, "width": 30, "height": 35},
            ]
        }
    )
    assessor = RoboflowBoxingAssessor(api_key="test", client=client)
    frame_result = assessor.infer_frame(object(), 1)

    red = assessor.assess_pair(
        frame_result,
        attacker_keypoints=_pose((140, 75), (100, 110)),
        defender_keypoints=_pose((190, 110), (195, 115)),
        attacker_box=(40, 40, 150, 180),
        defender_box=(145, 45, 230, 180),
    )
    blue = assessor.assess_pair(
        frame_result,
        attacker_keypoints=_pose((190, 110), (195, 115)),
        defender_keypoints=_pose((140, 75), (100, 110)),
        attacker_box=(145, 45, 230, 180),
        defender_box=(40, 40, 150, 180),
    )

    assert red is not None
    assert red.label == "landed_clean"
    assert red.target_zone == "Head"
    assert blue is None


def test_assessor_samples_frames_to_limit_remote_requests() -> None:
    client = FakeClient({"predictions": []})
    assessor = RoboflowBoxingAssessor(api_key="test", client=client, sample_every_frames=3)

    assert assessor.infer_frame(object(), 1) is not None
    assert assessor.infer_frame(object(), 2) is None
    assert assessor.infer_frame(object(), 3) is None
    assert assessor.infer_frame(object(), 4) is not None
    assert client.calls == 2
    assert assessor.diagnostics()["skipped_frames"] == 2


def test_http_client_posts_jpeg_to_model_endpoint(monkeypatch) -> None:
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self) -> bytes:
            return json.dumps({"predictions": []}).encode()

    def fake_urlopen(request, *, timeout, context):
        captured["url"] = request.full_url
        captured["body"] = request.data
        captured["timeout"] = timeout
        captured["context"] = context
        return Response()

    monkeypatch.setattr("boxing_analytics.detection.roboflow.urlopen", fake_urlopen)
    client = RoboflowHTTPClient(
        api_url="https://serverless.roboflow.com", api_key="secret", timeout_s=12
    )

    result = client.infer(np.zeros((20, 30, 3), dtype=np.uint8), model_id="boxing-vxhil/1")

    assert result == {"predictions": []}
    assert captured["url"].endswith("/boxing-vxhil/1?confidence=35&api_key=secret")
    assert captured["body"]
    assert captured["timeout"] == 12
    assert captured["context"].verify_mode != 0
