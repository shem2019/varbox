"""Roboflow-backed boxing action assessment.

The Universe model is an object detector, not a temporal punch classifier. This
adapter associates its ``punch``/``miss`` detections with tracked fighters and
normalizes them into the contact model used by the rest of VarBox.
"""

from __future__ import annotations

import base64
import json
import ssl
from dataclasses import dataclass
from typing import Any, Literal, Protocol
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import certifi

from boxing_analytics.detection.models import (
    ContactClassification,
    ContactLabel,
    TargetZone,
)

Box = tuple[int, int, int, int]


class InferenceClient(Protocol):
    def infer(self, image: Any, *, model_id: str) -> dict[str, Any]: ...


class RoboflowHTTPClient:
    """Minimal client for Roboflow's legacy-compatible inference endpoint."""

    def __init__(
        self,
        *,
        api_url: str,
        api_key: str,
        confidence_threshold: float = 0.35,
        timeout_s: float = 30.0,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.confidence_threshold = max(0.0, min(1.0, confidence_threshold))
        self.timeout_s = max(1.0, float(timeout_s))

    def infer(self, image: Any, *, model_id: str) -> dict[str, Any]:
        import cv2

        encoded_ok, jpg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not encoded_ok:
            raise ValueError("Could not JPEG-encode frame for Roboflow inference")
        body = base64.b64encode(jpg.tobytes())
        query_params: dict[str, str | int] = {
            "confidence": int(round(self.confidence_threshold * 100))
        }
        if self.api_key:
            query_params["api_key"] = self.api_key
        query = urlencode(query_params)
        url = f"{self.api_url}/{quote(model_id, safe='/')}"
        if query:
            url = f"{url}?{query}"
        request = Request(
            url,
            data=body,
            method="POST",
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "VarBox/roboflow-comparison",
            },
        )
        tls_context = ssl.create_default_context(cafile=certifi.where())
        with urlopen(  # noqa: S310
            request, timeout=self.timeout_s, context=tls_context
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Roboflow returned a non-object JSON response")
        return payload


@dataclass(frozen=True, slots=True)
class RoboflowDetection:
    label: str
    confidence: float
    box: Box

    @property
    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) // 2, (y1 + y2) // 2)


@dataclass(frozen=True, slots=True)
class RoboflowFrameResult:
    frame_index: int
    detections: tuple[RoboflowDetection, ...]


def _point(value: object) -> tuple[int, int] | None:
    if isinstance(value, tuple | list) and len(value) >= 2:
        return int(value[0]), int(value[1])
    return None


def _distance(a: tuple[int, int], b: tuple[int, int]) -> float:
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


def _box_diag(box: Box) -> float:
    x1, y1, x2, y2 = box
    return max(1.0, float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5))


def _inside(point: tuple[int, int], box: Box, padding: float = 0.1) -> bool:
    x1, y1, x2, y2 = box
    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    return x1 - pad_x <= point[0] <= x2 + pad_x and y1 - pad_y <= point[1] <= y2 + pad_y


def parse_detections(payload: dict[str, Any]) -> tuple[RoboflowDetection, ...]:
    """Parse the object-detection response returned by ``inference-sdk``."""
    raw_predictions = payload.get("predictions", [])
    if not isinstance(raw_predictions, list):
        return ()

    detections: list[RoboflowDetection] = []
    for row in raw_predictions:
        if not isinstance(row, dict):
            continue
        try:
            label = str(row.get("class", row.get("class_name", ""))).strip().lower()
            confidence = float(row.get("confidence", 0.0))
            x = float(row["x"])
            y = float(row["y"])
            width = max(0.0, float(row["width"]))
            height = max(0.0, float(row["height"]))
        except (KeyError, TypeError, ValueError):
            continue
        detections.append(
            RoboflowDetection(
                label=label,
                confidence=max(0.0, min(1.0, confidence)),
                box=(
                    int(round(x - width * 0.5)),
                    int(round(y - height * 0.5)),
                    int(round(x + width * 0.5)),
                    int(round(y + height * 0.5)),
                ),
            )
        )
    return tuple(detections)


class RoboflowBoxingAssessor:
    """Run and associate predictions from ``boxing-vxhil/1``."""

    def __init__(
        self,
        *,
        api_key: str,
        api_url: str = "https://serverless.roboflow.com",
        model_id: str = "boxing-vxhil/1",
        confidence_threshold: float = 0.35,
        sample_every_frames: int = 1,
        client: InferenceClient | None = None,
    ) -> None:
        if client is None:
            is_hosted = "roboflow.com" in api_url.lower()
            if is_hosted and not api_key.strip():
                raise ValueError(
                    "ROBOFLOW_API_KEY is required when VARBOX_STRIKE_BACKEND uses Roboflow"
                )
            client = RoboflowHTTPClient(
                api_url=api_url,
                api_key=api_key,
                confidence_threshold=confidence_threshold,
            )

        self.client = client
        self.model_id = model_id
        self.confidence_threshold = max(0.0, min(1.0, confidence_threshold))
        self.sample_every_frames = max(1, int(sample_every_frames))
        self.requests = 0
        self.failures = 0
        self.skipped_frames = 0

    def infer_frame(self, frame: Any, frame_index: int) -> RoboflowFrameResult | None:
        if (frame_index - 1) % self.sample_every_frames:
            self.skipped_frames += 1
            return None
        self.requests += 1
        try:
            payload = self.client.infer(frame, model_id=self.model_id)
        except Exception:
            self.failures += 1
            raise
        return RoboflowFrameResult(frame_index, parse_detections(payload))

    def assess_pair(
        self,
        result: RoboflowFrameResult | None,
        *,
        attacker_keypoints: dict[int, object],
        defender_keypoints: dict[int, object],
        attacker_box: Box,
        defender_box: Box,
    ) -> ContactClassification | None:
        """Associate one action detection with an attacker/defender direction."""
        if result is None:
            return None

        actions = [
            item
            for item in result.detections
            if item.label in {"punch", "miss"} and item.confidence >= self.confidence_threshold
        ]
        if not actions:
            return None

        attacker_wrists = [
            point
            for point in (_point(attacker_keypoints.get(15)), _point(attacker_keypoints.get(16)))
            if point is not None
        ]
        defender_wrists = [
            point
            for point in (_point(defender_keypoints.get(15)), _point(defender_keypoints.get(16)))
            if point is not None
        ]
        if not attacker_wrists:
            return None

        diag = max(_box_diag(attacker_box), _box_diag(defender_box))
        ranked: list[tuple[float, RoboflowDetection, tuple[int, int], float]] = []
        for action in actions:
            nearest_attacker = min(
                attacker_wrists, key=lambda point: _distance(point, action.center)
            )
            attacker_distance = _distance(nearest_attacker, action.center)
            defender_distance = (
                min(_distance(point, action.center) for point in defender_wrists)
                if defender_wrists
                else float("inf")
            )
            # Reject the opposite direction when its wrist is materially nearer.
            if defender_distance + 0.05 * diag < attacker_distance:
                continue
            association = max(0.0, 1.0 - attacker_distance / (0.75 * diag))
            ranked.append(
                (
                    action.confidence * (0.6 + 0.4 * association),
                    action,
                    nearest_attacker,
                    association,
                )
            )
        if not ranked:
            return None

        score, action, glove, association = max(ranked, key=lambda row: row[0])
        left = _point(attacker_keypoints.get(15))
        right = _point(attacker_keypoints.get(16))
        hand: Literal["L", "R"] = (
            "L"
            if left is not None
            and (right is None or _distance(left, glove) <= _distance(right, glove))
            else "R"
        )

        targets = [
            item
            for item in result.detections
            if item.label in {"head", "body"}
            and item.confidence >= self.confidence_threshold
            and _inside(item.center, defender_box)
        ]
        target = min(targets, key=lambda item: _distance(item.center, glove)) if targets else None
        target_zone: TargetZone = (
            "Head"
            if target is not None and target.label == "head"
            else "Body" if target is not None and target.label == "body" else "Unknown"
        )
        impact = target.center if target is not None else glove
        label: ContactLabel = "landed_clean" if action.label == "punch" else "missed"

        return ContactClassification(
            label=label,
            hand=hand,
            confidence=max(0.0, min(1.0, score)),
            impact_point=impact,
            target_zone=target_zone,
            glove_position=glove,
            features={
                "roboflow_action_confidence": action.confidence,
                "roboflow_association": association,
                "roboflow_target_confidence": target.confidence if target is not None else 0.0,
                "roboflow_source": 1.0,
            },
        )

    def diagnostics(self) -> dict[str, int | float | str]:
        return {
            "model_id": self.model_id,
            "confidence_threshold": self.confidence_threshold,
            "sample_every_frames": self.sample_every_frames,
            "requests": self.requests,
            "failures": self.failures,
            "skipped_frames": self.skipped_frames,
        }
