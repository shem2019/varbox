"""Find RED, BLUE (and optionally the referee) in one frame to seed mask tracking.

Automatic seeding runs a YOLO pose model, keeps the largest people near the ring centre and
assigns roles by kit colour sampled on the torso. Manual boxes override any role.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

NDArray = np.ndarray[Any, Any]
Box = tuple[float, float, float, float]

ROLE_COLOURS_BGR = {"red": (40, 40, 220), "blue": (220, 90, 30), "referee": (200, 200, 200)}


@dataclass
class Seed:
    role: str
    frame_index: int
    box: Box
    source: str  # auto | manual
    scores: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_manual_seeds(values: list[str] | None) -> dict[str, Box]:
    """'red=x1,y1,x2,y2' entries -> {role: box} in source pixels."""
    out: dict[str, Box] = {}
    for item in values or []:
        role, _, coords = item.partition("=")
        parts = [float(v) for v in coords.split(",")]
        if len(parts) != 4:
            raise ValueError(f"seed must be role=x1,y1,x2,y2, got {item!r}")
        out[role.strip().lower()] = (parts[0], parts[1], parts[2], parts[3])
    return out


def _torso_patch(frame: NDArray, box: Box, kps: NDArray | None) -> NDArray:
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    h, w = frame.shape[:2]
    if kps is not None and kps.shape[0] >= 13 and np.all(kps[[5, 6, 11, 12], 2] > 0.3):
        pts = kps[[5, 6, 11, 12], :2]
        tx1, ty1 = pts.min(axis=0)
        tx2, ty2 = pts.max(axis=0)
        pad_x = 0.15 * (tx2 - tx1)
        x1, x2 = int(tx1 + pad_x), int(tx2 - pad_x)
        y1, y2 = int(ty1), int(ty2)
    else:
        bw, bh = x2 - x1, y2 - y1
        x1, x2 = x1 + int(0.25 * bw), x2 - int(0.25 * bw)
        y1, y2 = y1 + int(0.2 * bh), y1 + int(0.55 * bh)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, max(x1 + 2, x2)), min(h, max(y1 + 2, y2))
    return frame[y1:y2, x1:x2]


def _glove_patches(frame: NDArray, kps: NDArray | None, radius: int) -> list[NDArray]:
    if kps is None or kps.shape[0] < 11:
        return []
    h, w = frame.shape[:2]
    out = []
    for idx in (9, 10):  # COCO wrists
        if kps[idx, 2] < 0.3:
            continue
        x, y = int(kps[idx, 0]), int(kps[idx, 1])
        out.append(
            frame[max(0, y - radius) : min(h, y + radius), max(0, x - radius) : min(w, x + radius)]
        )
    return out


def colour_scores(patch: NDArray) -> dict[str, float]:
    if patch.size == 0:
        return {"red": 0.0, "blue": 0.0, "white": 0.0}
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    h, s, v = hsv[:, 0], hsv[:, 1] / 255.0, hsv[:, 2] / 255.0
    chroma = (s > 0.35) & (v > 0.2)
    red = chroma & ((h <= 10) | (h >= 165))
    blue = chroma & (h >= 95) & (h <= 130)
    white = (s < 0.2) & (v > 0.7)
    n = float(h.size)
    return {"red": red.sum() / n, "blue": blue.sum() / n, "white": white.sum() / n}


def detect_people(frame: NDArray, model_path: str, device: str) -> list[tuple[Box, float, NDArray]]:
    from ultralytics import YOLO  # type: ignore[import-untyped]

    model = YOLO(model_path)
    result = model.predict(frame, device=device, verbose=False, conf=0.3, classes=[0])[0]
    people = []
    if result.boxes is None:
        return people
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    kps = None
    if result.keypoints is not None and result.keypoints.data is not None:
        kps = result.keypoints.data.cpu().numpy()
    for i in range(boxes.shape[0]):
        people.append(
            (tuple(float(v) for v in boxes[i]), float(confs[i]), None if kps is None else kps[i])
        )
    return people


def auto_seed(
    frame: NDArray,
    frame_index: int,
    *,
    model_path: str,
    device: str,
    with_referee: bool,
) -> dict[str, Seed]:
    """Assign red/blue (+referee) among the largest people in the central part of the frame."""
    h, w = frame.shape[:2]
    people = detect_people(frame, model_path, device)
    candidates = []
    for box, conf, kps in people:
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1) / float(w * h)
        cx = 0.5 * (x1 + x2) / w
        if area < 0.004 or not (0.08 < cx < 0.92):
            continue
        torso = colour_scores(_torso_patch(frame, box, kps))
        gloves = [
            colour_scores(p) for p in _glove_patches(frame, kps, max(6, int(0.04 * (y2 - y1))))
        ]
        glove = {
            k: float(np.mean([g[k] for g in gloves])) if gloves else 0.0
            for k in ("red", "blue", "white")
        }
        scores = {k: 0.65 * torso[k] + 0.35 * glove[k] for k in torso}
        scores["area"] = area
        scores["conf"] = conf
        candidates.append((box, scores))
    candidates.sort(key=lambda c: c[1]["area"], reverse=True)
    candidates = candidates[:5]
    seeds: dict[str, Seed] = {}
    used: set[int] = set()
    for role in ("red", "blue"):
        best_i, best_val = -1, -1.0
        for i, (_, sc) in enumerate(candidates):
            if i in used:
                continue
            val = sc[role] - 0.5 * sc["blue" if role == "red" else "red"]
            if val > best_val:
                best_i, best_val = i, val
        if best_i >= 0:
            used.add(best_i)
            box, sc = candidates[best_i]
            seeds[role] = Seed(
                role, frame_index, box, "auto", {k: round(v, 4) for k, v in sc.items()}
            )
    if with_referee:
        remaining = [(i, c) for i, c in enumerate(candidates) if i not in used]
        if remaining:
            i, (box, sc) = max(
                remaining, key=lambda item: item[1][1]["white"] + 0.2 * item[1][1]["area"]
            )
            seeds["referee"] = Seed(
                "referee", frame_index, box, "auto", {k: round(v, 4) for k, v in sc.items()}
            )
    return seeds


def draw_seeds(frame: NDArray, seeds: dict[str, Seed]) -> NDArray:
    out = frame.copy()
    for role, seed in seeds.items():
        x1, y1, x2, y2 = (int(round(v)) for v in seed.box)
        colour = ROLE_COLOURS_BGR.get(role, (0, 255, 0))
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 3)
        cv2.putText(
            out,
            f"{role} ({seed.source})",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            colour,
            2,
        )
    return out


def save_seeds(path: Path, seeds: dict[str, Seed]) -> None:
    path.write_text(
        json.dumps({r: s.to_dict() for r, s in seeds.items()}, indent=2), encoding="utf-8"
    )


def load_seeds(path: Path) -> dict[str, Seed]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        r: Seed(r, int(s["frame_index"]), tuple(s["box"]), s["source"], s.get("scores", {}))  # type: ignore[arg-type]
        for r, s in data.items()
    }
