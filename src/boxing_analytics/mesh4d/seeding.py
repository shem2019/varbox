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
    # Navy headgear and dark blue kit sit at low brightness, so blue accepts darker pixels.
    blue = (s > 0.3) & (v > 0.1) & (h >= 95) & (h <= 132)
    white = (s < 0.25) & (v > 0.55)
    n = float(h.size)
    return {"red": red.sum() / n, "blue": blue.sum() / n, "white": white.sum() / n}


_MODELS: dict[str, Any] = {}


def detect_people(
    frame: NDArray, model_path: str, device: str
) -> list[tuple[Box, float, NDArray | None]]:
    from ultralytics import YOLO

    if model_path not in _MODELS:
        _MODELS[model_path] = YOLO(model_path)
    result = _MODELS[model_path].predict(
        frame, device=device, verbose=False, conf=0.3, classes=[0]
    )[0]
    people: list[tuple[Box, float, NDArray | None]] = []
    if result.boxes is None:
        return people
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    kps = None
    if result.keypoints is not None and result.keypoints.data is not None:
        kps = result.keypoints.data.cpu().numpy()
    for i in range(boxes.shape[0]):
        people.append(
            (
                (float(boxes[i][0]), float(boxes[i][1]), float(boxes[i][2]), float(boxes[i][3])),
                float(confs[i]),
                None if kps is None else kps[i],
            )
        )
    return people


def _containment(inner: Box, outer: Box) -> float:
    x1, y1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    x2, y2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = max(1e-6, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter / area


def _iou(a: Box, b: Box) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(union, 1e-6)


def _head_patch(frame: NDArray, box: Box, kps: NDArray | None) -> NDArray:
    """Headgear region: around the face keypoints, or the top of the box."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    if kps is not None and kps.shape[0] >= 5 and np.any(kps[:5, 2] > 0.3):
        pts = kps[:5][kps[:5, 2] > 0.3, :2]
        cx, cy = pts.mean(axis=0)
        r = max(6.0, 0.07 * (y2 - y1))
    else:
        cx, cy, r = 0.5 * (x1 + x2), y1 + 0.08 * (y2 - y1), max(6.0, 0.07 * (y2 - y1))
    return frame[
        max(0, int(cy - r)) : min(h, int(cy + r)), max(0, int(cx - r)) : min(w, int(cx + r))
    ]


def plausible_boxer(box: Box, frame_size: tuple[int, int]) -> bool:
    """A standing person: taller than wide, clear of the side edges.

    A boxer close to the phone may run off the bottom of the frame; that is accepted when the box
    is tall. Short or wide shapes at the bottom edge (ring pads, a crouching photographer) are not.
    """
    w, h = frame_size
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if bh < 1.15 * bw:
        return False
    if x1 < 0.005 * w or x2 > 0.995 * w:
        return False
    if y2 > 0.985 * h and bh < 0.35 * h:
        return False
    return (bw * bh) / float(w * h) >= 0.004


def _candidates(frame: NDArray, model_path: str, device: str) -> list[tuple[Box, dict[str, float]]]:
    h, w = frame.shape[:2]
    people = detect_people(frame, model_path, device)
    people.sort(key=lambda p: (p[0][2] - p[0][0]) * (p[0][3] - p[0][1]), reverse=True)
    kept: list[tuple[Box, float, NDArray | None]] = []
    for person in people:
        # A partial box mostly inside a bigger one is the same person (or an occluded one).
        if any(_containment(person[0], k[0]) > 0.6 for k in kept):
            continue
        kept.append(person)
    out = []
    for box, conf, kps in kept:
        x1, y1, x2, y2 = box
        if not plausible_boxer(box, (w, h)):
            continue
        torso = colour_scores(_torso_patch(frame, box, kps))
        head = colour_scores(_head_patch(frame, box, kps))
        gloves = [
            colour_scores(p) for p in _glove_patches(frame, kps, max(6, int(0.04 * (y2 - y1))))
        ]
        glove = {
            k: float(np.mean([g[k] for g in gloves])) if gloves else 0.0
            for k in ("red", "blue", "white")
        }
        # Headgear carries the corner colour most reliably: tops vary and walls can be red.
        scores = {k: 0.35 * torso[k] + 0.2 * glove[k] + 0.45 * head[k] for k in torso}
        scores["torso_white"] = torso["white"]
        scores["head_kit"] = head["red"] + head["blue"]
        scores["area"] = (x2 - x1) * (y2 - y1) / float(w * h)
        scores["conf"] = conf
        out.append((box, scores))
    return out[:6]


def detect_roles(
    frame: NDArray, frame_index: int, *, model_path: str, device: str, min_separation: float = 0.12
) -> dict[str, Seed]:
    """Both fighters with a clear colour difference, for re-anchoring tracking mid-clip.

    When the two boxers look alike, nothing is returned: re-anchoring on a guess could swap them.
    """
    seeds, _ = _assign(_candidates(frame, model_path, device), frame_index, False)
    if "red" not in seeds or "blue" not in seeds:
        return {}
    if seeds["red"].scores.get("separation", 0.0) < min_separation:
        return {}
    return {r: seeds[r] for r in ("red", "blue")}


def auto_seed(
    frame: NDArray,
    frame_index: int,
    *,
    model_path: str,
    device: str,
    with_referee: bool,
) -> dict[str, Seed]:
    """Red/blue (+referee) seeds from a single frame."""
    return _assign(_candidates(frame, model_path, device), frame_index, with_referee)[0]


def _assign(
    candidates: list[tuple[Box, dict[str, float]]], frame_index: int, with_referee: bool
) -> tuple[dict[str, Seed], float]:
    """Pick the two fighters, then decide red and blue by comparing the two of them.

    Fighters are the largest fully framed people wearing sports kit (the white-shirted referee
    is set aside). Corner colour then comes from the relative blue-minus-red of shirt, gloves and
    headgear, so two boxers in similar singlets still get two distinct, consistent identities.
    Quality rewards clear kit, a clear colour difference and separated boxes.
    """

    def is_referee(sc: dict[str, float]) -> bool:
        return sc["torso_white"] > 0.12 and sc["red"] < 0.08 and sc["blue"] < 0.08

    pool = [(i, c) for i, c in enumerate(candidates) if not is_referee(c[1])]
    max_area = max((c[1]["area"] for _, c in pool), default=1.0)

    def fighter_score(sc: dict[str, float]) -> float:
        kit = min(1.0, (sc["red"] + sc["blue"]) / 0.4)
        return 0.6 * sc["area"] / max_area + 0.4 * kit

    def height(box: Box) -> float:
        return box[3] - box[1]

    def pair_score(a: tuple[Box, dict[str, float]], b: tuple[Box, dict[str, float]]) -> float:
        """Two sparring boxers: both in kit, similar size (similar distance from the camera),
        within sparring distance, with clearly different corner colours."""
        (box_a, sa), (box_b, sb) = a, b
        ha, hb = height(box_a), height(box_b)
        size_ratio = min(ha, hb) / max(ha, hb, 1e-6)
        cxa, cxb = 0.5 * (box_a[0] + box_a[2]), 0.5 * (box_b[0] + box_b[2])
        gap = abs(cxa - cxb) / max(0.5 * (ha + hb), 1e-6)
        separation = abs((sa["blue"] - sa["red"]) - (sb["blue"] - sb["red"]))
        score = fighter_score(sa) + fighter_score(sb) + 0.8 * min(separation, 1.0)
        # Sparring boxers wear red or blue headgear; coaches and spectators do not.
        score += 0.6 * (
            min(1.0, sa.get("head_kit", 0.0) / 0.25) + min(1.0, sb.get("head_kit", 0.0) / 0.25)
        )
        score -= 2.0 * max(0.0, 0.75 - size_ratio)  # a seated or distant person is much smaller
        # Boxers share the ring floor, so their feet sit at about the same image height.
        feet_gap = abs(box_a[3] - box_b[3]) / max(0.5 * (ha + hb), 1e-6)
        score -= 2.0 * max(0.0, feet_gap - 0.12)
        score -= 0.5 * max(0.0, gap - 2.0)  # farther apart than sparring distance
        score -= 2.0 * _iou(box_a, box_b)
        return score

    best_pair: tuple[Any, Any] | None = None
    best_score = -1e9
    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            value = pair_score(pool[i][1], pool[j][1])
            if value > best_score:
                best_pair, best_score = (pool[i], pool[j]), value
    fighters = list(best_pair) if best_pair else []
    seeds: dict[str, Seed] = {}
    quality = best_score if best_pair else -4.0
    if len(fighters) == 2:
        (ia, (box_a, sa)), (ib, (box_b, sb)) = fighters
        da, db = sa["blue"] - sa["red"], sb["blue"] - sb["red"]
        (blue_box, blue_sc), (red_box, red_sc) = (
            ((box_a, sa), (box_b, sb)) if da > db else ((box_b, sb), (box_a, sa))
        )
        separation = abs(da - db)
        for role, box, sc in (("red", red_box, red_sc), ("blue", blue_box, blue_sc)):
            scores = {k: round(v, 4) for k, v in sc.items()}
            scores["separation"] = round(separation, 4)
            seeds[role] = Seed(role, frame_index, box, "auto", scores)
    used = {i for i, _ in fighters}
    if with_referee:
        remaining = [(i, c) for i, c in enumerate(candidates) if i not in used and is_referee(c[1])]
        if remaining:
            _, (box, sc) = max(
                remaining, key=lambda item: item[1][1]["torso_white"] + 0.2 * item[1][1]["area"]
            )
            seeds["referee"] = Seed(
                "referee", frame_index, box, "auto", {k: round(v, 4) for k, v in sc.items()}
            )
            quality += 0.1
        else:
            quality -= 0.2
    boxes = [s.box for s in seeds.values()]
    overlap = max((_iou(a, b) for i, a in enumerate(boxes) for b in boxes[i + 1 :]), default=0.0)
    return seeds, quality - 2.0 * overlap


def scan_for_seed(
    video_path: str,
    start: int,
    stop: int,
    *,
    model_path: str,
    device: str,
    with_referee: bool,
    step: int = 5,
) -> tuple[dict[str, Seed], float]:
    """Best seed frame in [start, stop): everyone found, colours clear, nobody overlapping."""
    from boxing_analytics.mesh4d.video_io import iter_frames

    best: tuple[dict[str, Seed], float] = ({}, -1e9)
    for index, frame in iter_frames(video_path, start, stop, step):
        seeds, quality = _assign(_candidates(frame, model_path, device), index, with_referee)
        if quality > best[1]:
            best = (seeds, quality)
    return best


def has_fighters(seeds: dict[str, Seed]) -> bool:
    return "red" in seeds and "blue" in seeds


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
        r: Seed(r, int(s["frame_index"]), tuple(s["box"]), s["source"], s.get("scores", {}))
        for r, s in data.items()
    }
