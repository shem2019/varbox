"""Combine mesh punch events with the VideoMAE strike classifier.

The mesh pipeline is good at who threw, with which hand, and when. Whether the punch landed is a
pixel-level question that a single camera's geometry answers poorly. For every mesh event this
module cuts the same kind of clip VideoMAE was trained on (a 0.6 s window around the punch,
cropped to the glove-and-target box enlarged by the training margin), classifies it, and fuses
the classifier with a soft version of the mesh evidence:

    log p(outcome) = log p_videomae(outcome) + mesh_weight * log p_mesh(outcome)

The weight is fixed in advance rather than tuned on test clips.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from boxing_analytics.mesh4d.geometry import apply_transform, glove_centres
from boxing_analytics.mesh4d.reconstruct import ViewScene
from boxing_analytics.training.clip_extractor import uniform_frame_times
from boxing_analytics.training.interaction_crop import crop_and_pad, expanded_box

NDArray = np.ndarray[Any, Any]
LogFn = Callable[[str], None]

OUTCOMES = ("landed_head", "landed_body", "blocked", "missed")
GLOVE_RADIUS_M = 0.085


def _project(cam_from_world: NDArray, k: NDArray, points: NDArray) -> tuple[NDArray, NDArray]:
    p = apply_transform(cam_from_world, points)
    z = np.maximum(p[:, 2], 1e-6)
    uv = np.stack([p[:, 0] / z * k[0, 0] + k[0, 2], p[:, 1] / z * k[1, 1] + k[1, 2]], axis=1)
    return uv, z


def punch_box(scene: ViewScene, event: dict[str, Any]) -> tuple[float, float, float, float] | None:
    """Image box around the attacking glove and the nearest point of the opponent at contact."""
    frame = int(event["contact_frame"] or event["peak_frame"])
    i = frame - int(scene.frames[0])
    if not (0 <= i < scene.frames.shape[0]):
        return None
    attacker, defender = event["attacker"], event["defender"]
    if not (scene.valid[attacker][i] and scene.valid[defender][i]):
        return None
    left, right = glove_centres(scene.kp3d[attacker][i][None])
    glove = (left if event["hand"] == "left" else right)[0]
    verts = scene.verts[defender][i][::3]
    target = verts[np.argmin(np.linalg.norm(verts - glove, axis=1))]
    cam_from_world = np.linalg.inv(scene.world_from_cam)
    uv, z = _project(cam_from_world, scene.intrinsics, np.stack([glove, target]))
    r = scene.intrinsics[0, 0] * GLOVE_RADIUS_M / z
    x1 = float(min(uv[0, 0] - r[0], uv[1, 0] - r[1]))
    y1 = float(min(uv[0, 1] - r[0], uv[1, 1] - r[1]))
    x2 = float(max(uv[0, 0] + r[0], uv[1, 0] + r[1]))
    y2 = float(max(uv[0, 1] + r[0], uv[1, 1] + r[1]))
    return x1, y1, x2, y2


def decode_event_clip(
    video_path: str,
    event_time_s: float,
    box: tuple[float, float, float, float] | None,
    *,
    duration_s: float,
    frame_count: int,
    image_size: int,
    crop_margin: float = 1.5,
) -> list[NDArray]:
    """RGB frames prepared exactly like training (uniform times, expanded box, letterboxed)."""
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start = max(0.0, event_time_s - duration_s * 0.5)
    end = min(total / fps, start + duration_s)
    start = max(0.0, end - duration_s)
    crop = expanded_box(box, width, height, margin=crop_margin) if box is not None else None
    times = uniform_frame_times(start, end, frame_count)
    indices = [max(0, min(total - 1, int(round(t * fps)))) for t in times]
    wanted = set(indices)
    decoded: dict[int, NDArray] = {}
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(indices))
    index = min(indices)
    while index <= max(indices):
        ok, frame = cap.read()
        if not ok:
            break
        if index in wanted:
            decoded[index] = frame
        index += 1
    cap.release()
    frames = []
    last = None
    for idx in indices:
        frame = decoded.get(idx, last)
        if frame is None:
            raise ValueError(f"could not decode frame {idx} of {video_path}")
        last = frame
        frames.append(cv2.cvtColor(crop_and_pad(frame, crop, output_size=image_size), cv2.COLOR_BGR2RGB))
    return frames


class VideoMAEScorer:
    def __init__(self, model_dir: str, device: str = "cuda") -> None:
        from transformers import AutoConfig, AutoImageProcessor, AutoModelForVideoClassification

        self.model_dir = model_dir
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True, use_fast=False)
        self.config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForVideoClassification.from_pretrained(model_dir, local_files_only=True).to(device)
        self.model.eval()
        self.id2label = {int(k): str(v) for k, v in self.config.id2label.items()}
        self.frame_count = int(getattr(self.config, "num_frames", 16) or 16)
        self.image_size = int(getattr(self.config, "image_size", 224) or 224)
        meta = Path(model_dir) / "varbox_model_metadata.json"
        self.metadata = json.loads(meta.read_text(encoding="utf-8")) if meta.is_file() else {}
        self.clip_seconds = float(self.metadata.get("clip_duration_s", 0.6) or 0.6)

    def score(self, clips: list[list[NDArray]], batch_size: int = 8) -> list[dict[str, float]]:
        import torch

        out: list[dict[str, float]] = []
        for b in range(0, len(clips), batch_size):
            batch = clips[b : b + batch_size]
            pixel = torch.cat(
                [
                    self.processor(c, return_tensors="pt", do_resize=False, do_center_crop=False)["pixel_values"]
                    for c in batch
                ]
            ).to(self.device)
            with torch.inference_mode():
                probs = torch.softmax(self.model(pixel_values=pixel).logits, dim=-1).cpu().numpy()
            for row in probs:
                out.append({self.id2label[i]: float(row[i]) for i in range(row.shape[0])})
        return out


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def mesh_distribution(event: dict[str, Any], tol_m: float) -> dict[str, float]:
    """Soft outcome distribution from mesh gaps; deliberately broad (monocular depth is weak)."""
    land = _sigmoid((tol_m - float(event["min_gap_m"])) / 0.04)
    block = _sigmoid((tol_m - float(event["guard_gap_m"])) / 0.04) * (1.0 - 0.5 * land)
    miss = max(0.05, 1.0 - max(land, block))
    head_share = 0.75 if event.get("target") != "torso" else 0.3
    raw = {
        "landed_head": land * head_share,
        "landed_body": land * (1.0 - head_share),
        "blocked": block,
        "missed": miss,
    }
    total = sum(raw.values())
    return {k: max(1e-3, v / total) for k, v in raw.items()}


def fuse(p_video: dict[str, float], p_mesh: dict[str, float], mesh_weight: float) -> dict[str, float]:
    punch_mass = sum(p_video.get(k, 0.0) for k in OUTCOMES)
    logp = {
        k: math.log(max(1e-6, p_video.get(k, 0.0) / max(punch_mass, 1e-6))) + mesh_weight * math.log(p_mesh[k])
        for k in OUTCOMES
    }
    m = max(logp.values())
    exp = {k: math.exp(v - m) for k, v in logp.items()}
    total = sum(exp.values())
    return {k: v / total for k, v in exp.items()}


def combine_events(
    scene: ViewScene,
    events: list[dict[str, Any]],
    video_path: str,
    scorer: VideoMAEScorer,
    *,
    mesh_weight: float,
    contact_tol_m: float,
    log: LogFn = print,
) -> list[dict[str, Any]]:
    clips, boxes = [], []
    for e in events:
        box = punch_box(scene, e)
        t = int(e["contact_frame"] or e["peak_frame"]) / scene.fps
        clips.append(
            decode_event_clip(
                video_path,
                t,
                box,
                duration_s=scorer.clip_seconds,
                frame_count=scorer.frame_count,
                image_size=scorer.image_size,
            )
        )
        boxes.append(box)
    probs = scorer.score(clips)
    combined = []
    for e, p_video, box in zip(events, probs, boxes, strict=True):
        p_mesh = mesh_distribution(e, contact_tol_m)
        p = fuse(p_video, p_mesh, mesh_weight)
        label = max(p, key=lambda k: p[k])
        c = dict(e)
        c["mesh_outcome"] = e["outcome"] + (f"_{e['target']}" if e.get("target") else "")
        c["videomae_probabilities"] = {k: round(v, 4) for k, v in p_video.items()}
        c["mesh_probabilities"] = {k: round(v, 4) for k, v in p_mesh.items()}
        c["combined_probabilities"] = {k: round(v, 4) for k, v in p.items()}
        c["outcome"] = "landed" if label.startswith("landed") else label
        c["target"] = {"landed_head": "head", "landed_body": "torso"}.get(label)
        c["confidence"] = round(p[label], 3)
        c["punch_box"] = None if box is None else [round(v, 1) for v in box]
        c["no_punch_probability"] = round(p_video.get("no_punch", 0.0), 4)
        if c["no_punch_probability"] > 0.6:
            c.setdefault("notes", []).append("classifier thinks this may not be a punch")
        combined.append(c)
    log(f"combine: {len(combined)} events rescored with {Path(scorer.model_dir).name} (mesh weight {mesh_weight})")
    return combined
