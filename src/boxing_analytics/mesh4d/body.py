"""Per-frame SAM 3D Body meshes for every tracked person, prompted by their SAM 2.1 masks.

SAM 3D Body (Meta, SAM License) is cloned next to this repo by scripts/gpu/setup.sh; its
checkpoints are gated on Hugging Face. The camera is assumed fixed, so the focal length is
estimated once (MoGe-2) on a few frames and then reused for every frame of the window.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np

from boxing_analytics.mesh4d.geometry import project
from boxing_analytics.mesh4d.masklets import MaskStore
from boxing_analytics.mesh4d.video_io import VideoInfo, iter_frames, prefetch

NDArray = np.ndarray[Any, Any]
LogFn = Callable[[str], None]


@dataclass(frozen=True)
class BodyConfig:
    repo_dir: str = os.environ.get("SAM3D_BODY_DIR", "../sam-3d-body")
    hf_repo_id: str = "facebook/sam-3d-body-dinov3"
    device: str = "cuda"
    # Gloves hide the hands, so the separate hand decoder adds cost without useful detail.
    inference_type: str = "body"
    # The released checkpoints ignore mask prompts; boxes come from the SAM 2.1 masks anyway.
    use_mask: bool = False
    box_pad: float = 0.08
    chunk_frames: int = 250
    focal_samples: int = 6


def _import_sam3d(repo_dir: str) -> tuple[Any, Any, Any]:
    repo = str(Path(repo_dir).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from sam_3d_body import (  # type: ignore[import-not-found]
        SAM3DBodyEstimator,
        load_sam_3d_body_hf,
    )

    try:
        from tools.build_fov_estimator import FOVEstimator  # type: ignore[import-not-found]
    except Exception:  # MoGe missing: fall back to the model's default intrinsics
        FOVEstimator = None
    return SAM3DBodyEstimator, load_sam_3d_body_hf, FOVEstimator


def _quiet() -> contextlib.redirect_stdout[io.StringIO]:
    """Silence the estimator's per-image prints."""
    return contextlib.redirect_stdout(io.StringIO())


def _to_np(value: Any) -> NDArray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value)


def _padded_box(box: NDArray, pad: float, width: int, height: int) -> NDArray:
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    return np.array(
        [
            max(0.0, x1 - pad * bw),
            max(0.0, y1 - pad * bh),
            min(width - 1.0, x2 + pad * bw),
            min(height - 1.0, y2 + pad * bh),
        ],
        dtype=np.float32,
    )


# Candidate mappings from raw model output to OpenCV camera space. The first call picks the one
# whose projection matches the model's own 2D keypoints, so a convention change upstream is caught.
_FLIP = np.array([1.0, -1.0, -1.0])
_CONVENTIONS = ("plus_t", "flip_then_t", "t_then_flip", "raw", "flip")


def _apply_convention(points: NDArray, cam_t: NDArray, name: str) -> NDArray:
    if name == "plus_t":
        return points + cam_t
    if name == "flip_then_t":
        return points * _FLIP + cam_t
    if name == "t_then_flip":
        return (points + cam_t) * _FLIP
    if name == "flip":
        return points * _FLIP
    return points


def pick_convention(person: dict[str, Any], intrinsics: NDArray) -> tuple[str, float]:
    kp3d = _to_np(person["pred_keypoints_3d"]).reshape(-1, 3)
    kp2d = _to_np(person["pred_keypoints_2d"]).reshape(-1, 2)[:, :2]
    cam_t = _to_np(person["pred_cam_t"]).reshape(3)
    best_name, best_err = "plus_t", float("inf")
    for name in _CONVENTIONS:
        pts = _apply_convention(kp3d, cam_t, name)
        if np.median(pts[:, 2]) <= 0:
            continue
        err = float(np.median(np.linalg.norm(project(pts, intrinsics) - kp2d, axis=1)))
        if err < best_err:
            best_name, best_err = name, err
    return best_name, best_err


class BodyRunner:
    def __init__(self, config: BodyConfig, log: LogFn = print) -> None:
        import torch

        self.config = config
        self.log = log
        self.torch = torch
        # The estimator empties the CUDA cache on every image, which stalls a long sequence.
        torch.cuda.empty_cache = lambda: None
        estimator_cls, load_hf, fov_cls = _import_sam3d(config.repo_dir)
        with _quiet():
            model, model_cfg = load_hf(config.hf_repo_id, device=config.device)
            fov = fov_cls(name="moge2", device=config.device) if fov_cls is not None else None
        self.estimator = estimator_cls(
            sam_3d_body_model=model,
            model_cfg=model_cfg,
            human_detector=None,
            human_segmentor=None,
            fov_estimator=fov,
        )
        self._fov = fov
        self.faces = np.asarray(self.estimator.faces, dtype=np.int32)
        self.cam_int: Any = None
        self.intrinsics: NDArray | None = None
        self.convention: str | None = None
        self._captured: list[Any] = []

    def reset_camera(self) -> None:
        """Forget the fixed intrinsics so the next video gets its own focal estimate."""
        self.intrinsics = None
        self.cam_int = None
        self._captured = []
        self.estimator.fov_estimator = self._fov

    def _capture_fov(self) -> None:
        fov = self.estimator.fov_estimator
        if fov is None or getattr(fov, "_varbox_wrapped", False):
            return
        original = fov.get_cam_intrinsics

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            value = original(*args, **kwargs)
            self._captured.append(value.detach().clone())
            return value

        fov.get_cam_intrinsics = wrapped
        fov._varbox_wrapped = True

    def calibrate_focal(self, video: VideoInfo, frames: list[int], masks: MaskStore) -> NDArray:
        """Median intrinsics over a few frames; fixed for the rest of the window."""
        self._capture_fov()
        focal = []
        for index in frames:
            frame = next(iter_frames(video.path, index, index + 1))[1]
            entries = masks.get(index)
            boxes = [b for b, _ in entries.values() if b is not None]
            if not boxes:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            box = _padded_box(np.asarray(boxes[0]), self.config.box_pad, video.width, video.height)
            with _quiet():
                out = self.estimator.process_one_image(
                    rgb, bboxes=box[None], inference_type=self.config.inference_type
                )
            if out:
                focal.append(float(_to_np(out[0]["focal_length"]).reshape(-1)[0]))
        if self._captured:
            stacked = self.torch.stack(self._captured)
            self.cam_int = stacked.median(dim=0).values
        f = float(np.median(focal)) if focal else 0.9 * max(video.width, video.height)
        self.intrinsics = np.array(
            [[f, 0.0, video.width / 2.0], [0.0, f, video.height / 2.0], [0.0, 0.0, 1.0]]
        )
        if self.cam_int is not None:
            k = _to_np(self.cam_int).reshape(-1, 3, 3)[0]
            self.intrinsics = k.astype(np.float64)
        # Drop the estimator's FOV model: intrinsics are fixed from here on.
        self.estimator.fov_estimator = None
        assert self.intrinsics is not None
        return self.intrinsics

    def _run_frame(
        self, rgb: NDArray, roles: list[str], boxes: list[NDArray], masks: list[NDArray]
    ) -> dict[str, dict[str, Any]]:
        bboxes = np.stack(boxes).astype(np.float32)
        kwargs: dict[str, Any] = {"bboxes": bboxes, "inference_type": self.config.inference_type}
        if self.config.use_mask:
            kwargs["masks"] = np.stack(masks)[..., None].astype(np.uint8)
            kwargs["use_mask"] = True
        if self.cam_int is not None:
            kwargs["cam_int"] = self.cam_int.clone()
        with _quiet():
            outputs = self.estimator.process_one_image(rgb, **kwargs)
        # SAM 3D Body returns one result per input box, in input order.
        result: dict[str, dict[str, Any]] = {}
        if len(outputs) == len(roles):
            result = dict(zip(roles, outputs, strict=False))
        return result

    def run(self, video: VideoInfo, start: int, stop: int, masks: MaskStore, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        roles = masks.roles
        if self.intrinsics is None:
            sample = np.linspace(
                start, stop - 1, num=min(self.config.focal_samples, stop - start)
            ).astype(int)
            self.calibrate_focal(video, sample.tolist(), masks)
        assert self.intrinsics is not None
        np.save(out_dir / "faces.npy", self.faces)
        started = time.monotonic()
        chunk = self.config.chunk_frames
        for chunk_index, c_start in enumerate(range(start, stop, chunk)):
            c_stop = min(stop, c_start + chunk)
            path = out_dir / f"chunk_{chunk_index:04d}.npz"
            if path.exists():
                self.log(f"body: chunk {chunk_index} cached")
                continue
            n = c_stop - c_start
            store: dict[str, NDArray] = {"frames": np.arange(c_start, c_stop, dtype=np.int64)}
            arrays: dict[str, dict[str, Any]] = {r: {} for r in roles}
            frames = prefetch(
                iter_frames(video.path, c_start, c_stop),
                transform=lambda item: (item[0], cv2.cvtColor(item[1], cv2.COLOR_BGR2RGB)),
            )
            for local, (index, rgb) in enumerate(frames):
                entries = masks.get(index)
                present = [r for r in roles if entries[r][0] is not None]
                if not present:
                    continue
                boxes = [
                    _padded_box(
                        np.asarray(entries[r][0]), self.config.box_pad, video.width, video.height
                    )
                    for r in present
                ]
                full_masks: list[NDArray] = []
                if self.config.use_mask:
                    for r in present:
                        low_res = entries[r][1]
                        assert low_res is not None  # present roles always carry a mask
                        full_masks.append(
                            cv2.resize(
                                low_res.astype(np.uint8),
                                (video.width, video.height),
                                interpolation=cv2.INTER_NEAREST,
                            )
                        )
                people = self._run_frame(rgb, present, boxes, full_masks)
                for role, person in people.items():
                    if self.convention is None:
                        self.convention, err = pick_convention(person, self.intrinsics)
                        self.log(
                            f"body: output convention '{self.convention}' "
                            f"(reprojection {err:.1f}px)"
                        )
                    cam_t = _to_np(person["pred_cam_t"]).reshape(3)
                    verts = _apply_convention(
                        _to_np(person["pred_vertices"]).reshape(-1, 3), cam_t, self.convention
                    )
                    kp3d = _apply_convention(
                        _to_np(person["pred_keypoints_3d"]).reshape(-1, 3), cam_t, self.convention
                    )
                    a = arrays[role]
                    if "verts" not in a:
                        a["verts"] = np.full((n, verts.shape[0], 3), np.nan, dtype=np.float16)
                        a["kp3d"] = np.full((n, kp3d.shape[0], 3), np.nan, dtype=np.float32)
                        a["kp2d"] = np.full((n, kp3d.shape[0], 2), np.nan, dtype=np.float32)
                        a["valid"] = np.zeros(n, dtype=bool)
                        a["shape"] = np.full(
                            (n, _to_np(person["shape_params"]).size), np.nan, dtype=np.float32
                        )
                    a["verts"][local] = verts.astype(np.float16)
                    a["kp3d"][local] = kp3d
                    a["kp2d"][local] = _to_np(person["pred_keypoints_2d"]).reshape(-1, 2)[:, :2]
                    a["shape"][local] = _to_np(person["shape_params"]).reshape(-1)
                    a["valid"][local] = True
                if local % 50 == 0:
                    done = index - start + 1
                    rate = done / max(time.monotonic() - started, 1e-6)
                    self.log(
                        f"body: frame {index} ({rate:.2f} fps, "
                        f"eta {(stop - index) / max(rate, 1e-6):.0f}s)"
                    )
            for role in roles:
                for key, value in arrays[role].items():
                    store[f"{role}_{key}"] = value
            np.savez(path, **cast(dict[str, Any], store))
        meta = {
            "roles": roles,
            "intrinsics": self.intrinsics.tolist(),
            "convention": self.convention,
            "config": self.config.__dict__,
            "start": start,
            "stop": stop,
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_body(out_dir: Path, start: int, stop: int) -> dict[str, Any]:
    """Concatenate chunk files into dense arrays covering [start, stop)."""
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    roles: list[str] = meta["roles"]
    n = stop - start
    faces = np.load(out_dir / "faces.npy")
    data: dict[str, dict[str, NDArray]] = {r: {} for r in roles}
    for chunk in sorted(out_dir.glob("chunk_*.npz")):
        with np.load(chunk) as z:
            frames = z["frames"]
            sel = (frames >= start) & (frames < stop)
            if not sel.any():
                continue
            rows = frames[sel] - start
            for role in roles:
                if f"{role}_verts" not in z.files:
                    continue
                d = data[role]
                for key in ("verts", "kp3d", "kp2d", "valid", "shape"):
                    src = z[f"{role}_{key}"]
                    if key not in d:
                        shape = (n,) + src.shape[1:]
                        fill: Any = False if key == "valid" else np.nan
                        d[key] = np.full(shape, fill, dtype=np.float32 if key != "valid" else bool)
                    d[key][rows] = src[sel]
    return {
        "meta": meta,
        "faces": faces,
        "roles": roles,
        "data": data,
        "start": start,
        "stop": stop,
    }
