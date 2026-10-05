"""Dense per-frame SAM 2.1 masks for each tracked person (RED, BLUE, optionally referee).

Frames are processed in overlapping chunks. Each chunk is seeded with the previous chunk's last
mask, so identity carries across the whole window without holding every frame in GPU memory.
A role that disappears is re-found at the next chunk boundary by colour-based re-seeding.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np

from boxing_analytics.mesh4d.seeding import Seed, auto_seed, detect_roles
from boxing_analytics.mesh4d.video_io import VideoInfo, iter_frames, prefetch

NDArray = np.ndarray[Any, Any]
LogFn = Callable[[str], None]


@dataclass(frozen=True)
class MaskConfig:
    checkpoint: str
    model_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml"
    device: str = "cuda"
    max_side: int = 1024
    chunk_frames: int = 240
    min_area_frac: float = 0.0015
    yolo_model: str = "yolo11m-pose.pt"
    # SAM 2 processes tracking chunks at once (0 = from CPU cores; 1 = the sequential tracker).
    workers: int = 0


def mask_to_box(mask: NDArray) -> tuple[float, float, float, float] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def _box_iou(a: NDArray, b: NDArray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / max(union, 1e-6))


def pack(mask: NDArray) -> NDArray:
    return np.packbits(mask.astype(bool).reshape(-1))


def unpack(bits: NDArray, shape: tuple[int, int]) -> NDArray:
    return np.unpackbits(bits, count=shape[0] * shape[1]).reshape(shape).astype(bool)


class MaskStore:
    """Reads the chunked mask files written by `track_masks`."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        self.roles: list[str] = meta["roles"]
        self.shape: tuple[int, int] = tuple(meta["mask_shape"])
        self.scale: float = float(meta["scale"])
        self.frames: dict[int, tuple[Path, int]] = {}
        for chunk in sorted(directory.glob("chunk_*.npz")):
            with np.load(chunk) as data:
                for local, frame in enumerate(data["frames"].tolist()):
                    self.frames[int(frame)] = (chunk, local)
        self._cache: tuple[Path, dict[str, NDArray]] | None = None

    def _chunk(self, path: Path) -> dict[str, NDArray]:
        if self._cache is None or self._cache[0] != path:
            with np.load(path) as data:
                self._cache = (path, {k: data[k] for k in data.files})
        return self._cache[1]

    def get(self, frame: int) -> dict[str, tuple[NDArray | None, NDArray | None]]:
        """{role: (full-res box or None, low-res bool mask or None)}"""
        out: dict[str, tuple[NDArray | None, NDArray | None]] = {
            r: (None, None) for r in self.roles
        }
        if frame not in self.frames:
            return out
        path, local = self.frames[frame]
        data = self._chunk(path)
        for role in self.roles:
            box = data[f"{role}_box"][local]
            if np.isnan(box).any():
                continue
            out[role] = (box, unpack(data[f"{role}_bits"][local], self.shape))
        return out

    def pair_iou(self, frame: int, a: str = "red", b: str = "blue") -> float:
        """Overlap of two roles' masks; near 1 means both tracks sit on the same person."""
        entries = self.get(frame)
        ma, mb = entries.get(a, (None, None))[1], entries.get(b, (None, None))[1]
        if ma is None or mb is None:
            return 0.0
        union = float((ma | mb).sum())
        return float((ma & mb).sum()) / union if union else 0.0

    def occlusion(self, frame: int) -> dict[str, float]:
        """Fraction of each role's mask box covered by any other role's mask."""
        entries = self.get(frame)
        out: dict[str, float] = {}
        for role, (box, mask) in entries.items():
            if box is None or mask is None:
                out[role] = 1.0
                continue
            s = self.scale
            x1, y1, x2, y2 = (int(v * s) for v in box)
            region = np.zeros(self.shape, dtype=bool)
            region[y1:y2, x1:x2] = True
            others = np.zeros(self.shape, dtype=bool)
            for other, (_, m) in entries.items():
                if other != role and m is not None:
                    others |= m
            out[role] = float((others & region).sum()) / max(1.0, float(region.sum()))
        return out


def _write_jpegs(
    video: VideoInfo, start: int, stop: int, directory: Path, scale: float
) -> list[int]:
    frames: list[int] = []
    size = (int(round(video.width * scale)), int(round(video.height * scale)))

    def save(local: int, frame: NDArray) -> None:
        small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(directory / f"{local:05d}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 92])

    # Decode in one thread, resize and JPEG-encode on several (OpenCV releases the GIL).
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = []
        for local, (index, frame) in enumerate(prefetch(iter_frames(video.path, start, stop))):
            jobs.append(pool.submit(save, local, frame))
            frames.append(index)
        for job in jobs:
            job.result()
    return frames


def track_masks(
    video: VideoInfo,
    start: int,
    stop: int,
    seeds: dict[str, Seed],
    out_dir: Path,
    config: MaskConfig,
    log: LogFn = print,
) -> None:
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    out_dir.mkdir(parents=True, exist_ok=True)
    roles = list(seeds.keys())
    scale = min(1.0, config.max_side / float(max(video.width, video.height)))
    mask_shape = (int(round(video.height * scale)), int(round(video.width * scale)))
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "roles": roles,
                "mask_shape": list(mask_shape),
                "scale": scale,
                "start": start,
                "stop": stop,
                "config": config.__dict__,
                "seeds": {r: s.to_dict() for r, s in seeds.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    # One owner per pixel: an occluded boxer's mask shrinks instead of copying the visible one.
    predictor = build_sam2_video_predictor(
        config.model_config,
        config.checkpoint,
        device=config.device,
        hydra_overrides_extra=["++model.non_overlap_masks=true"],
    )
    obj_ids = {role: i + 1 for i, role in enumerate(roles)}
    id_roles = {v: k for k, v in obj_ids.items()}
    carry: dict[str, NDArray] = {}  # role -> last good low-res mask
    min_area = config.min_area_frac * mask_shape[0] * mask_shape[1]
    chunk_start = start
    chunk_index = 0
    started = time.monotonic()
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if config.device.startswith("cuda")
        else torch.autocast("cpu", enabled=False)
    )
    while chunk_start < stop - 1 or (chunk_index == 0 and chunk_start < stop):
        chunk_stop = min(stop, chunk_start + config.chunk_frames)
        chunk_path = out_dir / f"chunk_{chunk_index:04d}.npz"
        if chunk_path.exists():
            with np.load(chunk_path) as data:
                for role in roles:
                    bits = data[f"{role}_bits"][-1]
                    if not np.isnan(data[f"{role}_box"][-1]).any():
                        carry[role] = unpack(bits, mask_shape)
            log(f"masks: chunk {chunk_index} cached")
            chunk_index += 1
            chunk_start = chunk_stop - 1
            if chunk_stop >= stop:
                break
            continue
        with tempfile.TemporaryDirectory(prefix="varbox_m4d_") as tmp:
            tmp_dir = Path(tmp)
            frames = _write_jpegs(video, chunk_start, chunk_stop, tmp_dir, scale)
            if not frames:
                break
            with torch.inference_mode(), autocast:
                state = predictor.init_state(
                    video_path=str(tmp_dir), offload_video_to_cpu=True, async_loading_frames=False
                )
                found: dict[str, Seed] = {}
                if chunk_index > 0:
                    frame0 = next(iter_frames(video.path, chunk_start, chunk_start + 1))[1]
                    found = detect_roles(
                        frame0, chunk_start, model_path=config.yolo_model, device=config.device
                    )
                for role in roles:
                    if chunk_index == 0:
                        seed = seeds[role]
                        local = max(0, min(len(frames) - 1, seed.frame_index - chunk_start))
                        box = np.asarray(seed.box, dtype=np.float32) * scale
                        predictor.add_new_points_or_box(
                            state, frame_idx=local, obj_id=obj_ids[role], box=box
                        )
                        continue
                    carried = carry.get(role)
                    detected = found.get(role)
                    carried_box = mask_to_box(carried) if carried is not None else None
                    if detected is not None and (
                        carried_box is None
                        or _box_iou(np.asarray(carried_box) / scale, np.asarray(detected.box)) < 0.3
                    ):
                        # Tracking lost or drifted: lock back onto the colour-confirmed boxer.
                        box = np.asarray(detected.box, dtype=np.float32) * scale
                        predictor.add_new_points_or_box(
                            state, frame_idx=0, obj_id=obj_ids[role], box=box
                        )
                        log(f"masks: re-anchored {role} at frame {chunk_start}")
                    elif carried is not None:
                        predictor.add_new_mask(
                            state, frame_idx=0, obj_id=obj_ids[role], mask=carried
                        )
                    elif role == "referee":
                        frame0 = next(iter_frames(video.path, chunk_start, chunk_start + 1))[1]
                        ref = auto_seed(
                            frame0,
                            chunk_start,
                            model_path=config.yolo_model,
                            device=config.device,
                            with_referee=True,
                        ).get("referee")
                        if ref is not None:
                            box = np.asarray(ref.box, dtype=np.float32) * scale
                            predictor.add_new_points_or_box(
                                state, frame_idx=0, obj_id=obj_ids[role], box=box
                            )
                    else:
                        log(f"masks: {role} out of view at frame {chunk_start}")
                n = len(frames)
                boxes = {r: np.full((n, 4), np.nan, dtype=np.float32) for r in roles}
                bits = {
                    r: np.zeros((n, (mask_shape[0] * mask_shape[1] + 7) // 8), dtype=np.uint8)
                    for r in roles
                }

                def consume(
                    stream: Any, boxes: dict[str, NDArray], bits: dict[str, NDArray]
                ) -> None:
                    for local, ids, logits in stream:
                        for k, obj in enumerate(ids):
                            role = id_roles[int(obj)]
                            mask = (logits[k] > 0.0).squeeze(0).float().cpu().numpy().astype(bool)
                            if mask.sum() < min_area:
                                continue
                            b = mask_to_box(mask)
                            if b is None:
                                continue
                            boxes[role][local] = np.asarray(b, dtype=np.float32) / scale
                            bits[role][local] = pack(mask)

                consume(predictor.propagate_in_video(state), boxes, bits)
                if chunk_index == 0:
                    # Seeds may sit a little after the window start: fill those frames backwards.
                    first_seed = min(max(0, seeds[r].frame_index - chunk_start) for r in roles)
                    if first_seed > 0:
                        consume(
                            predictor.propagate_in_video(
                                state, start_frame_idx=first_seed, reverse=True
                            ),
                            boxes,
                            bits,
                        )
                predictor.reset_state(state)
        for role in roles:
            good = np.flatnonzero(~np.isnan(boxes[role][:, 0]))
            if good.size and good[-1] >= len(frames) - 5:
                carry[role] = unpack(bits[role][good[-1]], mask_shape)
            else:
                carry.pop(role, None)
        payload: dict[str, NDArray] = {"frames": np.asarray(frames, dtype=np.int64)}
        for role in roles:
            payload[f"{role}_box"] = boxes[role]
            payload[f"{role}_bits"] = bits[role]
        # Chunks overlap by one frame; drop the duplicate first frame after chunk 0.
        if chunk_index > 0:
            payload = {k: v[1:] for k, v in payload.items()}
        np.savez_compressed(chunk_path, **cast(dict[str, Any], payload))
        coverage = {r: float(np.mean(~np.isnan(boxes[r][:, 0]))) for r in roles}
        elapsed = time.monotonic() - started
        done = chunk_stop - start
        rate = done / max(elapsed, 1e-6)
        log(
            f"masks: chunk {chunk_index} frames {chunk_start}-{chunk_stop - 1} "
            f"coverage {', '.join(f'{r}={c:.0%}' for r, c in coverage.items())} "
            f"({rate:.1f} fps, eta {(stop - start - done) / max(rate, 1e-6):.0f}s)"
        )
        chunk_index += 1
        if chunk_stop >= stop:
            break
        chunk_start = chunk_stop - 1


# --------------------------------------------------------------------------- parallel chunks
#
# Each chunk is seeded by its own colour-confirmed detection of both boxers, so chunks track
# independently and run side by side in several SAM 2 processes. A chunk without a clear colour
# difference continues from the previous chunk's last mask afterwards, in order. Boundaries are
# checked for red/blue swaps.

_WORKER: dict[str, Any] = {}


def auto_mask_workers() -> int:
    env = os.environ.get("VARBOX_MASK_WORKERS")
    if env:
        return max(1, int(env))
    return max(1, min(6, (os.cpu_count() or 2) // 3))


def _init_worker(config: dict[str, Any]) -> None:
    from sam2.build_sam import build_sam2_video_predictor

    cfg = MaskConfig(**config)
    _WORKER["config"] = cfg
    _WORKER["predictor"] = build_sam2_video_predictor(
        cfg.model_config,
        cfg.checkpoint,
        device=cfg.device,
        hydra_overrides_extra=["++model.non_overlap_masks=true"],
    )


def _chunk_job(task: dict[str, Any]) -> dict[str, Any]:
    """Track one chunk in this worker's SAM 2. Prompts: seeds, detect (find both boxers), carry."""
    import torch

    from boxing_analytics.mesh4d.seeding import detect_roles

    cfg: MaskConfig = _WORKER["config"]
    predictor = _WORKER["predictor"]
    video = VideoInfo(**task["video"])
    c_start, c_stop = int(task["start"]), int(task["stop"])
    roles: list[str] = task["roles"]
    scale = float(task["scale"])
    shape = tuple(task["mask_shape"])
    started = time.monotonic()
    prompts: dict[str, tuple[str, int, Any]] = {}
    separation = None
    if task["prompt"] == "seeds":
        for role, seed in task["seeds"].items():
            local = max(0, min(c_stop - c_start - 1, int(seed["frame_index"]) - c_start))
            prompts[role] = ("box", local, np.asarray(seed["box"], dtype=np.float32) * scale)
    elif task["prompt"] == "detect":
        horizon = min(c_stop, c_start + int(2 * video.fps))
        step = max(1, int(video.fps // 5))
        for index, frame in iter_frames(video.path, c_start, horizon, step):
            found = detect_roles(frame, index, model_path=cfg.yolo_model, device=cfg.device)
            if found:
                separation = float(found["red"].scores.get("separation", 0.0))
                for role, seed in found.items():
                    prompts[role] = (
                        "box",
                        index - c_start,
                        np.asarray(seed.box, dtype=np.float32) * scale,
                    )
                break
        if not prompts:
            return {"index": task["index"], "status": "dependent"}
    else:  # carry: the previous chunk's last mask for each role
        with np.load(task["carry_from"]) as prev:
            for role in roles:
                boxes = prev[f"{role}_box"]
                good = np.flatnonzero(~np.isnan(boxes[:, 0]))
                if good.size:
                    prompts[role] = ("mask", 0, unpack(prev[f"{role}_bits"][good[-1]], shape))
    obj_ids = {role: i + 1 for i, role in enumerate(roles)}
    id_roles = {v: k for k, v in obj_ids.items()}
    n = c_stop - c_start
    boxes_out = {r: np.full((n, 4), np.nan, dtype=np.float32) for r in roles}
    bits_out = {r: np.zeros((n, (shape[0] * shape[1] + 7) // 8), dtype=np.uint8) for r in roles}
    min_area = cfg.min_area_frac * shape[0] * shape[1]
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if cfg.device.startswith("cuda")
        else torch.autocast("cpu", enabled=False)
    )
    with tempfile.TemporaryDirectory(prefix="varbox_m4d_") as tmp:
        frames = _write_jpegs(video, c_start, c_stop, Path(tmp), scale)
        with torch.inference_mode(), autocast:
            state = predictor.init_state(
                video_path=tmp, offload_video_to_cpu=True, async_loading_frames=False
            )
            for role, (kind, local, value) in prompts.items():
                if kind == "box":
                    predictor.add_new_points_or_box(
                        state, frame_idx=local, obj_id=obj_ids[role], box=value
                    )
                else:
                    predictor.add_new_mask(state, frame_idx=local, obj_id=obj_ids[role], mask=value)

            def consume(stream: Any) -> None:
                for local, ids, logits in stream:
                    for k, obj in enumerate(ids):
                        mask = (logits[k] > 0.0).squeeze(0).float().cpu().numpy().astype(bool)
                        if mask.sum() < min_area:
                            continue
                        box = mask_to_box(mask)
                        if box is None:
                            continue
                        role = id_roles[int(obj)]
                        boxes_out[role][local] = np.asarray(box, dtype=np.float32) / scale
                        bits_out[role][local] = pack(mask)

            consume(predictor.propagate_in_video(state))
            first = min(local for _, local, _ in prompts.values())
            if first > 0:
                consume(predictor.propagate_in_video(state, start_frame_idx=first, reverse=True))
            predictor.reset_state(state)
    payload: dict[str, NDArray] = {"frames": np.asarray(frames, dtype=np.int64)}
    for role in roles:
        payload[f"{role}_box"] = boxes_out[role][: len(frames)]
        payload[f"{role}_bits"] = bits_out[role][: len(frames)]
    np.savez_compressed(task["out"], **cast(dict[str, Any], payload))
    coverage = {r: float(np.mean(~np.isnan(boxes_out[r][: len(frames), 0]))) for r in roles}
    return {
        "index": task["index"],
        "status": "done",
        "coverage": coverage,
        "separation": separation,
        "prompt": task["prompt"],
        "seconds": time.monotonic() - started,
        "frames": len(frames),
    }


def _box_at(path: Path, role: str, last: bool) -> NDArray | None:
    with np.load(path) as z:
        boxes = z[f"{role}_box"]
    good = np.flatnonzero(~np.isnan(boxes[:, 0]))
    if not good.size:
        return None
    return boxes[good[-1] if last else good[0]]


def _fix_boundary_swaps(
    out_dir: Path, results: dict[int, dict[str, Any]], roles: list[str], log: LogFn
) -> None:
    """Where red and blue swap across a chunk boundary and the new chunk's colour evidence is weak,
    relabel the new chunk to follow the boxers' positions."""
    if not {"red", "blue"} <= set(roles):
        return
    chunks = sorted(out_dir.glob("chunk_*.npz"))
    for prev, cur in zip(chunks, chunks[1:], strict=False):
        idx = int(cur.stem.split("_")[1])
        er, eb = _box_at(prev, "red", last=True), _box_at(prev, "blue", last=True)
        sr, sb = _box_at(cur, "red", last=False), _box_at(cur, "blue", last=False)
        if er is None or eb is None or sr is None or sb is None:
            continue
        same = _box_iou(er, sr) + _box_iou(eb, sb)
        cross = _box_iou(er, sb) + _box_iou(eb, sr)
        if cross <= same + 0.3:
            continue
        sep = (results.get(idx) or {}).get("separation")
        if sep is not None and sep >= 0.25:
            log(f"masks: chunk {idx} disagrees with the previous chunk; colour evidence kept")
            continue
        with np.load(cur) as z:
            data = {k: z[k] for k in z.files}
        for key in ("box", "bits"):
            data[f"red_{key}"], data[f"blue_{key}"] = data[f"blue_{key}"], data[f"red_{key}"]
        np.savez_compressed(cur, **cast(dict[str, Any], data))
        log(f"masks: chunk {idx} relabelled to keep red and blue continuous")


def track_masks_parallel(
    video: VideoInfo,
    start: int,
    stop: int,
    seeds: dict[str, Seed],
    out_dir: Path,
    config: MaskConfig,
    log: LogFn = print,
) -> None:
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    out_dir.mkdir(parents=True, exist_ok=True)
    roles = list(seeds.keys())
    scale = min(1.0, config.max_side / float(max(video.width, video.height)))
    mask_shape = (int(round(video.height * scale)), int(round(video.width * scale)))
    workers = config.workers or auto_mask_workers()
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "roles": roles,
                "mask_shape": list(mask_shape),
                "scale": scale,
                "start": start,
                "stop": stop,
                "config": config.__dict__,
                "seeds": {r: s.to_dict() for r, s in seeds.items()},
                "parallel_workers": workers,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    spans = [
        (i, s, min(stop, s + config.chunk_frames))
        for i, s in enumerate(range(start, stop, config.chunk_frames))
    ]
    base = {
        "video": {k: getattr(video, k) for k in ("path", "width", "height", "fps", "frame_count")},
        "roles": roles,
        "scale": scale,
        "mask_shape": list(mask_shape),
    }
    seed_payload = {r: s.to_dict() for r, s in seeds.items()}
    results: dict[int, dict[str, Any]] = {}
    started = time.monotonic()
    log(f"masks: {len(spans)} chunks on {workers} SAM 2 workers")
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=ctx, initializer=_init_worker, initargs=(config.__dict__,)
    ) as pool:
        futures = {}
        for i, c_start, c_stop in spans:
            out = out_dir / f"chunk_{i:04d}.npz"
            if out.exists():
                results[i] = {"index": i, "status": "done", "cached": True}
                continue
            task = {**base, "index": i, "start": c_start, "stop": c_stop, "out": str(out)}
            (
                task.update(prompt="seeds", seeds=seed_payload)
                if i == 0
                else task.update(prompt="detect")
            )
            futures[i] = pool.submit(_chunk_job, task)
        for i, future in sorted(futures.items()):
            results[i] = future.result()
            r = results[i]
            if r["status"] == "done":
                cov = ", ".join(f"{k}={v:.0%}" for k, v in r["coverage"].items())
                log(f"masks: chunk {i} ({r['prompt']}) coverage {cov} in {r['seconds']:.0f}s")
        # Chunks without a clear colour difference continue from their predecessor, in order.
        for i, c_start, c_stop in spans:
            if results.get(i, {}).get("status") != "dependent":
                continue
            task = {
                **base,
                "index": i,
                "start": c_start,
                "stop": c_stop,
                "out": str(out_dir / f"chunk_{i:04d}.npz"),
                "prompt": "carry",
                "carry_from": str(out_dir / f"chunk_{i - 1:04d}.npz"),
            }
            results[i] = pool.submit(_chunk_job, task).result()
            cov = ", ".join(f"{k}={v:.0%}" for k, v in results[i]["coverage"].items())
            log(f"masks: chunk {i} (continued from previous) coverage {cov}")
    _fix_boundary_swaps(out_dir, results, roles, log)
    secs = time.monotonic() - started
    log(f"masks: {stop - start} frames in {secs:.0f}s ({(stop - start) / max(secs, 1e-6):.1f} fps)")
