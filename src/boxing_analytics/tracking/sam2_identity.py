"""Chunked SAM 2.1 mask tracking for stable fighter-role continuity on Apple MPS."""

from __future__ import annotations

import bisect
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import types
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2

ProgressFn = Callable[[str, int | None], None]
StopFn = Callable[[], bool]
Box = tuple[int, int, int, int]


def _force_float32_sam2_memory(predictor: Any) -> None:
    """Keep SAM 2 video memories compatible with float32 non-CUDA model weights.

    Upstream SAM 2 always stores video-memory features as bfloat16. Its documented
    CUDA inference path runs under bfloat16 autocast, but VAR Box deliberately does
    not enable MPS autocast. Feeding those bfloat16 memories into float32 linear
    layers can abort the process inside Metal instead of raising a Python exception.
    """

    import torch
    from sam2.utils.misc import fill_holes_in_mask_scores

    def run_single_frame_float32(
        self: Any,
        inference_state: dict[str, Any],
        output_dict: dict[str, Any],
        frame_idx: int,
        batch_size: int,
        is_init_cond_frame: bool,
        point_inputs: Any,
        mask_inputs: Any,
        reverse: bool,
        run_mem_encoder: bool,
        prev_sam_mask_logits: Any = None,
    ) -> tuple[dict[str, Any], Any]:
        (
            _,
            _,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self._get_image_feature(inference_state, frame_idx, batch_size)
        current_output = self.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            output_dict=output_dict,
            num_frames=inference_state["num_frames"],
            track_in_reverse=reverse,
            run_mem_encoder=run_mem_encoder,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )
        storage_device = inference_state["storage_device"]
        memory = current_output["maskmem_features"]
        if memory is not None:
            memory = memory.to(dtype=torch.float32, device=storage_device, non_blocking=True)
        prediction_gpu = current_output["pred_masks"]
        if self.fill_hole_area > 0:
            prediction_gpu = fill_holes_in_mask_scores(prediction_gpu, self.fill_hole_area)
        prediction = prediction_gpu.to(storage_device, non_blocking=True)
        position_encoding = self._get_maskmem_pos_enc(inference_state, current_output)
        compact_output = {
            "maskmem_features": memory,
            "maskmem_pos_enc": position_encoding,
            "pred_masks": prediction,
            "obj_ptr": current_output["obj_ptr"],
            "object_score_logits": current_output["object_score_logits"],
        }
        return compact_output, prediction_gpu

    def run_memory_encoder_float32(
        self: Any,
        inference_state: dict[str, Any],
        frame_idx: int,
        batch_size: int,
        high_res_masks: Any,
        object_score_logits: Any,
        is_mask_from_pts: bool,
    ) -> tuple[Any, Any]:
        _, _, current_vision_feats, _, feat_sizes = self._get_image_feature(
            inference_state, frame_idx, batch_size
        )
        memory, position_encoding = self._encode_new_memory(
            current_vision_feats=current_vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=high_res_masks,
            object_score_logits=object_score_logits,
            is_mask_from_pts=is_mask_from_pts,
        )
        memory = memory.to(
            dtype=torch.float32,
            device=inference_state["storage_device"],
            non_blocking=True,
        )
        position_encoding = self._get_maskmem_pos_enc(
            inference_state,
            {"maskmem_pos_enc": position_encoding},
        )
        return memory, position_encoding

    predictor._run_single_frame_inference = types.MethodType(
        run_single_frame_float32,
        predictor,
    )
    predictor._run_memory_encoder = types.MethodType(
        run_memory_encoder_float32,
        predictor,
    )


@dataclass(frozen=True, slots=True)
class SamRoleSample:
    frame_index: int
    role: str
    box: Box
    mask_area: int
    confidence: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _box_iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = float((ix2 - ix1) * (iy2 - iy1))
    area_a = float(max(1, (ax2 - ax1) * (ay2 - ay1)))
    area_b = float(max(1, (bx2 - bx1) * (by2 - by1)))
    return inter / max(1.0, area_a + area_b - inter)


def _center_distance(a: Box, b: Box) -> float:
    ac = ((a[0] + a[2]) * 0.5, (a[1] + a[3]) * 0.5)
    bc = ((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5)
    return float(((ac[0] - bc[0]) ** 2 + (ac[1] - bc[1]) ** 2) ** 0.5)


def _pose_box(row: dict[str, Any]) -> Box | None:
    raw = row.get("box")
    if not isinstance(raw, tuple | list) or len(raw) != 4:
        return None
    return (int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3]))


class Sam2FighterIdentityTrack:
    """Precomputed sampled SAM mask tracks matched to live YOLO pose detections."""

    def __init__(
        self,
        *,
        checkpoint_path: str,
        model_config: str = "configs/sam2.1/sam2.1_hiera_t.yaml",
        device: str = "mps",
        stride: int = 10,
        chunk_samples: int = 120,
        max_side: int = 768,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.model_config = model_config
        self.device = device
        self.stride = max(1, int(stride))
        self.chunk_samples = max(16, int(chunk_samples))
        self.max_side = max(320, int(max_side))
        self.model_image_size = min(1024, ((self.max_side + 15) // 16) * 16)
        self.model_sha256 = (
            _file_sha256(self.checkpoint_path) if self.checkpoint_path.is_file() else ""
        )
        self.samples: dict[str, list[SamRoleSample]] = {"RED": [], "BLUE": []}
        self._sample_frames: dict[str, list[int]] = {"RED": [], "BLUE": []}
        self._last_match_confidence: dict[str, float] = {"RED": 0.0, "BLUE": 0.0}
        self.status = "not_initialized"
        self.error = ""
        self.cache_path = ""
        self.mps_float32_memory = False
        self.float32_memory_compat = False
        self.mps_isolated_worker = False

    @staticmethod
    def _parse_seeds(payload: str) -> dict[str, dict[str, Any]]:
        try:
            raw = json.loads(payload)
        except json.JSONDecodeError:
            return {}
        if not isinstance(raw, dict):
            return {}
        return {
            role: dict(raw[role]) for role in ("RED", "BLUE") if isinstance(raw.get(role), dict)
        }

    def _cache_signature(self, video_path: str, seeds: dict[str, dict[str, Any]]) -> str:
        stat = os.stat(video_path)
        payload = {
            "video": str(Path(video_path).resolve()),
            "video_size": stat.st_size,
            "video_mtime_ns": stat.st_mtime_ns,
            "checkpoint_sha256": self.model_sha256,
            "config": self.model_config,
            "stride": self.stride,
            "chunk_samples": self.chunk_samples,
            "max_side": self.max_side,
            "seeds": seeds,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def _load_cache(self, cache_path: Path, signature: str) -> bool:
        if not cache_path.is_file():
            return False
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if payload.get("signature") != signature:
            return False
        role_rows = payload.get("samples", {})
        if not isinstance(role_rows, dict):
            return False
        loaded: dict[str, list[SamRoleSample]] = {"RED": [], "BLUE": []}
        for role in ("RED", "BLUE"):
            rows = role_rows.get(role, [])
            if not isinstance(rows, list):
                return False
            for row in rows:
                if not isinstance(row, dict):
                    continue
                raw_box = row.get("box")
                if not isinstance(raw_box, list | tuple) or len(raw_box) != 4:
                    continue
                loaded[role].append(
                    SamRoleSample(
                        frame_index=int(row["frame_index"]),
                        role=role,
                        box=(
                            int(raw_box[0]),
                            int(raw_box[1]),
                            int(raw_box[2]),
                            int(raw_box[3]),
                        ),
                        mask_area=int(row.get("mask_area", 0)),
                        confidence=float(row.get("confidence", 0.0)),
                    )
                )
        if not loaded["RED"] or not loaded["BLUE"]:
            return False
        self.samples = loaded
        self._refresh_frame_index()
        self.status = "cache_loaded"
        return True

    def _save_cache(self, cache_path: Path, signature: str) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "signature": signature,
            "tracker": self.diagnostics(),
            "samples": {
                role: [sample.to_dict() for sample in rows] for role, rows in self.samples.items()
            },
        }
        cache_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _refresh_frame_index(self) -> None:
        for role in ("RED", "BLUE"):
            self.samples[role] = sorted(self.samples[role], key=lambda row: row.frame_index)
            self._sample_frames[role] = [row.frame_index for row in self.samples[role]]

    def _extract_chunk_frames(
        self,
        capture: cv2.VideoCapture,
        frame_indices: list[int],
        output_dir: Path,
        source_width: int,
        source_height: int,
    ) -> tuple[float, float]:
        scale = min(1.0, self.max_side / max(source_width, source_height))
        out_width = max(2, int(round(source_width * scale)))
        out_height = max(2, int(round(source_height * scale)))
        wanted = {
            source_index: local_index for local_index, source_index in enumerate(frame_indices)
        }
        written: set[int] = set()
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_indices[0])
        decoder_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES) or frame_indices[0])
        while decoder_index <= frame_indices[-1]:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            local_index = wanted.get(decoder_index)
            if local_index is not None:
                if scale < 1.0:
                    frame = cv2.resize(
                        frame,
                        (out_width, out_height),
                        interpolation=cv2.INTER_AREA,
                    )
                if not cv2.imwrite(str(output_dir / f"{local_index:05d}.jpg"), frame):
                    raise RuntimeError(f"Could not write temporary SAM frame {decoder_index}")
                written.add(decoder_index)
            decoder_index += 1
        missing = sorted(set(frame_indices) - written)
        if missing:
            raise ValueError(f"Could not decode SAM frames: {missing[:5]}")
        return out_width / source_width, out_height / source_height

    @staticmethod
    def _seed_box(
        seed: dict[str, Any],
        source_width: int,
        source_height: int,
        scale_x: float,
        scale_y: float,
    ) -> Box:
        rel_box = seed.get("rel_box")
        if not isinstance(rel_box, list | tuple) or len(rel_box) != 4:
            raise ValueError("SAM manual seed requires rel_box with four values")
        return (
            int(round(float(rel_box[0]) * source_width * scale_x)),
            int(round(float(rel_box[1]) * source_height * scale_y)),
            int(round(float(rel_box[2]) * source_width * scale_x)),
            int(round(float(rel_box[3]) * source_height * scale_y)),
        )

    @staticmethod
    def _sample_from_logits(
        *,
        logits: Any,
        frame_index: int,
        role: str,
        scale_x: float,
        scale_y: float,
    ) -> SamRoleSample | None:
        import torch

        tensor = logits.detach().float().cpu().squeeze()
        mask = tensor > 0.0
        coordinates = torch.nonzero(mask, as_tuple=False)
        if coordinates.numel() == 0:
            return None
        y1 = int(coordinates[:, 0].min().item())
        y2 = int(coordinates[:, 0].max().item()) + 1
        x1 = int(coordinates[:, 1].min().item())
        x2 = int(coordinates[:, 1].max().item()) + 1
        confidence = float(torch.sigmoid(tensor[mask]).mean().item())
        return SamRoleSample(
            frame_index=frame_index,
            role=role,
            box=(
                int(round(x1 / max(scale_x, 1e-6))),
                int(round(y1 / max(scale_y, 1e-6))),
                int(round(x2 / max(scale_x, 1e-6))),
                int(round(y2 / max(scale_y, 1e-6))),
            ),
            mask_area=int(mask.sum().item()),
            confidence=confidence,
        )

    def precompute(
        self,
        *,
        video_path: str,
        manual_seeds_payload: str,
        cache_path: str,
        progress_cb: ProgressFn | None = None,
        should_stop: StopFn | None = None,
    ) -> bool:
        if not self.checkpoint_path.is_file():
            self.status = "unavailable"
            self.error = f"SAM checkpoint missing: {self.checkpoint_path}"
            return False
        seeds = self._parse_seeds(manual_seeds_payload)
        if set(seeds) != {"RED", "BLUE"}:
            self.status = "unavailable"
            self.error = "SAM identity requires manual RED and BLUE seeds"
            return False
        signature = self._cache_signature(video_path, seeds)
        cache = Path(cache_path)
        self.cache_path = str(cache.resolve())
        if self._load_cache(cache, signature):
            return True
        isolate_mps = os.getenv("VARBOX_SAM2_ISOLATE_MPS", "1").strip() != "0"
        inside_worker = os.getenv("VARBOX_SAM2_WORKER", "0").strip() == "1"
        if self.device == "mps" and isolate_mps and not inside_worker:
            return self._precompute_in_isolated_worker(
                video_path=video_path,
                manual_seeds_payload=manual_seeds_payload,
                cache_path=str(cache),
                signature=signature,
                progress_cb=progress_cb,
                should_stop=should_stop,
            )
        try:
            import torch
            from sam2.build_sam import build_sam2_video_predictor
        except (ImportError, ModuleNotFoundError) as exc:
            self.status = "unavailable"
            self.error = f"SAM 2 package unavailable: {exc}"
            return False

        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            self.status = "failed"
            self.error = f"Could not open video for SAM tracking: {video_path}"
            return False
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if total_frames <= 0 or source_width <= 0 or source_height <= 0:
            capture.release()
            self.status = "failed"
            self.error = "Invalid source video metadata for SAM tracking"
            return False

        predictor = build_sam2_video_predictor(
            self.model_config,
            str(self.checkpoint_path),
            device=self.device,
            hydra_overrides_extra=[
                f"++model.image_size={self.model_image_size}",
                (
                    "++model.memory_attention.layer.self_attention.feat_sizes="
                    f"[{self.model_image_size // 16},{self.model_image_size // 16}]"
                ),
                (
                    "++model.memory_attention.layer.cross_attention.feat_sizes="
                    f"[{self.model_image_size // 16},{self.model_image_size // 16}]"
                ),
            ],
            apply_postprocessing=False,
        )
        predictor.eval()
        if self.device.split(":", maxsplit=1)[0] != "cuda":
            predictor.float()
            _force_float32_sam2_memory(predictor)
            self.float32_memory_compat = True
            self.mps_float32_memory = self.device.split(":", maxsplit=1)[0] == "mps"
        seed_frames = {
            role: max(0, min(total_frames - 1, int(seed.get("frame_idx", 1)) - 1))
            for role, seed in seeds.items()
        }
        first_frame = min(seed_frames.values())
        sampled_indices = list(range(first_frame, total_frames, self.stride))
        for seed_frame in seed_frames.values():
            if seed_frame not in sampled_indices:
                bisect.insort(sampled_indices, seed_frame)
        if progress_cb is not None:
            progress_cb(
                f"SAM pre-analysis will track {len(sampled_indices)} sampled frames "
                f"before pose analysis (stride={self.stride})",
                0,
            )
        carry_boxes: dict[str, Box] = {}
        self.samples = {"RED": [], "BLUE": []}
        propagation_started = time.monotonic()
        try:
            for chunk_start in range(0, len(sampled_indices), self.chunk_samples):
                if should_stop is not None and should_stop():
                    self.status = "cancelled"
                    self.error = "SAM precomputation cancelled"
                    return False
                chunk_indices = sampled_indices[chunk_start : chunk_start + self.chunk_samples]
                with tempfile.TemporaryDirectory(prefix="varbox_sam2_chunk_") as temp_dir:
                    frames_dir = Path(temp_dir)
                    scale_x, scale_y = self._extract_chunk_frames(
                        capture,
                        chunk_indices,
                        frames_dir,
                        source_width,
                        source_height,
                    )
                    state = predictor.init_state(
                        video_path=str(frames_dir),
                        offload_video_to_cpu=True,
                        offload_state_to_cpu=True,
                        async_loading_frames=False,
                    )
                    with torch.inference_mode():
                        for object_id, role in enumerate(("RED", "BLUE"), start=1):
                            if chunk_start == 0:
                                source_seed_frame = seed_frames[role]
                                if source_seed_frame not in chunk_indices:
                                    continue
                                local_seed_frame = chunk_indices.index(source_seed_frame)
                                prompt_box = self._seed_box(
                                    seeds[role],
                                    source_width,
                                    source_height,
                                    scale_x,
                                    scale_y,
                                )
                            else:
                                if role not in carry_boxes:
                                    continue
                                local_seed_frame = 0
                                raw = carry_boxes[role]
                                prompt_box = (
                                    int(round(raw[0] * scale_x)),
                                    int(round(raw[1] * scale_y)),
                                    int(round(raw[2] * scale_x)),
                                    int(round(raw[3] * scale_y)),
                                )
                            predictor.add_new_points_or_box(
                                state,
                                frame_idx=local_seed_frame,
                                obj_id=object_id,
                                box=torch.tensor(
                                    prompt_box,
                                    dtype=torch.float32,
                                    device=self.device,
                                ),
                            )
                        for local_frame, object_ids, mask_logits in predictor.propagate_in_video(
                            state
                        ):
                            if should_stop is not None and should_stop():
                                self.status = "cancelled"
                                self.error = "SAM precomputation cancelled"
                                predictor.reset_state(state)
                                return False
                            source_frame = chunk_indices[int(local_frame)]
                            for output_index, object_id in enumerate(object_ids):
                                role = "RED" if int(object_id) == 1 else "BLUE"
                                sample = self._sample_from_logits(
                                    logits=mask_logits[output_index],
                                    frame_index=source_frame,
                                    role=role,
                                    scale_x=scale_x,
                                    scale_y=scale_y,
                                )
                                if sample is not None:
                                    self.samples[role].append(sample)
                                    carry_boxes[role] = sample.box
                            completed = min(
                                len(sampled_indices),
                                chunk_start + int(local_frame) + 1,
                            )
                            if progress_cb is not None and (
                                completed == 1
                                or completed % 5 == 0
                                or completed == len(sampled_indices)
                            ):
                                elapsed = max(0.001, time.monotonic() - propagation_started)
                                remaining = max(0, len(sampled_indices) - completed)
                                eta_seconds = int(round(remaining * elapsed / max(1, completed)))
                                eta_minutes, eta_remainder = divmod(eta_seconds, 60)
                                eta_text = (
                                    f"{eta_minutes}m {eta_remainder:02d}s"
                                    if eta_minutes
                                    else f"{eta_remainder}s"
                                )
                                percent = int(
                                    round(100.0 * completed / max(1, len(sampled_indices)))
                                )
                                progress_cb(
                                    f"SAM pre-analysis {completed}/{len(sampled_indices)} "
                                    f"sampled frames; pose analysis follows (ETA {eta_text})",
                                    percent,
                                )
                    predictor.reset_state(state)
                completed = min(len(sampled_indices), chunk_start + len(chunk_indices))
                percent = int(round(100.0 * completed / max(1, len(sampled_indices))))
                if progress_cb is not None:
                    progress_cb(
                        f"SAM 2.1 identity masks {completed}/{len(sampled_indices)} "
                        f"sampled frames (stride={self.stride})",
                        percent,
                    )
                if self.device == "mps" and hasattr(torch, "mps"):
                    torch.mps.empty_cache()
        except Exception as exc:
            self.status = "failed"
            self.error = str(exc)
            return False
        finally:
            capture.release()
        self._refresh_frame_index()
        if not self.samples["RED"] or not self.samples["BLUE"]:
            self.status = "failed"
            self.error = "SAM produced no usable RED/BLUE mask tracks"
            return False
        self.status = "ready"
        self._save_cache(cache, signature)
        return True

    def _precompute_in_isolated_worker(
        self,
        *,
        video_path: str,
        manual_seeds_payload: str,
        cache_path: str,
        signature: str,
        progress_cb: ProgressFn | None,
        should_stop: StopFn | None,
    ) -> bool:
        """Run Metal inference out of process so a native MPS abort is recoverable."""

        self.mps_isolated_worker = True
        if progress_cb is not None:
            progress_cb("Starting crash-contained SAM 2.1 MPS worker", 0)
        request = {
            "checkpoint_path": str(self.checkpoint_path),
            "model_config": self.model_config,
            "device": self.device,
            "stride": self.stride,
            "chunk_samples": self.chunk_samples,
            "max_side": self.max_side,
            "model_image_size": self.model_image_size,
            "video_path": video_path,
            "manual_seeds_payload": manual_seeds_payload,
            "cache_path": cache_path,
        }
        with tempfile.TemporaryDirectory(prefix="varbox_sam2_worker_") as temp_dir:
            temp_path = Path(temp_dir)
            request_path = temp_path / "request.json"
            result_path = temp_path / "result.json"
            progress_path = temp_path / "progress.json"
            log_path = temp_path / "worker.log"
            request["result_path"] = str(result_path)
            request["progress_path"] = str(progress_path)
            request_path.write_text(json.dumps(request), encoding="utf-8")

            environment = os.environ.copy()
            environment["VARBOX_SAM2_WORKER"] = "1"
            source_root = str(Path(__file__).resolve().parents[2])
            existing_pythonpath = environment.get("PYTHONPATH", "")
            environment["PYTHONPATH"] = (
                f"{source_root}{os.pathsep}{existing_pythonpath}"
                if existing_pythonpath
                else source_root
            )
            with log_path.open("w+", encoding="utf-8") as log:
                try:
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "boxing_analytics.tracking.sam2_worker",
                            str(request_path),
                        ],
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                except OSError as exc:
                    self.status = "failed"
                    self.error = (
                        f"Could not start isolated SAM MPS worker: {exc}. "
                        "VAR Box continued safely with HMM/ReID identity tracking."
                    )
                    return False
                last_progress = ""
                while process.poll() is None:
                    if should_stop is not None and should_stop():
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        self.status = "cancelled"
                        self.error = "SAM precomputation cancelled"
                        return False
                    try:
                        progress_payload = progress_path.read_text(encoding="utf-8")
                    except OSError:
                        progress_payload = ""
                    if progress_payload and progress_payload != last_progress:
                        last_progress = progress_payload
                        try:
                            progress_row = json.loads(progress_payload)
                            if progress_cb is not None:
                                progress_cb(
                                    str(progress_row.get("message", "SAM 2.1 MPS worker")),
                                    (
                                        int(progress_row["percent"])
                                        if progress_row.get("percent") is not None
                                        else None
                                    ),
                                )
                        except (json.JSONDecodeError, TypeError, ValueError):
                            pass
                    time.sleep(0.1)
                log.seek(0)
                worker_log = log.read()

            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                result = {}
            if process.returncode == 0 and bool(result.get("ok")):
                if self._load_cache(Path(cache_path), signature):
                    self.float32_memory_compat = True
                    self.mps_float32_memory = True
                    self.status = "ready"
                    return True
                self.status = "failed"
                self.error = "SAM worker completed without a readable identity cache"
                return False

            self.status = str(result.get("status", "failed"))
            worker_error = str(result.get("error", "")).strip()
            log_tail = worker_log.strip()[-1200:]
            detail = worker_error or log_tail or "no worker diagnostic was produced"
            self.error = (
                f"SAM MPS worker exited with code {process.returncode}: {detail}. "
                "VAR Box continued safely with HMM/ReID identity tracking. "
                "Set VARBOX_SAM2_DEVICE=cpu to retry SAM on CPU."
            )
            return False

    def box_for_role(self, role: str, frame_index: int) -> Box | None:
        role_key = role.upper()
        rows = self.samples.get(role_key, [])
        frames = self._sample_frames.get(role_key, [])
        if not rows:
            return None
        position = bisect.bisect_left(frames, frame_index)
        if position <= 0:
            return rows[0].box
        if position >= len(rows):
            return rows[-1].box
        left = rows[position - 1]
        right = rows[position]
        span = max(1, right.frame_index - left.frame_index)
        alpha = (frame_index - left.frame_index) / span
        return (
            int(round(left.box[0] * (1.0 - alpha) + right.box[0] * alpha)),
            int(round(left.box[1] * (1.0 - alpha) + right.box[1] * alpha)),
            int(round(left.box[2] * (1.0 - alpha) + right.box[2] * alpha)),
            int(round(left.box[3] * (1.0 - alpha) + right.box[3] * alpha)),
        )

    def role_ids_for_poses(
        self,
        frame_index: int,
        poses: dict[int, dict[str, Any]],
    ) -> dict[str, int | None]:
        candidates: list[tuple[float, str, int]] = []
        for role in ("RED", "BLUE"):
            sam_box = self.box_for_role(role, frame_index)
            if sam_box is None:
                continue
            sam_diag = max(
                1.0,
                float(((sam_box[2] - sam_box[0]) ** 2 + (sam_box[3] - sam_box[1]) ** 2) ** 0.5),
            )
            for track_id, row in poses.items():
                detection_box = _pose_box(row)
                if detection_box is None:
                    continue
                iou = _box_iou(sam_box, detection_box)
                distance = _center_distance(sam_box, detection_box) / sam_diag
                score = 1.5 * iou + max(0.0, 1.0 - distance)
                candidates.append((score, role, track_id))
        assigned_roles: set[str] = set()
        assigned_ids: set[int] = set()
        result: dict[str, int | None] = {"RED": None, "BLUE": None}
        self._last_match_confidence = {"RED": 0.0, "BLUE": 0.0}
        for score, role, track_id in sorted(candidates, reverse=True):
            if score < 0.30 or role in assigned_roles or track_id in assigned_ids:
                continue
            result[role] = track_id
            self._last_match_confidence[role] = min(0.98, max(0.0, score / 2.0))
            assigned_roles.add(role)
            assigned_ids.add(track_id)
        return result

    def match_confidence_for_role(self, role: str) -> float:
        return float(self._last_match_confidence.get(role.upper(), 0.0))

    def diagnostics(self) -> dict[str, object]:
        return {
            "backend": "sam2.1_video_segmentation",
            "status": self.status,
            "error": self.error,
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_sha256": self.model_sha256,
            "model_config": self.model_config,
            "device": self.device,
            "stride": self.stride,
            "chunk_samples": self.chunk_samples,
            "max_side": self.max_side,
            "mps_float32_memory": self.mps_float32_memory,
            "float32_memory_compat": self.float32_memory_compat,
            "mps_isolated_worker": self.mps_isolated_worker,
            "sample_count": {role: len(rows) for role, rows in self.samples.items()},
            "cache_path": self.cache_path,
            "offline": 1,
        }
