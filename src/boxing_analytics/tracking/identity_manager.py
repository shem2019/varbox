"""Identity manager using two-hypothesis Viterbi smoothing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from boxing_analytics.tracking.identity_hmm import (
    IdentityHMMConfig,
    TrackObservation,
    TwoFighterIdentityHMM,
)
from boxing_analytics.tracking.tracklet_stitcher import (
    TrackletStitcherConfig,
    TwoFighterTrackletStitcher,
)

FloatArray = NDArray[np.float32]
FrameArray = NDArray[np.uint8]


@dataclass(slots=True)
class TrackState:
    track_id: int
    center: tuple[float, float]
    velocity: tuple[float, float]
    embedding: FloatArray
    pose_signature: FloatArray
    seen_count: int
    last_seen: int
    avg_area: float


@dataclass(frozen=True, slots=True)
class DetectionState:
    track_id: int
    box: tuple[int, int, int, int]
    center: tuple[float, float]
    diag: float
    area: float
    embedding: FloatArray
    pose_signature: FloatArray
    mean_visibility: float
    det_conf: float


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class IdentityManager:
    """Assign fighter roles with object permanence and two-hypothesis smoothing."""

    def __init__(
        self,
        max_missing_frames: int = 60,
        embedding_alpha: float = 0.18,
        min_assignment_score: float = 0.28,
        viterbi_window: int = 25,
        switch_penalty: float = 3.0,
        base_switch_penalty: float = 4.0,
        occlusion_switch_boost: float = 8.0,
        w_motion: float = 1.2,
        w_appear: float = 0.5,
        w_pose: float = 0.3,
        sigma_motion_px: float = 45.0,
        iou_occlusion_full: float = 0.25,
        occlusion_app_suppress: float = 0.9,
        occlusion_update_max: float = 0.15,  # Deprecated.
        update_max_occlusion: float = 0.08,
        min_det_conf: float = 0.4,
        min_kp_vis: float = 0.60,
        max_coast_frames: int = 60,
        miss_penalty: float = 8.0,
        flow_max_frames: int = 12,
        commit_p_threshold: float = 0.98,
        commit_seconds: float = 1.0,
        commit_max_occlusion: float = 0.10,
        stitch_max_dist_px: float = 80.0,
        stitch_embed_max: float = 0.35,
        stitch_w_motion: float = 1.0,
        stitch_w_appear: float = 0.6,
        stitch_w_pose: float = 0.2,
        # Deprecated knobs kept for compatibility.
        switch_margin: float | None = None,
        min_frames_before_switch: int | None = None,
        clinch_iou_freeze_threshold: float | None = None,
        clinch_freeze_frames: int | None = None,
        lock_role_orientation: bool = False,
    ) -> None:
        self.max_missing_frames = max_missing_frames
        self.embedding_alpha = embedding_alpha
        self.min_assignment_score = min_assignment_score

        # Deprecated compatibility mappings.
        if switch_margin is not None:
            base_switch_penalty = max(base_switch_penalty, 1.0 + 10.0 * float(switch_margin))
        if min_frames_before_switch is not None:
            viterbi_window = max(viterbi_window, int(min_frames_before_switch) * 2)
        if clinch_iou_freeze_threshold is not None:
            iou_occlusion_full = min(iou_occlusion_full, float(clinch_iou_freeze_threshold))
        if clinch_freeze_frames is not None:
            max_coast_frames = max(max_coast_frames, int(clinch_freeze_frames))
        update_max_occlusion = min(update_max_occlusion, occlusion_update_max)

        self._frame_idx = 0
        self.lock_role_orientation = bool(lock_role_orientation)
        self._tracks: dict[int, TrackState] = {}
        self._role_to_id: dict[str, int | None] = {"RED": None, "BLUE": None, "REF": None}
        self._id_to_role: dict[int, str] = {}
        self._live_role_to_id: dict[str, int | None] = {"RED": None, "BLUE": None, "REF": None}
        self._live_id_to_role: dict[int, str] = {}
        self._role_missing: dict[str, int] = {"RED": 0, "BLUE": 0}
        self._role_confidence: dict[str, float] = {"RED": 0.0, "BLUE": 0.0}
        self._swap_log: list[dict[str, Any]] = []
        self._role_change_log: list[dict[str, Any]] = []
        self._hypothesis_log: list[dict[str, Any]] = []
        self._stitcher = TwoFighterTrackletStitcher(
            config=TrackletStitcherConfig(
                stitch_max_dist_px=stitch_max_dist_px,
                stitch_embed_max=stitch_embed_max,
                stitch_w_motion=stitch_w_motion,
                stitch_w_appear=stitch_w_appear,
                stitch_w_pose=stitch_w_pose,
                update_max_occlusion=update_max_occlusion,
                min_det_conf=min_det_conf,
                min_kp_vis=min_kp_vis,
                occlusion_iou_full=iou_occlusion_full,
            )
        )

        self._hmm = TwoFighterIdentityHMM(
            config=IdentityHMMConfig(
                viterbi_window=viterbi_window,
                switch_penalty=switch_penalty,
                base_switch_penalty=base_switch_penalty,
                occlusion_switch_boost=occlusion_switch_boost,
                w_motion=w_motion,
                w_appear=w_appear,
                w_pose=w_pose,
                sigma_motion_px=sigma_motion_px,
                iou_occlusion_full=iou_occlusion_full,
                occlusion_app_suppress=occlusion_app_suppress,
                update_max_occlusion=update_max_occlusion,
                min_det_conf=min_det_conf,
                min_kp_vis=min_kp_vis,
                max_coast_frames=max_coast_frames,
                miss_penalty=miss_penalty,
                flow_max_frames=flow_max_frames,
                commit_p_threshold=commit_p_threshold,
                commit_seconds=commit_seconds,
                commit_max_occlusion=commit_max_occlusion,
                commit_lag=max(1, min(6, viterbi_window // 3)),
            ),
            logger=self._append_hypothesis_change_log,
        )

    @staticmethod
    def _crop(frame: FrameArray, box: tuple[int, int, int, int]) -> FrameArray:
        x1, y1, x2, y2 = box
        h, w = frame.shape[:2]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            return frame[0:0, 0:0]
        return frame[y1:y2, x1:x2]

    @staticmethod
    def _compute_embedding(crop: FrameArray) -> FloatArray:
        if crop.size == 0:
            return np.zeros((32,), dtype=np.float32)
        patch = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        hist_ang = cv2.calcHist([ang.astype(np.float32)], [0], None, [16], [0, 360]).flatten()
        hist_int = cv2.calcHist([gray], [0], None, [16], [0, 256]).flatten()
        vec = np.concatenate([hist_ang, hist_int]).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-6:
            return np.zeros((32,), dtype=np.float32)
        return cast(FloatArray, vec / norm)

    @staticmethod
    def _center(box: tuple[int, int, int, int]) -> tuple[float, float]:
        x1, y1, x2, y2 = box
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    @staticmethod
    def _as_float(value: object, default: float = 0.0) -> float:
        if isinstance(value, int | float | np.integer | np.floating):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return default
        return default

    @staticmethod
    def _diag(box: tuple[int, int, int, int]) -> float:
        x1, y1, x2, y2 = box
        return math.hypot(max(1, x2 - x1), max(1, y2 - y1))

    @staticmethod
    def _pose_signature(
        keypoints: dict[int, object],
    ) -> tuple[FloatArray, float]:
        required = [0, 11, 12, 23, 24, 13, 14, 15, 16]
        values: list[float] = []
        vis_values: list[float] = []
        anchor = None
        if 0 in keypoints:
            raw = keypoints[0]
            if isinstance(raw, tuple | list) and len(raw) >= 2:
                anchor = np.array([float(raw[0]), float(raw[1])], dtype=np.float32)
        shoulder_dist = 1.0
        if 11 in keypoints and 12 in keypoints:
            left_shoulder = keypoints[11]
            right_shoulder = keypoints[12]
            if (
                isinstance(left_shoulder, tuple | list)
                and isinstance(right_shoulder, tuple | list)
                and len(left_shoulder) >= 2
                and len(right_shoulder) >= 2
            ):
                shoulder_dist = max(
                    1.0,
                    float(
                        np.linalg.norm(
                            np.array(
                                [float(left_shoulder[0]), float(left_shoulder[1])], dtype=np.float32
                            )
                            - np.array(
                                [float(right_shoulder[0]), float(right_shoulder[1])],
                                dtype=np.float32,
                            )
                        )
                    ),
                )

        for idx in required:
            raw = keypoints.get(idx)
            if isinstance(raw, tuple | list) and len(raw) >= 2:
                x = float(raw[0])
                y = float(raw[1])
                conf = float(raw[2]) if len(raw) >= 3 else 1.0
                conf = _clamp(conf, 0.0, 1.0)
                vis_values.append(conf)
                if anchor is None:
                    values.extend([0.0, 0.0])
                else:
                    values.extend(
                        [
                            (x - float(anchor[0])) / shoulder_dist,
                            (y - float(anchor[1])) / shoulder_dist,
                        ]
                    )
            else:
                vis_values.append(0.0)
                values.extend([0.0, 0.0])

        signature = np.asarray(values, dtype=np.float32)
        norm = float(np.linalg.norm(signature))
        if norm > 1e-6:
            signature = signature / norm
        mean_visibility = float(np.mean(vis_values)) if vis_values else 0.0
        return signature, _clamp(mean_visibility, 0.0, 1.0)

    def _build_detections(
        self,
        frame: FrameArray,
        poses: dict[int, dict[str, object]],
    ) -> list[DetectionState]:
        detections: list[DetectionState] = []
        for track_id, payload in poses.items():
            raw_box = payload.get("box")
            if not isinstance(raw_box, tuple) or len(raw_box) != 4:
                continue
            box = (
                int(raw_box[0]),
                int(raw_box[1]),
                int(raw_box[2]),
                int(raw_box[3]),
            )
            center = self._center(box)
            diag = self._diag(box)
            area = float(max(1, (box[2] - box[0]) * (box[3] - box[1])))
            embedding = self._compute_embedding(self._crop(frame, box))
            keypoints_raw = payload.get("keypoints")
            keypoints = keypoints_raw if isinstance(keypoints_raw, dict) else {}
            pose_sig, mean_vis = self._pose_signature(keypoints)
            det_conf = self._as_float(payload.get("det_conf"), 1.0)
            detections.append(
                DetectionState(
                    track_id=int(track_id),
                    box=box,
                    center=center,
                    diag=diag,
                    area=area,
                    embedding=embedding,
                    pose_signature=pose_sig,
                    mean_visibility=mean_vis,
                    det_conf=_clamp(det_conf, 0.0, 1.0),
                )
            )
        return detections

    def _update_track_memory(self, detections: list[DetectionState], frame_idx: int) -> None:
        for det in detections:
            prev = self._tracks.get(det.track_id)
            if prev is None:
                self._tracks[det.track_id] = TrackState(
                    track_id=det.track_id,
                    center=det.center,
                    velocity=(0.0, 0.0),
                    embedding=det.embedding,
                    pose_signature=det.pose_signature,
                    seen_count=1,
                    last_seen=frame_idx,
                    avg_area=det.area,
                )
                continue

            dt = max(1, frame_idx - prev.last_seen)
            vx = (det.center[0] - prev.center[0]) / dt
            vy = (det.center[1] - prev.center[1]) / dt
            self._tracks[det.track_id] = TrackState(
                track_id=det.track_id,
                center=det.center,
                velocity=(0.78 * prev.velocity[0] + 0.22 * vx, 0.78 * prev.velocity[1] + 0.22 * vy),
                embedding=(
                    (1 - self.embedding_alpha) * prev.embedding
                    + self.embedding_alpha * det.embedding
                ).astype(np.float32),
                pose_signature=(
                    (1 - self.embedding_alpha) * prev.pose_signature
                    + self.embedding_alpha * det.pose_signature
                ).astype(np.float32),
                seen_count=prev.seen_count + 1,
                last_seen=frame_idx,
                avg_area=0.82 * prev.avg_area + 0.18 * det.area,
            )

    def _seen_count(self, track_id: int) -> int:
        state = self._tracks.get(track_id)
        return 0 if state is None else state.seen_count

    def _pair_score(self, det: DetectionState) -> float:
        area_term = math.log(max(1.0, det.area))
        persistence = min(1.0, self._seen_count(det.track_id) / 50.0)
        pred_red = self._hmm.role_predicted_center("RED")
        pred_blue = self._hmm.role_predicted_center("BLUE")
        motion_term = 0.0
        for pred in (pred_red, pred_blue):
            if pred is None:
                continue
            dist = math.hypot(det.center[0] - pred[0], det.center[1] - pred[1])
            motion_term = max(motion_term, math.exp(-((dist / max(40.0, 1.2 * det.diag)) ** 2)))
        return 0.52 * area_term + 0.28 * persistence + 0.20 * motion_term

    def _select_fighter_pair(
        self,
        detections: list[DetectionState],
    ) -> tuple[DetectionState | None, DetectionState | None]:
        if not detections:
            return None, None
        det_by_id = {det.track_id: det for det in detections}
        selected: list[DetectionState] = []
        for role in ("RED", "BLUE"):
            role_id = self._role_to_id.get(role)
            if role_id is not None and role_id in det_by_id:
                selected.append(det_by_id[role_id])

        if len(selected) < 2:
            remaining = [
                det for det in detections if det.track_id not in {d.track_id for d in selected}
            ]
            remaining.sort(key=self._pair_score, reverse=True)
            for det in remaining:
                selected.append(det)
                if len(selected) == 2:
                    break

        if len(selected) == 1:
            return selected[0], None
        if len(selected) >= 2:
            selected = sorted(selected[:2], key=lambda det: det.track_id)
            return selected[0], selected[1]
        return None, None

    def _referee_candidate(
        self,
        detections: list[DetectionState],
        assigned_ids: set[int],
    ) -> int | None:
        remaining = [d for d in detections if d.track_id not in assigned_ids]
        if not remaining:
            return None
        red_id = self._role_to_id.get("RED")
        blue_id = self._role_to_id.get("BLUE")
        red_det = next((d for d in detections if d.track_id == red_id), None)
        blue_det = next((d for d in detections if d.track_id == blue_id), None)
        if red_det is None or blue_det is None:
            remaining.sort(key=lambda det: (det.area, self._seen_count(det.track_id)))
            return remaining[0].track_id
        mid = (
            (red_det.center[0] + blue_det.center[0]) * 0.5,
            (red_det.center[1] + blue_det.center[1]) * 0.5,
        )
        fighter_area = max(red_det.area, blue_det.area)
        best_track = None
        best_score = float("inf")
        for det in remaining:
            between_dist = math.hypot(det.center[0] - mid[0], det.center[1] - mid[1])
            area_penalty = 0.0 if det.area <= 1.25 * fighter_area else 35.0
            score = between_dist + area_penalty
            if score < best_score:
                best_score = score
                best_track = det.track_id
        return best_track

    def _append_hypothesis_change_log(self, message: str) -> None:
        self._swap_log.append(
            {
                "frame": self._frame_idx,
                "type": "hypothesis_change",
                "message": message,
            }
        )

    def update(
        self,
        frame: FrameArray,
        poses: dict[int, dict[str, object]],
        frame_idx: int | None = None,
        timestamp_s: float | None = None,
    ) -> None:
        self._frame_idx = self._frame_idx + 1 if frame_idx is None else int(frame_idx)
        ts = float(timestamp_s) if timestamp_s is not None else float(self._frame_idx) / 30.0

        detections = self._build_detections(frame, poses)
        self._update_track_memory(detections, self._frame_idx)
        prev_red = self._role_to_id.get("RED")
        prev_blue = self._role_to_id.get("BLUE")

        raw_observations = [
            TrackObservation(
                track_id=det.track_id,
                bbox=det.box,
                center=det.center,
                det_conf=det.det_conf,
                timestamp_s=ts,
                appearance_embed=det.embedding,
                pose_embed=det.pose_signature,
                mean_visibility=det.mean_visibility,
            )
            for det in detections
        ]
        obs_a, obs_b, stitch_debug = self._stitcher.update(ts, raw_observations)

        gray = np.asarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), dtype=np.uint8)
        result = self._hmm.update(
            timestamp_s=ts,
            track_a_obs=obs_a,
            track_b_obs=obs_b,
            frame_gray=gray,
            external_debug={
                "churn_events_last_2s": int(stitch_debug.get("churn_events", {}).get("last_2s", 0)),
                "bank_updates_last_2s": int(stitch_debug.get("bank_updates_last_2s", 0)),
            },
        )
        red_raw = result.get("RED")
        blue_raw = result.get("BLUE")
        new_red = int(red_raw) if isinstance(red_raw, int | np.integer) else None
        new_blue = int(blue_raw) if isinstance(blue_raw, int | np.integer) else None

        # Keep orientation stable after initial lock if enabled.
        if (
            self.lock_role_orientation
            and prev_red is not None
            and prev_blue is not None
            and new_red is not None
            and new_blue is not None
            and (new_red != prev_red or new_blue != prev_blue)
        ):
            new_red = prev_red
            new_blue = prev_blue

        confidence = self._as_float(result.get("confidence"), 0.0)
        debug_raw = result.get("debug")
        debug: dict[str, object] = dict(debug_raw) if isinstance(debug_raw, dict) else {}
        if debug:
            debug["stitcher"] = stitch_debug
            self._hypothesis_log.append(debug)
            if len(self._hypothesis_log) > 300:
                self._hypothesis_log = self._hypothesis_log[-300:]

        for role, role_id in (("RED", new_red), ("BLUE", new_blue)):
            if role_id is None:
                if (
                    self._role_missing[role] < self.max_missing_frames
                    and self._role_to_id[role] is not None
                ):
                    self._role_missing[role] += 1
                else:
                    self._role_to_id[role] = None
                continue
            self._role_missing[role] = 0
            self._role_to_id[role] = role_id
            self._role_confidence[role] = confidence

        self._live_role_to_id["RED"] = None
        self._live_role_to_id["BLUE"] = None
        if self._hmm._current_state == 0:
            self._live_role_to_id["RED"] = obs_a.track_id if obs_a is not None else None
            self._live_role_to_id["BLUE"] = obs_b.track_id if obs_b is not None else None
        else:
            self._live_role_to_id["RED"] = obs_b.track_id if obs_b is not None else None
            self._live_role_to_id["BLUE"] = obs_a.track_id if obs_a is not None else None

        if (
            prev_red is not None
            and prev_blue is not None
            and self._role_to_id["RED"] == prev_blue
            and self._role_to_id["BLUE"] == prev_red
        ):
            self._swap_log.append(
                {
                    "frame": self._frame_idx,
                    "type": "role_swap",
                    "timestamp_s": round(ts, 3),
                    "prev_red": prev_red,
                    "prev_blue": prev_blue,
                    "new_red": self._role_to_id["RED"],
                    "new_blue": self._role_to_id["BLUE"],
                    "debug": debug,
                }
            )

        for role, prev in (("RED", prev_red), ("BLUE", prev_blue)):
            curr = self._role_to_id[role]
            if prev is not None and curr is not None and prev != curr:
                self._role_change_log.append(
                    {
                        "frame": self._frame_idx,
                        "timestamp_s": round(ts, 3),
                        "role": role,
                        "prev_id": prev,
                        "new_id": curr,
                        "confidence": round(self._role_confidence[role], 3),
                        "debug": debug,
                    }
                )

        assigned_ids = {
            role_id
            for role_id in (self._live_role_to_id["RED"], self._live_role_to_id["BLUE"])
            if role_id is not None
        }
        self._live_role_to_id["REF"] = self._referee_candidate(detections, assigned_ids)
        self._role_to_id["REF"] = self._live_role_to_id["REF"]

        self._id_to_role = {}
        for role, role_id in self._role_to_id.items():
            if role_id is not None:
                self._id_to_role[role_id] = role
        self._live_id_to_role = {}
        for role, role_id in self._live_role_to_id.items():
            if role_id is not None:
                self._live_id_to_role[role_id] = role

    def id_for_role(self, role: str) -> int | None:
        return self._role_to_id.get(role.upper())

    def live_id_for_role(self, role: str) -> int | None:
        return self._live_role_to_id.get(role.upper())

    def role_for_id(self, track_id: int) -> str | None:
        role = self._live_id_to_role.get(track_id)
        if role is not None:
            return role
        return self._id_to_role.get(track_id)

    def role_status(self) -> dict[str, int | None]:
        return dict(self._role_to_id)

    def live_role_status(self) -> dict[str, int | None]:
        return dict(self._live_role_to_id)

    def role_confidence(self) -> dict[str, float]:
        return {
            "RED": round(self._role_confidence["RED"], 3),
            "BLUE": round(self._role_confidence["BLUE"], 3),
        }

    def tracking_stats(self) -> dict[str, dict[str, int | float]]:
        last_debug = self._hypothesis_log[-1] if self._hypothesis_log else {}
        return {
            "RED": {
                "missing_frames": self._role_missing["RED"],
                "confidence": round(self._role_confidence["RED"], 3),
            },
            "BLUE": {
                "missing_frames": self._role_missing["BLUE"],
                "confidence": round(self._role_confidence["BLUE"], 3),
            },
            "GLOBAL": {
                "swap_events": len(
                    [row for row in self._swap_log if row.get("type") == "role_swap"]
                ),
                "role_changes": len(self._role_change_log),
                "tracked_ids": len(self._tracks),
                "last_logp_h0": float(last_debug.get("logp_h0", 0.0) or 0.0),
                "last_logp_h1": float(last_debug.get("logp_h1", 0.0) or 0.0),
                "last_delta": float(last_debug.get("delta", 0.0) or 0.0),
                "last_occlusion_score": float(last_debug.get("occlusion_score", 0.0) or 0.0),
                "status_uncertain": int(str(last_debug.get("status", "stable")) == "uncertain"),
            },
        }

    def swap_log(self) -> list[dict[str, Any]]:
        return list(self._swap_log)

    def role_change_log(self) -> list[dict[str, Any]]:
        return list(self._role_change_log)

    def hypothesis_log(self) -> list[dict[str, Any]]:
        return list(self._hypothesis_log)
