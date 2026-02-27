"""Two-hypothesis identity resolver with Viterbi smoothing for two fighters."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]
GrayFrame = NDArray[np.uint8]
LoggerFn = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class IdentityHMMConfig:
    viterbi_window: int = 25
    switch_penalty: float = 3.0  # Deprecated fallback.
    base_switch_penalty: float = 4.0
    occlusion_switch_boost: float = 8.0
    w_motion: float = 1.2
    w_appear: float = 0.5
    w_pose: float = 0.3
    sigma_motion_px: float = 45.0
    iou_occlusion_full: float = 0.25
    occlusion_app_suppress: float = 0.9
    update_max_occlusion: float = 0.08
    min_det_conf: float = 0.4
    min_kp_vis: float = 0.60
    max_coast_frames: int = 60
    miss_penalty: float = 8.0
    flow_max_frames: int = 12
    commit_p_threshold: float = 0.98
    commit_seconds: float = 1.0
    commit_max_occlusion: float = 0.10
    min_flip_interval_s: float = 5.0
    commit_lag: int = 3


@dataclass(frozen=True, slots=True)
class TrackObservation:
    track_id: int
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    det_conf: float
    timestamp_s: float
    appearance_embed: FloatArray | None = None
    pose_embed: FloatArray | None = None
    mean_visibility: float = 1.0
    keypoints: dict[int, tuple[float, float, float]] | None = None


@dataclass(slots=True)
class _RoleObjectFile:
    kf: cv2.KalmanFilter
    initialized: bool = False
    pred_center: tuple[float, float] | None = None
    embed_ema: FloatArray | None = None
    pose_ema: FloatArray | None = None
    last_seen_ts: float = 0.0
    current_track_id: int | None = None
    coast_frames: int = 0
    last_bbox: tuple[int, int, int, int] | None = None
    flow_points: FloatArray | None = None
    flow_active_frames: int = 0


@dataclass(frozen=True, slots=True)
class _Step:
    dp0: float
    dp1: float
    back0: int
    back1: int
    l0: float
    l1: float
    occlusion_score: float
    appearance_suppressed: int


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _cosine_distance(a: FloatArray | None, b: FloatArray | None) -> float:
    if a is None or b is None or a.size == 0 or b.size == 0 or a.shape != b.shape:
        return 1.0
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an <= 1e-6 or bn <= 1e-6:
        return 1.0
    cos = float(np.dot(a / an, b / bn))
    return _clamp(1.0 - cos, 0.0, 2.0)


def _l2_distance(a: FloatArray | None, b: FloatArray | None) -> float:
    if a is None or b is None or a.size == 0 or b.size == 0 or a.shape != b.shape:
        return 1.0
    return float(np.linalg.norm(a - b))


def _bbox_iou(
    box_a: tuple[int, int, int, int] | None,
    box_b: tuple[int, int, int, int] | None,
) -> float:
    if box_a is None or box_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = float(iw * ih)
    if inter <= 0.0:
        return 0.0
    area_a = float(max(1, (ax2 - ax1) * (ay2 - ay1)))
    area_b = float(max(1, (bx2 - bx1) * (by2 - by1)))
    return inter / max(1.0, area_a + area_b - inter)


class TwoFighterIdentityHMM:
    """Track two fighter identities with delayed commitment and object permanence."""

    def __init__(
        self,
        config: IdentityHMMConfig | None = None,
        logger: LoggerFn | None = None,
    ) -> None:
        self.cfg = config or IdentityHMMConfig()
        self._logger = logger
        self._roles: dict[str, _RoleObjectFile] = {
            "RED": _RoleObjectFile(kf=self._build_kf()),
            "BLUE": _RoleObjectFile(kf=self._build_kf()),
        }
        self._steps: deque[_Step] = deque(maxlen=max(3, int(self.cfg.viterbi_window)))
        self._dp0 = 0.0
        self._dp1 = 0.0
        self._last_ts: float | None = None
        self._current_state = 0
        self._pending_state: int | None = None
        self._pending_since: float | None = None
        self._last_flip_ts = -1e9
        self._missing_streak = 0
        self._prev_gray: GrayFrame | None = None
        self._bank_update_ts: deque[float] = deque(maxlen=400)
        self._debug_history: deque[dict[str, Any]] = deque(maxlen=300)

    @staticmethod
    def _build_kf() -> cv2.KalmanFilter:
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.04
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.2
        kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0
        return kf

    def _log(self, message: str) -> None:
        if self._logger is not None:
            self._logger(message)

    def _predict_role(self, role: str, dt_s: float) -> None:
        obj = self._roles[role]
        kf = obj.kf
        kf.transitionMatrix[0, 2] = float(dt_s)
        kf.transitionMatrix[1, 3] = float(dt_s)
        if not obj.initialized:
            return
        pred = kf.predict()
        obj.pred_center = (float(pred[0, 0]), float(pred[1, 0]))

    def _correct_role(
        self, role: str, center: tuple[float, float], noise_scale: float = 1.0
    ) -> None:
        obj = self._roles[role]
        meas = np.array([[center[0]], [center[1]]], dtype=np.float32)
        base_noise = np.eye(2, dtype=np.float32) * 1.2
        obj.kf.measurementNoiseCov = base_noise * max(0.2, noise_scale)
        if not obj.initialized:
            obj.kf.statePost = np.array([[center[0]], [center[1]], [0.0], [0.0]], dtype=np.float32)
            obj.initialized = True
            obj.pred_center = center
            return
        corrected = obj.kf.correct(meas)
        obj.pred_center = (float(corrected[0, 0]), float(corrected[1, 0]))

    def _occlusion_score(
        self,
        obs_a: TrackObservation | None,
        obs_b: TrackObservation | None,
    ) -> tuple[float, float, float]:
        iou = _bbox_iou(obs_a.bbox if obs_a else None, obs_b.bbox if obs_b else None)
        iou_norm = min(iou / max(1e-6, self.cfg.iou_occlusion_full), 1.0)
        if obs_a is None and obs_b is None:
            mean_vis = 0.0
        elif obs_a is None:
            mean_vis = _clamp(float(obs_b.mean_visibility) if obs_b is not None else 0.0, 0.0, 1.0)
        elif obs_b is None:
            mean_vis = _clamp(float(obs_a.mean_visibility), 0.0, 1.0)
        else:
            mean_vis = _clamp(
                0.5 * (float(obs_a.mean_visibility) + float(obs_b.mean_visibility)),
                0.0,
                1.0,
            )
        kp_drop = _clamp(1.0 - mean_vis, 0.0, 1.0)
        occlusion_score = _clamp(max(iou_norm, kp_drop), 0.0, 1.0)
        return occlusion_score, iou_norm, kp_drop

    def _likelihood(
        self,
        role: str,
        obs: TrackObservation | None,
        occlusion_score: float,
    ) -> tuple[float, dict[str, float], int]:
        obj = self._roles[role]
        if obs is None:
            miss = -float(self.cfg.miss_penalty)
            return miss, {"motion": miss, "appearance": 0.0, "pose": 0.0}, 0

        pred_center = obj.pred_center if obj.pred_center is not None else obs.center
        dx = float(obs.center[0] - pred_center[0])
        dy = float(obs.center[1] - pred_center[1])
        d = math.hypot(dx, dy)
        sigma = max(1e-3, float(self.cfg.sigma_motion_px))
        l_motion = -((d * d) / (2.0 * sigma * sigma))

        l_app = 0.0
        l_pose = 0.0
        app_available = obs.appearance_embed is not None and obj.embed_ema is not None
        pose_available = obs.pose_embed is not None and obj.pose_ema is not None
        if app_available:
            l_app = -_cosine_distance(obs.appearance_embed, obj.embed_ema)
        if pose_available:
            l_pose = -_l2_distance(obs.pose_embed, obj.pose_ema)

        w_a_eff = float(self.cfg.w_appear) * (
            1.0 - float(occlusion_score) * float(self.cfg.occlusion_app_suppress)
        )
        appearance_suppressed = int(w_a_eff < float(self.cfg.w_appear))
        total = (
            float(self.cfg.w_motion) * l_motion + w_a_eff * l_app + float(self.cfg.w_pose) * l_pose
        )
        terms = {
            "motion": float(self.cfg.w_motion) * l_motion,
            "appearance": w_a_eff * l_app,
            "pose": float(self.cfg.w_pose) * l_pose,
        }
        return total, terms, appearance_suppressed

    def _hypothesis_likelihood(
        self,
        hyp_state: int,
        obs_a: TrackObservation | None,
        obs_b: TrackObservation | None,
        occlusion_score: float,
    ) -> tuple[float, dict[str, float], int]:
        if hyp_state == 0:
            red_obs, blue_obs = obs_a, obs_b
        else:
            red_obs, blue_obs = obs_b, obs_a

        red_l, red_terms, red_supp = self._likelihood("RED", red_obs, occlusion_score)
        blue_l, blue_terms, blue_supp = self._likelihood("BLUE", blue_obs, occlusion_score)
        terms = {
            "motion": red_terms["motion"] + blue_terms["motion"],
            "appearance": red_terms["appearance"] + blue_terms["appearance"],
            "pose": red_terms["pose"] + blue_terms["pose"],
        }
        return red_l + blue_l, terms, int(red_supp or blue_supp)

    def _viterbi_step(self, l0: float, l1: float, switch_penalty: float) -> _Step:
        from_00 = self._dp0
        from_10 = self._dp1 - switch_penalty
        if from_00 >= from_10:
            new_dp0 = from_00 + l0
            back0 = 0
        else:
            new_dp0 = from_10 + l0
            back0 = 1

        from_11 = self._dp1
        from_01 = self._dp0 - switch_penalty
        if from_11 >= from_01:
            new_dp1 = from_11 + l1
            back1 = 1
        else:
            new_dp1 = from_01 + l1
            back1 = 0

        self._dp0 = float(new_dp0)
        self._dp1 = float(new_dp1)
        norm = max(self._dp0, self._dp1)
        self._dp0 -= norm
        self._dp1 -= norm
        return _Step(
            dp0=self._dp0,
            dp1=self._dp1,
            back0=back0,
            back1=back1,
            l0=l0,
            l1=l1,
            occlusion_score=0.0,
            appearance_suppressed=0,
        )

    def _effective_switch_penalty(self, occlusion_score: float) -> float:
        base = max(float(self.cfg.base_switch_penalty), float(self.cfg.switch_penalty))
        return base + float(self.cfg.occlusion_switch_boost) * float(occlusion_score)

    @staticmethod
    def _recent_count(history: deque[float], now_s: float, window_s: float) -> int:
        while history and now_s - history[0] > max(10.0, window_s * 2.0):
            history.popleft()
        return len([ts for ts in history if now_s - ts <= window_s])

    def _traceback_state(self) -> tuple[int, list[int]]:
        if not self._steps:
            return 0, [0]
        state = 0 if self._steps[-1].dp0 >= self._steps[-1].dp1 else 1
        path_rev = [state]
        for idx in range(len(self._steps) - 1, 0, -1):
            step = self._steps[idx]
            state = step.back0 if state == 0 else step.back1
            path_rev.append(state)
        path = list(reversed(path_rev))
        lag = max(0, min(int(self.cfg.commit_lag), len(path) - 1))
        committed = path[-1 - lag]
        return committed, path

    def _refresh_flow_points(
        self,
        obj: _RoleObjectFile,
        gray: GrayFrame,
        bbox: tuple[int, int, int, int],
    ) -> None:
        x1, y1, x2, y2 = bbox
        h, w = gray.shape[:2]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            obj.flow_points = None
            return
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            obj.flow_points = None
            return
        points = cv2.goodFeaturesToTrack(
            roi,
            maxCorners=28,
            qualityLevel=0.02,
            minDistance=4,
            blockSize=5,
        )
        if points is None or len(points) == 0:
            obj.flow_points = None
            return
        points_arr = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        points_arr[:, 0, 0] = points_arr[:, 0, 0] + float(x1)
        points_arr[:, 0, 1] = points_arr[:, 0, 1] + float(y1)
        obj.flow_points = points_arr
        obj.flow_active_frames = 0

    @staticmethod
    def _flow_center(
        prev_gray: GrayFrame,
        gray: GrayFrame,
        points: FloatArray,
    ) -> tuple[tuple[float, float], FloatArray] | None:
        next_points_seed = points.copy()
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            gray,
            points,
            next_points_seed,
            winSize=(19, 19),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        if next_points.size == 0 or status.size == 0:
            return None
        good = next_points[status.flatten() == 1]
        if good.size == 0:
            return None
        cx = float(np.median(good[:, 0]))
        cy = float(np.median(good[:, 1]))
        return (cx, cy), good.reshape(-1, 1, 2).astype(np.float32)

    def _update_object_file(
        self,
        role: str,
        obs: TrackObservation | None,
        timestamp_s: float,
        occlusion_score: float,
        frame_gray: GrayFrame | None,
    ) -> bool:
        obj = self._roles[role]
        if obs is not None:
            obj.current_track_id = obs.track_id
            obj.last_seen_ts = timestamp_s
            obj.coast_frames = 0
            obj.last_bbox = obs.bbox
            self._correct_role(role, obs.center, noise_scale=max(0.8, 1.3 - obs.det_conf))

            appearance = obs.appearance_embed
            updated_any = False
            should_update_app = (
                obs.det_conf >= self.cfg.min_det_conf
                and occlusion_score < self.cfg.update_max_occlusion
                and appearance is not None
            )
            if should_update_app and appearance is not None:
                if obj.embed_ema is None or obj.embed_ema.shape != appearance.shape:
                    obj.embed_ema = appearance.copy()
                else:
                    obj.embed_ema = (0.88 * obj.embed_ema + 0.12 * appearance).astype(np.float32)
                updated_any = True

            pose_embed = obs.pose_embed
            should_update_pose = (
                obs.det_conf >= self.cfg.min_det_conf
                and occlusion_score < self.cfg.update_max_occlusion
                and obs.mean_visibility >= self.cfg.min_kp_vis
                and pose_embed is not None
            )
            if should_update_pose and pose_embed is not None:
                if obj.pose_ema is None or obj.pose_ema.shape != pose_embed.shape:
                    obj.pose_ema = pose_embed.copy()
                else:
                    obj.pose_ema = (0.88 * obj.pose_ema + 0.12 * pose_embed).astype(np.float32)
                updated_any = True

            if frame_gray is not None:
                self._refresh_flow_points(obj, frame_gray, obs.bbox)
            return updated_any

        obj.coast_frames += 1
        if obj.coast_frames > self.cfg.max_coast_frames:
            obj.current_track_id = None
        if (
            frame_gray is not None
            and self._prev_gray is not None
            and obj.flow_points is not None
            and obj.flow_active_frames < self.cfg.flow_max_frames
        ):
            flowed = self._flow_center(self._prev_gray, frame_gray, obj.flow_points)
            if flowed is not None:
                flow_center, next_points = flowed
                obj.flow_points = next_points
                obj.flow_active_frames += 1
                self._correct_role(role, flow_center, noise_scale=3.5)
            else:
                obj.flow_points = None
        return False

    def role_embedding(self, role: str) -> FloatArray | None:
        return self._roles[role.upper()].embed_ema

    def role_predicted_center(self, role: str) -> tuple[float, float] | None:
        return self._roles[role.upper()].pred_center

    def update(
        self,
        timestamp_s: float,
        track_a_obs: TrackObservation | None,
        track_b_obs: TrackObservation | None,
        frame_gray: GrayFrame | None = None,
        external_debug: dict[str, Any] | None = None,
    ) -> dict[str, object]:
        ts = float(timestamp_s)
        dt = 1.0 / 30.0 if self._last_ts is None else max(1e-3, min(0.5, ts - self._last_ts))
        self._last_ts = ts

        self._predict_role("RED", dt)
        self._predict_role("BLUE", dt)

        any_missing = track_a_obs is None or track_b_obs is None
        if any_missing:
            self._missing_streak += 1
        else:
            self._missing_streak = 0

        occlusion_score, iou_norm, kp_drop = self._occlusion_score(track_a_obs, track_b_obs)
        l0, terms0, app_sup0 = self._hypothesis_likelihood(
            0, track_a_obs, track_b_obs, occlusion_score
        )
        l1, terms1, app_sup1 = self._hypothesis_likelihood(
            1, track_a_obs, track_b_obs, occlusion_score
        )

        switch_penalty = self._effective_switch_penalty(occlusion_score)
        if any_missing and self._missing_streak <= self.cfg.max_coast_frames:
            switch_penalty += float(self.cfg.miss_penalty) * 4.0

        step = self._viterbi_step(l0, l1, switch_penalty)
        step = _Step(
            dp0=step.dp0,
            dp1=step.dp1,
            back0=step.back0,
            back1=step.back1,
            l0=step.l0,
            l1=step.l1,
            occlusion_score=occlusion_score,
            appearance_suppressed=int(app_sup0 or app_sup1),
        )
        self._steps.append(step)
        best_state, _ = self._traceback_state()

        if best_state == 0:
            best_logp = self._dp0
            other_logp = self._dp1
        else:
            best_logp = self._dp1
            other_logp = self._dp0

        delta_best = float(best_logp - other_logp)
        delta_for_prob = max(-60.0, min(60.0, delta_best))
        p_best = 1.0 / (1.0 + math.exp(-delta_for_prob))

        old_state = self._current_state
        can_commit = (
            best_state != self._current_state
            and p_best >= self.cfg.commit_p_threshold
            and occlusion_score <= self.cfg.commit_max_occlusion
        )
        if best_state == self._current_state:
            self._pending_state = None
            self._pending_since = None
        elif can_commit:
            if self._pending_state != best_state:
                self._pending_state = best_state
                self._pending_since = ts
            elif self._pending_since is not None:
                held = ts - self._pending_since
                can_flip_now = ts - self._last_flip_ts > self.cfg.min_flip_interval_s
                if held >= self.cfg.commit_seconds and can_flip_now:
                    self._current_state = best_state
                    self._last_flip_ts = ts
                    self._pending_state = None
                    self._pending_since = None
        else:
            self._pending_state = None
            self._pending_since = None

        pending_seconds = (
            ts - self._pending_since
            if self._pending_since is not None and self._pending_state == best_state
            else 0.0
        )
        status = (
            "uncertain"
            if (
                best_state != self._current_state
                or p_best < self.cfg.commit_p_threshold
                or occlusion_score > self.cfg.commit_max_occlusion
            )
            else "stable"
        )

        if self._current_state == 0:
            red_obs, blue_obs = track_a_obs, track_b_obs
            chosen_label = "H0"
            chosen_terms = terms0
        else:
            red_obs, blue_obs = track_b_obs, track_a_obs
            chosen_label = "H1"
            chosen_terms = terms1

        red_updated = self._update_object_file("RED", red_obs, ts, occlusion_score, frame_gray)
        blue_updated = self._update_object_file("BLUE", blue_obs, ts, occlusion_score, frame_gray)
        self._prev_gray = frame_gray.copy() if frame_gray is not None else None
        updated_any = red_updated or blue_updated
        if updated_any:
            self._bank_update_ts.append(ts)

        delta = float(self._dp0 - self._dp1)
        confidence = float(
            np.tanh(abs(delta) / max(0.1, self._effective_switch_penalty(occlusion_score)))
        )
        dominant_term = max(chosen_terms.items(), key=lambda item: abs(item[1]))[0]

        debug = {
            "timestamp_s": round(ts, 3),
            "logp_h0": round(self._dp0, 4),
            "logp_h1": round(self._dp1, 4),
            "chosen_hypothesis": chosen_label,
            "delta": round(delta, 4),
            "occlusion_score": round(occlusion_score, 4),
            "iou_norm": round(iou_norm, 4),
            "kp_drop_norm": round(kp_drop, 4),
            "appearance_suppressed": int(step.appearance_suppressed),
            "dominant_term": dominant_term,
            "current_state": int(self._current_state),
            "best_state": int(best_state),
            "p_best": round(p_best, 4),
            "status": status,
            "pending_seconds": round(float(max(0.0, pending_seconds)), 3),
            "bank_updated": int(updated_any),
            "bank_updates_last_2s": self._recent_count(self._bank_update_ts, ts, 2.0),
            "terms": {
                "motion": round(chosen_terms["motion"], 4),
                "appearance": round(chosen_terms["appearance"], 4),
                "pose": round(chosen_terms["pose"], 4),
            },
        }
        if external_debug:
            debug["external"] = dict(external_debug)
        self._debug_history.append(debug)

        hypothesis_changed = old_state != self._current_state
        if hypothesis_changed:
            churn_last_2s = 0
            if external_debug is not None:
                raw_churn = external_debug.get("churn_events_last_2s")
                if isinstance(raw_churn, int):
                    churn_last_2s = raw_churn
            bank_updates_last_2s = self._recent_count(self._bank_update_ts, ts, 2.0)
            self._log(
                f"designation_flip t={ts:.3f} {old_state}->{self._current_state} "
                f"dp0={self._dp0:.3f} dp1={self._dp1:.3f} "
                f"delta={delta_best:.3f} p_best={p_best:.4f} "
                f"occ={occlusion_score:.3f} churn_last2s={churn_last_2s} "
                f"banks_updated_last2s={bank_updates_last_2s}"
            )

        return {
            "RED": self._roles["RED"].current_track_id,
            "BLUE": self._roles["BLUE"].current_track_id,
            "confidence": round(confidence, 4),
            "debug": {**debug, "hypothesis_changed": int(hypothesis_changed)},
        }

    def debug_history(self) -> list[dict[str, Any]]:
        return list(self._debug_history)
