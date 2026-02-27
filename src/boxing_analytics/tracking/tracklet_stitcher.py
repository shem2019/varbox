"""Tracklet stitching for two-fighter candidate stabilization."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from boxing_analytics.tracking.identity_hmm import TrackObservation

FloatArray = NDArray[np.float32]


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


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
    if inter <= 0:
        return 0.0
    area_a = float(max(1, (ax2 - ax1) * (ay2 - ay1)))
    area_b = float(max(1, (bx2 - bx1) * (by2 - by1)))
    return inter / max(1.0, area_a + area_b - inter)


def _cosine_distance(a: FloatArray | None, b: FloatArray | None) -> float:
    if a is None or b is None or a.shape != b.shape or a.size == 0 or b.size == 0:
        return 1.0
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an <= 1e-6 or bn <= 1e-6:
        return 1.0
    cos = float(np.dot(a / an, b / bn))
    return _clamp(1.0 - cos, 0.0, 2.0)


@dataclass(frozen=True, slots=True)
class TrackletStitcherConfig:
    stitch_max_dist_px: float = 80.0
    stitch_embed_max: float = 0.35
    stitch_w_motion: float = 1.0
    stitch_w_appear: float = 0.6
    stitch_w_pose: float = 0.2
    update_max_occlusion: float = 0.08
    min_det_conf: float = 0.4
    min_kp_vis: float = 0.60
    occlusion_iou_full: float = 0.25


@dataclass(slots=True)
class _CandidateState:
    kf: cv2.KalmanFilter
    initialized: bool = False
    pred_center: tuple[float, float] | None = None
    embed_ema: FloatArray | None = None
    pose_ema: FloatArray | None = None
    last_seen_ts: float = 0.0
    raw_track_id: int | None = None


class TwoFighterTrackletStitcher:
    """Stitch unstable raw track IDs into persistent candidate slots."""

    def __init__(self, config: TrackletStitcherConfig | None = None) -> None:
        self.cfg = config or TrackletStitcherConfig()
        self._cand0 = _CandidateState(kf=self._build_kf())
        self._cand1 = _CandidateState(kf=self._build_kf())
        self._last_ts: float | None = None
        self._churn_events: deque[float] = deque(maxlen=400)
        self._bank_updates: deque[float] = deque(maxlen=400)

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

    def _predict(self, cand: _CandidateState, dt_s: float) -> None:
        cand.kf.transitionMatrix[0, 2] = float(dt_s)
        cand.kf.transitionMatrix[1, 3] = float(dt_s)
        if not cand.initialized:
            return
        pred = cand.kf.predict()
        cand.pred_center = (float(pred[0, 0]), float(pred[1, 0]))

    def _correct(self, cand: _CandidateState, center: tuple[float, float]) -> None:
        measurement = np.array([[center[0]], [center[1]]], dtype=np.float32)
        if not cand.initialized:
            cand.kf.statePost = np.array([[center[0]], [center[1]], [0.0], [0.0]], dtype=np.float32)
            cand.initialized = True
            cand.pred_center = center
            return
        corrected = cand.kf.correct(measurement)
        cand.pred_center = (float(corrected[0, 0]), float(corrected[1, 0]))

    def _track_occlusion(self, idx: int, tracks: list[TrackObservation], iou_full: float) -> float:
        obs = tracks[idx]
        max_iou = 0.0
        for jdx, other in enumerate(tracks):
            if jdx == idx:
                continue
            max_iou = max(max_iou, _bbox_iou(obs.bbox, other.bbox))
        iou_norm = min(max_iou / max(1e-6, iou_full), 1.0)
        kp_drop = _clamp(1.0 - float(obs.mean_visibility), 0.0, 1.0)
        return _clamp(max(iou_norm, kp_drop), 0.0, 1.0)

    def _cost(
        self,
        cand: _CandidateState,
        obs: TrackObservation,
        occlusion_score: float,
    ) -> tuple[float, dict[str, float]]:
        if cand.initialized and cand.pred_center is not None:
            dx = float(obs.center[0] - cand.pred_center[0])
            dy = float(obs.center[1] - cand.pred_center[1])
            motion = math.hypot(dx, dy)
            if motion > self.cfg.stitch_max_dist_px:
                return float("inf"), {"motion": float("inf"), "appearance": 0.0, "pose": 0.0}
        else:
            motion = 0.0

        appearance = 0.0
        if (
            cand.embed_ema is not None
            and obs.appearance_embed is not None
            and occlusion_score < self.cfg.update_max_occlusion
        ):
            appearance = _cosine_distance(obs.appearance_embed, cand.embed_ema)
            if appearance > self.cfg.stitch_embed_max:
                return float("inf"), {"motion": motion, "appearance": float("inf"), "pose": 0.0}

        pose = 0.0
        if (
            cand.pose_ema is not None
            and obs.pose_embed is not None
            and obs.mean_visibility >= self.cfg.min_kp_vis
        ):
            pose = float(np.linalg.norm(obs.pose_embed - cand.pose_ema))

        w_a_eff = self.cfg.stitch_w_appear * (1.0 - occlusion_score)
        total = (
            self.cfg.stitch_w_motion * motion + w_a_eff * appearance + self.cfg.stitch_w_pose * pose
        )
        terms = {
            "motion": self.cfg.stitch_w_motion * motion,
            "appearance": w_a_eff * appearance,
            "pose": self.cfg.stitch_w_pose * pose,
        }
        return float(total), terms

    def _update_candidate(
        self,
        cand: _CandidateState,
        obs: TrackObservation | None,
        occlusion_score: float,
        timestamp_s: float,
    ) -> bool:
        if obs is None:
            return False

        prev_raw = cand.raw_track_id
        self._correct(cand, obs.center)
        cand.last_seen_ts = timestamp_s
        cand.raw_track_id = obs.track_id
        churned = prev_raw is not None and prev_raw != obs.track_id
        if churned:
            self._churn_events.append(timestamp_s)

        updated_any = False
        if (
            obs.appearance_embed is not None
            and obs.det_conf >= self.cfg.min_det_conf
            and occlusion_score < self.cfg.update_max_occlusion
        ):
            if cand.embed_ema is None or cand.embed_ema.shape != obs.appearance_embed.shape:
                cand.embed_ema = obs.appearance_embed.copy()
            else:
                cand.embed_ema = (0.88 * cand.embed_ema + 0.12 * obs.appearance_embed).astype(
                    np.float32
                )
            updated_any = True

        if (
            obs.pose_embed is not None
            and obs.det_conf >= self.cfg.min_det_conf
            and obs.mean_visibility >= self.cfg.min_kp_vis
            and occlusion_score < self.cfg.update_max_occlusion
        ):
            if cand.pose_ema is None or cand.pose_ema.shape != obs.pose_embed.shape:
                cand.pose_ema = obs.pose_embed.copy()
            else:
                cand.pose_ema = (0.88 * cand.pose_ema + 0.12 * obs.pose_embed).astype(np.float32)
            updated_any = True

        if updated_any:
            self._bank_updates.append(timestamp_s)
        return churned

    @staticmethod
    def _prune_recent(history: deque[float], now_s: float, window_s: float) -> int:
        while history and now_s - history[0] > max(10.0, window_s * 2.0):
            history.popleft()
        return len([ts for ts in history if now_s - ts <= window_s])

    def update(
        self,
        timestamp_s: float,
        tracks: list[TrackObservation],
    ) -> tuple[TrackObservation | None, TrackObservation | None, dict[str, Any]]:
        ts = float(timestamp_s)
        dt = 1.0 / 30.0 if self._last_ts is None else max(1e-3, min(0.5, ts - self._last_ts))
        self._last_ts = ts

        self._predict(self._cand0, dt)
        self._predict(self._cand1, dt)

        if not tracks:
            debug = {
                "raw_ids": [],
                "assignment_scores": {"cand0": None, "cand1": None},
                "churn_events": {
                    "cand0": 0,
                    "cand1": 0,
                    "last_2s": self._prune_recent(self._churn_events, ts, 2.0),
                },
                "bank_updates_last_2s": self._prune_recent(self._bank_updates, ts, 2.0),
            }
            return None, None, debug

        occlusion_scores = [
            self._track_occlusion(idx, tracks, self.cfg.occlusion_iou_full)
            for idx in range(len(tracks))
        ]

        best_pair: tuple[int | None, int | None] = (None, None)
        best_scores: tuple[float, float] = (float("inf"), float("inf"))
        best_total = float("inf")
        best_terms: dict[str, dict[str, float]] = {
            "cand0": {"motion": 0.0, "appearance": 0.0, "pose": 0.0},
            "cand1": {"motion": 0.0, "appearance": 0.0, "pose": 0.0},
        }

        if len(tracks) >= 2:
            for idx0 in range(len(tracks)):
                for idx1 in range(len(tracks)):
                    if idx0 == idx1:
                        continue
                    score0, terms0 = self._cost(self._cand0, tracks[idx0], occlusion_scores[idx0])
                    score1, terms1 = self._cost(self._cand1, tracks[idx1], occlusion_scores[idx1])
                    if math.isinf(score0) or math.isinf(score1):
                        continue
                    total = score0 + score1
                    if total < best_total:
                        best_total = total
                        best_pair = (idx0, idx1)
                        best_scores = (score0, score1)
                        best_terms = {"cand0": terms0, "cand1": terms1}

        if best_pair == (None, None):
            if len(tracks) == 1:
                score0, terms0 = self._cost(self._cand0, tracks[0], occlusion_scores[0])
                score1, terms1 = self._cost(self._cand1, tracks[0], occlusion_scores[0])
                if score0 <= score1 and not math.isinf(score0):
                    best_pair = (0, None)
                    best_scores = (score0, float("inf"))
                    best_terms = {
                        "cand0": terms0,
                        "cand1": {"motion": 0.0, "appearance": 0.0, "pose": 0.0},
                    }
                elif not math.isinf(score1):
                    best_pair = (None, 0)
                    best_scores = (float("inf"), score1)
                    best_terms = {
                        "cand0": {"motion": 0.0, "appearance": 0.0, "pose": 0.0},
                        "cand1": terms1,
                    }
            else:
                # Fall back to independent minimum-cost assignment.
                best0_idx = None
                best0_score = float("inf")
                best0_terms: dict[str, float] = {"motion": 0.0, "appearance": 0.0, "pose": 0.0}
                for idx, obs in enumerate(tracks):
                    score, terms = self._cost(self._cand0, obs, occlusion_scores[idx])
                    if score < best0_score:
                        best0_idx = idx
                        best0_score = score
                        best0_terms = terms

                best1_idx = None
                best1_score = float("inf")
                best1_terms: dict[str, float] = {"motion": 0.0, "appearance": 0.0, "pose": 0.0}
                for idx, obs in enumerate(tracks):
                    if idx == best0_idx:
                        continue
                    score, terms = self._cost(self._cand1, obs, occlusion_scores[idx])
                    if score < best1_score:
                        best1_idx = idx
                        best1_score = score
                        best1_terms = terms

                if best0_idx is not None and not math.isinf(best0_score):
                    best_pair = (
                        best0_idx,
                        (
                            best1_idx
                            if best1_idx is not None and not math.isinf(best1_score)
                            else None
                        ),
                    )
                    best_scores = (best0_score, best1_score)
                    best_terms = {"cand0": best0_terms, "cand1": best1_terms}

        cand0_obs = tracks[best_pair[0]] if best_pair[0] is not None else None
        cand1_obs = tracks[best_pair[1]] if best_pair[1] is not None else None

        churn0 = self._update_candidate(
            self._cand0,
            cand0_obs,
            occlusion_scores[best_pair[0]] if best_pair[0] is not None else 0.0,
            ts,
        )
        churn1 = self._update_candidate(
            self._cand1,
            cand1_obs,
            occlusion_scores[best_pair[1]] if best_pair[1] is not None else 0.0,
            ts,
        )

        debug = {
            "raw_ids": [int(obs.track_id) for obs in tracks],
            "assignment_scores": {
                "cand0": None if math.isinf(best_scores[0]) else round(float(best_scores[0]), 4),
                "cand1": None if math.isinf(best_scores[1]) else round(float(best_scores[1]), 4),
            },
            "assignment_terms": best_terms,
            "churn_events": {
                "cand0": int(churn0),
                "cand1": int(churn1),
                "last_2s": self._prune_recent(self._churn_events, ts, 2.0),
            },
            "bank_updates_last_2s": self._prune_recent(self._bank_updates, ts, 2.0),
            "candidate_raw_ids": {
                "cand0": self._cand0.raw_track_id,
                "cand1": self._cand1.raw_track_id,
            },
        }
        return cand0_obs, cand1_obs, debug
