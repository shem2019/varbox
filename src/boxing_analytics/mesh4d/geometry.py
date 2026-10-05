"""Pure numpy geometry for the 4D pipeline: floor plane, ground anchoring, alignment, smoothing.

Conventions
-----------
Camera space follows OpenCV: +x right, +y down, +z forward, metres.
World space is built per view from the fitted floor: +y up, floor at y = 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

NDArray = np.ndarray[Any, Any]

# MHR70 keypoint indices used across the pipeline.
KP = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_hip": 9,
    "right_hip": 10,
    "left_knee": 11,
    "right_knee": 12,
    "left_ankle": 13,
    "right_ankle": 14,
    "left_big_toe": 15,
    "left_small_toe": 16,
    "left_heel": 17,
    "right_big_toe": 18,
    "right_small_toe": 19,
    "right_heel": 20,
    "right_wrist": 41,
    "left_wrist": 62,
    "neck": 69,
}
FOOT_KPS = (13, 14, 15, 16, 17, 18, 19, 20)
# Knuckle ("third joint") keypoints of index, middle, ring and pinky fingers.
RIGHT_KNUCKLES = (28, 32, 36, 40)
LEFT_KNUCKLES = (49, 53, 57, 61)


def normalize(v: NDArray, axis: int = -1) -> NDArray:
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, 1e-9)


@dataclass(frozen=True)
class Plane:
    """Plane n . x + c = 0 with unit normal n pointing up (away from the floor)."""

    normal: NDArray
    offset: float

    def height(self, points: NDArray) -> NDArray:
        return points @ self.normal + self.offset

    def to_dict(self) -> dict[str, Any]:
        return {"normal": self.normal.tolist(), "offset": float(self.offset)}

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> Plane:
        return Plane(np.asarray(payload["normal"], dtype=np.float64), float(payload["offset"]))


def estimate_up_direction(kp3d: NDArray, valid: NDArray) -> NDArray:
    """Median body axis (mid-ankle -> neck) over all valid person-frames, as a unit vector.

    kp3d: (..., 70, 3) camera-space keypoints, valid: matching boolean mask over leading dims.
    """
    pts = kp3d[valid]
    if pts.shape[0] == 0:
        return np.array([0.0, -1.0, 0.0])
    mid_ankle = 0.5 * (pts[:, KP["left_ankle"]] + pts[:, KP["right_ankle"]])
    axis = normalize(pts[:, KP["neck"]] - mid_ankle)
    up = normalize(np.median(axis, axis=0))
    return up


def lowest_foot_points(kp3d: NDArray, up: NDArray) -> NDArray:
    """For each person-frame, the foot keypoint lowest along `up`. kp3d: (N, 70, 3) -> (N, 3)."""
    feet = kp3d[:, FOOT_KPS, :]
    heights = feet @ up
    idx = np.argmin(heights, axis=1)
    return feet[np.arange(feet.shape[0]), idx]


def fit_floor_plane(
    points: NDArray,
    up_hint: NDArray,
    *,
    iterations: int = 2000,
    inlier_m: float = 0.04,
    max_tilt_deg: float = 30.0,
    seed: int = 0,
) -> tuple[Plane, NDArray]:
    """RANSAC plane through foot contact points, normal constrained near `up_hint`.

    Returns the refined plane and the inlier mask.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 3:
        raise ValueError("floor fit needs at least three foot points")
    up_hint = normalize(np.asarray(up_hint, dtype=np.float64))
    rng = np.random.default_rng(seed)
    cos_tilt = np.cos(np.deg2rad(max_tilt_deg))
    best_inliers = np.zeros(pts.shape[0], dtype=bool)
    best_score = -1.0
    for _ in range(iterations):
        sample = pts[rng.choice(pts.shape[0], 3, replace=False)]
        n = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        if n @ up_hint < 0:
            n = -n
        if n @ up_hint < cos_tilt:
            continue
        c = -float(n @ sample[0])
        residual = np.abs(pts @ n + c)
        inliers = residual < inlier_m
        # Prefer planes with many inliers and few points well below the floor.
        below = np.sum((pts @ n + c) < -3 * inlier_m)
        score = float(inliers.sum()) - 2.0 * float(below)
        if score > best_score:
            best_score = score
            best_inliers = inliers
    if best_inliers.sum() < 3:
        # Degenerate scene: fall back to a plane through the lowest decile along the hint.
        heights = pts @ up_hint
        cut = np.quantile(heights, 0.1)
        n = up_hint
        c = -float(np.median(heights[heights <= cut + 0.02]))
        return Plane(n, c), np.abs(pts @ n + c) < inlier_m
    inlier_pts = pts[best_inliers]
    centroid = inlier_pts.mean(axis=0)
    _, _, vt = np.linalg.svd(inlier_pts - centroid)
    n = vt[-1]
    if n @ up_hint < 0:
        n = -n
    c = -float(n @ centroid)
    plane = Plane(n, c)
    return plane, np.abs(plane.height(pts)) < inlier_m


def ground_scale_factors(foot_points: NDArray, plane: Plane) -> NDArray:
    """Per person-frame scale k about the camera centre that puts the lowest foot on the floor.

    Scaling about the camera centre leaves every pixel projection unchanged, so it only resolves
    the monocular depth/size ambiguity. k = -c / (n . p).
    """
    denom = foot_points @ plane.normal
    k = -plane.offset / np.where(np.abs(denom) < 1e-9, np.nan, denom)
    return k


def smooth_series(values: NDArray, valid: NDArray, window: int) -> NDArray:
    """Median-then-mean smoothing along axis 0 that ignores invalid entries.

    values: (T, ...) array, valid: (T,) bool. Invalid entries are filled from neighbours.
    """
    t = values.shape[0]
    out = np.array(values, dtype=np.float64, copy=True)
    if t == 0 or window <= 1:
        return out
    half = window // 2
    idx_valid = np.flatnonzero(valid)
    if idx_valid.size == 0:
        return out
    flat = out.reshape(t, -1)
    result = np.empty_like(flat)
    for i in range(t):
        lo, hi = max(0, i - half), min(t, i + half + 1)
        sel = idx_valid[(idx_valid >= lo) & (idx_valid < hi)]
        if sel.size == 0:
            nearest = idx_valid[np.argmin(np.abs(idx_valid - i))]
            result[i] = flat[nearest]
            continue
        window_vals = flat[sel]
        med = np.median(window_vals, axis=0)
        spread = np.median(np.abs(window_vals - med), axis=0) * 3.0 + 1e-6
        keep = np.abs(window_vals - med) <= spread
        weighted = np.where(keep, window_vals, 0.0).sum(axis=0) / np.maximum(keep.sum(axis=0), 1)
        result[i] = weighted
    return result.reshape(values.shape)


def one_euro(
    values: NDArray,
    fps: float,
    *,
    min_cutoff: float = 1.0,
    beta: float = 2.0,
    d_cutoff: float = 1.0,
) -> NDArray:
    """One Euro filter along axis 0. Keeps fast motion (punches) sharp, removes jitter at rest."""
    x = np.asarray(values, dtype=np.float64)
    if x.shape[0] < 2:
        return x.copy()
    te = 1.0 / fps

    def alpha(cutoff: NDArray | float) -> NDArray | float:
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    out = np.empty_like(x)
    out[0] = x[0]
    dx_prev = np.zeros_like(x[0])
    for i in range(1, x.shape[0]):
        dx = (x[i] - out[i - 1]) / te
        a_d = alpha(d_cutoff)
        dx_hat = a_d * dx + (1 - a_d) * dx_prev
        cutoff = min_cutoff + beta * np.abs(dx_hat)
        a = alpha(cutoff)
        out[i] = a * x[i] + (1 - a) * out[i - 1]
        dx_prev = dx_hat
    return out


def world_from_camera(plane: Plane, anchor_cam: NDArray) -> NDArray:
    """4x4 transform camera -> world. World +y = floor normal, origin = anchor dropped to floor.

    World +x follows the camera's +x projected onto the floor, so the view is "behind" the camera.
    """
    up = normalize(plane.normal)
    x_cam = np.array([1.0, 0.0, 0.0])
    x_axis = normalize(x_cam - (x_cam @ up) * up)
    z_axis = normalize(np.cross(x_axis, up))
    rot = np.stack([x_axis, up, z_axis], axis=0)  # rows: world axes expressed in camera coords
    origin = anchor_cam - plane.height(anchor_cam[None])[0] * up
    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[:3, 3] = -rot @ origin
    return transform


def apply_transform(transform: NDArray, points: NDArray) -> NDArray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def umeyama(
    src: NDArray, dst: NDArray, *, with_scale: bool = True
) -> tuple[float, NDArray, NDArray]:
    """Least-squares similarity dst ~= s * R @ src + t. Returns (s, R, t)."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / src.shape[0]
    u, d, vt = np.linalg.svd(cov)
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2, 2] = -1
    rot = u @ sign @ vt
    var_s = (xs**2).sum() / src.shape[0]
    scale = float(np.trace(np.diag(d) @ sign) / var_s) if with_scale and var_s > 0 else 1.0
    trans = mu_d - scale * rot @ mu_s
    return scale, rot, trans


def ransac_umeyama(
    src_groups: list[NDArray],
    dst_groups: list[NDArray],
    *,
    iterations: int = 500,
    inlier_m: float = 0.15,
    sample_groups: int = 3,
    with_scale: bool = True,
    seed: int = 0,
) -> tuple[float, NDArray, NDArray, NDArray]:
    """Robust similarity between two views from paired keypoint groups (one group per frame).

    A frame is an inlier when its median keypoint residual is under `inlier_m`.
    Returns (s, R, t, inlier_mask over groups).
    """
    n = len(src_groups)
    if n == 0:
        raise ValueError("no paired frames for cross-view alignment")
    rng = np.random.default_rng(seed)
    best = None
    best_count = -1
    for _ in range(iterations):
        pick = rng.choice(n, min(sample_groups, n), replace=False)
        s, r, t = umeyama(
            np.concatenate([src_groups[i] for i in pick]),
            np.concatenate([dst_groups[i] for i in pick]),
            with_scale=with_scale,
        )
        residuals = np.array(
            [
                np.median(np.linalg.norm(s * src_groups[i] @ r.T + t - dst_groups[i], axis=1))
                for i in range(n)
            ]
        )
        inliers = residuals < inlier_m
        if inliers.sum() > best_count:
            best_count = int(inliers.sum())
            best = inliers
    assert best is not None
    if best.sum() < 2:
        best = np.ones(n, dtype=bool)
    s, r, t = umeyama(
        np.concatenate([src_groups[i] for i in np.flatnonzero(best)]),
        np.concatenate([dst_groups[i] for i in np.flatnonzero(best)]),
        with_scale=with_scale,
    )
    return s, r, t, best


def project(points_cam: NDArray, intrinsics: NDArray) -> NDArray:
    z = np.maximum(points_cam[..., 2:3], 1e-6)
    uv = points_cam[..., :2] / z
    return uv * np.array([intrinsics[0, 0], intrinsics[1, 1]]) + np.array(
        [intrinsics[0, 2], intrinsics[1, 2]]
    )


def glove_centres(kp3d: NDArray, glove_offset_m: float = 0.03) -> tuple[NDArray, NDArray]:
    """Left and right glove centres from wrist + knuckles, nudged forward along the forearm.

    kp3d: (..., 70, 3). Returns (left, right), each (..., 3).
    """

    def one(wrist_i: int, elbow_i: int, knuckles: tuple[int, ...]) -> NDArray:
        wrist = kp3d[..., wrist_i, :]
        knuck = kp3d[..., list(knuckles), :].mean(axis=-2)
        centre = 0.5 * (wrist + knuck)
        forward = normalize(wrist - kp3d[..., elbow_i, :])
        return centre + glove_offset_m * forward

    left = one(KP["left_wrist"], KP["left_elbow"], LEFT_KNUCKLES)
    right = one(KP["right_wrist"], KP["right_elbow"], RIGHT_KNUCKLES)
    return left, right
