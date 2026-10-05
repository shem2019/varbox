"""Turn per-frame camera-space meshes into a floor-anchored 4D scene, then fuse two views.

Per view
  1. Fit the ring floor from the fighters' lowest foot points (RANSAC).
  2. Rescale each person about the camera centre so the lowest foot touches the floor. This only
     moves people along their camera rays, so every pixel projection is unchanged; it fixes the
     main weakness of monocular meshes: wrong relative depth between the two fighters.
  3. Smooth over time with a One Euro filter (keeps punches sharp, removes jitter).
  4. Express everything in a world frame: floor at y = 0, +y up.

Two views
  The same fighters seen by both cameras give thousands of 3D keypoint correspondences, so the
  view-to-view transform is solved directly (RANSAC Umeyama). No checkerboard is needed. Fused
  meshes are visibility-weighted averages, which works because both views share one topology.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from boxing_analytics.mesh4d.geometry import (
    KP,
    Plane,
    apply_transform,
    estimate_up_direction,
    fit_floor_plane,
    ground_scale_factors,
    lowest_foot_points,
    one_euro,
    ransac_umeyama,
    smooth_series,
    world_from_camera,
)

NDArray = np.ndarray[Any, Any]
LogFn = Callable[[str], None]

# Body joints that both views estimate reliably (no fingers, no face details).
ALIGN_KPS = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 41, 62, 69]


@dataclass
class ViewScene:
    """Floor-anchored world-space sequence for one camera. Arrays are indexed by local frame."""

    roles: list[str]
    fps: float
    frames: NDArray  # (T,) source frame index
    verts: dict[str, NDArray]  # (T, V, 3) float32 world
    kp3d: dict[str, NDArray]  # (T, 70, 3) world
    valid: dict[str, NDArray]  # (T,)
    occlusion: dict[str, NDArray]  # (T,)
    plane_cam: Plane
    world_from_cam: NDArray
    intrinsics: NDArray
    faces: NDArray
    scale_k: dict[str, NDArray]

    def save(self, path: Path) -> None:
        payload: dict[str, NDArray] = {
            "frames": self.frames,
            "world_from_cam": self.world_from_cam,
            "intrinsics": self.intrinsics,
            "faces": self.faces,
        }
        for r in self.roles:
            payload[f"{r}_verts"] = self.verts[r].astype(np.float16)
            payload[f"{r}_kp3d"] = self.kp3d[r].astype(np.float32)
            payload[f"{r}_valid"] = self.valid[r]
            payload[f"{r}_occlusion"] = self.occlusion[r].astype(np.float32)
            payload[f"{r}_scale_k"] = self.scale_k[r].astype(np.float32)
        np.savez(path, **cast(dict[str, Any], payload))
        path.with_suffix(".json").write_text(
            json.dumps(
                {"roles": self.roles, "fps": self.fps, "plane_cam": self.plane_cam.to_dict()},
                indent=2,
            ),
            encoding="utf-8",
        )

    @staticmethod
    def load(path: Path) -> ViewScene:
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        roles = meta["roles"]
        with np.load(path) as z:
            return ViewScene(
                roles=roles,
                fps=float(meta["fps"]),
                frames=z["frames"],
                verts={r: z[f"{r}_verts"].astype(np.float32) for r in roles},
                kp3d={r: z[f"{r}_kp3d"] for r in roles},
                valid={r: z[f"{r}_valid"] for r in roles},
                occlusion={r: z[f"{r}_occlusion"] for r in roles},
                plane_cam=Plane.from_dict(meta["plane_cam"]),
                world_from_cam=z["world_from_cam"],
                intrinsics=z["intrinsics"],
                faces=z["faces"],
                scale_k={r: z[f"{r}_scale_k"] for r in roles},
            )


def build_view_scene(
    body: dict[str, Any],
    occlusion: dict[str, NDArray],
    fps: float,
    log: LogFn = print,
    *,
    smooth: bool = True,
) -> ViewScene:
    roles: list[str] = [r for r in body["roles"] if "kp3d" in body["data"][r]]
    data = body["data"]
    start, stop = body["start"], body["stop"]
    t_len = stop - start
    intrinsics = np.asarray(body["meta"]["intrinsics"], dtype=np.float64)

    all_kp = np.concatenate([data[r]["kp3d"] for r in roles])
    all_valid = np.concatenate(
        [data[r]["valid"] & np.isfinite(data[r]["kp3d"][:, 0, 0]) for r in roles]
    )
    up = estimate_up_direction(all_kp, all_valid)
    feet = lowest_foot_points(all_kp[all_valid], up)
    plane, inliers = fit_floor_plane(feet, up)
    log(f"world: floor fit from {feet.shape[0]} foot points, {inliers.mean():.0%} inliers")

    verts: dict[str, NDArray] = {}
    kps: dict[str, NDArray] = {}
    valid: dict[str, NDArray] = {}
    scale_k: dict[str, NDArray] = {}
    pelvis_samples = []
    for r in roles:
        v = data[r]["verts"].astype(np.float32)
        kp = data[r]["kp3d"].astype(np.float64)
        ok = data[r]["valid"] & np.isfinite(kp[:, 0, 0])
        k = np.ones(t_len)
        if ok.any():
            k_raw = ground_scale_factors(lowest_foot_points(kp[ok], up), plane)
            k_full = np.full(t_len, np.nan)
            k_full[ok] = k_raw
            k_ok = ok & np.isfinite(k_full) & (k_full > 0.6) & (k_full < 1.6)
            if k_ok.any():
                # Half-second median window ignores brief hops where no foot touches the floor.
                k = smooth_series(np.nan_to_num(k_full, nan=1.0), k_ok, max(3, int(0.5 * fps)) | 1)
        kp = kp * k[:, None, None]
        v = v * k[:, None, None].astype(np.float32)
        if smooth and ok.sum() > 2:
            kp[ok] = one_euro(kp[ok], fps)
            v[ok] = one_euro(v[ok].astype(np.float64), fps, min_cutoff=1.5).astype(np.float32)
        verts[r], kps[r], valid[r], scale_k[r] = v, kp, ok, k
        if ok.any():
            pelvis_samples.append(0.5 * (kp[ok, KP["left_hip"]] + kp[ok, KP["right_hip"]]))
        k_median = float(np.median(k[ok])) if ok.any() else 1.0
        log(f"world: {r} valid {ok.mean():.0%}, depth scale k median {k_median:.3f}")

    anchor = (
        np.median(np.concatenate(pelvis_samples), axis=0)
        if pelvis_samples
        else np.array([0, 0, 5.0])
    )
    t_wc = world_from_camera(plane, anchor)
    for r in roles:
        verts[r] = apply_transform(t_wc, verts[r]).astype(np.float32)
        kps[r] = apply_transform(t_wc, kps[r])
    occ = {r: np.asarray(occlusion.get(r, np.zeros(t_len)), dtype=np.float32) for r in roles}
    return ViewScene(
        roles=roles,
        fps=fps,
        frames=np.arange(start, stop),
        verts=verts,
        kp3d=kps,
        valid=valid,
        occlusion=occ,
        plane_cam=plane,
        world_from_cam=t_wc,
        intrinsics=intrinsics,
        faces=body["faces"],
        scale_k=scale_k,
    )


@dataclass
class FusionResult:
    scale: float
    rotation: NDArray
    translation: NDArray
    role_map: dict[str, str]  # role in A -> role in B
    inlier_frames: int
    paired_frames: int
    median_residual_m: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "rotation": self.rotation.tolist(),
            "translation": self.translation.tolist(),
            "role_map": self.role_map,
            "inlier_frames": self.inlier_frames,
            "paired_frames": self.paired_frames,
            "median_residual_m": self.median_residual_m,
        }


def _pairs(
    a: ViewScene, b: ViewScene, b_index: NDArray, role_map: dict[str, str], max_occ: float
) -> tuple[list[NDArray], list[NDArray]]:
    src, dst = [], []
    for i in range(a.frames.shape[0]):
        j = int(b_index[i])
        if j < 0:
            continue
        s_parts, d_parts = [], []
        for ra, rb in role_map.items():
            if (
                a.valid[ra][i]
                and b.valid[rb][j]
                and a.occlusion[ra][i] <= max_occ
                and b.occlusion[rb][j] <= max_occ
            ):
                s_parts.append(b.kp3d[rb][j, ALIGN_KPS])
                d_parts.append(a.kp3d[ra][i, ALIGN_KPS])
        if s_parts:
            src.append(np.concatenate(s_parts))
            dst.append(np.concatenate(d_parts))
    return src, dst


def align_views(
    a: ViewScene, b: ViewScene, b_index: NDArray, log: LogFn = print, max_occ: float = 0.25
) -> FusionResult:
    """Similarity transform mapping view B world -> view A world. Tries both role assignments."""
    fighters = [r for r in a.roles if r in ("red", "blue")]
    candidates = [{r: r for r in a.roles if r in b.roles}]
    if len(fighters) == 2:
        swapped = {fighters[0]: fighters[1], fighters[1]: fighters[0]}
        swapped.update({r: r for r in a.roles if r not in fighters and r in b.roles})
        candidates.append(swapped)
    best: FusionResult | None = None
    for role_map in candidates:
        src, dst = _pairs(a, b, b_index, role_map, max_occ)
        if len(src) < 10:
            continue
        stride = max(1, len(src) // 400)
        s, rot, t, inl = ransac_umeyama(src[::stride], dst[::stride])
        res = [
            np.median(np.linalg.norm(s * x @ rot.T + t - y, axis=1))
            for x, y in zip(src, dst, strict=False)
        ]
        result = FusionResult(
            s,
            rot,
            t,
            role_map,
            int(np.sum(np.asarray(res) < 0.15)),
            len(src),
            float(np.median(res)),
        )
        log(
            f"fuse: roles {role_map} -> median residual {result.median_residual_m:.3f} m, "
            f"{result.inlier_frames}/{result.paired_frames} frames within 15 cm, scale {s:.3f}"
        )
        # Two boxers facing each other are nearly symmetric under a half turn, so the swapped
        # assignment can fit almost as well. Colour seeding wins unless the swap fits far better.
        if best is None or result.median_residual_m < 0.6 * best.median_residual_m:
            best = result
    if best is None:
        raise RuntimeError("not enough frames where both cameras see the fighters clearly")
    return best


def fuse_scenes(a: ViewScene, b: ViewScene, b_index: NDArray, fusion: FusionResult) -> ViewScene:
    """Visibility-weighted average of the two views in view A's world frame."""
    verts, kps, valid, occ = {}, {}, {}, {}
    s, rot, t = fusion.scale, fusion.rotation, fusion.translation
    t_len = a.frames.shape[0]
    for ra in a.roles:
        rb = fusion.role_map.get(ra)
        v_out = a.verts[ra].copy()
        k_out = a.kp3d[ra].copy()
        ok_out = a.valid[ra].copy()
        o_out = a.occlusion[ra].copy()
        if rb is not None:
            for i in range(t_len):
                j = int(b_index[i])
                if j < 0 or not b.valid[rb][j]:
                    continue
                wa = float(a.valid[ra][i]) * (1.0 - min(1.0, float(a.occlusion[ra][i]))) ** 2
                wb = (1.0 - min(1.0, float(b.occlusion[rb][j]))) ** 2
                vb = s * b.verts[rb][j] @ rot.T + t
                kb = s * b.kp3d[rb][j] @ rot.T + t
                if wa + wb <= 1e-6:
                    wa = wb = 0.5
                if not a.valid[ra][i]:
                    v_out[i], k_out[i] = vb, kb
                else:
                    v_out[i] = (wa * a.verts[ra][i] + wb * vb) / (wa + wb)
                    k_out[i] = (wa * a.kp3d[ra][i] + wb * kb) / (wa + wb)
                ok_out[i] = True
                o_out[i] = min(o_out[i], b.occlusion[rb][j])
        verts[ra], kps[ra], valid[ra], occ[ra] = v_out, k_out, ok_out, o_out
    return ViewScene(
        roles=a.roles,
        fps=a.fps,
        frames=a.frames,
        verts=verts,
        kp3d=kps,
        valid=valid,
        occlusion=occ,
        plane_cam=a.plane_cam,
        world_from_cam=a.world_from_cam,
        intrinsics=a.intrinsics,
        faces=a.faces,
        scale_k=a.scale_k,
    )
