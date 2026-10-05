"""Mesh rendering without OpenGL: dense point splatting with a z-buffer, on GPU when available.

Two outputs:
  overlay  - each fighter's mesh tinted over the original camera frames, with glove markers,
             contact flashes and a running tally.
  isolated - one fighter alone on a dark stage, seen from a slowly orbiting virtual camera.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from boxing_analytics.mesh4d.geometry import apply_transform, glove_centres
from boxing_analytics.mesh4d.reconstruct import ViewScene
from boxing_analytics.mesh4d.video_io import iter_frames, prefetch, transcode_h264

NDArray = np.ndarray[Any, Any]
LogFn = Callable[[str], None]

ROLE_RGB = {"red": (225, 60, 70), "blue": (60, 120, 235), "referee": (205, 205, 205)}


class Splatter:
    def __init__(self, faces: NDArray, device: str = "cuda") -> None:
        import torch

        self.torch = torch
        self.device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        self.faces = torch.as_tensor(faces.astype(np.int64), device=self.device)

    def _dense(self, verts: Any) -> tuple[Any, Any]:
        """Vertices + face centroids, with per-point normals, so splats close the surface."""
        torch = self.torch
        f = self.faces
        v0, v1, v2 = verts[f[:, 0]], verts[f[:, 1]], verts[f[:, 2]]
        fn = torch.linalg.cross(v1 - v0, v2 - v0)
        vn = torch.zeros_like(verts)
        for k in range(3):
            vn.index_add_(0, f[:, k], fn)
        vn = vn / vn.norm(dim=1, keepdim=True).clamp_min(1e-9)
        fn = fn / fn.norm(dim=1, keepdim=True).clamp_min(1e-9)
        centroids = (v0 + v1 + v2) / 3.0
        return torch.cat([verts, centroids]), torch.cat([vn, fn])

    def render(
        self,
        layers: list[tuple[NDArray, tuple[int, int, int]]],
        intrinsics: NDArray,
        size: tuple[int, int],
        radius: int = 2,
    ) -> tuple[NDArray, NDArray]:
        """layers: [(camera-space verts (V,3), rgb)]. Returns (rgb image uint8, alpha mask bool)."""
        torch = self.torch
        h, w = size
        zbuf = torch.full((h * w,), float("inf"), device=self.device)
        colour = torch.zeros((h * w, 3), device=self.device)
        offsets = torch.stack(
            torch.meshgrid(
                torch.arange(-radius, radius + 1, device=self.device),
                torch.arange(-radius, radius + 1, device=self.device),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(-1, 2)
        offsets = offsets[(offsets**2).sum(1) <= radius * radius + 1]
        fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
        all_idx, all_z, all_rgb = [], [], []
        for verts_np, rgb in layers:
            verts = torch.as_tensor(np.nan_to_num(verts_np).astype(np.float32), device=self.device)
            pts, nrm = self._dense(verts)
            z = pts[:, 2]
            keep = z > 0.1
            pts, nrm, z = pts[keep], nrm[keep], z[keep]
            u = (pts[:, 0] / z * fx + cx).round().long()
            v = (pts[:, 1] / z * fy + cy).round().long()
            view = -pts / pts.norm(dim=1, keepdim=True).clamp_min(1e-9)
            lambert = (nrm * view).sum(1).abs().clamp(0, 1)
            shade = 0.3 + 0.7 * lambert
            base = torch.tensor(rgb, device=self.device, dtype=torch.float32)
            c = shade[:, None] * base[None]
            uu = (u[:, None] + offsets[None, :, 0]).reshape(-1)
            vv = (v[:, None] + offsets[None, :, 1]).reshape(-1)
            zz = z[:, None].expand(-1, offsets.shape[0]).reshape(-1)
            cc = c[:, None, :].expand(-1, offsets.shape[0], -1).reshape(-1, 3)
            inside = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
            all_idx.append(vv[inside] * w + uu[inside])
            all_z.append(zz[inside])
            all_rgb.append(cc[inside])
        if not all_idx:
            return np.zeros((h, w, 3), np.uint8), np.zeros((h, w), bool)
        idx = torch.cat(all_idx)
        z = torch.cat(all_z)
        rgb_all = torch.cat(all_rgb)
        zbuf.scatter_reduce_(0, idx, z, reduce="amin")
        front = z <= zbuf[idx] + 1e-4
        colour[idx[front]] = rgb_all[front]
        alpha = torch.isfinite(zbuf).reshape(h, w)
        img = colour.reshape(h, w, 3).clamp(0, 255).byte().cpu().numpy()
        return img, alpha.cpu().numpy()


def _hud(frame: NDArray, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    y = 36
    for text, bgr in lines:
        cv2.putText(frame, text, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(frame, text, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2, cv2.LINE_AA)
        y += 34


def render_overlay(
    scene: ViewScene,
    video_path: str,
    out_path: Path,
    events: list[dict[str, Any]],
    *,
    device: str = "cuda",
    alpha: float = 0.55,
    log: LogFn = print,
) -> Path:
    """Meshes drawn back onto the source camera. scene must be this camera's own world frame."""
    splat = Splatter(scene.faces, device)
    cam_from_world = np.linalg.inv(scene.world_from_cam)
    start, stop = int(scene.frames[0]), int(scene.frames[-1]) + 1
    first = next(iter_frames(video_path, start, start + 1))[1]
    h, w = first.shape[:2]
    raw_path = out_path.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter.fourcc(*"mp4v"), scene.fps, (w, h))
    flashes: dict[int, list[dict[str, Any]]] = {}
    for e in events:
        if e.get("contact_frame") is not None:
            for d in range(int(0.25 * scene.fps)):
                flashes.setdefault(int(e["contact_frame"]) - start + d, []).append(e)
    counts = {r: {"landed": 0, "thrown": 0} for r in scene.roles}
    by_start: dict[int, list[dict[str, Any]]] = {}
    for e in events:
        by_start.setdefault(int(e["contact_frame"] or e["peak_frame"]) - start, []).append(e)
    for local, (index, frame) in enumerate(prefetch(iter_frames(video_path, start, stop))):
        layers = []
        for r in scene.roles:
            if scene.valid[r][local]:
                layers.append(
                    (
                        apply_transform(cam_from_world, scene.verts[r][local]),
                        ROLE_RGB.get(r, (0, 255, 0)),
                    )
                )
        img, mask = splat.render(layers, scene.intrinsics, (h, w))
        out = frame.copy()
        bgr = img[..., ::-1]
        out[mask] = (alpha * bgr[mask] + (1 - alpha) * frame[mask]).astype(np.uint8)
        for e in by_start.get(local, []):
            counts[e["attacker"]]["thrown"] += 1
            if e["outcome"] == "landed" and e["target"] in ("head", "torso"):
                counts[e["attacker"]]["landed"] += 1
        for e in flashes.get(local, []):
            kp = scene.kp3d[e["attacker"]][local]
            left, right = glove_centres(kp[None])
            g = (left if e["hand"] == "left" else right)[0]
            p = apply_transform(cam_from_world, g[None])[0]
            if p[2] > 0.1:
                u = int(p[0] / p[2] * scene.intrinsics[0, 0] + scene.intrinsics[0, 2])
                v = int(p[1] / p[2] * scene.intrinsics[1, 1] + scene.intrinsics[1, 2])
                colour = {
                    "landed": (60, 230, 60),
                    "blocked": (0, 200, 255),
                    "missed": (180, 180, 180),
                }[e["outcome"]]
                cv2.circle(out, (u, v), 38, colour, 4, cv2.LINE_AA)
                label = e["outcome"] + (f" {e['target']}" if e["target"] else "")
                cv2.putText(
                    out, label, (u + 42, v), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2, cv2.LINE_AA
                )
        lines = [(f"frame {index}  t={index / scene.fps:6.2f}s", (255, 255, 255))]
        for r in scene.roles:
            if r in counts:
                c = ROLE_RGB.get(r, (255, 255, 255))
                lines.append(
                    (
                        f"{r.upper():7s} landed {counts[r]['landed']:3d} "
                        f"/ thrown {counts[r]['thrown']:3d}",
                        (c[2], c[1], c[0]),
                    )
                )
        lines.append(("decision support only - not an official score", (200, 200, 200)))
        _hud(out, lines)
        writer.write(out)
        if local % 100 == 0:
            log(f"render overlay: frame {index}/{stop - 1}")
    writer.release()
    if transcode_h264(raw_path, out_path):
        raw_path.unlink(missing_ok=True)
        return out_path
    return raw_path


def _look_at(eye: NDArray, target: NDArray) -> NDArray:
    """4x4 world -> camera (OpenCV) for a camera at `eye` looking at `target`, world +y up."""
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    rot = np.stack([right, down, fwd])
    m = np.eye(4)
    m[:3, :3] = rot
    m[:3, 3] = -rot @ eye
    return m


def render_isolated(
    scene: ViewScene,
    role: str,
    out_path: Path,
    *,
    device: str = "cuda",
    size: tuple[int, int] = (720, 720),
    orbit_seconds: float = 12.0,
    log: LogFn = print,
) -> Path:
    """One fighter alone, centred, with the camera orbiting at chest height."""
    splat = Splatter(scene.faces, device)
    h, w = size
    f = 1.2 * w
    k = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    raw_path = out_path.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter.fourcc(*"mp4v"), scene.fps, (w, h))
    valid = scene.valid[role]
    pelvis = 0.5 * (scene.kp3d[role][:, 9] + scene.kp3d[role][:, 10])
    centre = pelvis[valid].mean(axis=0) if valid.any() else np.zeros(3)
    grid = []
    for x in np.linspace(-3, 3, 13):
        grid.append((np.array([x, 0, -3]), np.array([x, 0, 3])))
        grid.append((np.array([-3, 0, x]), np.array([3, 0, x])))
    track = centre.copy()
    for local in range(scene.frames.shape[0]):
        if valid[local]:
            track = 0.9 * track + 0.1 * pelvis[local]
        angle = 2 * np.pi * (local / scene.fps) / orbit_seconds
        target = np.array([track[0], 1.0, track[2]])
        eye = target + np.array([3.6 * np.sin(angle), 0.4, 3.6 * np.cos(angle)])
        cam = _look_at(eye, target)
        frame = np.full((h, w, 3), 24, np.uint8)
        for a, b in grid:
            pa, pb = apply_transform(
                cam, np.stack([a + [track[0], 0, track[2]], b + [track[0], 0, track[2]]])
            )
            if pa[2] > 0.1 and pb[2] > 0.1:
                ua = (int(pa[0] / pa[2] * f + w / 2), int(pa[1] / pa[2] * f + h / 2))
                ub = (int(pb[0] / pb[2] * f + w / 2), int(pb[1] / pb[2] * f + h / 2))
                cv2.line(frame, ua, ub, (60, 60, 60), 1, cv2.LINE_AA)
        if valid[local]:
            img, mask = splat.render(
                [(apply_transform(cam, scene.verts[role][local]), ROLE_RGB.get(role, (0, 255, 0)))],
                k,
                (h, w),
            )
            frame[mask] = img[..., ::-1][mask]
        cv2.putText(
            frame,
            f"{role.upper()}  frame {int(scene.frames[local])}",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (230, 230, 230),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
        if local % 200 == 0:
            log(f"render isolated {role}: {local}/{scene.frames.shape[0]}")
    writer.release()
    if transcode_h264(raw_path, out_path):
        raw_path.unlink(missing_ok=True)
        return out_path
    return raw_path
