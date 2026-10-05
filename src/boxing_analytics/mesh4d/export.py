"""Export a clip of the 4D scene for the browser viewer (three.js, no build step).

Meshes are decimated once by vertex clustering on a template frame; because the topology is fixed,
every later frame just indexes the same vertex subset. Positions are quantised to int16.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from boxing_analytics.mesh4d.contact import REGIONS, FrameContact
from boxing_analytics.mesh4d.geometry import glove_centres
from boxing_analytics.mesh4d.reconstruct import ViewScene

NDArray = np.ndarray[Any, Any]
VIEWER_DIR = Path(__file__).parent / "viewer"


def cluster_decimate(vertices: NDArray, faces: NDArray, cell_m: float) -> tuple[NDArray, NDArray]:
    """Vertex-clustering decimation. Returns (kept original vertex indices, new faces)."""
    keys = np.floor(vertices / cell_m).astype(np.int64)
    _, cluster, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    cluster = cluster.reshape(-1)
    n_clusters = counts.shape[0]
    # Representative = original vertex closest to the cluster mean.
    sums = np.zeros((n_clusters, 3))
    np.add.at(sums, cluster, vertices)
    means = sums / counts[:, None]
    dist = np.linalg.norm(vertices - means[cluster], axis=1)
    order = np.lexsort((dist, cluster))
    first = np.ones(order.shape[0], dtype=bool)
    first[1:] = cluster[order][1:] != cluster[order][:-1]
    rep = np.empty(n_clusters, dtype=np.int64)
    rep[cluster[order][first]] = order[first]
    new_faces = cluster[faces]
    good = (
        (new_faces[:, 0] != new_faces[:, 1])
        & (new_faces[:, 1] != new_faces[:, 2])
        & (new_faces[:, 0] != new_faces[:, 2])
    )
    new_faces = new_faces[good]
    _, first_idx = np.unique(np.sort(new_faces, axis=1), axis=0, return_index=True)
    return rep, new_faces[np.sort(first_idx)].astype(np.int32)


def export_viewer(
    scene: ViewScene,
    contacts: FrameContact | None,
    events: list[dict[str, Any]],
    out_dir: Path,
    *,
    target_fps: float = 25.0,
    max_frames: int = 1500,
    cell_m: float = 0.035,
    summary: dict[str, Any] | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    step = max(1, int(round(scene.fps / target_fps)))
    local = np.arange(0, scene.frames.shape[0], step)[:max_frames]
    manifest: dict[str, Any] = {
        "fps": scene.fps / step,
        "source_fps": scene.fps,
        "frames": scene.frames[local].tolist(),
        "roles": [],
        "events": events,
        "regions": list(REGIONS),
        "summary": summary or {},
    }
    for role in scene.roles:
        valid = scene.valid[role][local]
        if not valid.any():
            continue
        template_i = int(local[np.flatnonzero(valid)[0]])
        rep, faces = cluster_decimate(scene.verts[role][template_i], scene.faces, cell_m)
        seq = scene.verts[role][local][:, rep, :]  # (F, Vd, 3)
        seq = np.where(valid[:, None, None], seq, np.nan)
        finite = np.isfinite(seq).all(axis=2)
        lo = np.nanmin(seq.reshape(-1, 3), axis=0) - 0.05
        hi = np.nanmax(seq.reshape(-1, 3), axis=0) + 0.05
        q = np.zeros(seq.shape, dtype=np.int16)
        norm = (np.nan_to_num(seq, nan=0.0) - lo) / (hi - lo)
        q[...] = np.clip(np.round(norm * 65534 - 32767), -32767, 32767).astype(np.int16)
        q[~finite] = -32768
        (out_dir / f"{role}_verts.bin").write_bytes(q.tobytes())
        (out_dir / f"{role}_faces.bin").write_bytes(faces.astype(np.uint32).tobytes())
        left, right = glove_centres(scene.kp3d[role][local])
        gloves = np.stack([left, right], axis=1).astype(np.float32)
        gloves[~valid] = np.nan
        entry: dict[str, Any] = {
            "name": role,
            "vertex_count": int(rep.shape[0]),
            "face_count": int(faces.shape[0]),
            "bounds_lo": lo.tolist(),
            "bounds_hi": hi.tolist(),
            "valid": valid.astype(int).tolist(),
            "gloves": np.round(np.nan_to_num(gloves, nan=0.0), 4).tolist(),
        }
        if contacts is not None:
            entry["gaps"] = {
                hand: np.round(
                    np.where(
                        np.isfinite(contacts.gaps[(role, hand)][local]),
                        contacts.gaps[(role, hand)][local],
                        9.0,
                    ),
                    3,
                ).tolist()
                for hand in ("left", "right")
                if (role, hand) in contacts.gaps
            }
        manifest["roles"].append(entry)
    (out_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    shutil.copy(VIEWER_DIR / "index.html", out_dir / "index.html")
    return out_dir
