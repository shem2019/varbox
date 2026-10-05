"""Package a finished run for the VAR Box web dashboard.

The browser composites per-person layers itself, so the package carries:

- layers.mp4: footage on top, a label map underneath (same frames, perfectly in sync). Each
  tracked person has one grey level in the label map, which the viewer turns into switchable
  layers: real pixels, solid silhouette, 3D body, or hidden.
- plate.jpg: the empty ring, built from the fixed camera, used to hide a person.
- camera.json: intrinsics and camera pose, so meshes can be drawn over the footage.
- mesh/: decimated per-frame meshes (the 4D viewer data).
- events, tallies and the rendered videos.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from boxing_analytics.mesh4d.jsonsafe import safe_dumps
from boxing_analytics.mesh4d.masklets import MaskStore
from boxing_analytics.mesh4d.video_io import iter_frames

NDArray = np.ndarray[Any, Any]
LogFn = Callable[[str], None]

LABEL_LEVELS = {"red": 80, "blue": 160, "referee": 240}
WEB_WIDTH = 960


def _even(value: float) -> int:
    return int(round(value / 2.0)) * 2


def _label_frame(store: MaskStore, frame: int, size: tuple[int, int]) -> NDArray:
    w, h = size
    out = np.zeros((h, w), dtype=np.uint8)
    for role, (_, mask) in store.get(frame).items():
        if mask is None or role not in LABEL_LEVELS:
            continue
        small = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        out[small] = LABEL_LEVELS[role]
    return out


def build_layers_video(
    video_path: str, store: MaskStore, start: int, stop: int, fps: float, out_path: Path
) -> tuple[int, int]:
    """Stacked footage (top) and label map (bottom), H.264, browser friendly."""
    cap = cv2.VideoCapture(video_path)
    src_w, src_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    w = WEB_WIDTH
    h = _even(src_h * w / src_w)
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{2 * h}", "-r", f"{fps}", "-i", "-",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(out_path),
    ]  # fmt: skip
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    for index, frame in iter_frames(video_path, start, stop):
        top = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        label = _label_frame(store, index, (w, h))
        stacked = np.vstack([top, cv2.cvtColor(label, cv2.COLOR_GRAY2BGR)])
        proc.stdin.write(stacked.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed while writing the layers video")
    return w, h


def build_plate(
    video_path: str,
    store: MaskStore,
    start: int,
    stop: int,
    size: tuple[int, int],
    samples: int = 48,
) -> NDArray:
    """Empty-ring background: per-pixel median over frames where no tracked person covers it."""
    w, h = size
    picks = set(np.linspace(start, stop - 1, num=min(samples, stop - start)).astype(int).tolist())
    stack, covered = [], []
    for index, frame in iter_frames(video_path, min(picks), max(picks) + 1):
        if index not in picks:
            continue
        stack.append(cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA))
        label = _label_frame(store, index, (w, h))
        covered.append(cv2.dilate((label > 0).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)
    frames = np.stack(stack).astype(np.float32)
    mask = np.stack(covered)[..., None].repeat(3, axis=3)
    frames[mask] = np.nan
    plate = np.nanmedian(frames, axis=0)
    fallback = np.median(np.stack(stack), axis=0)
    plate = np.where(np.isnan(plate), fallback, plate)
    return plate.astype(np.uint8)


def _tally(events: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for e in events:
        row = out.setdefault(
            e["attacker"],
            {"thrown": 0, "landed_head": 0, "landed_torso": 0, "blocked": 0, "missed": 0},
        )
        row["thrown"] += 1
        if e["outcome"] == "landed" and e.get("target") in ("head", "torso"):
            row[f"landed_{e['target']}"] += 1
        elif e["outcome"] in ("blocked", "missed"):
            row[e["outcome"]] += 1
    return out


def export_web(run_dir: Path, title: str | None = None, log: LogFn = print) -> Path:
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    out = run_dir / "web"
    if out.exists():
        shutil.rmtree(out)
    (out / "mesh").mkdir(parents=True)
    store = MaskStore(run_dir / "view_A" / "masks")
    scene_path = run_dir / "fused.npz" if (run_dir / "fused.npz").exists() else None
    scene_path = scene_path or run_dir / "view_A" / "scene.npz"
    with np.load(scene_path) as z:
        frames = z["frames"]
        world_from_cam = z["world_from_cam"]
        intrinsics = z["intrinsics"]
    start, stop = int(frames[0]), int(frames[-1]) + 1
    cap = cv2.VideoCapture(meta["video_a"])
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    src_w, src_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    started = time.monotonic()
    w, h = build_layers_video(meta["video_a"], store, start, stop, fps, out / "layers.mp4")
    log(f"web: layers video {w}x{2 * h} in {time.monotonic() - started:.0f}s")
    plate = build_plate(meta["video_a"], store, start, stop, (w, h))
    cv2.imwrite(str(out / "plate.jpg"), plate, [cv2.IMWRITE_JPEG_QUALITY, 90])
    poster = next(iter_frames(meta["video_a"], (start + stop) // 2, (start + stop) // 2 + 1))[1]
    cv2.imwrite(
        str(out / "poster.jpg"),
        cv2.resize(poster, (w, h), interpolation=cv2.INTER_AREA),
        [cv2.IMWRITE_JPEG_QUALITY, 85],
    )
    (out / "camera.json").write_text(
        safe_dumps(
            {
                "source_size": [src_w, src_h],
                "web_size": [w, h],
                "intrinsics": np.asarray(intrinsics).tolist(),
                "world_from_cam": np.asarray(world_from_cam).tolist(),
                "fps": fps,
                "start_frame": start,
                "frame_count": stop - start,
                "label_levels": LABEL_LEVELS,
                "roles": store.roles,
            }
        ),
        encoding="utf-8",
    )

    viewer = run_dir / "viewer_combined"
    if not viewer.exists():
        viewer = run_dir / "viewer"
    for f in viewer.iterdir():
        if f.suffix in (".bin", ".json"):
            shutil.copy(f, out / "mesh" / f.name)

    def read(path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    events_mesh = read(run_dir / "report" / "events.json") or []
    events_after = read(run_dir / "report_combined" / "events.json")
    (out / "events_before.json").write_text(safe_dumps(events_mesh), encoding="utf-8")
    (out / "events.json").write_text(safe_dumps(events_after or events_mesh), encoding="utf-8")

    videos = {}
    render = run_dir / "render"
    for name in (
        "overlay_A.mp4",
        "overlay_A_combined.mp4",
        "overlay_B.mp4",
        "isolated_red.mp4",
        "isolated_blue.mp4",
    ):
        if (render / name).exists():
            shutil.copy(render / name, out / name)
            videos[name.removesuffix(".mp4")] = name
    if (out / "overlay_A.mp4").exists() and (out / "overlay_A_combined.mp4").exists():
        subprocess.run(
            [
                "ffmpeg", "-v", "error", "-y",
                "-i", str(out / "overlay_A.mp4"), "-i", str(out / "overlay_A_combined.mp4"),
                "-filter_complex", "[0:v]scale=960:-2[a];[1:v]scale=960:-2[b];[a][b]hstack",
                "-c:v", "libx264", "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(out / "before_after.mp4"),
            ],
            check=False,
        )  # fmt: skip
        if (out / "before_after.mp4").exists():
            videos["before_after"] = "before_after.mp4"

    analysis = {
        "title": title or run_dir.name,
        "run": run_dir.name,
        "source_video": Path(meta["video_a"]).name,
        "second_camera": Path(meta["video_b"]).name if meta.get("video_b") else None,
        "views": 2 if (run_dir / "fused.npz").exists() else 1,
        "fps": fps,
        "start_s": round(start / fps, 3),
        "duration_s": round((stop - start) / fps, 3),
        "frames": stop - start,
        "tally_before": _tally(events_mesh),
        "tally_after": _tally(events_after) if events_after else None,
        "punches": len(events_after or events_mesh),
        "videos": videos,
        "has_mesh": (out / "mesh" / "manifest.json").exists(),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    evaluation = run_dir.parent / "evaluation_heldout.json"
    if evaluation.exists():
        analysis["evaluation_note"] = "Scored against the dataset's hand labels"
    (out / "analysis.json").write_text(safe_dumps(analysis, indent=2), encoding="utf-8")
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    log(f"web: package ready at {out} ({size / 1e6:.0f} MB)")
    return out
