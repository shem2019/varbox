"""Contact analysis over a 4D scene, and the reviewer files: per-frame JSONL, events, summary."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from boxing_analytics.mesh4d.contact import (
    REGIONS,
    ContactConfig,
    FrameContact,
    PunchEvent,
    compute_frame_contacts,
    detect_punches,
    label_vertices,
    tally,
)
from boxing_analytics.mesh4d.geometry import KP
from boxing_analytics.mesh4d.reconstruct import ViewScene

NDArray = np.ndarray[Any, Any]
LogFn = Callable[[str], None]

DISCLAIMER = (
    "Assistive judging evidence only. Contact events come from estimated 3D body meshes and "
    "need review by licensed officials; they are not an official score."
)


def _template_frame(scene: ViewScene, role: str) -> int:
    """Frame with the least occlusion among valid frames, used to label body regions."""
    valid = np.flatnonzero(scene.valid[role])
    if valid.size == 0:
        raise ValueError(f"{role} never has a valid mesh")
    return int(valid[np.argmin(scene.occlusion[role][valid])])


def analyse(
    scene: ViewScene, n_views: int, config: ContactConfig | None = None, log: LogFn = print
) -> tuple[FrameContact, list[PunchEvent], dict[str, NDArray]]:
    fighters = tuple(r for r in ("red", "blue") if r in scene.roles)
    if len(fighters) != 2:
        raise ValueError(f"need both red and blue meshes, have {scene.roles}")
    cfg = config or ContactConfig.for_views(n_views)
    labels = {}
    for r in fighters:
        i = _template_frame(scene, r)
        labels[r] = label_vertices(scene.verts[r][i].astype(np.float64), scene.kp3d[r][i])
        counts = {reg: int(np.sum(labels[r] == k)) for k, reg in enumerate(REGIONS)}
        log(f"contact: {r} region labels from frame {int(scene.frames[i])}: {counts}")
    contacts = compute_frame_contacts(
        fighters,
        {r: scene.verts[r] for r in fighters},
        {r: scene.kp3d[r] for r in fighters},
        {r: scene.valid[r] for r in fighters},
        labels,
        scene.fps,
        cfg,
    )
    events = detect_punches(fighters, contacts, scene.fps, cfg, occlusion=scene.occlusion)
    # Report frames in source-video numbering.
    offset = int(scene.frames[0])
    for e in events:
        e.start_frame += offset
        e.peak_frame += offset
        e.end_frame += offset
        if e.contact_frame is not None:
            e.contact_frame += offset
    log(f"contact: {len(events)} punch events ({cfg.contact_tol_m * 100:.0f} cm contact tolerance)")
    return contacts, events, labels


def round_suggestion(counts: dict[str, dict[str, int]], events: list[PunchEvent]) -> dict[str, Any]:
    """Clean landed punches weighted by confidence; a 10-9 lean only with a clear margin."""
    score = {r: 0.0 for r in counts}
    for e in events:
        if e.outcome == "landed" and e.target in ("head", "torso"):
            score[e.attacker] += e.confidence
    if len(score) != 2:
        return {"lean": None, "reason": "needs exactly two fighters"}
    (a, sa), (b, sb) = sorted(score.items(), key=lambda kv: kv[1], reverse=True)
    margin = sa - sb
    if margin < 1.5:
        return {
            "lean": "even",
            "weighted_landed": score,
            "reason": "margin under 1.5 weighted clean punches",
        }
    return {
        "lean": a,
        "proposal": {a: 10, b: 9},
        "weighted_landed": score,
        "reason": f"{a} landed {margin:.1f} more confidence-weighted clean punches",
    }


def write_reports(
    out_dir: Path,
    scene: ViewScene,
    contacts: FrameContact,
    events: list[PunchEvent],
    *,
    n_views: int,
    run_meta: dict[str, Any],
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fighters = [r for r in ("red", "blue") if r in scene.roles]
    event_dicts = [e.to_dict() for e in events]
    (out_dir / "events.json").write_text(json.dumps(event_dicts, indent=2), encoding="utf-8")
    offset = int(scene.frames[0])
    running = {r: {"thrown": 0, "landed": 0} for r in fighters}
    by_frame: dict[int, list[PunchEvent]] = {}
    for e in events:
        by_frame.setdefault((e.contact_frame or e.peak_frame) - offset, []).append(e)
    with (out_dir / "frames.jsonl").open("w", encoding="utf-8") as fh:
        for i in range(scene.frames.shape[0]):
            for e in by_frame.get(i, []):
                running[e.attacker]["thrown"] += 1
                if e.outcome == "landed" and e.target in ("head", "torso"):
                    running[e.attacker]["landed"] += 1
            row: dict[str, Any] = {
                "frame": int(scene.frames[i]),
                "t": round(float(scene.frames[i]) / scene.fps, 4),
                "fighters": {},
                "events": [e.to_dict() for e in by_frame.get(i, [])],
                "running": {r: dict(v) for r, v in running.items()},
            }
            for r in fighters:
                if not scene.valid[r][i]:
                    row["fighters"][r] = None
                    continue
                kp = scene.kp3d[r][i]
                pelvis = 0.5 * (kp[KP["left_hip"]] + kp[KP["right_hip"]])
                entry: dict[str, Any] = {
                    "pelvis": np.round(pelvis, 3).tolist(),
                    "occlusion": round(float(scene.occlusion[r][i]), 3),
                }
                for hand in ("left", "right"):
                    key = (r, hand)
                    g = contacts.gaps[key][i]
                    entry[f"{hand}_glove"] = {
                        "pos": np.round(contacts.gloves[key][i], 3).tolist(),
                        "speed_mps": round(float(contacts.speed[key][i]), 2),
                        "gap_m": {
                            reg: (round(float(g[k]), 3) if np.isfinite(g[k]) else None)
                            for k, reg in enumerate(REGIONS)
                        },
                    }
                row["fighters"][r] = entry
            fh.write(json.dumps(row) + "\n")
    counts = tally(events, tuple(fighters))
    summary = {
        "disclaimer": DISCLAIMER,
        "views": n_views,
        "frames": int(scene.frames.shape[0]),
        "fps": scene.fps,
        "window_s": [float(scene.frames[0]) / scene.fps, float(scene.frames[-1] + 1) / scene.fps],
        "valid_fraction": {r: round(float(scene.valid[r].mean()), 3) for r in scene.roles},
        "mean_occlusion": {r: round(float(scene.occlusion[r].mean()), 3) for r in scene.roles},
        "tally": counts,
        "suggestion": round_suggestion(counts, events),
        "run": run_meta,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
