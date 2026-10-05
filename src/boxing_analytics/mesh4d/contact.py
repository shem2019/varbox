"""Glove-to-opponent contact analysis on per-frame body meshes.

Every frame yields, for each attacking glove, the gap (metres) between the glove surface and the
opponent's head, torso, guard (arms) and below-belt regions. Punch events come from glove motion:
a fast extension toward the opponent. Each event is then classified from the smallest gaps reached
inside its window: landed (head/body), blocked (guard first) or missed.

The output is evidence for a human judge, never an official score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from boxing_analytics.mesh4d.geometry import KP, glove_centres, normalize

NDArray = np.ndarray[Any, Any]

REGIONS = ("head", "torso", "guard", "low")
LANDING_REGIONS = ("head", "torso")

# Skeleton segments used to label template vertices. Order matters only for readability.
_SEGMENTS: dict[str, list[tuple[str, str]]] = {
    "head": [("neck", "nose"), ("left_ear", "right_ear"), ("left_eye", "right_eye")],
    "torso": [
        ("neck", "mid_hip"),
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
    ],
    "guard": [
        ("left_shoulder", "left_elbow"),
        ("right_shoulder", "right_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_elbow", "right_wrist"),
        ("left_wrist", "left_hand"),
        ("right_wrist", "right_hand"),
    ],
    "low": [
        ("left_hip", "left_knee"),
        ("right_hip", "right_knee"),
        ("left_knee", "left_ankle"),
        ("right_knee", "right_ankle"),
    ],
}


@dataclass(frozen=True)
class ContactConfig:
    glove_radius_m: float = 0.085
    contact_tol_m: float = 0.05
    guard_margin_m: float = 0.02
    punch_speed_mps: float = 3.0
    punch_min_extension_m: float = 0.12
    toward_cos: float = 0.5
    min_gap_frames: int = 6
    vertex_stride: int = 3

    @staticmethod
    def for_views(n_views: int) -> ContactConfig:
        # A single camera cannot separate "in front of" from "touching", so allow more slack.
        return ContactConfig(contact_tol_m=0.05 if n_views >= 2 else 0.09)


@dataclass
class PunchEvent:
    attacker: str
    defender: str
    hand: str
    start_frame: int
    peak_frame: int
    end_frame: int
    contact_frame: int | None
    outcome: str  # landed | blocked | missed
    target: str | None  # head | torso | low | None
    min_gap_m: float
    guard_gap_m: float
    peak_speed_mps: float
    deceleration: float
    confidence: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _named_point(kp: NDArray, name: str) -> NDArray:
    if name == "mid_hip":
        return 0.5 * (kp[KP["left_hip"]] + kp[KP["right_hip"]])
    if name == "left_hand":
        return kp[[49, 53, 57, 61]].mean(axis=0)
    if name == "right_hand":
        return kp[[28, 32, 36, 40]].mean(axis=0)
    return kp[KP[name]]


def _point_segment_distance(points: NDArray, a: NDArray, b: NDArray) -> NDArray:
    ab = b - a
    denom = max(float(ab @ ab), 1e-12)
    t = np.clip((points - a) @ ab / denom, 0.0, 1.0)
    closest = a + t[:, None] * ab
    return np.linalg.norm(points - closest, axis=1)


def label_vertices(vertices: NDArray, kp3d: NDArray) -> NDArray:
    """Assign every template vertex to a region index (into REGIONS) by nearest skeleton segment.

    Run once on a clean, unoccluded frame; the mesh topology is fixed so labels carry over.
    """
    best = np.full(vertices.shape[0], np.inf)
    labels = np.zeros(vertices.shape[0], dtype=np.int8)
    for region_index, region in enumerate(REGIONS):
        for a_name, b_name in _SEGMENTS[region]:
            d = _point_segment_distance(
                vertices, _named_point(kp3d, a_name), _named_point(kp3d, b_name)
            )
            closer = d < best
            best[closer] = d[closer]
            labels[closer] = region_index
    # Anything above the shoulders and near the head axis is head, whatever the segments said.
    neck = kp3d[KP["neck"]]
    up = normalize(neck - 0.5 * (kp3d[KP["left_hip"]] + kp3d[KP["right_hip"]]))
    above = (vertices - neck) @ up > 0.03
    near_axis = (
        np.linalg.norm((vertices - neck) - np.outer((vertices - neck) @ up, up), axis=1) < 0.14
    )
    labels[above & near_axis] = REGIONS.index("head")
    return labels


def region_gaps(
    glove: NDArray,
    opponent_vertices: NDArray,
    labels: NDArray,
    glove_radius_m: float,
) -> dict[str, float]:
    """Gap from a glove sphere surface to each opponent region. Negative means overlap."""
    d = np.linalg.norm(opponent_vertices - glove, axis=1) - glove_radius_m
    out: dict[str, float] = {}
    for region_index, region in enumerate(REGIONS):
        sel = labels == region_index
        out[region] = float(d[sel].min()) if np.any(sel) else float("inf")
    return out


@dataclass
class FrameContact:
    """Per-frame, per attacker-hand gaps plus kinematics, kept as dense arrays."""

    gaps: dict[tuple[str, str], NDArray]  # (attacker, hand) -> (T, len(REGIONS))
    speed: dict[tuple[str, str], NDArray]  # (attacker, hand) -> (T,)
    toward: dict[tuple[str, str], NDArray]  # cosine of velocity vs direction to opponent head
    extension: dict[tuple[str, str], NDArray]  # shoulder -> glove distance
    gloves: dict[tuple[str, str], NDArray]  # (T, 3)
    valid: NDArray  # (T,) both fighters present


def compute_frame_contacts(
    roles: tuple[str, str],
    vertices: dict[str, NDArray],
    kp3d: dict[str, NDArray],
    valid: dict[str, NDArray],
    labels: dict[str, NDArray],
    fps: float,
    config: ContactConfig,
) -> FrameContact:
    """Dense contact signals for both fighters. vertices[role]: (T, V, 3) world metres."""
    a, b = roles
    t_len = kp3d[a].shape[0]
    both = valid[a] & valid[b]
    stride = max(1, config.vertex_stride)
    gaps: dict[tuple[str, str], NDArray] = {}
    speed: dict[tuple[str, str], NDArray] = {}
    toward: dict[tuple[str, str], NDArray] = {}
    extension: dict[tuple[str, str], NDArray] = {}
    gloves: dict[tuple[str, str], NDArray] = {}
    for attacker, defender in ((a, b), (b, a)):
        left, right = glove_centres(kp3d[attacker])
        sub_labels = labels[defender][::stride]
        for hand, centre, shoulder_i in (
            ("left", left, KP["left_shoulder"]),
            ("right", right, KP["right_shoulder"]),
        ):
            key = (attacker, hand)
            g = np.full((t_len, len(REGIONS)), np.inf)
            for i in np.flatnonzero(both):
                rg = region_gaps(
                    centre[i], vertices[defender][i, ::stride], sub_labels, config.glove_radius_m
                )
                g[i] = [rg[r] for r in REGIONS]
            vel = np.gradient(centre, 1.0 / fps, axis=0)
            spd = np.linalg.norm(vel, axis=1)
            target = kp3d[defender][:, KP["nose"]] * 0.5 + kp3d[defender][:, KP["neck"]] * 0.5
            to_target = normalize(target - centre)
            cos = np.sum(normalize(vel) * to_target, axis=1)
            ext = np.linalg.norm(centre - kp3d[attacker][:, shoulder_i], axis=1)
            gaps[key] = g
            speed[key] = np.where(both, spd, 0.0)
            toward[key] = np.where(both, cos, 0.0)
            extension[key] = ext
            gloves[key] = centre
    return FrameContact(gaps, speed, toward, extension, gloves, both)


def _punch_windows(
    spd: NDArray, cos: NDArray, ext: NDArray, valid: NDArray, config: ContactConfig
) -> list[tuple[int, int, int]]:
    """(start, peak, end) frame windows where the glove drives toward the opponent and extends."""
    driving = (spd >= config.punch_speed_mps) & (cos >= config.toward_cos) & valid
    windows: list[tuple[int, int, int]] = []
    i, t_len = 0, spd.shape[0]
    while i < t_len:
        if not driving[i]:
            i += 1
            continue
        start = i
        while i < t_len and driving[i]:
            i += 1
        end = min(t_len - 1, i + 2)  # contact usually lands just after peak speed
        lo = max(0, start - 4)
        if ext[start : end + 1].max() - ext[lo : start + 1].min() < config.punch_min_extension_m:
            continue
        peak = start + int(np.argmax(spd[start:i]))
        windows.append((lo, peak, end))
    merged: list[tuple[int, int, int]] = []
    for w in windows:
        if merged and w[0] - merged[-1][2] < config.min_gap_frames:
            prev = merged[-1]
            peak = prev[1] if spd[prev[1]] >= spd[w[1]] else w[1]
            merged[-1] = (prev[0], peak, w[2])
        else:
            merged.append(w)
    return merged


def detect_punches(
    roles: tuple[str, str],
    contacts: FrameContact,
    fps: float,
    config: ContactConfig,
    occlusion: dict[str, NDArray] | None = None,
) -> list[PunchEvent]:
    events: list[PunchEvent] = []
    head_i, torso_i, guard_i, low_i = (REGIONS.index(r) for r in REGIONS)
    for attacker, defender in ((roles[0], roles[1]), (roles[1], roles[0])):
        for hand in ("left", "right"):
            key = (attacker, hand)
            spd, cos, ext = contacts.speed[key], contacts.toward[key], contacts.extension[key]
            gaps = contacts.gaps[key]
            for start, peak, end in _punch_windows(spd, cos, ext, contacts.valid, config):
                window = gaps[start : end + 1]
                landing = window[:, [head_i, torso_i]]
                best_flat = int(np.argmin(landing))
                best_frame_local, best_region_col = divmod(best_flat, 2)
                min_gap = float(landing.min())
                guard_gap = float(window[:, guard_i].min())
                low_gap = float(window[:, low_i].min())
                contact_frame = start + best_frame_local
                notes: list[str] = []
                # Deceleration: speed just after contact relative to peak.
                after = spd[contact_frame : min(spd.shape[0], contact_frame + 3)]
                decel = (
                    1.0 - float(after.min()) / max(float(spd[peak]), 1e-6) if after.size else 0.0
                )
                guard_first = guard_gap + config.guard_margin_m < min_gap
                if min_gap <= config.contact_tol_m and not guard_first:
                    outcome = "landed"
                    target = ("head", "torso")[best_region_col]
                elif guard_gap <= config.contact_tol_m:
                    outcome = "blocked"
                    target = None
                    contact_frame = start + int(np.argmin(window[:, guard_i]))
                elif low_gap <= config.contact_tol_m:
                    outcome = "landed"
                    target = "low"
                    notes.append("below the belt")
                    contact_frame = start + int(np.argmin(window[:, low_i]))
                else:
                    outcome = "missed"
                    target = None
                margin = config.contact_tol_m - min(min_gap, guard_gap, low_gap)
                confidence = float(np.clip(0.5 + margin / (2 * config.contact_tol_m), 0.05, 0.95))
                if outcome == "landed" and decel < 0.2:
                    confidence *= 0.7
                    notes.append("little glove deceleration at contact")
                if occlusion is not None:
                    occ = float(
                        max(
                            occlusion[attacker][start : end + 1].max(initial=0.0),
                            occlusion[defender][start : end + 1].max(initial=0.0),
                        )
                    )
                    if occ > 0.4:
                        confidence *= 1.0 - 0.6 * min(1.0, occ)
                        notes.append(f"occlusion {occ:.2f} in window")
                events.append(
                    PunchEvent(
                        attacker=attacker,
                        defender=defender,
                        hand=hand,
                        start_frame=int(start),
                        peak_frame=int(peak),
                        end_frame=int(end),
                        contact_frame=None if outcome == "missed" else int(contact_frame),
                        outcome=outcome,
                        target=target,
                        min_gap_m=round(min_gap, 4),
                        guard_gap_m=round(guard_gap, 4),
                        peak_speed_mps=round(float(spd[peak]), 3),
                        deceleration=round(decel, 3),
                        confidence=round(confidence, 3),
                        notes=notes,
                    )
                )
    events.sort(key=lambda e: (e.start_frame, e.attacker, e.hand))
    return events


def tally(events: list[PunchEvent], roles: tuple[str, ...]) -> dict[str, dict[str, int]]:
    out = {
        r: {"thrown": 0, "landed_head": 0, "landed_torso": 0, "blocked": 0, "missed": 0, "low": 0}
        for r in roles
    }
    for e in events:
        row = out[e.attacker]
        row["thrown"] += 1
        if e.outcome == "landed" and e.target in ("head", "torso"):
            row[f"landed_{e.target}"] += 1
        elif e.outcome == "landed" and e.target == "low":
            row["low"] += 1
        elif e.outcome == "blocked":
            row["blocked"] += 1
        else:
            row["missed"] += 1
    return out
