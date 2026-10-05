from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2

from multi_person_tracker import MultiPersonPoseTracker


@dataclass(frozen=True)
class SystemCheckItem:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class GuidedSystemCheckResult:
    passed: bool
    summary: str
    checks: tuple[SystemCheckItem, ...]
    tracker_mode: str
    tracker_issue: str | None
    frames_scanned: int
    frames_with_track_ids: int
    frames_with_required_roles: int
    required_roles: tuple[str, ...]
    lock_status: dict[str, int | None]


def _required_roles(manual_seeds: dict[str, Any] | None) -> tuple[str, ...]:
    seeds = manual_seeds if isinstance(manual_seeds, dict) else {}
    return tuple(role for role in ("RED", "BLUE") if isinstance(seeds.get(role), dict))


def summarize_guided_system_check(
    *,
    tracker_diagnostics: dict[str, Any],
    lock_status: dict[str, Any] | None,
    manual_seeds: dict[str, Any] | None,
    ring_roi: dict[str, Any] | None,
    frames_scanned: int,
    frames_with_track_ids: int,
    frames_with_required_roles: int,
    min_required_role_frames: int = 3,
) -> GuidedSystemCheckResult:
    required_roles = _required_roles(manual_seeds)
    lock_status = dict(lock_status or {})

    ring_ok = isinstance(ring_roi, dict) and len(ring_roi.get("normalized_points", [])) >= 3
    seeds_ok = len(required_roles) == 2
    tracker_ok = bool(tracker_diagnostics.get("enabled"))
    distinct_roles = (
        len(required_roles) == 2
        and lock_status.get("RED") is not None
        and lock_status.get("BLUE") is not None
        and lock_status.get("RED") != lock_status.get("BLUE")
    )
    evidence_ok = frames_with_required_roles >= min_required_role_frames

    checks = (
        SystemCheckItem(
            "Ring ROI",
            ring_ok,
            (
                "Manual polygon available for ring-gated detection."
                if ring_ok
                else "Manual ring ROI was not captured."
            ),
        ),
        SystemCheckItem(
            "Boxer Labels",
            seeds_ok,
            (
                "RED and BLUE annotations are present."
                if seeds_ok
                else "RED and BLUE annotations are both required."
            ),
        ),
        SystemCheckItem(
            "Tracker IDs",
            tracker_ok,
            (
                "Ultralytics tracker is ready to emit live track IDs."
                if tracker_ok
                else str(tracker_diagnostics.get("issue") or "Tracker is not ready.")
            ),
        ),
        SystemCheckItem(
            "Role Locks",
            distinct_roles,
            (
                f"RED={lock_status.get('RED')} BLUE={lock_status.get('BLUE')}"
                if distinct_roles
                else "RED/BLUE did not lock to two distinct track IDs."
            ),
        ),
        SystemCheckItem(
            "Verification Frames",
            evidence_ok,
            (
                f"Verified {frames_with_required_roles} frames with both fighters live "
                "inside the tracked window."
                if evidence_ok
                else (
                    f"Only verified {frames_with_required_roles} frame(s); "
                    f"need at least {min_required_role_frames}."
                )
            ),
        ),
    )

    passed = all(item.passed for item in checks)
    if passed:
        summary = (
            f"PASS: RED/BLUE locked to live track IDs on {frames_with_required_roles} "
            f"frames after manual setup."
        )
    else:
        failing = next((item for item in checks if not item.passed), None)
        summary = f"FAIL: {failing.detail if failing else 'System check did not complete.'}"

    return GuidedSystemCheckResult(
        passed=passed,
        summary=summary,
        checks=checks,
        tracker_mode=str(tracker_diagnostics.get("mode") or "unknown"),
        tracker_issue=tracker_diagnostics.get("issue"),
        frames_scanned=int(frames_scanned),
        frames_with_track_ids=int(frames_with_track_ids),
        frames_with_required_roles=int(frames_with_required_roles),
        required_roles=required_roles,
        lock_status=lock_status,
    )


def format_guided_system_check(result: GuidedSystemCheckResult) -> str:
    lines = [f"System Check {('PASS' if result.passed else 'FAIL')}: {result.summary}"]
    for item in result.checks:
        lines.append(f"{'PASS' if item.passed else 'FAIL'} {item.name}: {item.detail}")
    lines.append(
        f"Tracker mode: {result.tracker_mode} | frames scanned: {result.frames_scanned} | "
        f"frames with track IDs: {result.frames_with_track_ids}"
    )
    return "\n".join(lines)


def run_guided_system_check(
    video_path: str,
    *,
    backend: str,
    ring_roi: dict[str, Any] | None,
    manual_seeds: dict[str, Any] | None,
    min_required_role_frames: int = 3,
) -> GuidedSystemCheckResult:
    try:
        tracker = MultiPersonPoseTracker(
            backend=backend,
            manual_ring_roi=ring_roi,
            manual_seeds=manual_seeds,
        )
    except Exception as exc:
        return summarize_guided_system_check(
            tracker_diagnostics={
                "enabled": False,
                "mode": "error",
                "issue": f"{type(exc).__name__}: {exc}",
            },
            lock_status={},
            manual_seeds=manual_seeds,
            ring_roi=ring_roi,
            frames_scanned=0,
            frames_with_track_ids=0,
            frames_with_required_roles=0,
            min_required_role_frames=min_required_role_frames,
        )

    tracker_diagnostics = tracker.tracking_diagnostics()
    if not tracker_diagnostics.get("enabled"):
        return summarize_guided_system_check(
            tracker_diagnostics=tracker_diagnostics,
            lock_status=tracker.lock_status(),
            manual_seeds=manual_seeds,
            ring_roi=ring_roi,
            frames_scanned=0,
            frames_with_track_ids=0,
            frames_with_required_roles=0,
            min_required_role_frames=min_required_role_frames,
        )

    required_roles = _required_roles(manual_seeds)
    seed_frames = []
    for role in required_roles:
        seed = manual_seeds.get(role) if isinstance(manual_seeds, dict) else None
        if isinstance(seed, dict):
            seed_frames.append(int(seed.get("frame_idx", 0) or 0))
    start_frame = max(0, min(seed_frames) - 15) if seed_frames else 0
    end_frame = max(seed_frames) + 90 if seed_frames else 180

    frames_scanned = 0
    frames_with_track_ids = 0
    frames_with_required_roles = 0

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            return summarize_guided_system_check(
                tracker_diagnostics={
                    "enabled": False,
                    "mode": "error",
                    "issue": f"Unable to open video for system check: {video_path}",
                },
                lock_status={},
                manual_seeds=manual_seeds,
                ring_roi=ring_roi,
                frames_scanned=0,
                frames_with_track_ids=0,
                frames_with_required_roles=0,
                min_required_role_frames=min_required_role_frames,
            )

        frame_idx = 0
        while frame_idx <= end_frame:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx < start_frame:
                frame_idx += 1
                continue

            poses = tracker.process_frame(frame, frame_idx)
            frames_scanned += 1
            if tracker.latest_tracks():
                frames_with_track_ids += 1

            live_roles = tracker.live_role_status()
            if required_roles and all(
                live_roles.get(role) in poses and live_roles.get(role) is not None
                for role in required_roles
            ):
                frames_with_required_roles += 1
                locked = tracker.lock_status()
                if (
                    locked.get("RED") is not None
                    and locked.get("BLUE") is not None
                    and locked.get("RED") != locked.get("BLUE")
                    and frames_with_required_roles >= min_required_role_frames
                ):
                    break

            frame_idx += 1
    finally:
        cap.release()

    return summarize_guided_system_check(
        tracker_diagnostics=tracker.tracking_diagnostics(),
        lock_status=tracker.lock_status(),
        manual_seeds=manual_seeds,
        ring_roi=ring_roi,
        frames_scanned=frames_scanned,
        frames_with_track_ids=frames_with_track_ids,
        frames_with_required_roles=frames_with_required_roles,
        min_required_role_frames=min_required_role_frames,
    )
