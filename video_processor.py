# video_processor.py
import json
import os
import sys
from collections import defaultdict, deque
from hashlib import sha256
from typing import Callable, Dict, List, Optional, Tuple

# Reduce noisy MediaPipe/absl warnings in production logs.
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import mediapipe as mp
import numpy as np

ROOT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(ROOT_DIR, "src")
if os.path.isdir(SRC_DIR) and SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from boxing_analytics.video.rounds import (
    BoutConfig,
    bout_end_timestamp,
    build_round_segments,
    ended_rounds_by_timestamp,
    round_for_timestamp,
)
from boxing_analytics.video.timeline import (
    CanonicalTimeline,
    TimelineMarker,
    resolve_frame_timestamp,
)
from boxing_analytics.video.progress import ProgressSnapshot, progress_message, progress_percent
from boxing_analytics.calibration import load_profile
from boxing_analytics.detection import EventDeduplicator, evaluate_strike
from boxing_analytics.review import (
    apply_manual_corrections,
    export_evidence_clip,
    parse_manual_corrections,
)
from boxing_analytics.scoring import (
    build_round_criteria,
    evaluate_scoring_gate,
    propose_round_points,
)
from boxing_analytics.tracking import IdentityManager

import config
from multi_person_tracker import MultiPersonPoseTracker
from score_tracker import ScoreTracker
from stats_aggregator import StatsAggregator

POSE = mp.solutions.pose.PoseLandmark
ROLE_COLOR = {"BLUE": (255, 110, 90), "RED": (20, 35, 230), None: (90, 220, 90)}

# Tunables
WARMUP_FRAMES = 90
ATTEMPT_CONFIDENCE = 0.36
IMPACT_MARK_TTL = 10
LANDED_LABELS = {"landed_clean", "landed_glancing"}
ProgressCallback = Callable[[str, Optional[int]], None]
StopCallback = Callable[[], bool]


def _emit_progress(
    progress_cb: Optional[ProgressCallback], message: str, percent: Optional[int] = None
) -> None:
    if progress_cb is None:
        return
    try:
        progress_cb(message, percent)
    except Exception:
        # UI callbacks should never break processing.
        pass


def _as_point(v) -> Optional[Tuple[int, int]]:
    if v is None:
        return None
    if isinstance(v, (tuple, list)) and len(v) >= 2:
        return int(v[0]), int(v[1])
    return None


def _safe_circle(
    image: np.ndarray,
    center: object,
    radius: int,
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    pxy = _as_point(center)
    if pxy is None:
        return
    cv2.circle(image, pxy, radius, color, thickness)


def _apply_output_orientation(frame: np.ndarray) -> np.ndarray:
    mode = str(getattr(config, "OUTPUT_ORIENTATION", "source") or "source").lower()
    direction = str(
        getattr(config, "OUTPUT_ROTATION_DIRECTION", "clockwise") or "clockwise"
    ).lower()
    h, w = frame.shape[:2]

    if mode == "portrait" and w > h:
        rot = (
            cv2.ROTATE_90_COUNTERCLOCKWISE
            if direction == "counterclockwise"
            else cv2.ROTATE_90_CLOCKWISE
        )
        return cv2.rotate(frame, rot)
    if mode == "landscape" and h > w:
        rot = (
            cv2.ROTATE_90_COUNTERCLOCKWISE
            if direction == "counterclockwise"
            else cv2.ROTATE_90_CLOCKWISE
        )
        return cv2.rotate(frame, rot)
    return frame


def _warmup_assign(poses):
    if len(poses) < 2:
        return None, None
    center_x = []
    for bid, d in poses.items():
        x1, _, x2, _ = d["box"]
        center_x.append(((x1 + x2) // 2, bid))
    center_x.sort(key=lambda t: t[0])
    red_id = center_x[0][1]
    blue_id = center_x[-1][1]
    if red_id == blue_id:
        return None, None
    return red_id, blue_id


def _target_point_in_panel(impact, defender_box, panel_w, panel_h):
    x1, y1, x2, y2 = defender_box
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    rel_x = max(0.0, min(1.0, (impact[0] - x1) / bw))
    rel_y = max(0.0, min(1.0, (impact[1] - y1) / bh))
    px = int(70 + rel_x * (panel_w - 140))
    py = int(60 + rel_y * (panel_h - 120))
    return px, py


def _target_zone_or_unknown(zone: str) -> str:
    z = str(zone).strip().lower()
    if z.startswith("head"):
        return "Head"
    if z.startswith("body"):
        return "Body"
    return "Unknown"


def _save_punch_evidence(
    frame,
    frame_idx: int,
    attacker_role: str,
    defender_role: str,
    attacker_box,
    defender_box,
    wrist_pt,
    impact_pt,
    target_zone: str,
    confidence: float,
    out_dir: str,
):
    os.makedirs(out_dir, exist_ok=True)
    annotated = frame.copy()

    cv2.rectangle(
        annotated,
        (attacker_box[0], attacker_box[1]),
        (attacker_box[2], attacker_box[3]),
        ROLE_COLOR[attacker_role],
        3,
    )
    cv2.rectangle(
        annotated,
        (defender_box[0], defender_box[1]),
        (defender_box[2], defender_box[3]),
        ROLE_COLOR[defender_role],
        3,
    )
    cv2.line(annotated, wrist_pt, impact_pt, (0, 255, 255), 3)
    _safe_circle(annotated, wrist_pt, 8, ROLE_COLOR[attacker_role], -1)
    _safe_circle(annotated, impact_pt, 12, (30, 210, 250), 2)
    cv2.putText(
        annotated,
        f"{attacker_role} -> {defender_role}  |  {target_zone}  |  conf {confidence:.2f}",
        (24, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
    )

    h = annotated.shape[0]
    panel_w = 320
    panel = np.full((h, panel_w, 3), 242, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (panel_w - 1, h - 1), (40, 45, 56), 2)
    cv2.putText(panel, "Target Map", (88, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (34, 39, 52), 2)

    # basic silhouette
    head_c = (panel_w // 2, int(h * 0.22))
    head_r = 42
    _safe_circle(panel, head_c, head_r, (80, 88, 102), 2)
    body_top = int(h * 0.30)
    body_bot = int(h * 0.82)
    cv2.rectangle(
        panel, (int(panel_w * 0.32), body_top), (int(panel_w * 0.68), body_bot), (80, 88, 102), 2
    )

    px, py = _target_point_in_panel(impact_pt, defender_box, panel_w, h)
    _safe_circle(panel, (px, py), 11, (24, 24, 210), -1)
    _safe_circle(panel, (px, py), 17, (24, 24, 210), 2)
    cv2.putText(panel, target_zone, (30, h - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (34, 39, 52), 2)

    composite = np.hstack([annotated, panel])
    out_path = os.path.join(out_dir, f"punch_{frame_idx:06d}_{attacker_role}.jpg")
    cv2.imwrite(out_path, composite)
    return out_path


def _draw_wrist_trails(frame, trails):
    for role in ("RED", "BLUE"):
        color = ROLE_COLOR[role]
        for hand in ("L", "R"):
            pts = list(trails[role][hand])
            for i in range(1, len(pts)):
                a = _as_point(pts[i - 1])
                b = _as_point(pts[i])
                if not a or not b:
                    continue
                thickness = max(1, int(1 + (i / max(1, len(pts))) * 2))
                cv2.line(frame, a, b, color, thickness)


def _draw_recent_impacts(frame, impacts):
    keep = []
    for item in impacts:
        pt = item["point"]
        ttl = item["ttl"]
        role = item["role"]
        if ttl <= 0:
            continue
        rad = 8 + (IMPACT_MARK_TTL - ttl)
        _safe_circle(frame, pt, rad, ROLE_COLOR[role], 2)
        _safe_circle(frame, pt, 3, (255, 255, 255), -1)
        item["ttl"] = ttl - 1
        keep.append(item)
    return keep


def _overlay_top_bar(frame, score_tracker, red_name, blue_name, lock_status):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 52), (17, 20, 28), -1)
    cv2.putText(
        frame,
        f"{red_name} (RED)  P:{score_tracker.get_score('RED')}  A:{score_tracker.attempts.get('RED', 0)}",
        (12, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        ROLE_COLOR["RED"],
        2,
    )
    cv2.putText(
        frame,
        f"{blue_name} (BLUE)  P:{score_tracker.get_score('BLUE')}  A:{score_tracker.attempts.get('BLUE', 0)}",
        (12, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        ROLE_COLOR["BLUE"],
        2,
    )

    r_acc = 100.0 * score_tracker.get_score("RED") / max(1, score_tracker.attempts.get("RED", 0))
    b_acc = 100.0 * score_tracker.get_score("BLUE") / max(1, score_tracker.attempts.get("BLUE", 0))
    cv2.putText(
        frame,
        f"Accuracy  R:{r_acc:.1f}%  B:{b_acc:.1f}%",
        (w - 330, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        (190, 215, 245),
        2,
    )
    locked = lock_status.get("RED") is not None and lock_status.get("BLUE") is not None
    lock_text = "Corners Locked" if locked else "Corner Locking..."
    lock_col = (76, 210, 120) if locked else (70, 180, 255)
    cv2.putText(frame, lock_text, (w - 250, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.6, lock_col, 2)


def _timestamp_from_seconds(timestamp_s: float) -> str:
    sec = max(0, int(timestamp_s))
    mm = sec // 60
    ss = sec % 60
    return f"{mm:02}:{ss:02}"


def _decoder_pts_seconds(cap) -> Optional[float]:
    pts_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
    if pts_msec <= 0:
        return None
    return pts_msec / 1000.0


def _load_bout_config() -> BoutConfig:
    return BoutConfig(
        rounds_count=int(
            os.getenv("VARBOX_ROUNDS_COUNT", str(config.BOUT_ROUNDS_COUNT))
            or config.BOUT_ROUNDS_COUNT
        ),
        round_seconds=float(
            os.getenv("VARBOX_ROUND_SECONDS", str(config.BOUT_ROUND_SECONDS))
            or config.BOUT_ROUND_SECONDS
        ),
        rest_seconds=float(
            os.getenv("VARBOX_REST_SECONDS", str(config.BOUT_REST_SECONDS))
            or config.BOUT_REST_SECONDS
        ),
        warmup_seconds=float(
            os.getenv("VARBOX_WARMUP_SECONDS", str(config.BOUT_WARMUP_SECONDS))
            or config.BOUT_WARMUP_SECONDS
        ),
    )


def _load_calibration_status(profile_path: str) -> dict:
    path = profile_path.strip()
    if not path:
        return {
            "loaded": 0,
            "status": "unverified_no_profile",
            "message": "No calibration profile selected.",
        }
    try:
        profile = load_profile(path)
    except Exception as exc:
        return {
            "loaded": 0,
            "status": "invalid_profile",
            "message": str(exc),
            "profile_path": path,
        }
    return {
        "loaded": 1,
        "status": "loaded",
        "profile_path": path,
        "profile_name": profile.profile_name,
        "resolution": f"{profile.image_width}x{profile.image_height}",
        "reprojection_error": round(float(profile.reprojection_error), 6),
        "has_ring_homography": int(profile.ring_homography is not None),
        "message": "Calibration loaded. Outputs still require validation for official use.",
    }


def _parse_confirmed_ref_events(
    payload: str,
    round_segments,
    total_rounds: int,
) -> list[dict]:
    if not payload:
        return []
    try:
        raw = json.loads(payload)
    except Exception:
        return []
    if not isinstance(raw, list):
        return []

    out: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        event_type = str(item.get("event_type", "")).strip().lower()
        role = str(item.get("role", "")).strip().upper()
        if event_type not in {"knockdown", "foul", "deduction"}:
            continue
        if role not in {"RED", "BLUE"}:
            continue

        raw_round = int(item.get("round", 0) or 0)
        ts_s = float(item.get("timestamp_s", 0.0) or 0.0)
        round_no = raw_round
        if round_no <= 0:
            mapped = round_for_timestamp(ts_s, round_segments)
            round_no = mapped if mapped is not None else 0
        if round_no <= 0 or round_no > total_rounds:
            continue

        points = max(1, int(item.get("points", 1) or 1))
        count = max(1, int(item.get("count", 1) or 1))
        out.append(
            {
                "event_type": event_type,
                "role": role,
                "round": round_no,
                "timestamp_s": round(ts_s, 3),
                "points": points,
                "count": count,
                "note": str(item.get("note", "")).strip(),
            }
        )
    return out


def _apply_confirmed_ref_events(stats: StatsAggregator, events: list[dict]) -> None:
    for event in events:
        event_type = event["event_type"]
        role = event["role"]
        round_no = int(event["round"])
        if event_type == "knockdown":
            stats.add_knockdown(role_down=role, round_no=round_no, count=int(event.get("count", 1)))
        elif event_type == "deduction":
            stats.add_deduction(role=role, round_no=round_no, points=int(event.get("points", 1)))
        elif event_type == "foul":
            stats.add_foul(role=role, round_no=round_no, count=int(event.get("count", 1)))


def _rebuild_round_stats(stats: StatsAggregator, punch_log: list[dict], total_rounds: int) -> None:
    stats.round_stats = {
        r: {"RED": {"landed": 0}, "BLUE": {"landed": 0}} for r in range(1, total_rounds + 1)
    }
    for row in punch_log:
        if int(row.get("invalidated_by_review", 0) or 0):
            continue
        role = str(row.get("role", "")).strip().upper()
        if role not in {"RED", "BLUE"}:
            continue
        round_no = int(row.get("round", 0) or 0)
        if round_no <= 0 or round_no > total_rounds:
            continue
        label = str(row.get("classification_label", "")).strip().lower()
        if label in LANDED_LABELS:
            stats.round_stats[round_no][role]["landed"] += 1


def _manual_corrections_hash(corrections_applied: list[dict]) -> str:
    encoded = json.dumps(corrections_applied, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def _write_analysis_snapshot(
    *,
    snapshot_path: str,
    score_tracker: ScoreTracker,
    output_video: str,
) -> None:
    if not snapshot_path.strip():
        return
    payload = {
        "generated_monotonic_s": round(
            float(cv2.getTickCount() / max(1.0, cv2.getTickFrequency())),
            6,
        ),
        "output_video": output_video,
        "metadata": score_tracker.metadata,
        "punch_log": score_tracker.punch_log,
        "round_points": score_tracker.round_points,
        "ten_point_totals": score_tracker.ten_point_totals,
    }
    try:
        os.makedirs(os.path.dirname(snapshot_path) or ".", exist_ok=True)
        with open(snapshot_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    except Exception:
        pass


def _evaluate_and_record_strike(
    *,
    attacker_role: str,
    defender_role: str,
    attacker_name: str,
    defender_name: str,
    attacker_data: dict,
    defender_data: dict,
    last_wrists: dict,
    attempt_dedup: EventDeduplicator,
    contact_dedup: EventDeduplicator,
    score_tracker: ScoreTracker,
    stats: StatsAggregator,
    frame,
    frame_idx: int,
    timestamp_s: float,
    in_round: bool,
    current_round: int | None,
    evidence_dir: str,
    clip_dir: str,
    input_video: str,
) -> tuple[list[dict], list[dict]]:
    events: list[dict] = []
    impacts: list[dict] = []

    attacker_k = attacker_data["keypoints"]
    defender_k = defender_data["keypoints"]
    result = evaluate_strike(
        attacker_keypoints=attacker_k,
        defender_keypoints=defender_k,
        prev_wrists=last_wrists[attacker_role],
        attacker_box=attacker_data["box"],
        defender_box=defender_data["box"],
    )

    hand = result.hand if result.hand in ("L", "R") else "ANY"
    glove_pos = result.glove_position
    if in_round and result.confidence >= ATTEMPT_CONFIDENCE:
        if attempt_dedup.allow_attempt(attacker_role, hand, timestamp_s, glove_pos):
            score_tracker.register_attempt(attacker_role)

    event_row = {
        "timestamp_s": round(timestamp_s, 3),
        "role": attacker_role,
        "opponent_role": defender_role,
        "label": result.label,
        "confidence": round(float(result.confidence), 3),
        "hand": hand,
        "target_zone": _target_zone_or_unknown(result.target_zone),
        "round": current_round or 0,
        "features": {k: round(float(v), 4) for k, v in result.features.items()},
    }
    events.append(event_row)

    if not in_round or result.label not in LANDED_LABELS:
        return events, impacts
    if not contact_dedup.allow_contact(attacker_role, hand, result.label, timestamp_s, glove_pos):
        return events, impacts

    impact_pt = result.impact_point
    if impact_pt is None:
        impact_pt = _as_point(defender_k.get(POSE.NOSE))
    if impact_pt is None:
        return events, impacts

    wrist_pt = glove_pos
    if wrist_pt is None:
        fallback = POSE.LEFT_WRIST if hand == "L" else POSE.RIGHT_WRIST
        wrist_pt = _as_point(attacker_k.get(fallback))
    if wrist_pt is None:
        wrist_pt = impact_pt

    clip_path = export_evidence_clip(
        input_video=input_video,
        timestamp_s=timestamp_s,
        out_dir=clip_dir,
        tag=f"{attacker_role}_{result.label}",
        pre_s=0.5,
        post_s=0.5,
    )
    if not clip_path:
        return events, impacts

    details = {
        "confidence": round(float(result.confidence), 3),
        "classification_label": result.label,
        "target_zone": _target_zone_or_unknown(result.target_zone),
        "opponent_role": defender_role,
        "opponent_name": defender_name,
        "fighter_name": attacker_name,
        "round": current_round or 0,
        "feature_speed": round(float(result.features.get("speed", 0.0)), 3),
        "feature_extension": round(float(result.features.get("extension", 0.0)), 3),
        "feature_target_dist": round(float(result.features.get("target_dist", 0.0)), 3),
        "feature_guard_cover": round(float(result.features.get("guard_cover", 0.0)), 3),
        "feature_clinch_score": round(float(result.features.get("clinch_score", 0.0)), 3),
        "feature_overlap": round(float(result.features.get("overlap", 0.0)), 3),
        "evidence_clip": clip_path,
    }
    accepted = score_tracker.update(
        frame_idx=frame_idx,
        fighter_id=attacker_role,
        timestamp=_timestamp_from_seconds(timestamp_s),
        hand=hand,
        event_time_s=timestamp_s,
        details=details,
    )
    if not accepted:
        return events, impacts

    evidence_path = _save_punch_evidence(
        frame=frame,
        frame_idx=frame_idx,
        attacker_role=attacker_role,
        defender_role=defender_role,
        attacker_box=attacker_data["box"],
        defender_box=defender_data["box"],
        wrist_pt=wrist_pt,
        impact_pt=impact_pt,
        target_zone=details["target_zone"],
        confidence=details["confidence"],
        out_dir=evidence_dir,
    )
    score_tracker.punch_log[-1]["evidence_image"] = evidence_path
    score_tracker.punch_log[-1]["evidence_clip"] = clip_path
    impacts.append({"point": impact_pt, "ttl": IMPACT_MARK_TTL, "role": attacker_role})

    if current_round is not None:
        stats.add_punch(attacker_role, current_round)

    return events, impacts


def process_video(
    score_tracker: ScoreTracker,
    progress_cb: Optional[ProgressCallback] = None,
    should_stop: Optional[StopCallback] = None,
):
    input_video = os.getenv("VARBOX_INPUT", config.INPUT_VIDEO)
    output_video = os.getenv("VARBOX_OUTPUT", config.OUTPUT_VIDEO)
    metadata_snapshot = os.getenv(
        "VARBOX_METADATA_PATH",
        os.path.join(os.path.dirname(output_video), "analysis_metadata.json"),
    )
    evidence_dir = os.getenv("VARBOX_EVIDENCE_DIR", config.PUNCH_EVIDENCE_DIR)
    clip_dir = os.path.join(os.path.dirname(evidence_dir), "event_clips")
    backend = os.getenv("VARBOX_BACKEND", config.BACKEND)
    fps_override = int(os.getenv("VARBOX_FPS_OVERRIDE", "0") or "0")
    red_name = os.getenv("VARBOX_RED_NAME", "Red Corner")
    blue_name = os.getenv("VARBOX_BLUE_NAME", "Blue Corner")
    unknown_corners_mode = bool(int(os.getenv("VARBOX_UNKNOWN_CORNERS", "0") or "0"))
    round_start_offset_s = float(
        os.getenv("VARBOX_ROUND_START_OFFSET_SECONDS", str(config.ROUND_START_OFFSET_SECONDS))
        or config.ROUND_START_OFFSET_SECONDS
    )
    calibration_profile_path = os.getenv("VARBOX_CALIBRATION_PROFILE", "").strip()
    stop_at_bout_end = bool(
        int(
            os.getenv("VARBOX_STOP_AT_BOUT_END", str(config.STOP_AT_BOUT_END))
            or config.STOP_AT_BOUT_END
        )
    )
    bout_config = _load_bout_config()
    bout_config.validate()
    round_segments = build_round_segments(
        start_s=max(0.0, round_start_offset_s), config=bout_config
    )
    final_round_end_s = bout_end_timestamp(round_segments)

    os.makedirs(os.path.dirname(output_video) or ".", exist_ok=True)
    os.makedirs(evidence_dir, exist_ok=True)
    os.makedirs(clip_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_video}")

    native_fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = fps_override or native_fps
    progress_interval_frames = max(1, int(max(1, native_fps) * 5))
    id_viterbi_window = int(os.getenv("VARBOX_ID_VITERBI_WINDOW", "25") or "25")
    id_switch_penalty = float(os.getenv("VARBOX_ID_SWITCH_PENALTY", "3.0") or "3.0")
    id_base_switch_penalty = float(os.getenv("VARBOX_ID_BASE_SWITCH_PENALTY", "4.0") or "4.0")
    id_occlusion_switch_boost = float(os.getenv("VARBOX_ID_OCCLUSION_SWITCH_BOOST", "8.0") or "8.0")
    id_w_motion = float(os.getenv("VARBOX_ID_W_MOTION", "1.2") or "1.2")
    id_w_appear = float(os.getenv("VARBOX_ID_W_APPEAR", "0.5") or "0.5")
    id_w_pose = float(os.getenv("VARBOX_ID_W_POSE", "0.3") or "0.3")
    id_sigma_motion_px = float(os.getenv("VARBOX_ID_SIGMA_MOTION_PX", "45.0") or "45.0")
    id_iou_occlusion_full = float(os.getenv("VARBOX_ID_IOU_OCCLUSION_FULL", "0.25") or "0.25")
    id_occlusion_app_suppress = float(os.getenv("VARBOX_ID_OCCLUSION_APP_SUPPRESS", "0.9") or "0.9")
    id_occlusion_update_max = float(os.getenv("VARBOX_ID_OCCLUSION_UPDATE_MAX", "0.15") or "0.15")
    id_update_max_occlusion = float(os.getenv("VARBOX_ID_UPDATE_MAX_OCCLUSION", "0.08") or "0.08")
    id_min_det_conf = float(os.getenv("VARBOX_ID_MIN_DET_CONF", "0.4") or "0.4")
    id_min_kp_vis = float(os.getenv("VARBOX_ID_MIN_KP_VIS", "0.60") or "0.60")
    id_max_coast_frames = int(os.getenv("VARBOX_ID_MAX_COAST_FRAMES", "60") or "60")
    id_miss_penalty = float(os.getenv("VARBOX_ID_MISS_PENALTY", "8.0") or "8.0")
    id_flow_max_frames = int(os.getenv("VARBOX_ID_FLOW_MAX_FRAMES", "12") or "12")
    id_commit_p_threshold = float(os.getenv("VARBOX_ID_COMMIT_P_THRESHOLD", "0.98") or "0.98")
    id_commit_seconds = float(os.getenv("VARBOX_ID_COMMIT_SECONDS", "1.0") or "1.0")
    id_commit_max_occlusion = float(os.getenv("VARBOX_ID_COMMIT_MAX_OCCLUSION", "0.10") or "0.10")
    stitch_max_dist_px = float(os.getenv("VARBOX_STITCH_MAX_DIST_PX", "80.0") or "80.0")
    stitch_embed_max = float(os.getenv("VARBOX_STITCH_EMBED_MAX", "0.35") or "0.35")
    stitch_w_motion = float(os.getenv("VARBOX_STITCH_W_MOTION", "1.0") or "1.0")
    stitch_w_appear = float(os.getenv("VARBOX_STITCH_W_APPEAR", "0.6") or "0.6")
    stitch_w_pose = float(os.getenv("VARBOX_STITCH_W_POSE", "0.2") or "0.2")

    dep_switch_margin = os.getenv("VARBOX_ID_SWITCH_MARGIN", "").strip()
    dep_min_frames_before_switch = os.getenv("VARBOX_ID_MIN_FRAMES_BEFORE_SWITCH", "").strip()
    dep_clinch_iou_freeze = os.getenv("VARBOX_ID_CLINCH_IOU_FREEZE", "").strip()
    dep_clinch_freeze_frames = os.getenv("VARBOX_ID_CLINCH_FREEZE_FRAMES", "").strip()
    lock_corner_orientation = bool(
        int(
            os.getenv(
                "VARBOX_LOCK_CORNER_ORIENTATION",
                str(getattr(config, "LOCK_CORNER_ORIENTATION", 1)),
            )
            or "1"
        )
    )

    _emit_progress(
        progress_cb,
        (
            f"Opened video stream | fps={native_fps} "
            f"frames={total_frames if total_frames > 0 else 'unknown'}"
        ),
        0 if total_frames > 0 else None,
    )

    writer_size = None
    out = None

    pose_tracker = MultiPersonPoseTracker(bootstrap_frames=30, backend=backend)
    identity = IdentityManager(
        max_missing_frames=max(45, int(1.5 * fps)),
        viterbi_window=id_viterbi_window,
        switch_penalty=id_switch_penalty,
        base_switch_penalty=id_base_switch_penalty,
        occlusion_switch_boost=id_occlusion_switch_boost,
        w_motion=id_w_motion,
        w_appear=id_w_appear,
        w_pose=id_w_pose,
        sigma_motion_px=id_sigma_motion_px,
        iou_occlusion_full=id_iou_occlusion_full,
        occlusion_app_suppress=id_occlusion_app_suppress,
        occlusion_update_max=id_occlusion_update_max,
        update_max_occlusion=id_update_max_occlusion,
        min_det_conf=id_min_det_conf,
        min_kp_vis=id_min_kp_vis,
        max_coast_frames=id_max_coast_frames,
        miss_penalty=id_miss_penalty,
        flow_max_frames=id_flow_max_frames,
        commit_p_threshold=id_commit_p_threshold,
        commit_seconds=id_commit_seconds,
        commit_max_occlusion=id_commit_max_occlusion,
        stitch_max_dist_px=stitch_max_dist_px,
        stitch_embed_max=stitch_embed_max,
        stitch_w_motion=stitch_w_motion,
        stitch_w_appear=stitch_w_appear,
        stitch_w_pose=stitch_w_pose,
        switch_margin=float(dep_switch_margin) if dep_switch_margin else None,
        min_frames_before_switch=(
            int(dep_min_frames_before_switch) if dep_min_frames_before_switch else None
        ),
        clinch_iou_freeze_threshold=float(dep_clinch_iou_freeze) if dep_clinch_iou_freeze else None,
        clinch_freeze_frames=int(dep_clinch_freeze_frames) if dep_clinch_freeze_frames else None,
        lock_role_orientation=lock_corner_orientation,
    )
    stats = StatsAggregator(total_rounds=bout_config.rounds_count)
    timeline = CanonicalTimeline()

    score_tracker.metadata.setdefault("title", "VAR Box Match Analysis")
    score_tracker.metadata.setdefault("subtitle", os.path.basename(input_video))
    score_tracker.metadata["red_name"] = red_name
    score_tracker.metadata["blue_name"] = blue_name
    score_tracker.metadata["unknown_corners_mode"] = int(unknown_corners_mode)
    score_tracker.metadata["backend"] = backend
    score_tracker.metadata["identity_tuning"] = {
        "viterbi_window": id_viterbi_window,
        "switch_penalty": id_switch_penalty,
        "base_switch_penalty": id_base_switch_penalty,
        "occlusion_switch_boost": id_occlusion_switch_boost,
        "w_motion": id_w_motion,
        "w_appear": id_w_appear,
        "w_pose": id_w_pose,
        "sigma_motion_px": id_sigma_motion_px,
        "iou_occlusion_full": id_iou_occlusion_full,
        "occlusion_app_suppress": id_occlusion_app_suppress,
        "occlusion_update_max": id_occlusion_update_max,
        "update_max_occlusion": id_update_max_occlusion,
        "min_det_conf": id_min_det_conf,
        "min_kp_vis": id_min_kp_vis,
        "max_coast_frames": id_max_coast_frames,
        "miss_penalty": id_miss_penalty,
        "flow_max_frames": id_flow_max_frames,
        "commit_p_threshold": id_commit_p_threshold,
        "commit_seconds": id_commit_seconds,
        "commit_max_occlusion": id_commit_max_occlusion,
        "stitch_max_dist_px": stitch_max_dist_px,
        "stitch_embed_max": stitch_embed_max,
        "stitch_w_motion": stitch_w_motion,
        "stitch_w_appear": stitch_w_appear,
        "stitch_w_pose": stitch_w_pose,
        "deprecated_switch_margin": dep_switch_margin,
        "deprecated_min_frames_before_switch": dep_min_frames_before_switch,
        "deprecated_clinch_iou_freeze": dep_clinch_iou_freeze,
        "deprecated_clinch_freeze_frames": dep_clinch_freeze_frames,
        "lock_corner_orientation": int(lock_corner_orientation),
    }
    score_tracker.metadata["input_video"] = input_video
    score_tracker.metadata["decision_support_notice"] = (
        "Assistive judging analytics only. Requires human judge/referee confirmation."
    )
    score_tracker.metadata["calibration"] = _load_calibration_status(calibration_profile_path)
    score_tracker.metadata["bout_config"] = {
        "rounds_count": bout_config.rounds_count,
        "round_seconds": bout_config.round_seconds,
        "rest_seconds": bout_config.rest_seconds,
        "warmup_seconds": bout_config.warmup_seconds,
        "round_start_offset_seconds": round_start_offset_s,
    }
    score_tracker.metadata["round_segments"] = [
        {"round": seg.round_index, "start_s": round(seg.start_s, 3), "end_s": round(seg.end_s, 3)}
        for seg in round_segments
    ]
    timeline.add_marker(
        TimelineMarker(timestamp_s=max(0.0, round_start_offset_s), label="round_start_manual")
    )

    confirmed_ref_events = _parse_confirmed_ref_events(
        payload=os.getenv("VARBOX_REF_EVENTS", "").strip(),
        round_segments=round_segments,
        total_rounds=bout_config.rounds_count,
    )
    manual_corrections = parse_manual_corrections(
        os.getenv("VARBOX_MANUAL_CORRECTIONS", "").strip()
    )
    _apply_confirmed_ref_events(stats, confirmed_ref_events)
    score_tracker.metadata["confirmed_ref_events"] = confirmed_ref_events
    score_tracker.metadata["confirmed_ref_event_flags_present"] = 1
    score_tracker.metadata["manual_corrections_requested"] = [
        {
            "timestamp_s": c.timestamp_s,
            "match_role": c.match_role or "",
            "new_role": c.new_role or "",
            "new_label": c.new_label or "",
            "new_target_zone": c.new_target_zone or "",
            "invalidate": int(c.invalidate),
            "note": c.note,
        }
        for c in manual_corrections
    ]

    if os.getenv("VARBOX_MANUAL_SEEDS", "").strip():
        score_tracker.metadata["manual_seeds_applied"] = []
        score_tracker.metadata["manual_seed_notice"] = (
            "Manual fingerprint seeds ignored: identity now uses motion + appearance re-ID without color priors."
        )

    frame_idx = 0
    last_wrists = defaultdict(lambda: {"L": None, "R": None})
    trails = defaultdict(lambda: {"L": deque(maxlen=10), "R": deque(maxlen=10)})
    impacts = []
    classified_events: List[Dict] = []
    attempt_dedup = EventDeduplicator(
        attempt_window_s=0.22, contact_window_s=0.30, min_travel_px=16.0
    )
    contact_dedup = EventDeduplicator(
        attempt_window_s=0.22, contact_window_s=0.32, min_travel_px=22.0
    )
    scored_rounds = set()
    current_round = None
    in_round = False
    last_timestamp_s = None
    cancelled = False

    while cap.isOpened():
        if should_stop is not None and should_stop():
            cancelled = True
            _emit_progress(
                progress_cb, "Cancellation requested. Stopping frame processing...", None
            )
            break
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        frame = _apply_output_orientation(frame)

        ts_row = resolve_frame_timestamp(
            frame_index=frame_idx,
            pts_seconds=_decoder_pts_seconds(cap),
            fallback_fps=float(native_fps),
            last_timestamp_s=last_timestamp_s,
        )
        last_timestamp_s = ts_row.timestamp_s
        timeline.add_frame_timestamp(ts_row)
        timestamp_s = ts_row.timestamp_s
        if frame_idx == 1 or frame_idx % progress_interval_frames == 0:
            snapshot = ProgressSnapshot(
                frame_idx=frame_idx,
                total_frames=total_frames,
                timestamp_s=timestamp_s,
                events_count=len(score_tracker.punch_log),
            )
            _emit_progress(
                progress_cb,
                progress_message(snapshot),
                progress_percent(frame_idx, total_frames),
            )

        fh, fw = frame.shape[:2]
        if out is None:
            writer_size = (int(fw), int(fh))
            out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, writer_size)
            score_tracker.metadata["input_resolution"] = f"{fw}x{fh}"
            score_tracker.metadata["output_resolution"] = f"{fw}x{fh}"
            score_tracker.metadata["output_orientation_mode"] = str(
                getattr(config, "OUTPUT_ORIENTATION", "source") or "source"
            ).lower()
            score_tracker.metadata["output_rotation_direction"] = str(
                getattr(config, "OUTPUT_ROTATION_DIRECTION", "clockwise") or "clockwise"
            ).lower()
            score_tracker.metadata["preserved_orientation"] = int(
                score_tracker.metadata["output_orientation_mode"] == "source"
            )
        elif writer_size is not None and (fw, fh) != writer_size:
            frame = cv2.resize(frame, writer_size, interpolation=cv2.INTER_LINEAR)
            fh, fw = frame.shape[:2]

        current_round = round_for_timestamp(timestamp_s, round_segments)
        in_round = current_round is not None
        due_rounds = ended_rounds_by_timestamp(
            timestamp_s=timestamp_s, segments=round_segments, already_scored=scored_rounds
        )
        bout_over = stop_at_bout_end and timestamp_s >= final_round_end_s > 0
        poses = pose_tracker.process_frame(frame, frame_idx)

        identity.update(frame, poses, frame_idx=frame_idx, timestamp_s=timestamp_s)
        red_id = identity.id_for_role("RED")
        blue_id = identity.id_for_role("BLUE")

        if (red_id is None or blue_id is None) and frame_idx <= WARMUP_FRAMES:
            f_red, f_blue = _warmup_assign(poses)
            red_id = red_id if red_id is not None else f_red
            blue_id = blue_id if blue_id is not None else f_blue

        if red_id is not None and blue_id is not None and red_id in poses and blue_id in poses:
            d_red, d_blue = poses[red_id], poses[blue_id]
            k_red, k_blue = d_red["keypoints"], d_blue["keypoints"]

            red_lw = _as_point(k_red.get(POSE.LEFT_WRIST))
            red_rw = _as_point(k_red.get(POSE.RIGHT_WRIST))
            blue_lw = _as_point(k_blue.get(POSE.LEFT_WRIST))
            blue_rw = _as_point(k_blue.get(POSE.RIGHT_WRIST))

            if red_lw:
                trails["RED"]["L"].append(red_lw)
            if red_rw:
                trails["RED"]["R"].append(red_rw)
            if blue_lw:
                trails["BLUE"]["L"].append(blue_lw)
            if blue_rw:
                trails["BLUE"]["R"].append(blue_rw)

            red_events, red_impacts = _evaluate_and_record_strike(
                attacker_role="RED",
                defender_role="BLUE",
                attacker_name=red_name,
                defender_name=blue_name,
                attacker_data=d_red,
                defender_data=d_blue,
                last_wrists=last_wrists,
                attempt_dedup=attempt_dedup,
                contact_dedup=contact_dedup,
                score_tracker=score_tracker,
                stats=stats,
                frame=frame,
                frame_idx=frame_idx,
                timestamp_s=timestamp_s,
                in_round=in_round,
                current_round=current_round,
                evidence_dir=evidence_dir,
                clip_dir=clip_dir,
                input_video=input_video,
            )
            classified_events.extend(red_events)
            impacts.extend(red_impacts)

            blue_events, blue_impacts = _evaluate_and_record_strike(
                attacker_role="BLUE",
                defender_role="RED",
                attacker_name=blue_name,
                defender_name=red_name,
                attacker_data=d_blue,
                defender_data=d_red,
                last_wrists=last_wrists,
                attempt_dedup=attempt_dedup,
                contact_dedup=contact_dedup,
                score_tracker=score_tracker,
                stats=stats,
                frame=frame,
                frame_idx=frame_idx,
                timestamp_s=timestamp_s,
                in_round=in_round,
                current_round=current_round,
                evidence_dir=evidence_dir,
                clip_dir=clip_dir,
                input_video=input_video,
            )
            classified_events.extend(blue_events)
            impacts.extend(blue_impacts)

            if red_lw:
                last_wrists["RED"]["L"] = red_lw
            if red_rw:
                last_wrists["RED"]["R"] = red_rw
            if blue_lw:
                last_wrists["BLUE"]["L"] = blue_lw
            if blue_rw:
                last_wrists["BLUE"]["R"] = blue_rw

        for ended_round in due_rounds:
            scored_rounds.add(ended_round)

            cv2.rectangle(frame, (0, fh - 44), (fw, fh), (17, 20, 28), -1)
            cv2.putText(
                frame,
                (
                    f"Round {ended_round} ended | assistant review pending "
                    "(decision support, human confirmation required)"
                ),
                (12, fh - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (82, 218, 255),
                2,
            )

        for bid, data in poses.items():
            x1, y1, x2, y2 = data["box"]
            role = identity.role_for_id(bid)
            if role is None and frame_idx <= WARMUP_FRAMES:
                if red_id == bid:
                    role = "RED"
                elif blue_id == bid:
                    role = "BLUE"
            color = ROLE_COLOR.get(role, ROLE_COLOR[None])

            mask = data.get("mask")
            if mask is not None:
                try:
                    mr = cv2.resize(mask, (x2 - x1, y2 - y1))
                    mb = (mr > 0.1).astype("uint8") * 255
                    m_rgb = cv2.applyColorMap(mb, cv2.COLORMAP_OCEAN)
                    roi = frame[y1:y2, x1:x2]
                    frame[y1:y2, x1:x2] = cv2.addWeighted(roi, 0.7, m_rgb, 0.3, 0)
                except Exception:
                    pass

            p_count = score_tracker.get_score(role) if role in ("RED", "BLUE") else 0
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{role or 'TRACK'} | ID {bid} | P:{p_count}",
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                color,
                2,
            )

            for pt in data["keypoints"].values():
                pxy = _as_point(pt)
                if pxy is None:
                    continue
                _safe_circle(frame, pxy, 3, color, -1)

        _draw_wrist_trails(frame, trails)
        impacts = _draw_recent_impacts(frame, impacts)
        _overlay_top_bar(frame, score_tracker, red_name, blue_name, identity.role_status())

        out.write(frame)
        if bout_over:
            break

    cap.release()
    if out is not None:
        out.release()

    if cancelled:
        score_tracker.metadata["cancelled"] = 1
        score_tracker.metadata["cancelled_reason"] = "cancelled_by_operator"
        score_tracker.metadata["cancelled_at_frame"] = frame_idx
        score_tracker.metadata["cancelled_at_timestamp_s"] = round(
            float(last_timestamp_s or 0.0), 3
        )
        score_tracker.metadata["scoring_mode"] = "cancelled_no_score"
        score_tracker.metadata["no_official_score"] = 1
        score_tracker.metadata["no_official_score_reason"] = "cancelled_by_operator"
        score_tracker.metadata["round_stats"] = stats.round_stats
        score_tracker.metadata["kd"] = stats.kd
        score_tracker.metadata["deductions"] = stats.deductions
        score_tracker.metadata["fouls"] = stats.fouls
        score_tracker.metadata["classified_events"] = classified_events
        score_tracker.metadata["timeline_markers"] = [
            {"timestamp_s": round(m.timestamp_s, 3), "label": m.label} for m in timeline.markers
        ]
        score_tracker.metadata["timebase_samples"] = len(timeline.frame_timestamps)
        score_tracker.metadata["timebase_source"] = (
            timeline.frame_timestamp_rows[0].source
            if timeline.frame_timestamp_rows
            else "monotonic_fallback"
        )
        score_tracker.metadata["analysis_metadata_path"] = metadata_snapshot
        score_tracker.round_stats = stats.round_stats
        score_tracker.kd = stats.kd
        score_tracker.deductions = stats.deductions
        score_tracker.fouls = stats.fouls
        _write_analysis_snapshot(
            snapshot_path=metadata_snapshot,
            score_tracker=score_tracker,
            output_video=output_video,
        )
        _emit_progress(progress_cb, "Run cancelled by operator. Partial artifacts saved.", None)
        return output_video

    _emit_progress(
        progress_cb, "Applying manual corrections and recomputing scoring signals...", None
    )

    corrected_events, corrected_punch_log, corrections_applied = apply_manual_corrections(
        classified_events=classified_events,
        punch_log=score_tracker.punch_log,
        corrections=manual_corrections,
    )
    classified_events = corrected_events
    score_tracker.punch_log = corrected_punch_log
    _rebuild_round_stats(stats, score_tracker.punch_log, bout_config.rounds_count)

    applied_count = len([row for row in corrections_applied if int(row.get("applied", 0) or 0)])
    score_tracker.metadata["manual_corrections_applied"] = corrections_applied
    score_tracker.metadata["manual_corrections_count"] = applied_count
    if corrections_applied:
        score_tracker.metadata["manual_corrections_hash"] = _manual_corrections_hash(
            corrections_applied
        )
        score_tracker.metadata["scoring_recomputed_from_manual_corrections"] = 1
    else:
        score_tracker.metadata["scoring_recomputed_from_manual_corrections"] = 0

    scoring_events = [
        row for row in classified_events if not int(row.get("invalidated_by_review", 0) or 0)
    ]
    scoring_punch_log = [
        row for row in score_tracker.punch_log if not int(row.get("invalidated_by_review", 0) or 0)
    ]

    criteria_by_round = build_round_criteria(
        classified_events=scoring_events,
        total_rounds=bout_config.rounds_count,
    )
    score_tracker.metadata["criteria_by_round"] = {
        rnd: {
            role: {
                "clean_punching_score": crit.clean_punching_score,
                "effective_aggressiveness_score": crit.effective_aggressiveness_score,
                "ring_generalship_score": crit.ring_generalship_score,
                "defense_score": crit.defense_score,
                "landed_clean": crit.landed_clean,
                "landed_glancing": crit.landed_glancing,
                "blocked_guarded": crit.blocked_guarded,
                "missed": crit.missed,
                "clinch": crit.clinch,
            }
            for role, crit in roles.items()
        }
        for rnd, roles in criteria_by_round.items()
    }

    scoring_gate = evaluate_scoring_gate(
        metadata=score_tracker.metadata,
        classified_events=scoring_events,
        punch_log=scoring_punch_log,
    )
    score_tracker.metadata["scoring_gate"] = {
        "round_alignment_ready": int(scoring_gate.round_alignment_ready),
        "classification_ready": int(scoring_gate.classification_ready),
        "evidence_clips_ready": int(scoring_gate.evidence_clips_ready),
        "ref_events_flag_ready": int(scoring_gate.ref_events_flag_ready),
        "can_propose_ten_point": int(scoring_gate.can_propose_ten_point),
        "missing_reasons": scoring_gate.missing_reasons(),
    }

    if scoring_gate.can_propose_ten_point:
        proposals = propose_round_points(
            criteria=criteria_by_round,
            kd=stats.kd,
            deductions=stats.deductions,
            total_rounds=bout_config.rounds_count,
        )
        for round_no in sorted(scored_rounds):
            if round_no not in proposals:
                continue
            red_pts, blue_pts, rationale = proposals[round_no]
            score_tracker.add_round_points(round_no, red_pts, blue_pts, rationale)
        if score_tracker.round_points:
            score_tracker.metadata["scoring_mode"] = "provisional_10_point_assistant"
            score_tracker.metadata["no_official_score"] = 0
        else:
            score_tracker.metadata["scoring_mode"] = "analytics_only_no_completed_rounds"
            score_tracker.metadata["no_official_score"] = 1
            score_tracker.metadata["no_official_score_reason"] = "no_completed_rounds"
    else:
        score_tracker.metadata["scoring_mode"] = "analytics_only"
        score_tracker.metadata["no_official_score"] = 1
        score_tracker.metadata["no_official_score_reason"] = ",".join(
            scoring_gate.missing_reasons()
        )

    # expose artifacts for scorecard
    score_tracker.metadata["round_stats"] = stats.round_stats
    score_tracker.metadata["kd"] = stats.kd
    score_tracker.metadata["deductions"] = stats.deductions
    score_tracker.metadata["fouls"] = stats.fouls
    score_tracker.metadata["corner_lock"] = {
        "RED": identity.role_status().get("RED"),
        "BLUE": identity.role_status().get("BLUE"),
    }
    score_tracker.metadata["corner_locked"] = (
        identity.id_for_role("RED") is not None and identity.id_for_role("BLUE") is not None
    )
    score_tracker.metadata["tracking_stats"] = identity.tracking_stats()
    score_tracker.metadata["identity_confidence"] = identity.role_confidence()
    score_tracker.metadata["swap_events"] = identity.swap_log()
    score_tracker.metadata["role_change_events"] = identity.role_change_log()
    score_tracker.metadata["classified_events"] = classified_events
    score_tracker.metadata["attempts"] = dict(score_tracker.attempts)
    score_tracker.metadata["accuracy"] = {
        "RED": round(
            100.0 * score_tracker.get_score("RED") / max(1, score_tracker.attempts.get("RED", 0)), 2
        ),
        "BLUE": round(
            100.0 * score_tracker.get_score("BLUE") / max(1, score_tracker.attempts.get("BLUE", 0)),
            2,
        ),
    }
    score_tracker.metadata["evidence_dir"] = evidence_dir
    score_tracker.metadata["evidence_clip_dir"] = clip_dir
    score_tracker.metadata["timeline_markers"] = [
        {"timestamp_s": round(m.timestamp_s, 3), "label": m.label} for m in timeline.markers
    ]
    score_tracker.metadata["timebase_samples"] = len(timeline.frame_timestamps)
    score_tracker.metadata["timebase_source"] = (
        timeline.frame_timestamp_rows[0].source
        if timeline.frame_timestamp_rows
        else "monotonic_fallback"
    )
    score_tracker.metadata["rounds_scored"] = sorted(scored_rounds)
    score_tracker.metadata["analysis_metadata_path"] = metadata_snapshot

    # compatibility with extractors that read attributes directly
    score_tracker.round_stats = stats.round_stats
    score_tracker.kd = stats.kd
    score_tracker.deductions = stats.deductions
    score_tracker.fouls = stats.fouls

    _write_analysis_snapshot(
        snapshot_path=metadata_snapshot,
        score_tracker=score_tracker,
        output_video=output_video,
    )
    _emit_progress(progress_cb, "Analysis artifacts finalized.", 100)

    print(f"Saved annotated fight video: {output_video}")
    return output_video
