# video_processor.py
import os
import cv2
import mediapipe as mp
import math
from collections import defaultdict
from typing import Callable, Optional

from config import FRAME_RATE
from punch_detector import detect_punch
from multi_person_tracker import MultiPersonPoseTracker
from score_tracker import ScoreTracker
from participant_manager import ParticipantManager

from round_timer import RoundTimer
from stats_aggregator import StatsAggregator
from judge_10point import judge_round

POSE = mp.solutions.pose.PoseLandmark
ROLE_COLOR = {"BLUE": (255, 80, 60), "RED": (0, 0, 255), None: (0, 255, 0)}

# ---------- Tunables (safe defaults) ----------
SAFE_PUNCH_DIST_PX = 48          # contact threshold fallback (px), auto-scales by shoulder width
MIN_WRIST_SPEED_PX = 2.0         # minimal wrist speed (px/frame) to count impact
WARMUP_FRAMES = 90               # allow left/right assignment early (~3s @30fps)
DEBUG_DRAW = False               # set True to see heuristic visuals

ProgressCB = Callable[[int, int], None]  # (current_frame, total_frames)
CancelCB   = Callable[[], bool]
LogCB      = Callable[[str], None]

# ---------- helpers ----------
def _parse_punch_result(result):
    """
    Accepts:
      - bool
      - (landed: bool, hand: 'L'|'R'|None)
      - {'landed': bool, 'hand': 'L'|'R'|None}
    Returns (landed: bool, hand_key: 'L'|'R'|'ANY')
    """
    landed, hand = False, "ANY"
    if isinstance(result, bool):
        landed = result
    elif isinstance(result, tuple) and len(result) >= 1:
        landed = bool(result[0])
        hand = ("ANY" if len(result) < 2 or result[1] is None else str(result[1]).upper())
    elif isinstance(result, dict):
        landed = bool(result.get("landed", False))
        h = result.get("hand", "ANY")
        hand = "ANY" if h is None else str(h).upper()
    if hand not in ("L", "R", "ANY"):
        hand = "ANY"
    return landed, hand

def _euclid(a, b):
    ax, ay = a; bx, by = b
    return math.hypot(ax - bx, ay - by)

def _shoulder_scale(kpts):
    # Use shoulder width as a scene scale reference
    if POSE.LEFT_SHOULDER in kpts and POSE.RIGHT_SHOULDER in kpts:
        return _euclid(kpts[POSE.LEFT_SHOULDER], kpts[POSE.RIGHT_SHOULDER])
    return 0.0

def _safe_detect_punch(attacker_k, head_xy, last_wrists, role):
    """
    Fallback heuristic when detect_punch() returns False/None.
    Signal contact if (wrist near head) AND (wrist moving) towards target.
    Returns (landed: bool, hand: 'L'|'R').
    """
    lw = attacker_k.get(POSE.LEFT_WRIST)
    rw = attacker_k.get(POSE.RIGHT_WRIST)
    if lw is None and rw is None:
        return False, "ANY"

    # scale-aware threshold
    scale = max(_shoulder_scale(attacker_k), 1.0)
    thr = max(SAFE_PUNCH_DIST_PX, 0.6 * scale)  # at least SAFE_PUNCH_DIST_PX or 0.6*shoulder width

    best_hand, best_dist, moving = "ANY", 1e9, False
    for hand_key, w in (("L", lw), ("R", rw)):
        if w is None:
            continue
        d = _euclid(w, head_xy)
        # speed
        last = last_wrists[role][hand_key]
        spd = 0.0 if last is None else _euclid(w, last)
        if d < best_dist:
            best_dist = d
            best_hand = hand_key
            moving = spd >= MIN_WRIST_SPEED_PX

    landed = (best_dist <= thr) and moving
    return landed, best_hand if landed else ("ANY")

def _warmup_assign(poses):
    """
    Fallback mapping by x-position (left=RED, right=BLUE) used during warmup or when roles missing.
    """
    if len(poses) < 2:
        return None, None
    center_x = []
    for bid, d in poses.items():
        x1, y1, x2, y2 = d["box"]
        center_x.append(((x1 + x2) // 2, bid))
    center_x.sort(key=lambda t: t[0])
    red_id = center_x[0][1]
    blue_id = center_x[-1][1]
    if red_id == blue_id:
        return None, None
    return red_id, blue_id

# Track last wrist positions for speed check (role -> {'L': (x,y)|None, 'R': ...})
_last_wrists = defaultdict(lambda: {"L": None, "R": None})

def process_video(
    score_tracker: ScoreTracker,
    input_video: str,
    output_video: str,
    progress_cb: Optional[ProgressCB] = None,
    cancel_cb: Optional[CancelCB] = None,
    log_cb: Optional[LogCB] = None
):
    """
    Processes a video and writes an annotated output. Updates score_tracker in place.

    Args:
        score_tracker: ScoreTracker instance to log punches and round decisions.
        input_video:   Path to input video file.
        output_video:  Path to output annotated video file.
        progress_cb:   Optional callback(current_frame, total_frames).
        cancel_cb:     Optional callback() -> bool. If returns True, processing stops gracefully.
        log_cb:        Optional callback(msg: str) for GUI logging.
    """
    def _log(msg: str):
        if log_cb:
            log_cb(msg)
        else:
            print(msg)

    if not input_video or not os.path.isfile(input_video):
        raise FileNotFoundError(f"❌ Could not open video: {input_video}")

    os.makedirs(os.path.dirname(output_video) or ".", exist_ok=True)

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise FileNotFoundError(f"❌ Could not open video: {input_video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps_src = cap.get(cv2.CAP_PROP_FPS)
    fps  = int(fps_src) if fps_src and fps_src > 0 else (FRAME_RATE if FRAME_RATE > 0 else 30)

    W    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out  = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

    _log(f"▶️ Processing: {input_video}")
    _log(f"   Resolution: {W}x{H} @ {fps} fps, Frames: {total_frames if total_frames>0 else 'unknown'}")
    _log(f"   Writing to: {output_video}")

    if progress_cb:
        progress_cb(0, total_frames)

    pose_tracker = MultiPersonPoseTracker(bootstrap_frames=30)
    participants = ParticipantManager(min_color_fraction=0.03, min_sim_accept=0.30,
                                      smooth_alpha=0.12, max_missing_frames=45,
                                      freeze_anchors_after_seed=True)

    # Round timing + per-round stats (10-Point reads these; does not alter punch logging)
    timer = RoundTimer(fps=fps, round_secs=180, rest_secs=60, total_rounds=12)
    stats = StatsAggregator(total_rounds=12)

    # Optional metadata for scorecard
    score_tracker.metadata.setdefault("title", "Boxing Match Scorecard")
    score_tracker.metadata.setdefault("subtitle", "Automated VAR Box Scoring")

    frame_idx = 0
    canceled = False

    while cap.isOpened():
        if cancel_cb and cancel_cb():
            _log("⏹ Cancel requested. Stopping gracefully...")
            canceled = True
            break

        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        round_no, in_round, just_ended_round, bout_over = timer.step()

        # Detect
        poses = pose_tracker.process_frame(frame, frame_idx)

        # Update roles
        participants.update(frame, poses)
        red_id  = participants.id_for_role("RED")
        blue_id = participants.id_for_role("BLUE")

        # Fallback left/right assignment early
        if (red_id is None or blue_id is None) and frame_idx <= WARMUP_FRAMES:
            f_red, f_blue = _warmup_assign(poses)
            red_id  = red_id  if red_id  is not None else f_red
            blue_id = blue_id if blue_id is not None else f_blue

        # ----- Punch detection -----
        if red_id is not None and blue_id is not None and red_id in poses and blue_id in poses:
            d_red, d_blue = poses[red_id], poses[blue_id]
            k_red, k_blue = d_red["keypoints"], d_blue["keypoints"]

            need = [POSE.LEFT_WRIST, POSE.RIGHT_WRIST, POSE.NOSE]
            have_red  = all(k in k_red  for k in need)
            have_blue = all(k in k_blue for k in need)

            if have_red and have_blue:
                # NOTE: keep original timestamp approach for continuity
                tstamp = f"{(frame_idx // fps):02}:{(frame_idx % fps):02}"

                # RED -> BLUE
                red_att = {"left_wrist": k_red[POSE.LEFT_WRIST], "right_wrist": k_red[POSE.RIGHT_WRIST], "head": k_red[POSE.NOSE]}
                blue_head = k_blue[POSE.NOSE]
                landed, hand = _parse_punch_result(detect_punch(red_att, blue_head))
                if not landed:
                    landed, hand = _safe_detect_punch(k_red, blue_head, _last_wrists, "RED")
                if landed:
                    accepted = score_tracker.update(frame_idx, fighter_id="RED", timestamp=tstamp, hand=hand)
                    if accepted and in_round:
                        stats.add_punch("RED", round_no)

                # BLUE -> RED
                blue_att = {"left_wrist": k_blue[POSE.LEFT_WRIST], "right_wrist": k_blue[POSE.RIGHT_WRIST], "head": k_blue[POSE.NOSE]}
                red_head = k_red[POSE.NOSE]
                landed, hand = _parse_punch_result(detect_punch(blue_att, red_head))
                if not landed:
                    landed, hand = _safe_detect_punch(k_blue, red_head, _last_wrists, "BLUE")
                if landed:
                    accepted = score_tracker.update(frame_idx, fighter_id="BLUE", timestamp=tstamp, hand=hand)
                    if accepted and in_round:
                        stats.add_punch("BLUE", round_no)

                # update last wrists for speed
                if POSE.LEFT_WRIST in k_red:  _last_wrists["RED"]["L"]  = k_red[POSE.LEFT_WRIST]
                if POSE.RIGHT_WRIST in k_red: _last_wrists["RED"]["R"]  = k_red[POSE.RIGHT_WRIST]
                if POSE.LEFT_WRIST in k_blue: _last_wrists["BLUE"]["L"] = k_blue[POSE.LEFT_WRIST]
                if POSE.RIGHT_WRIST in k_blue:_last_wrists["BLUE"]["R"] = k_blue[POSE.RIGHT_WRIST]

                # optional debug draw
                if DEBUG_DRAW:
                    for p, q, col in [(k_red.get(POSE.LEFT_WRIST), blue_head, (0, 0, 255)),
                                      (k_red.get(POSE.RIGHT_WRIST), blue_head, (0, 0, 255)),
                                      (k_blue.get(POSE.LEFT_WRIST), red_head, (255, 80, 60)),
                                      (k_blue.get(POSE.RIGHT_WRIST), red_head, (255, 80, 60))]:
                        if p and q:
                            cv2.line(frame, p, q, col, 1)

        # ----- End-of-round -> 10-point decision -----
        if just_ended_round:
            rs    = stats.get_round(round_no)
            kd_r  = stats.kd[round_no]["RED"]
            kd_b  = stats.kd[round_no]["BLUE"]
            ded_r = stats.deductions[round_no]["RED"]
            ded_b = stats.deductions[round_no]["BLUE"]
            red_pts, blue_pts, note = judge_round(rs, kd_red=kd_r, kd_blue=kd_b, ded_red=ded_r, ded_blue=ded_b)
            score_tracker.add_round_points(round_no, red_pts, blue_pts, note)

            cv2.rectangle(frame, (0, H - 40), (W, H), (20, 20, 20), -1)
            cv2.putText(frame, f"Round {round_no}  |  RED {red_pts} - {blue_pts} BLUE  |  {note}",
                        (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # ----- Overlays -----
        for bid, data in poses.items():
            x1, y1, x2, y2 = data["box"]
            role = participants.role_for_id(bid)
            if role is None and frame_idx <= WARMUP_FRAMES:
                # show provisional roles too
                if red_id == bid:  role = "RED"
                if blue_id == bid: role = "BLUE"
            color = ROLE_COLOR.get(role, ROLE_COLOR[None])

            # seg mask overlay (optional)
            mask = data.get("mask")
            if mask is not None:
                try:
                    mr = cv2.resize(mask, (x2 - x1, y2 - y1))
                    mb = (mr > 0.1).astype('uint8') * 255
                    m_rgb = cv2.applyColorMap(mb, cv2.COLORMAP_OCEAN)
                    roi = frame[y1:y2, x1:x2]
                    frame[y1:y2, x1:x2] = cv2.addWeighted(roi, 0.6, m_rgb, 0.4, 0)
                except Exception:
                    pass

            p_count = score_tracker.get_score(role) if role in ("RED", "BLUE") else 0
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{role or 'IGN'} | ID {bid} | P:{p_count}",
                        (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            for pt in data["keypoints"].values():
                cv2.circle(frame, tuple(pt), 4, color, -1)

        # top bar
        cv2.rectangle(frame, (0, 0), (W, 30), (20, 20, 20), -1)
        cv2.putText(frame, f"RED P:{score_tracker.get_score('RED')}",  (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, ROLE_COLOR["RED"], 2)
        cv2.putText(frame, f"BLUE P:{score_tracker.get_score('BLUE')}", (160, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, ROLE_COLOR["BLUE"], 2)
        if hasattr(score_tracker, "ten_point_totals"):
            rt = score_tracker.ten_point_totals.get("RED", 0)
            bt = score_tracker.ten_point_totals.get("BLUE", 0)
            cv2.putText(frame, f"10pt — R:{rt} B:{bt}", (320, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 220, 255), 2)

        out.write(frame)

        if progress_cb:
            progress_cb(frame_idx, total_frames)

        if bout_over:
            break

    cap.release()
    out.release()

    # Expose artifacts for scorecard
    score_tracker.metadata["round_stats"] = stats.round_stats
    score_tracker.metadata["kd"] = stats.kd
    score_tracker.metadata["deductions"] = stats.deductions

    if not canceled:
        _log(f"✅ Saved: {output_video}")
