# multi_person_tracker.py

import json
import math
import os
import cv2
import numpy as np

from boxer_registry import BoxerRegistry
from identity_bootstrap import IdentityBootstrap
from color_signature import compute_hist_signature, signature_similarity
from config import (
    BACKEND,
    DNN_MODEL,
    DNN_PROTO,
    IS_APPLE_SILICON,
    POSE_ENABLE_SEGMENTATION,
    POSE_MODEL_COMPLEXITY,
    POSE_TASK_MODEL,
    YOLO_DEVICE,
    YOLO_HALF,
    YOLO_IMGSZ,
    YOLOV8_WEIGHTS,
)
from mediapipe_compat import PoseLandmark, create_pose_estimator
from runtime_profile import resolve_backend

POSE = PoseLandmark


def pad_to_square(image):
    h, w = image.shape[:2]
    size = max(h, w)
    top = (size - h) // 2
    bottom = size - h - top
    left = (size - w) // 2
    right = size - w - left
    padded = cv2.copyMakeBorder(
        image, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    return padded, top, left


class MultiPersonPoseTracker:
    def __init__(self, confidence=0.5, bootstrap_frames=30, backend=None):
        requested_backend = backend or BACKEND
        self.backend = resolve_backend(
            requested_backend,
            is_apple_silicon=IS_APPLE_SILICON,
        )
        self.registry = BoxerRegistry(max_distance=0.34)
        self.pose = create_pose_estimator(
            confidence=confidence,
            enable_segmentation=bool(POSE_ENABLE_SEGMENTATION),
            model_complexity=POSE_MODEL_COMPLEXITY,
            task_model_path=POSE_TASK_MODEL,
        )
        self.bootstrap = IdentityBootstrap(frames=bootstrap_frames, min_samples=5)
        self.role_map = {}  # boxer_id -> role
        self.id_color_sig = {}  # boxer_id -> running avg hist signature
        self.person_model = None
        self.person_model_kind = None
        self.max_people = 8
        self.max_ring_candidates = 5
        self.min_box_area = 80 * 80
        self.last_tracks = []
        self.current_role_to_id = {}
        self.persistent_role_to_id = {}
        self.locked_role_to_id = {"RED": None, "BLUE": None, "REF": None}
        self.manual_seed_window = 75
        self.manual_seed_max_missing = 180
        self.yolo_device = YOLO_DEVICE
        self.yolo_imgsz = YOLO_IMGSZ
        self.yolo_half = bool(YOLO_HALF)
        self.manual_ring_roi = self._load_manual_ring_roi()
        self.manual_seeds = self._load_manual_seeds()
        self.manual_role_state = {
            role: {
                "track_id": None,
                "last_center": None,
                "last_diag": None,
                "velocity": (0.0, 0.0),
                "last_seen": 0,
                "signature": None,
                "missing_frames": 0,
                "seed_pending": True,
            }
            for role in self.manual_seeds
        }
        self._init_detector()

    @staticmethod
    def _load_manual_ring_roi():
        raw = os.getenv("VARBOX_RING_ROI", "").strip()
        if not raw:
            return None
        try:
            payload = json.loads(raw)
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        points = payload.get("normalized_points")
        if not isinstance(points, list) or len(points) < 3:
            return None
        normalized_points = []
        for row in points:
            if not isinstance(row, list) or len(row) < 2:
                return None
            normalized_points.append(
                (
                    float(max(0.0, min(1.0, row[0]))),
                    float(max(0.0, min(1.0, row[1]))),
                )
            )
        return {"normalized_points": normalized_points}

    def _manual_ring_polygon(self, frame_shape):
        if not self.manual_ring_roi:
            return None
        h, w = frame_shape[:2]
        points = np.array(
            [
                [int(round(px * w)), int(round(py * h))]
                for px, py in self.manual_ring_roi["normalized_points"]
            ],
            dtype=np.int32,
        )
        if len(points) < 3:
            return None
        return points

    @staticmethod
    def _load_manual_seeds():
        raw = os.getenv("VARBOX_MANUAL_SEEDS", "").strip()
        if not raw:
            return {}
        try:
            payload = json.loads(raw)
        except Exception:
            return {}
        if not isinstance(payload, dict):
            return {}

        seeds = {}
        for role in ("RED", "BLUE", "REF"):
            row = payload.get(role)
            if not isinstance(row, dict):
                continue
            rel_box = row.get("rel_box")
            if not isinstance(rel_box, list) or len(rel_box) < 4:
                continue
            try:
                x1, y1, x2, y2 = [float(v) for v in rel_box[:4]]
            except Exception:
                continue
            seeds[role] = {
                "frame_idx": int(row.get("frame_idx", 0) or 0) + 1,
                "rel_box": [
                    max(0.0, min(1.0, x1)),
                    max(0.0, min(1.0, y1)),
                    max(0.0, min(1.0, x2)),
                    max(0.0, min(1.0, y2)),
                ],
            }
        return seeds

    @staticmethod
    def _box_center(box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    @staticmethod
    def _box_diag(box):
        x1, y1, x2, y2 = box
        return float(max(1.0, math.hypot(max(1, x2 - x1), max(1, y2 - y1))))

    @staticmethod
    def _box_iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = float(iw * ih)
        if inter <= 0:
            return 0.0
        area_a = float(max(1, ax2 - ax1) * max(1, ay2 - ay1))
        area_b = float(max(1, bx2 - bx1) * max(1, by2 - by1))
        return inter / max(1.0, area_a + area_b - inter)

    def _manual_seed_box(self, role, frame_shape):
        seed = self.manual_seeds.get(role)
        if not seed:
            return None
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = seed["rel_box"]
        box = (
            int(round(x1 * w)),
            int(round(y1 * h)),
            int(round(x2 * w)),
            int(round(y2 * h)),
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            return None
        return box

    def _bind_role(self, role, track_id, locked=False):
        if track_id is None:
            return
        role = str(role).upper()

        old_track = self.persistent_role_to_id.get(role)
        if old_track is not None and old_track != track_id:
            self.role_map.pop(old_track, None)

        for other_role, other_track in list(self.persistent_role_to_id.items()):
            if other_role != role and other_track == track_id:
                self.persistent_role_to_id[other_role] = None
                if other_role in self.locked_role_to_id:
                    self.locked_role_to_id[other_role] = None

        self.role_map[track_id] = role
        self.persistent_role_to_id[role] = track_id
        self.current_role_to_id[role] = track_id
        if locked:
            self.locked_role_to_id[role] = track_id

    def _update_role_state(self, role, track_id, box, frame_num):
        state = self.manual_role_state.get(role)
        if state is None:
            self._bind_role(role, track_id, locked=False)
            return

        center = self._box_center(box)
        diag = self._box_diag(box)
        prev_center = state["last_center"]
        last_seen = int(state["last_seen"] or 0)
        dt = max(1, frame_num - last_seen)
        if prev_center is not None:
            vx = (center[0] - prev_center[0]) / dt
            vy = (center[1] - prev_center[1]) / dt
            pvx, pvy = state["velocity"]
            state["velocity"] = (0.7 * pvx + 0.3 * vx, 0.7 * pvy + 0.3 * vy)
        else:
            state["velocity"] = (0.0, 0.0)

        state["track_id"] = track_id
        state["last_center"] = center
        state["last_diag"] = diag
        state["last_seen"] = frame_num
        state["missing_frames"] = 0
        state["seed_pending"] = False
        sig = self.id_color_sig.get(track_id)
        if sig is not None:
            prev_sig = state["signature"]
            state["signature"] = sig if prev_sig is None or prev_sig.shape != sig.shape else (0.82 * prev_sig + 0.18 * sig)

        self._bind_role(role, track_id, locked=True)

    def _seed_match_score(self, role, det, frame_shape, frame_num):
        expected = self._manual_seed_box(role, frame_shape)
        seed = self.manual_seeds.get(role)
        if expected is None or seed is None:
            return None
        if abs(frame_num - int(seed["frame_idx"])) > self.manual_seed_window:
            return None

        det_box = det["bbox"]
        iou = self._box_iou(expected, det_box)
        ex_cx, ex_cy = self._box_center(expected)
        det_cx, det_cy = self._box_center(det_box)
        dist = math.hypot(det_cx - ex_cx, det_cy - ex_cy)
        scale = max(40.0, 0.6 * max(self._box_diag(expected), self._box_diag(det_box)))
        proximity = max(0.0, 1.0 - dist / scale)
        score = 1.8 * iou + 1.2 * proximity + 0.15 * float(det.get("det_conf", 0.0))
        if iou < 0.05 and proximity < 0.28:
            return None
        return score

    def _reacquire_score(self, role, det, frame_num):
        state = self.manual_role_state.get(role)
        if not state or state["last_center"] is None:
            return None

        center = det["center"]
        diag = self._box_diag(det["bbox"])
        dt = max(1, frame_num - int(state["last_seen"] or frame_num))
        vx, vy = state["velocity"]
        pred = (state["last_center"][0] + vx * dt, state["last_center"][1] + vy * dt)
        dist = math.hypot(center[0] - pred[0], center[1] - pred[1])
        max_travel = max(70.0, 2.4 * max(diag, float(state["last_diag"] or diag)) * dt)
        if dist > max_travel:
            return None

        motion = math.exp(-((dist / max(1.0, 0.55 * max_travel)) ** 2))
        ref_sig = state["signature"]
        cand_sig = self.id_color_sig.get(det["track_id"])
        appearance = signature_similarity(cand_sig, ref_sig) if cand_sig is not None and ref_sig is not None else 0.0
        size_ratio = min(diag, float(state["last_diag"] or diag)) / max(diag, float(state["last_diag"] or diag))
        score = 0.58 * motion + 0.30 * appearance + 0.12 * size_ratio
        if motion < 0.16 and appearance < 0.45:
            return None
        if score < 0.34:
            return None
        return score

    def _apply_manual_role_locks(self, detections, frame_shape, frame_num):
        if not self.manual_role_state:
            return

        det_by_id = {row["track_id"]: row for row in detections}
        used_ids = set()

        for role, state in self.manual_role_state.items():
            track_id = state.get("track_id")
            if track_id in det_by_id:
                det = det_by_id[track_id]
                self._update_role_state(role, track_id, det["bbox"], frame_num)
                used_ids.add(track_id)
            else:
                state["missing_frames"] = int(state.get("missing_frames", 0)) + 1
                self.current_role_to_id.pop(role, None)

        for role, state in self.manual_role_state.items():
            if self.current_role_to_id.get(role) is not None:
                continue

            best = None
            best_score = float("-inf")
            for det in detections:
                track_id = det["track_id"]
                if track_id in used_ids:
                    continue
                assigned_role = self.role_map.get(track_id)
                if assigned_role and assigned_role != role and assigned_role in self.manual_role_state:
                    continue

                score = None
                if state.get("seed_pending", True):
                    score = self._seed_match_score(role, det, frame_shape, frame_num)
                if score is None and state.get("track_id") is not None:
                    if int(state.get("missing_frames", 0)) > self.manual_seed_max_missing:
                        continue
                    score = self._reacquire_score(role, det, frame_num)
                if score is None:
                    continue
                if score > best_score:
                    best = det
                    best_score = score

            if best is None:
                continue

            self._update_role_state(role, best["track_id"], best["bbox"], frame_num)
            used_ids.add(best["track_id"])

    def _merge_bootstrap_roles(self, bootstrap_roles):
        if not bootstrap_roles:
            return
        for track_id, role in bootstrap_roles.items():
            role = str(role).upper()
            if role in self.manual_role_state:
                locked = self.locked_role_to_id.get(role)
                if locked is not None and locked != track_id:
                    continue
            self._bind_role(role, track_id, locked=False)

    def _ring_gate_boxes(self, frame, boxes):
        if len(boxes) <= 2:
            return boxes

        h, w = frame.shape[:2]
        diag = max(1.0, math.hypot(w, h))
        cx0 = w * 0.5
        cy0 = h * 0.5
        scored = []
        polygon = self._manual_ring_polygon(frame.shape)
        inside_boxes = []

        for box in boxes:
            x1, y1, x2, y2, conf = box
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            area = float(bw * bh)
            aspect = bh / max(1.0, float(bw))
            center_dist = math.hypot(cx - cx0, cy - cy0)
            centrality = max(0.0, 1.0 - center_dist / (0.45 * diag))
            edge_margin = min(cx, w - cx, cy, h - cy) / diag
            aspect_score = max(0.0, 1.0 - abs(aspect - 2.0) / 2.0)
            vertical_band = 1.0 if 0.10 * h <= cy <= 0.92 * h else 0.0
            boundary_penalty = 0.8 if y2 >= 0.985 * h or x1 <= 0 or x2 >= w - 1 else 0.0
            ring_bonus = 0.0
            if polygon is not None:
                foot_pt = (float(cx), float(y2 - 1))
                center_pt = (float(cx), float(cy))
                inside = (
                    cv2.pointPolygonTest(polygon, foot_pt, False) >= 0
                    or cv2.pointPolygonTest(polygon, center_pt, False) >= 0
                )
                if inside:
                    ring_bonus = 3.0
                    inside_boxes.append(box)
                else:
                    ring_bonus = -3.5
            score = (
                1.4 * centrality
                + 0.9 * edge_margin
                + 0.6 * aspect_score
                + 0.35 * vertical_band
                + 0.4 * float(conf)
                + 0.00003 * area
                + ring_bonus
                - boundary_penalty
            )
            scored.append((score, box))

        if len(inside_boxes) >= 2:
            inside_boxes.sort(key=lambda b: ((b[3] - b[1]) * (b[2] - b[0]), b[4]), reverse=True)
            return inside_boxes[: self.max_ring_candidates]

        scored.sort(key=lambda item: item[0], reverse=True)
        kept = [box for _, box in scored[: self.max_ring_candidates]]
        if len(kept) >= 2:
            return kept
        return boxes[: self.max_ring_candidates]

    def _init_detector(self):
        requested = self.backend
        if requested == "opencv" and os.path.isfile(DNN_PROTO) and os.path.isfile(DNN_MODEL):
            self.person_model = cv2.dnn.readNetFromCaffe(DNN_PROTO, DNN_MODEL)
            self.person_model_kind = "opencv"
            return

        try:
            from ultralytics import YOLO  # pylint: disable=import-outside-toplevel

            candidates = [YOLOV8_WEIGHTS, os.path.join(os.path.dirname(__file__), "yolov8n.pt")]
            weights = next((p for p in candidates if p and os.path.isfile(p)), "yolov8n.pt")
            self.person_model = YOLO(weights)
            self.person_model_kind = "yolov8"
            return
        except Exception as exc:
            if requested == "opencv" and os.path.isfile(DNN_PROTO) and os.path.isfile(DNN_MODEL):
                self.person_model = cv2.dnn.readNetFromCaffe(DNN_PROTO, DNN_MODEL)
                self.person_model_kind = "opencv"
                return
            raise RuntimeError(
                f"Unable to initialize person detector. backend={requested}, error={exc}"
            ) from exc

    def _detect_people_yolov8(self, frame):
        results = self.person_model.predict(
            frame,
            classes=[0],
            verbose=False,
            device=self.yolo_device,
            imgsz=self.yolo_imgsz,
            half=self.yolo_half,
            max_det=self.max_people,
        )[0]
        boxes = []
        for det in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, *_ = det
            if conf < 0.25:
                continue
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if (x2 - x1) * (y2 - y1) < self.min_box_area:
                continue
            boxes.append((x1, y1, x2, y2, float(conf)))
        boxes.sort(key=lambda b: b[4], reverse=True)
        return boxes[: self.max_people]

    def _detect_people_opencv(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            scalefactor=0.007843,
            size=(300, 300),
            mean=127.5,
        )
        self.person_model.setInput(blob)
        detections = self.person_model.forward()
        boxes = []
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            cls = int(detections[0, 0, i, 1])
            if cls != 15 or conf < 0.35:  # person class
                continue
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 - x1 < 20 or y2 - y1 < 20:
                continue
            boxes.append((x1, y1, x2, y2, conf))
        boxes.sort(key=lambda b: ((b[2] - b[0]) * (b[3] - b[1]), b[4]), reverse=True)
        return boxes[: self.max_people]

    def detect_people(self, frame):
        if self.person_model_kind == "opencv":
            boxes = self._detect_people_opencv(frame)
        else:
            boxes = self._detect_people_yolov8(frame)
        return self._ring_gate_boxes(frame, boxes)

    def _update_color_sig(self, boxer_id, frame, box, alpha=0.15):
        x1, y1, x2, y2 = box
        crop = frame[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
        sig = compute_hist_signature(crop)
        prev = self.id_color_sig.get(boxer_id)
        if prev is None or prev.shape != sig.shape:
            self.id_color_sig[boxer_id] = sig
        else:
            self.id_color_sig[boxer_id] = (1 - alpha) * prev + alpha * sig

    def process_frame(self, frame, frame_num):
        people_boxes = self.detect_people(frame)
        poses_by_id = {}
        track_rows = []
        self.current_role_to_id = {}

        for detection_index, entry in enumerate(people_boxes):
            if len(entry) >= 5:
                x1, y1, x2, y2, det_conf = entry
            else:
                x1, y1, x2, y2 = entry[:4]
                det_conf = 1.0
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            padded_crop, pad_top, pad_left = pad_to_square(crop)
            rgb_crop = cv2.cvtColor(padded_crop, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_crop, frame_num * 1000 + detection_index)

            if not results.pose_landmarks:
                continue

            landmarks = results.pose_landmarks.landmark
            h, w = padded_crop.shape[:2]
            keypoints = {
                idx: [
                    int(landmark.x * w + x1 - pad_left),
                    int(landmark.y * h + y1 - pad_top),
                    float(max(0.0, min(1.0, landmark.visibility))),
                ]
                for idx, landmark in enumerate(landmarks)
            }

            required = [
                POSE.NOSE,
                POSE.LEFT_WRIST,
                POSE.RIGHT_WRIST,
                POSE.LEFT_SHOULDER,
                POSE.RIGHT_SHOULDER,
            ]
            boxer_id = self.registry.match_or_register(keypoints, frame_num, required)
            if boxer_id is None:
                continue

            self._update_color_sig(boxer_id, frame, (x1, y1, x2, y2))

            if not self.bootstrap.finalized:
                self.bootstrap.add_observation(frame_num, boxer_id, frame, (x1, y1, x2, y2))
                if self.bootstrap.ready(frame_num):
                    bootstrap_roles = self.bootstrap.finalize()
                    self._merge_bootstrap_roles(bootstrap_roles)
                    if bootstrap_roles:
                        print(f"Bootstrap roles: {bootstrap_roles}")

            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            track_rows.append(
                {
                    "track_id": boxer_id,
                    "bbox": (x1, y1, x2, y2),
                    "center": center,
                    "keypoints": keypoints,
                    "mask": results.segmentation_mask,
                    "det_conf": float(det_conf),
                }
            )

        self._apply_manual_role_locks(track_rows, frame.shape, frame_num)
        current_id_to_role = {
            track_id: role for role, track_id in self.current_role_to_id.items() if track_id is not None
        }

        for row in track_rows:
            boxer_id = row["track_id"]
            role = current_id_to_role.get(boxer_id)
            if role is None:
                role = self.role_map.get(boxer_id)
            poses_by_id[boxer_id] = {
                "track_id": boxer_id,
                "keypoints": row["keypoints"],
                "box": row["bbox"],
                "bbox": row["bbox"],
                "center": row["center"],
                "mask": row["mask"],
                "role": role,
                "det_conf": float(row["det_conf"]),
            }

        self.registry.clean_old_ids(frame_num)
        self.last_tracks = track_rows
        return poses_by_id

    def latest_tracks(self):
        return list(self.last_tracks)

    def role_status(self):
        status = dict(self.persistent_role_to_id)
        for role in ("RED", "BLUE", "REF"):
            status.setdefault(role, None)
        return status

    def live_role_status(self):
        status = {}
        for role, track_id in self.current_role_to_id.items():
            if track_id is not None:
                status[role] = track_id
        return status

    def lock_status(self):
        status = dict(self.locked_role_to_id)
        for role in ("RED", "BLUE", "REF"):
            if status.get(role) is None:
                status[role] = self.persistent_role_to_id.get(role)
        return status

    def manual_seed_status(self):
        return {
            role: {
                "requested": int(role in self.manual_seeds),
                "locked_track_id": self.locked_role_to_id.get(role),
                "last_seen_frame": int(self.manual_role_state.get(role, {}).get("last_seen", 0) or 0),
                "missing_frames": int(self.manual_role_state.get(role, {}).get("missing_frames", 0) or 0),
            }
            for role in ("RED", "BLUE", "REF")
        }
