# multi_person_tracker.py

import importlib.util
import json
import math
import os

import cv2
import numpy as np

from color_signature import compute_hist_signature, signature_similarity
from config import (
    BACKEND,
    DNN_MODEL,
    DNN_PROTO,
    IS_APPLE_SILICON,
    YOLO_DEVICE,
    YOLO_HALF,
    YOLO_IMGSZ,
    YOLO_POSE_WEIGHTS,
    YOLO_TRACK_CONF,
    YOLO_TRACK_IOU,
    YOLO_TRACK_PERSIST,
    YOLO_TRACKER,
    YOLO_TRACKER_CONFIG,
)
from mediapipe_compat import PoseLandmark
from runtime_profile import resolve_backend

POSE = PoseLandmark
_COCO17_TO_MEDIAPIPE = {
    0: int(POSE.NOSE),
    1: 2,
    2: 5,
    3: 7,
    4: 8,
    5: int(POSE.LEFT_SHOULDER),
    6: int(POSE.RIGHT_SHOULDER),
    7: int(POSE.LEFT_ELBOW),
    8: int(POSE.RIGHT_ELBOW),
    9: int(POSE.LEFT_WRIST),
    10: int(POSE.RIGHT_WRIST),
    11: int(POSE.LEFT_HIP),
    12: int(POSE.RIGHT_HIP),
    13: 25,
    14: 26,
    15: int(POSE.LEFT_ANKLE),
    16: int(POSE.RIGHT_ANKLE),
}


class MultiPersonPoseTracker:
    def __init__(
        self,
        confidence=0.5,
        bootstrap_frames=30,
        backend=None,
        manual_ring_roi=None,
        manual_seeds=None,
    ):
        requested_backend = backend or BACKEND
        self.backend = resolve_backend(
            requested_backend,
            is_apple_silicon=IS_APPLE_SILICON,
        )
        self.role_map = {}  # boxer_id -> role
        self.id_color_sig = {}  # boxer_id -> running avg hist signature
        self.person_model = None
        self.person_model_kind = None
        self.max_people = 8
        self.max_ring_candidates = 5
        self.min_box_area = 80 * 80
        self.ring_min_box_inside_ratio = 0.55
        self.ring_min_pose_points_inside = 2
        self.last_tracks = []
        self.current_role_to_id = {}
        self.persistent_role_to_id = {}
        self.locked_role_to_id = {"RED": None, "BLUE": None, "REF": None}
        self.manual_seed_window = 75
        self.manual_seed_max_missing = 180
        self.yolo_device = YOLO_DEVICE
        self.yolo_imgsz = YOLO_IMGSZ
        self.yolo_half = bool(YOLO_HALF)
        tracker_name = (YOLO_TRACKER or "bytetrack").strip().lower()
        self.yolo_tracker = tracker_name if tracker_name in {"bytetrack", "botsort"} else "bytetrack"
        self.yolo_track_persist = bool(YOLO_TRACK_PERSIST)
        self.yolo_track_conf = float(max(0.01, min(0.95, YOLO_TRACK_CONF)))
        self.yolo_track_iou = float(max(0.05, min(0.95, YOLO_TRACK_IOU)))
        self.yolo_tracker_config = self._resolve_tracker_config_path(
            requested_path=YOLO_TRACKER_CONFIG,
            tracker_name=self.yolo_tracker,
        )
        self._yolo_track_issue = None
        self.manual_ring_roi = (
            manual_ring_roi if manual_ring_roi is not None else self._load_manual_ring_roi()
        )
        self.manual_seeds = manual_seeds if manual_seeds is not None else self._load_manual_seeds()
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
    def _lap_available():
        return importlib.util.find_spec("lap") is not None

    @staticmethod
    def _expand_box(box, frame_shape, x_margin=0.18, y_margin_top=0.18, y_margin_bottom=0.10):
        x1, y1, x2, y2 = [int(v) for v in box[:4]]
        h, w = frame_shape[:2]
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        mx = int(round(bw * x_margin))
        my_top = int(round(bh * y_margin_top))
        my_bottom = int(round(bh * y_margin_bottom))
        ex1 = max(0, x1 - mx)
        ey1 = max(0, y1 - my_top)
        ex2 = min(w, x2 + mx)
        ey2 = min(h, y2 + my_bottom)
        return ex1, ey1, ex2, ey2

    @staticmethod
    def _tracker_assets_dir():
        return os.path.join(os.path.dirname(__file__), "assets", "trackers")

    @classmethod
    def _resolve_tracker_config_path(cls, requested_path, tracker_name):
        if requested_path and os.path.isfile(requested_path):
            return requested_path
        filename = f"{tracker_name}_fixedcam.yaml"
        candidate = os.path.join(cls._tracker_assets_dir(), filename)
        if os.path.isfile(candidate):
            return candidate
        return None

    @staticmethod
    def _as_numpy(data):
        if data is None:
            return None
        if hasattr(data, "detach"):
            data = data.detach()
        if hasattr(data, "cpu"):
            data = data.cpu()
        if hasattr(data, "numpy"):
            try:
                return data.numpy()
            except Exception:
                return None
        return np.asarray(data)

    @staticmethod
    def _remap_pose_keypoints(keypoints_xy, keypoints_conf=None):
        xy = np.asarray(keypoints_xy, dtype=np.float32)
        conf = None if keypoints_conf is None else np.asarray(keypoints_conf, dtype=np.float32)
        if xy.ndim != 2 or xy.shape[1] < 2:
            return {}

        mapped = {}
        for coco_idx, pose_idx in _COCO17_TO_MEDIAPIPE.items():
            if coco_idx >= len(xy):
                continue
            x = float(xy[coco_idx, 0])
            y = float(xy[coco_idx, 1])
            score = 1.0
            if conf is not None and coco_idx < len(conf):
                score = float(conf[coco_idx])
            if not np.isfinite(x) or not np.isfinite(y) or score <= 0.0:
                continue
            mapped[int(pose_idx)] = [int(round(x)), int(round(y)), float(max(0.0, min(1.0, score)))]
        return mapped

    @staticmethod
    def _entry_box(entry):
        if isinstance(entry, dict):
            return (
                int(entry["x1"]),
                int(entry["y1"]),
                int(entry["x2"]),
                int(entry["y2"]),
                float(entry.get("conf", 1.0)),
            )
        row = tuple(entry)
        if len(row) >= 5:
            return int(row[0]), int(row[1]), int(row[2]), int(row[3]), float(row[4])
        return int(row[0]), int(row[1]), int(row[2]), int(row[3]), 1.0

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

    def _manual_ring_mask(self, frame_shape):
        polygon = self._manual_ring_polygon(frame_shape)
        if polygon is None:
            return None
        h, w = frame_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, polygon, 255)
        return mask

    def _box_inside_ring_ratio(self, box, ring_mask):
        if ring_mask is None:
            return 1.0
        x1, y1, x2, y2 = box[:4]
        h, w = ring_mask.shape[:2]
        x1 = max(0, min(w - 1, int(x1)))
        x2 = max(0, min(w, int(x2)))
        y1 = max(0, min(h - 1, int(y1)))
        y2 = max(0, min(h, int(y2)))
        if x2 <= x1 or y2 <= y1:
            return 0.0
        roi = ring_mask[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0
        inside = float(cv2.countNonZero(roi))
        return inside / float(roi.shape[0] * roi.shape[1])

    def _pose_inside_ring(self, keypoints, frame_shape):
        polygon = self._manual_ring_polygon(frame_shape)
        if polygon is None:
            return True
        checks = [
            POSE.NOSE,
            POSE.LEFT_SHOULDER,
            POSE.RIGHT_SHOULDER,
            POSE.LEFT_HIP,
            POSE.RIGHT_HIP,
            POSE.LEFT_ANKLE,
            POSE.RIGHT_ANKLE,
        ]
        inside_count = 0
        for idx in checks:
            row = keypoints.get(idx)
            if not isinstance(row, (tuple, list)) or len(row) < 2:
                continue
            pt = (float(row[0]), float(row[1]))
            if cv2.pointPolygonTest(polygon, pt, False) >= 0:
                inside_count += 1
        return inside_count >= self.ring_min_pose_points_inside

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
        polygon = self._manual_ring_polygon(frame.shape)
        if polygon is not None:
            ring_mask = self._manual_ring_mask(frame.shape)
            inside_boxes = []
            for entry in boxes:
                x1, y1, x2, y2, _ = self._entry_box(entry)
                cx = (x1 + x2) * 0.5
                cy = (y1 + y2) * 0.5
                foot_pt = (float(cx), float(y2 - 1))
                center_pt = (float(cx), float(cy))
                inside_ratio = self._box_inside_ring_ratio((x1, y1, x2, y2), ring_mask)
                inside = (
                    inside_ratio >= self.ring_min_box_inside_ratio
                    or (
                        inside_ratio >= 0.25
                        and (
                            cv2.pointPolygonTest(polygon, foot_pt, False) >= 0
                            or cv2.pointPolygonTest(polygon, center_pt, False) >= 0
                        )
                    )
                )
                if inside:
                    inside_boxes.append(entry)
            boxes = inside_boxes

        if len(boxes) <= 2:
            return boxes

        h, w = frame.shape[:2]
        diag = max(1.0, math.hypot(w, h))
        cx0 = w * 0.5
        cy0 = h * 0.5
        scored = []

        for entry in boxes:
            x1, y1, x2, y2, conf = self._entry_box(entry)
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
            score = (
                1.4 * centrality
                + 0.9 * edge_margin
                + 0.6 * aspect_score
                + 0.35 * vertical_band
                + 0.4 * float(conf)
                + 0.00003 * area
                - boundary_penalty
            )
            scored.append((score, entry))

        scored.sort(key=lambda item: item[0], reverse=True)
        kept = [entry for _, entry in scored[: self.max_ring_candidates]]
        if len(kept) >= 2:
            return kept
        return boxes[: self.max_ring_candidates]

    def _init_detector(self):
        requested = self.backend
        try:
            from ultralytics import YOLO  # pylint: disable=import-outside-toplevel

            candidates = [YOLO_POSE_WEIGHTS, os.path.join(os.path.dirname(__file__), "yolo11m-pose.pt")]
            weights = next((p for p in candidates if p and os.path.isfile(p)), "yolo11m-pose.pt")
            self.person_model = YOLO(weights)
            self.person_model_kind = "ultralytics_pose"
            return
        except Exception as exc:
            if requested == "opencv" and os.path.isfile(DNN_PROTO) and os.path.isfile(DNN_MODEL):
                self.person_model = cv2.dnn.readNetFromCaffe(DNN_PROTO, DNN_MODEL)
                self.person_model_kind = "opencv"
                return
            raise RuntimeError(
                f"Unable to initialize pose tracker. backend={requested}, error={exc}"
            ) from exc

    def _detect_people_yolo_pose(self, frame):
        if self.person_model is None:
            return []

        common_kwargs = {
            "classes": [0],
            "verbose": False,
            "device": self.yolo_device,
            "imgsz": self.yolo_imgsz,
            "half": self.yolo_half,
            "max_det": self.max_people,
            "conf": self.yolo_track_conf,
            "iou": self.yolo_track_iou,
        }

        result = None
        tracking = self.tracking_diagnostics()
        if tracking["enabled"]:
            try:
                tracked = self.person_model.track(
                    frame,
                    persist=self.yolo_track_persist,
                    tracker=self.yolo_tracker_config,
                    **common_kwargs,
                )
                result = tracked[0] if isinstance(tracked, list) and tracked else tracked
            except Exception as exc:
                self._yolo_track_issue = f"{type(exc).__name__}: {exc}"
                result = None

        if result is None:
            predicted = self.person_model.predict(frame, **common_kwargs)
            result = predicted[0] if isinstance(predicted, list) and predicted else predicted

        boxes_attr = getattr(result, "boxes", None)
        if boxes_attr is None:
            return []

        xyxy = self._as_numpy(getattr(boxes_attr, "xyxy", None))
        conf = self._as_numpy(getattr(boxes_attr, "conf", None))
        track_ids = self._as_numpy(getattr(boxes_attr, "id", None))
        classes = self._as_numpy(getattr(boxes_attr, "cls", None))
        if xyxy is None:
            return []

        detections = []
        keypoints_attr = getattr(result, "keypoints", None)
        keypoints_xy = self._as_numpy(getattr(keypoints_attr, "xy", None))
        keypoints_conf = self._as_numpy(getattr(keypoints_attr, "conf", None))
        for idx, coords in enumerate(np.asarray(xyxy)):
            if len(coords) < 4:
                continue
            cls_id = int(classes[idx]) if classes is not None and idx < len(classes) else 0
            if cls_id != 0:
                continue
            det_conf = float(conf[idx]) if conf is not None and idx < len(conf) else 1.0
            if det_conf < self.yolo_track_conf:
                continue
            x1, y1, x2, y2 = [int(v) for v in coords[:4]]
            if (x2 - x1) * (y2 - y1) < self.min_box_area:
                continue
            track_id = None
            if track_ids is not None and idx < len(track_ids):
                raw_id = float(track_ids[idx])
                if np.isfinite(raw_id):
                    track_id = int(raw_id)
            pose_points = None
            if keypoints_xy is not None and idx < len(keypoints_xy):
                pose_points = self._remap_pose_keypoints(
                    keypoints_xy[idx],
                    None if keypoints_conf is None or idx >= len(keypoints_conf) else keypoints_conf[idx],
                )
            detections.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "conf": det_conf,
                    "track_id": track_id,
                    "keypoints": pose_points or {},
                    "mask": None,
                }
            )
        detections.sort(
            key=lambda row: (
                row["track_id"] is None,
                -float(row["conf"]),
                -float((row["x2"] - row["x1"]) * (row["y2"] - row["y1"])),
            )
        )
        return detections[: self.max_people]

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
        ring_mask = self._manual_ring_mask(frame.shape)
        detector_frame = frame
        if ring_mask is not None:
            detector_frame = cv2.bitwise_and(frame, frame, mask=ring_mask)
        if self.person_model_kind == "opencv":
            boxes = self._detect_people_opencv(detector_frame)
        else:
            boxes = self._detect_people_yolo_pose(detector_frame)
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

        for entry in people_boxes:
            source_track_id = None
            keypoints = {}
            mask = None
            if isinstance(entry, dict):
                x1, y1, x2, y2, det_conf = self._entry_box(entry)
                source_track_id = entry.get("track_id")
                keypoints = entry.get("keypoints") or {}
                mask = entry.get("mask")
            else:
                if len(entry) >= 5:
                    x1, y1, x2, y2, det_conf = self._entry_box(entry)
                else:
                    x1, y1, x2, y2 = entry[:4]
                    det_conf = 1.0
            if source_track_id is None:
                continue
            if not keypoints:
                continue
            if not self._pose_inside_ring(keypoints, frame.shape):
                continue

            boxer_id = int(source_track_id)

            self._update_color_sig(boxer_id, frame, (x1, y1, x2, y2))

            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            track_rows.append(
                {
                    "track_id": boxer_id,
                    "bbox": (x1, y1, x2, y2),
                    "center": center,
                    "keypoints": keypoints,
                    "mask": mask,
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

        self.last_tracks = track_rows
        return poses_by_id

    def latest_tracks(self):
        return list(self.last_tracks)

    def tracking_diagnostics(self):
        if self.person_model_kind != "ultralytics_pose":
            return {
                "enabled": False,
                "mode": "unavailable",
                "issue": "Active backend does not provide Ultralytics pose track IDs.",
                "tracker": self.yolo_tracker,
                "tracker_config": self.yolo_tracker_config,
                "person_model_kind": self.person_model_kind,
            }
        if not self.yolo_tracker_config:
            return {
                "enabled": False,
                "mode": "unavailable",
                "issue": "No tracker configuration file was resolved.",
                "tracker": self.yolo_tracker,
                "tracker_config": self.yolo_tracker_config,
                "person_model_kind": self.person_model_kind,
            }
        if self._yolo_track_issue:
            return {
                "enabled": False,
                "mode": "predict_only",
                "issue": self._yolo_track_issue,
                "tracker": self.yolo_tracker,
                "tracker_config": self.yolo_tracker_config,
                "person_model_kind": self.person_model_kind,
            }
        if not self._lap_available():
            return {
                "enabled": False,
                "mode": "predict_only",
                "issue": "Missing optional dependency 'lap' required by the Ultralytics tracker.",
                "tracker": self.yolo_tracker,
                "tracker_config": self.yolo_tracker_config,
                "person_model_kind": self.person_model_kind,
            }
        return {
            "enabled": True,
            "mode": "track",
            "issue": None,
            "tracker": self.yolo_tracker,
            "tracker_config": self.yolo_tracker_config,
            "person_model_kind": self.person_model_kind,
        }

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
