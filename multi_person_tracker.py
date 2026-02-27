# multi_person_tracker.py

import os
import cv2
import mediapipe as mp

from boxer_registry import BoxerRegistry
from identity_bootstrap import IdentityBootstrap
from color_signature import compute_hist_signature
from config import BACKEND, DNN_PROTO, DNN_MODEL, YOLOV8_WEIGHTS

POSE = mp.solutions.pose.PoseLandmark


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
        self.backend = (backend or BACKEND or "opencv").strip().lower()
        self.registry = BoxerRegistry(max_distance=0.34)
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            enable_segmentation=True,
            smooth_segmentation=False,
            min_detection_confidence=confidence,
            min_tracking_confidence=0.5,
        )
        self.bootstrap = IdentityBootstrap(frames=bootstrap_frames, min_samples=5)
        self.role_map = {}  # boxer_id -> role
        self.id_color_sig = {}  # boxer_id -> running avg hist signature
        self.person_model = None
        self.person_model_kind = None
        self.max_people = 8
        self.min_box_area = 80 * 80
        self.last_tracks = []
        self._init_detector()

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
        results = self.person_model.predict(frame, classes=[0], verbose=False)[0]
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
            return self._detect_people_opencv(frame)
        return self._detect_people_yolov8(frame)

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

        for entry in people_boxes:
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
            results = self.pose.process(rgb_crop)

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
                    self.role_map = self.bootstrap.finalize()
                    if self.role_map:
                        print(f"Bootstrap roles: {self.role_map}")

            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            poses_by_id[boxer_id] = {
                "track_id": boxer_id,
                "keypoints": keypoints,
                "box": (x1, y1, x2, y2),
                "bbox": (x1, y1, x2, y2),
                "center": center,
                "mask": results.segmentation_mask,
                "role": self.role_map.get(boxer_id),
                "det_conf": float(det_conf),
            }
            track_rows.append(
                {
                    "track_id": boxer_id,
                    "bbox": (x1, y1, x2, y2),
                    "center": center,
                    "keypoints": keypoints,
                    "det_conf": float(det_conf),
                }
            )

        self.registry.clean_old_ids(frame_num)
        self.last_tracks = track_rows
        return poses_by_id

    def latest_tracks(self):
        return list(self.last_tracks)
