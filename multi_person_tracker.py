# multi_person_tracker.py

import cv2
import numpy as np
import mediapipe as mp
from boxer_registry import BoxerRegistry
from identity_bootstrap import IdentityBootstrap
from color_signature import compute_hist_signature, signature_similarity

POSE = mp.solutions.pose.PoseLandmark

def pad_to_square(image):
    h, w = image.shape[:2]
    size = max(h, w)
    top = (size - h) // 2
    bottom = size - h - top
    left = (size - w) // 2
    right = size - w - left
    return cv2.copyMakeBorder(image, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)), top, left

class MultiPersonPoseTracker:
    def __init__(self, confidence=0.5, bootstrap_frames=30):
        from ultralytics import YOLO
        self.model = YOLO("yolov8n.pt")
        self.registry = BoxerRegistry()
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            enable_segmentation=True,
            smooth_segmentation=False,  # important with varying crop sizes
            min_detection_confidence=confidence,
            min_tracking_confidence=0.5
        )
        self.bootstrap = IdentityBootstrap(frames=bootstrap_frames, min_samples=5)
        self.role_map = {}               # boxer_id -> role
        self.id_color_sig = {}           # boxer_id -> running avg hist signature

    def detect_people(self, frame):
        results = self.model.predict(frame, classes=[0], verbose=False)[0]
        boxes = []
        for det in results.boxes.data.tolist():
            x1, y1, x2, y2, *_ = det
            boxes.append((int(x1), int(y1), int(x2), int(y2)))
        return boxes

    def _update_color_sig(self, boxer_id, frame, box, alpha=0.15):
        x1, y1, x2, y2 = box
        crop = frame[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
        sig = compute_hist_signature(crop)
        prev = self.id_color_sig.get(boxer_id)
        if prev is None or prev.shape != sig.shape:
            self.id_color_sig[boxer_id] = sig
        else:
            self.id_color_sig[boxer_id] = (1 - alpha) * prev + alpha * sig

    def process_frame(self, frame, frame_num):
        people_boxes = self.detect_people(frame)
        poses_by_id = {}

        for (x1, y1, x2, y2) in people_boxes:
            crop = frame[y1:y2, x1:x2]
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
                    int(landmark.y * h + y1 - pad_top)
                ]
                for idx, landmark in enumerate(landmarks)
            }

            required = [POSE.NOSE, POSE.LEFT_WRIST, POSE.RIGHT_WRIST, POSE.LEFT_SHOULDER, POSE.RIGHT_SHOULDER]
            boxer_id = self.registry.match_or_register(keypoints, frame_num, required)
            if boxer_id is None:
                continue

            # Update color signature per ID
            self._update_color_sig(boxer_id, frame, (x1, y1, x2, y2))

            # Bootstrap roles from early frames
            if not self.bootstrap.finalized:
                self.bootstrap.add_observation(frame_num, boxer_id, frame, (x1, y1, x2, y2))
                if self.bootstrap.ready(frame_num):
                    self.role_map = self.bootstrap.finalize()
                    print(f"🎯 Bootstrap roles: {self.role_map}")

            poses_by_id[boxer_id] = {
                "keypoints": keypoints,
                "box": (x1, y1, x2, y2),
                "mask": results.segmentation_mask,
                "role": self.role_map.get(boxer_id)  # may be None before finalize
            }

        self.registry.clean_old_ids(frame_num)
        return poses_by_id
