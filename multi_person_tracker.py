# multi_person_tracker.py

import cv2
import numpy as np
import mediapipe as mp
from boxer_registry import BoxerRegistry

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
    def __init__(self, confidence=0.5):
        from ultralytics import YOLO
        self.model = YOLO("yolov8n.pt")
        self.tracker = BoxerRegistry()
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            enable_segmentation=True,
            smooth_segmentation=False,  # ✅ this line fixes the crash
            min_detection_confidence=confidence,
            min_tracking_confidence=0.5
        )

    def detect_people(self, frame):
        results = self.model.predict(frame, classes=[0], verbose=False)[0]
        boxes = []
        for box in results.boxes.data.tolist():
            x1, y1, x2, y2, *_ = map(int, box[:6])
            boxes.append((x1, y1, x2, y2))
        return boxes

    def process_frame(self, frame, frame_num):
        people_boxes = self.detect_people(frame)
        poses_by_id = {}

        for (x1, y1, x2, y2) in people_boxes:
            crop = frame[y1:y2, x1:x2]

            # ✅ Square padded crop
            padded_crop, pad_top, pad_left = pad_to_square(crop)
            rgb_crop = cv2.cvtColor(padded_crop, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_crop)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                h, w = padded_crop.shape[:2]

                keypoints = {
                    idx: [
                        int(landmark.x * w + x1 - pad_left),
                        int(landmark.y * h + y1 - pad_top)
                    ]
                    for idx, landmark in enumerate(landmarks)
                }

                # Required for Re-ID
                required = [
                    POSE.NOSE,
                    POSE.LEFT_WRIST,
                    POSE.RIGHT_WRIST,
                    POSE.LEFT_SHOULDER,
                    POSE.RIGHT_SHOULDER,
                ]
                boxer_id = self.tracker.match_or_register(keypoints, frame_num, required)
                if boxer_id is not None:
                    poses_by_id[boxer_id] = {
                        "keypoints": keypoints,
                        "box": (x1, y1, x2, y2),   # for overlay
                        "mask": results.segmentation_mask
                    }

        self.tracker.clean_old_ids(frame_num)
        return poses_by_id
