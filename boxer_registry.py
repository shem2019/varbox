import math

import mediapipe as mp
import numpy as np
from scipy.spatial.distance import cosine

from config import COOLDOWN_FRAMES

POSE = mp.solutions.pose.PoseLandmark


class BoxerRegistry:
    """
    Persistent person ID registry with pose + spatial matching.
    """

    def __init__(self, max_distance=0.34, max_age_frames=None):
        self.next_id = 0
        self.registry = {}  # boxer_id -> {'pose', 'center', 'scale', 'last_seen'}
        self.max_distance = max_distance
        self.max_age_frames = int(max_age_frames or max(COOLDOWN_FRAMES * 80, 900))

    def _pose_vector(self, keypoints, required_indices):
        vector = []
        if POSE.NOSE not in keypoints:
            return np.zeros(len(required_indices) * 2, dtype=np.float32)

        center = np.array(keypoints[POSE.NOSE], dtype=np.float32)
        if POSE.LEFT_SHOULDER in keypoints and POSE.RIGHT_SHOULDER in keypoints:
            scale = np.linalg.norm(
                np.array(keypoints[POSE.LEFT_SHOULDER], dtype=np.float32)
                - np.array(keypoints[POSE.RIGHT_SHOULDER], dtype=np.float32)
            )
        else:
            scale = 1.0
        scale = scale if scale > 0 else 1.0

        for idx in required_indices:
            if idx in keypoints:
                pt = np.array(keypoints[idx], dtype=np.float32)
                rel = (pt - center) / scale
                vector.extend(rel.tolist())
            else:
                vector.extend([0.0, 0.0])
        return np.array(vector, dtype=np.float32)

    @staticmethod
    def _center_scale(keypoints):
        if POSE.NOSE not in keypoints:
            return None, 1.0
        center = np.array(keypoints[POSE.NOSE], dtype=np.float32)
        if POSE.LEFT_SHOULDER in keypoints and POSE.RIGHT_SHOULDER in keypoints:
            scale = float(
                np.linalg.norm(
                    np.array(keypoints[POSE.LEFT_SHOULDER], dtype=np.float32)
                    - np.array(keypoints[POSE.RIGHT_SHOULDER], dtype=np.float32)
                )
            )
        else:
            scale = 40.0
        return center, max(20.0, scale)

    @staticmethod
    def safe_cosine(a, b):
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            return 1.0
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0 or b_norm == 0:
            return 1.0
        a = np.clip(a, -1e6, 1e6)
        b = np.clip(b, -1e6, 1e6)
        return float(cosine(a, b))

    def _combined_distance(self, new_pose, old_pose, new_center, old_center, new_scale, old_scale):
        pose_dist = self.safe_cosine(new_pose, old_pose)
        if new_center is None or old_center is None:
            spatial = 1.0
        else:
            dist = float(np.linalg.norm(new_center - old_center))
            denom = max(30.0, 1.8 * max(new_scale, old_scale))
            spatial = min(1.0, dist / denom)
        return 0.68 * pose_dist + 0.32 * spatial

    def match_or_register(self, keypoints, current_frame, required_indices):
        new_vector = self._pose_vector(keypoints, required_indices)
        if not np.all(np.isfinite(new_vector)) or np.linalg.norm(new_vector) == 0:
            return None

        new_center, new_scale = self._center_scale(keypoints)
        best_id = None
        best_distance = float("inf")

        for boxer_id, entry in self.registry.items():
            age = current_frame - entry["last_seen"]
            if age > self.max_age_frames:
                continue
            old_vector = entry["pose"]
            if not np.all(np.isfinite(old_vector)) or np.linalg.norm(old_vector) == 0:
                continue
            dist = self._combined_distance(
                new_vector,
                old_vector,
                new_center,
                entry.get("center"),
                new_scale,
                entry.get("scale", 40.0),
            )
            # Mild age penalty to prefer recently seen tracks.
            dist += min(0.08, age * 0.0008)
            if dist < best_distance and dist < self.max_distance:
                best_distance = dist
                best_id = boxer_id

        if best_id is not None:
            self.registry[best_id]["pose"] = new_vector
            self.registry[best_id]["center"] = new_center
            self.registry[best_id]["scale"] = new_scale
            self.registry[best_id]["last_seen"] = current_frame
            return best_id

        new_id = self.next_id
        self.registry[new_id] = {
            "pose": new_vector,
            "center": new_center,
            "scale": new_scale,
            "last_seen": current_frame,
        }
        self.next_id += 1
        return new_id

    def clean_old_ids(self, current_frame, max_age=None):
        keep_age = int(max_age or self.max_age_frames)
        to_delete = [
            bid for bid, data in self.registry.items() if current_frame - data["last_seen"] > keep_age
        ]
        for bid in to_delete:
            del self.registry[bid]
