# boxer_registry.py

import numpy as np
from scipy.spatial.distance import cosine
from config import COOLDOWN_FRAMES
import mediapipe as mp
POSE = mp.solutions.pose.PoseLandmark


class BoxerRegistry:
    def __init__(self, max_distance=0.3):
        self.next_id = 0
        self.registry = {}  # boxer_id -> {'pose': keypoint_vector, 'last_seen': frame_num}
        self.max_distance = max_distance

    def _pose_vector(self, keypoints, required_indices):
        vector = []
        if POSE.NOSE not in keypoints:
            return np.zeros(len(required_indices) * 2)

        center = np.array(keypoints[POSE.NOSE])

        # Use shoulder width as scale reference
        if POSE.LEFT_SHOULDER in keypoints and POSE.RIGHT_SHOULDER in keypoints:
            scale = np.linalg.norm(
                np.array(keypoints[POSE.LEFT_SHOULDER]) - np.array(keypoints[POSE.RIGHT_SHOULDER]))
        else:
            scale = 1.0
        scale = scale if scale > 0 else 1.0

        for idx in required_indices:
            if idx in keypoints:
                pt = np.array(keypoints[idx])
                rel = (pt - center) / scale
                vector.extend(rel.tolist())
            else:
                vector.extend([0, 0])
        return np.array(vector)

    def safe_cosine(self, a, b):
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            return 1.0
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0 or b_norm == 0:
            return 1.0
        a = np.clip(a, -1e6, 1e6)
        b = np.clip(b, -1e6, 1e6)
        return cosine(a, b)

    def match_or_register(self, keypoints, current_frame, required_indices):
        new_vector = self._pose_vector(keypoints, required_indices)

        if not np.all(np.isfinite(new_vector)) or np.linalg.norm(new_vector) == 0:
            print(f"⚠️ Skipping invalid pose vector at frame {current_frame}")
            return None

        best_id = None
        best_distance = float('inf')

        for boxer_id, entry in self.registry.items():
            old_vector = entry['pose']
            if not np.all(np.isfinite(old_vector)) or np.linalg.norm(old_vector) == 0:
                continue
            dist = self.safe_cosine(new_vector, old_vector)
            if dist < best_distance and dist < self.max_distance:
                best_distance = dist
                best_id = boxer_id

        if best_id is not None:
            self.registry[best_id]['pose'] = new_vector
            self.registry[best_id]['last_seen'] = current_frame
            return best_id
        else:
            self.registry[self.next_id] = {'pose': new_vector, 'last_seen': current_frame}
            self.next_id += 1
            return self.next_id - 1

    def clean_old_ids(self, current_frame, max_age=COOLDOWN_FRAMES * 4):
        to_delete = [bid for bid, data in self.registry.items()
                     if current_frame - data['last_seen'] > max_age]
        for bid in to_delete:
            del self.registry[bid]
