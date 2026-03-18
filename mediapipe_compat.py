import os
from types import SimpleNamespace

import numpy as np
import mediapipe as mp


try:
    _POSE_MODULE = mp.solutions.pose
    PoseLandmark = _POSE_MODULE.PoseLandmark
except AttributeError:
    _POSE_MODULE = None
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision import (
        PoseLandmark,
        PoseLandmarker,
        PoseLandmarkerOptions,
        RunningMode,
    )


class _LegacyPoseAdapter:
    def __init__(self, confidence, enable_segmentation, model_complexity):
        self._pose = _POSE_MODULE.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=enable_segmentation,
            smooth_segmentation=False,
            min_detection_confidence=confidence,
            min_tracking_confidence=0.5,
        )

    def process(self, rgb_frame, timestamp_ms=None):
        del timestamp_ms
        return self._pose.process(rgb_frame)


class _TaskPoseAdapter:
    def __init__(self, confidence, enable_segmentation, model_path):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"MediaPipe pose model not found at {model_path}. "
                "Set VARBOX_POSE_TASK_MODEL to a valid .task file."
            )
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=model_path,
                delegate=BaseOptions.Delegate.CPU,
            ),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=confidence,
            min_pose_presence_confidence=confidence,
            min_tracking_confidence=0.5,
            output_segmentation_masks=enable_segmentation,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)

    def process(self, rgb_frame, timestamp_ms=None):
        frame = np.ascontiguousarray(rgb_frame)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        ts_value = 0 if timestamp_ms is None else int(timestamp_ms)
        result = self._landmarker.detect_for_video(mp_image, ts_value)
        if not result.pose_landmarks:
            return SimpleNamespace(pose_landmarks=None, segmentation_mask=None)

        segmentation_mask = None
        if result.segmentation_masks:
            segmentation_mask = np.array(result.segmentation_masks[0].numpy_view(), copy=True)

        return SimpleNamespace(
            pose_landmarks=SimpleNamespace(landmark=result.pose_landmarks[0]),
            segmentation_mask=segmentation_mask,
        )


def create_pose_estimator(
    confidence,
    enable_segmentation,
    model_complexity,
    task_model_path,
):
    if _POSE_MODULE is not None:
        return _LegacyPoseAdapter(
            confidence=confidence,
            enable_segmentation=enable_segmentation,
            model_complexity=model_complexity,
        )
    return _TaskPoseAdapter(
        confidence=confidence,
        enable_segmentation=enable_segmentation,
        model_path=task_model_path,
    )
