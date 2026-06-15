"""PoseEngine implementation using Google MediaPipe PoseLandmarker."""

from __future__ import annotations

import os

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

from surfanalysis.extraction.engine import PoseEngine
from surfanalysis.extraction.schema import EngineInfo, Keypoints


class MediaPipeEngine(PoseEngine):
    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._params = {
            "model_complexity": model_complexity,
            "min_detection_confidence": min_detection_confidence,
            "min_tracking_confidence": min_tracking_confidence,
        }
        # Monotonic timestamp guard for VIDEO mode (see detect()).
        self._last_ts_ms = -1
        model_path = os.environ.get(
            "MEDIAPIPE_MODEL_PATH",
            "/Users/shuk/.mediapipe/pose_landmarker.task",
        )
        base_options = BaseOptions(model_asset_path=model_path)
        options = PoseLandmarkerOptions(
            base_options=base_options,
            # VIDEO mode keeps a tracking prior across frames so the pose
            # carries over momentary detection dropouts. IMAGE mode re-detects
            # each frame from scratch and ignores min_tracking_confidence,
            # producing on/off flicker on small/fast-moving subjects.
            running_mode=VisionTaskRunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._pose = PoseLandmarker.create_from_options(options)

    def detect(self, frame: np.ndarray, timestamp_ms: float = 0.0) -> Keypoints | None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # detect_for_video requires strictly increasing integer ms timestamps;
        # clamp upward if the source ever repeats or regresses a timestamp.
        ts = int(timestamp_ms)
        if ts <= self._last_ts_ms:
            ts = self._last_ts_ms + 1
        self._last_ts_ms = ts
        result = self._pose.detect_for_video(mp_image, ts)
        if result.pose_landmarks is None or len(result.pose_landmarks) == 0:
            return None
        h, w = frame.shape[:2]
        points = [
            (lm.x, lm.y, lm.z, lm.visibility)
            for lm in result.pose_landmarks[0]
        ]
        return Keypoints(points=points, image_size=(w, h))

    def info(self) -> EngineInfo:
        return EngineInfo(name="mediapipe", version=mp.__version__, params=self._params)

    def close(self) -> None:
        self._pose.close()
