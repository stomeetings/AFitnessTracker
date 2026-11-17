"""MediaPipe Pose Landmarker (CPU on Windows). Variants: lite | full."""
from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

from .base import Capabilities, Keypoint, PoseEngine, PoseResult, bbox_from_keypoints

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

# MediaPipe landmark index -> common joint name
_MP_INDEX = {
    0: "nose",
    11: "l_shoulder", 12: "r_shoulder", 13: "l_elbow", 14: "r_elbow", 15: "l_wrist", 16: "r_wrist",
    23: "l_hip", 24: "r_hip", 25: "l_knee", 26: "r_knee", 27: "l_ankle", 28: "r_ankle",
    29: "l_heel", 30: "r_heel", 31: "l_foot_index", 32: "r_foot_index",
}


class MediaPipeEngine(PoseEngine):
    capabilities = Capabilities(has_3d=True, has_feet=True, native_bbox=False, device="cpu")
    stateless = False       # VIDEO mode tracks the person between frames: one instance per patient, never shared

    def __init__(self, variant: str = "full", min_conf: float = 0.5):
        assert variant in ("lite", "full"), variant
        self.variant = variant
        self.name = f"mediapipe-{variant}"
        self.model_path = MODELS_DIR / f"pose_landmarker_{variant}.task"
        self.min_conf = min_conf
        self._lm = None
        self._last_ts = -1

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"{self.model_path} missing; run scripts/download_models.py")
        self._lm = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=self.min_conf,
            min_pose_presence_confidence=self.min_conf,
            min_tracking_confidence=self.min_conf,
        ))

    def infer(self, frame_bgr: np.ndarray, timestamp_ms: int) -> PoseResult | None:
        # VIDEO mode requires strictly increasing timestamps.
        timestamp_ms = max(int(timestamp_ms), self._last_ts + 1)
        self._last_ts = timestamp_ms
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self._lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
        if not res.pose_landmarks:
            return None
        h, w = frame_bgr.shape[:2]
        lms, world = res.pose_landmarks[0], res.pose_world_landmarks[0]
        kps = {
            name: Keypoint(lms[i].x * w, lms[i].y * h, float(lms[i].visibility or 0.0),
                           (world[i].x, world[i].y, world[i].z))
            for i, name in _MP_INDEX.items()
        }
        bbox = bbox_from_keypoints(kps, frame_bgr.shape, self.min_conf)
        if bbox is None:
            return None
        return PoseResult(kps, bbox, timestamp_ms)

    def close(self) -> None:
        if self._lm is not None:
            self._lm.close()
            self._lm = None
