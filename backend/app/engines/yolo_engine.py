"""Ultralytics YOLO11n-pose (COCO 17 keypoints). AGPL-3.0: licence decision deferred to P3."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import BBox, Capabilities, Keypoint, PoseEngine, PoseResult

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

# COCO keypoint index -> common joint name (no feet)
_COCO_INDEX = {
    0: "nose",
    5: "l_shoulder", 6: "r_shoulder", 7: "l_elbow", 8: "r_elbow", 9: "l_wrist", 10: "r_wrist",
    11: "l_hip", 12: "r_hip", 13: "l_knee", 14: "r_knee", 15: "l_ankle", 16: "r_ankle",
}


class YoloEngine(PoseEngine):
    def __init__(self, device: str = "cuda:0", weights: str = "yolo11n-pose.pt", imgsz: int = 640, min_conf: float = 0.25,
                 half: bool = True):
        self.device = device
        self.weights = MODELS_DIR / weights
        self.imgsz = imgsz
        self.min_conf = min_conf
        # FP16 on a CUDA GPU: about twice the speed for practically the same keypoints. (INT8 needs TensorRT calibration and
        # costs keypoint accuracy; on a GPU FP16 is the right trade.) CPU has no half precision, so it is never asked for there.
        self.half = bool(half and device.startswith("cuda"))
        self._precision: dict = {}              # the keyword that asks Ultralytics for FP16; worked out in load() (needs the import)
        self.name = f"yolo11n-pose-{device.split(':')[0]}"
        self.capabilities = Capabilities(has_3d=False, has_feet=False, native_bbox=True, device=device)
        self._model = None

    def _precision_args(self) -> dict:
        """FP16 as keyword arguments for predict(). Newer Ultralytics calls it `quantize=16` and deprecates `half`, and WARNS ON EVERY
        CALL that `half` is deprecated (it would flood the log at 20+ frames a second), while older versions only know `half`."""
        if not self.half:
            return {}
        from ultralytics.utils import DEFAULT_CFG_DICT
        return {"quantize": 16} if "quantize" in DEFAULT_CFG_DICT else {"half": True}

    def load(self) -> None:
        from ultralytics import YOLO  # slow import, keep it lazy
        if not self.weights.exists():
            raise FileNotFoundError(f"{self.weights} missing; run scripts/download_models.py")
        self._precision = self._precision_args()
        self._model = YOLO(str(self.weights))
        self._model.to(self.device)
        # warm-up: the first predict pays for CUDA/cuDNN init and model fusing; do it here, not on a live frame
        self._model.predict(np.zeros((480, 640, 3), np.uint8), device=self.device, imgsz=self.imgsz, verbose=False, **self._precision)

    def infer(self, frame_bgr: np.ndarray, timestamp_ms: int) -> PoseResult | None:
        r = self._model.predict(frame_bgr, device=self.device, imgsz=self.imgsz, conf=self.min_conf, verbose=False,
                                **self._precision)[0]
        return self._to_pose(r, timestamp_ms)

    def infer_batch(self, frames_bgr, timestamps_ms) -> list[PoseResult | None]:
        """The whole batch goes through the GPU in one call (the frames may differ in size)."""
        if not len(frames_bgr):
            return []
        results = self._model.predict(list(frames_bgr), device=self.device, imgsz=self.imgsz, conf=self.min_conf,
                                      verbose=False, **self._precision)
        return [self._to_pose(r, ts) for r, ts in zip(results, timestamps_ms)]

    def _to_pose(self, r, timestamp_ms: int) -> PoseResult | None:
        if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None:
            return None
        xyxy = r.boxes.xyxy
        i = int(((xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])).argmax())  # single-patient: largest person
        x0, y0, x1, y1 = xyxy[i].tolist()
        xy = r.keypoints.xy[i].cpu().numpy()
        conf = r.keypoints.conf[i].cpu().numpy() if r.keypoints.conf is not None else np.ones(len(xy))
        kps = {name: Keypoint(float(xy[j, 0]), float(xy[j, 1]), float(conf[j])) for j, name in _COCO_INDEX.items()}
        return PoseResult(kps, BBox(x0, y0, x1 - x0, y1 - y0, float(r.boxes.conf[i])), int(timestamp_ms))

    def close(self) -> None:
        self._model = None
