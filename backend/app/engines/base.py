"""Engine-agnostic pose contract. Analytics code only ever sees PoseResult."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

# Common joint schema. Every engine fills the core 13; feet are optional (MediaPipe only).
CORE_JOINTS = (
    "nose",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
    "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle",
)
FOOT_JOINTS = ("l_heel", "r_heel", "l_foot_index", "r_foot_index")

SKELETON = (
    ("l_shoulder", "r_shoulder"), ("l_hip", "r_hip"),
    ("l_shoulder", "l_hip"), ("r_shoulder", "r_hip"),
    ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
    ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"),
    ("l_hip", "l_knee"), ("l_knee", "l_ankle"),
    ("r_hip", "r_knee"), ("r_knee", "r_ankle"),
    ("l_ankle", "l_heel"), ("l_heel", "l_foot_index"),
    ("r_ankle", "r_heel"), ("r_heel", "r_foot_index"),
)


# Every joint name a pose can carry (the MediaPipe set; YOLO's COCO joints are a subset). A browser that runs the pose model
# itself may send only these.
KEYPOINT_NAMES = frozenset({
    "nose", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip",
    "l_knee", "r_knee", "l_ankle", "r_ankle", "l_heel", "r_heel", "l_foot_index", "r_foot_index",
})


@dataclass
class Keypoint:
    x: float            # image pixels
    y: float
    conf: float         # visibility / confidence 0..1
    world: tuple[float, float, float] | None = None  # metres, hip-centred (MediaPipe only)


@dataclass
class BBox:
    x: float
    y: float
    w: float
    h: float
    conf: float


@dataclass
class PoseResult:
    keypoints: dict[str, Keypoint]
    bbox: BBox
    timestamp_ms: int

    def get(self, name: str, min_conf: float = 0.5) -> Keypoint | None:
        kp = self.keypoints.get(name)
        return kp if kp is not None and kp.conf >= min_conf else None


def bbox_from_keypoints(kps: dict[str, Keypoint], frame_shape, min_conf=0.5, pad=0.05) -> BBox | None:
    pts = np.array([(k.x, k.y) for k in kps.values() if k.conf >= min_conf])
    if len(pts) < 4:
        return None
    h_img, w_img = frame_shape[:2]
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    x0, y0 = max(0.0, x0 - px), max(0.0, y0 - py)
    x1, y1 = min(float(w_img), x1 + px), min(float(h_img), y1 + py)
    return BBox(x0, y0, x1 - x0, y1 - y0, float(np.mean([k.conf for k in kps.values()])))


@dataclass
class Capabilities:
    has_3d: bool
    has_feet: bool
    native_bbox: bool
    device: str = "cpu"


class PoseEngine(ABC):
    name: str = "base"
    capabilities: Capabilities | None = None
    # True when infer() carries nothing from one frame to the next (YOLO). A stateful engine (MediaPipe tracks the
    # person between video frames) must never see the frames of two different patients, so the inference service
    # gives each session its own instance of it, and shares one instance of a stateless engine between all sessions.
    stateless: bool = True

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def infer(self, frame_bgr: np.ndarray, timestamp_ms: int) -> PoseResult | None:
        """Return the single most confident pose, or None if nobody is detected."""

    def infer_batch(self, frames_bgr, timestamps_ms) -> list[PoseResult | None]:
        """One pose (or None) per frame, in the same order. The default runs the frames one by one; an engine that can
        do better (YOLO sends the whole batch through the GPU at once) overrides it. Frames may differ in size."""
        return [self.infer(frame, ts) for frame, ts in zip(frames_bgr, timestamps_ms)]

    def close(self) -> None:
        pass

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, *exc):
        self.close()
