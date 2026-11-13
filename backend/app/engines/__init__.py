from .base import SKELETON, BBox, Capabilities, Keypoint, PoseEngine, PoseResult


def make_engine(name: str, **kw) -> PoseEngine:
    """name: 'mediapipe' (kw: variant=lite|full) or 'yolo' (kw: device=cuda:0|cpu)."""
    if name == "mediapipe":
        from .mediapipe_engine import MediaPipeEngine
        return MediaPipeEngine(**kw)
    if name == "yolo":
        from .yolo_engine import YoloEngine
        return YoloEngine(**kw)
    raise ValueError(f"unknown engine {name!r}")
