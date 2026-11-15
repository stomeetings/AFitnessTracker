"""Joint angles from a PoseResult (3D world coords when available, else 2D) + EMA smoothing."""
from __future__ import annotations

import numpy as np

from app.engines import PoseResult

# joint name -> (a, b, c); the angle is measured at b.
TRIPLES = {
    "elbow": ("shoulder", "elbow", "wrist"),
    "shoulder": ("hip", "shoulder", "elbow"),
    "knee": ("hip", "knee", "ankle"),
    "hip": ("shoulder", "hip", "knee"),
}


def angle_deg(a, b, c) -> float | None:
    a, b, c = (np.asarray(p, dtype=float) for p in (a, b, c))
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-9:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0))))


def joint_angles(res: PoseResult, min_conf: float = 0.5, use_3d: bool = True) -> dict[str, float]:
    """Returns e.g. {'l_elbow': 92.1, 'r_knee': 171.0, 'l_hip_abd': 93.0, 'knee_mean': ...}. Missing joints are omitted.

    `{side}_hip_abd` is the angle at the hip between the pelvis line (to the other hip) and the thigh:
    about 90 standing, larger as the leg lifts out to the side.
    """
    out: dict[str, float] = {}

    def put(key: str, names: tuple[str, str, str]) -> None:
        kps = [res.get(n, min_conf) for n in names]
        if not all(kps):
            return
        pts = [k.world for k in kps] if use_3d and all(k.world for k in kps) else [(k.x, k.y) for k in kps]
        ang = angle_deg(*pts)
        if ang is not None:
            out[key] = ang

    for side, other in ("lr", "rl"):
        for joint, names in TRIPLES.items():
            put(f"{side}_{joint}", tuple(f"{side}_{n}" for n in names))
        put(f"{side}_hip_abd", (f"{other}_hip", f"{side}_hip", f"{side}_knee"))
    for joint in ("knee", "hip", "elbow"):
        vals = [out[k] for k in (f"l_{joint}", f"r_{joint}") if k in out]
        if vals:
            out[f"{joint}_mean"] = sum(vals) / len(vals)
    return out


class EMA:
    """Per-key exponential moving average. Keys absent in a frame are not updated or returned (FSMs freeze)."""

    def __init__(self, alpha: float = 0.35):
        self.alpha = alpha
        self.s: dict[str, float] = {}

    def update(self, vals: dict[str, float]) -> dict[str, float]:
        for k, v in vals.items():
            self.s[k] = v if k not in self.s else self.s[k] + self.alpha * (v - self.s[k])
        return {k: self.s[k] for k in vals}

    def reset(self) -> None:
        self.s.clear()
