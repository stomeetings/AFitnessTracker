"""Walking / step-cadence estimate from ankle separation oscillation and hip displacement (2D image space).

All lengths are normalised by torso length so the estimate is distance-independent. Side view works best.
"""
from __future__ import annotations

import math
from collections import deque

from app.engines import PoseResult

SEP_HI, SEP_LO = 0.45, 0.25   # ankle separation / torso length; a step = rise above HI then fall below LO
MIN_STEP_S = 0.3
CADENCE_WINDOW_S = 10.0
DISP_WINDOW_S = 2.0
WALK_DISP = 0.5               # hip moved > 0.5 torso lengths within DISP_WINDOW_S
RECENT_STEP_S = 2.0


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class Mobility:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.state = "Stationary"
        self._hip_hist: deque[tuple[float, tuple[float, float]]] = deque()
        self._step_times: deque[float] = deque()
        self._armed = False
        self._last_step_t = -1e9

    def update(self, res: PoseResult, t: float, exercising: bool) -> dict:
        kp = {n: res.get(n) for n in ("l_ankle", "r_ankle", "l_hip", "r_hip", "l_shoulder", "r_shoulder")}
        if not all(kp.values()):
            return self.snapshot()
        hip = ((kp["l_hip"].x + kp["r_hip"].x) / 2, (kp["l_hip"].y + kp["r_hip"].y) / 2)
        sh = ((kp["l_shoulder"].x + kp["r_shoulder"].x) / 2, (kp["l_shoulder"].y + kp["r_shoulder"].y) / 2)
        torso = max(_dist(hip, sh), 1e-6)
        sep = _dist((kp["l_ankle"].x, kp["l_ankle"].y), (kp["r_ankle"].x, kp["r_ankle"].y)) / torso

        if sep > SEP_HI:
            self._armed = True
        elif self._armed and sep < SEP_LO and t - self._last_step_t > MIN_STEP_S:
            self.steps += 1
            self._step_times.append(t)
            self._last_step_t = t
            self._armed = False
        while self._step_times and t - self._step_times[0] > CADENCE_WINDOW_S:
            self._step_times.popleft()

        self._hip_hist.append((t, hip))
        while self._hip_hist and t - self._hip_hist[0][0] > DISP_WINDOW_S:
            self._hip_hist.popleft()
        disp = _dist(self._hip_hist[0][1], hip) / torso

        if disp > WALK_DISP and t - self._last_step_t < RECENT_STEP_S:
            self.state = "Walking"
        elif exercising:
            self.state = "Active Exercise"
        else:
            self.state = "Stationary"
        return self.snapshot()

    def snapshot(self) -> dict:
        return {
            "state": self.state,
            "steps": self.steps,
            "cadence_spm": round(len(self._step_times) * 60.0 / CADENCE_WINDOW_S),
        }
