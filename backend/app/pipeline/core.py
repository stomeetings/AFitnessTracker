"""One shared analytics core for live and (later) upload mode. Time-based, engine-agnostic."""
from __future__ import annotations

from app.engines import PoseResult
from app.pipeline.escalation import Escalation
from app.pipeline.fall_detector import FallDetector
from app.pipeline.kinematics import EMA, joint_angles
from app.pipeline.mobility import Mobility
from app.pipeline.rep_counter import RepCounter

SITTING_KNEE_DEG = 130.0


class Pipeline:
    def __init__(self, exercise: str):
        self.ema = EMA(alpha=0.35)
        self.reps = RepCounter(exercise)
        self.mobility = Mobility()
        self.fall = FallDetector()
        self.escalation = Escalation()      # not cleared by reset(): resetting reps must not hide an alarm
        self.last: dict = {}

    @property
    def exercise(self) -> str:
        return self.reps.exercise

    def set_exercise(self, exercise: str) -> None:
        self.reps = RepCounter(exercise)
        self.ema.reset()

    def reset(self) -> None:
        self.reps.reset()
        self.mobility.reset()
        self.ema.reset()

    def process(self, res: PoseResult | None, t: float) -> dict:
        """t in seconds (wall clock live, frame_index / fps on upload)."""
        angles: dict[str, float] = {}
        if res is not None:
            angles = self.ema.update(joint_angles(res))
            self.reps.update(angles, t)
            self.mobility.update(res, t, self.reps.active())
        fall = self.fall.update(res, t)
        fall["alert"] = self.escalation.update(fall, t)

        if res is None:
            posture = "unknown"
        elif fall["on_floor"] or (fall["tilt_deg"] is not None and fall["tilt_deg"] > 55):
            posture = "lying"
        elif angles.get("knee_mean", 180) < SITTING_KNEE_DEG:
            posture = "sitting"
        else:
            posture = "standing"

        self.last = {
            "exercise": self.exercise,
            "angles": {k: round(v, 1) for k, v in angles.items()},
            "reps": self.reps.snapshot(),
            "mobility": self.mobility.snapshot(),
            "fall": fall,
            "posture": posture,
        }
        return self.last
