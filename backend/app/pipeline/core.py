"""One shared analytics core for live and (later) upload mode. Time-based, engine-agnostic."""
from __future__ import annotations

import statistics
from collections import deque

from app.engines import PoseResult
from app.voice import settings
from app.pipeline.escalation import Escalation
from app.pipeline.fall_detector import FallDetector
from app.pipeline.kinematics import EMA, joint_angles
from app.pipeline.mobility import Mobility
from app.pipeline.rep_counter import EXERCISES, RepCounter, calibrated_limits

SITTING_KNEE_DEG = 130.0
# Angle smoothing is time based: the filter reacts the same over seconds at 10 fps and at 30. 0.08 s is what the old per-frame
# alpha of 0.35 amounted to at 30 fps (tau = -dt / ln(1 - 0.35)).
SMOOTH_TAU_S = 0.08

# Calibrating the rest angle from the READY standby (see Pipeline.calibrate)
CAL_WINDOW_S = 1.5       # the median of the last this-many seconds of standby
CAL_MIN_SAMPLES = 8       # and at least this many angles ...
CAL_MIN_SPAN_S = 0.8      # ... spread over at least this long, or there is not enough to trust
CAL_MAX_STALE_S = 0.5     # the newest sample must be this recent when counting starts (the person is still in view)
CAL_SIDE_DIFF_DEG = 30.0  # two-sided exercises: the sides' measured rests may differ by this much, else keep the table for both
CAL_MIN_SHIFT_DEG = 25.0  # a measured rest this close to the table's rest is the table's posture: keep the table values
CAL_STABLE_DEG = 20.0   # the angle must have stayed within this band: someone still moving has no "rest" to measure


class Pipeline:
    def __init__(self, exercise: str):
        # Switches (process environment or backend/.env, read when a session starts) to compare against the behaviour before Phase A:
        # ANGLES_3D=0 measures angles in the image (2D); SMOOTH_TAU_S=0 smooths with the old fixed per-frame alpha; CALIBRATE=0 keeps
        # the table's rest/peak. Defaults: all on.
        self.angles_3d = settings.get("ANGLES_3D", "1") != "0"
        self.calibrating = settings.get("CALIBRATE", "1") != "0"
        self.ema = EMA(tau=settings.get_float("SMOOTH_TAU_S", SMOOTH_TAU_S) or None)
        self.reps = RepCounter(exercise)
        self.mobility = Mobility()
        self.fall = FallDetector()
        self.escalation = Escalation()      # not cleared by reset(): resetting reps must not hide an alarm
        # False = standby: reps are NOT counted (a person still stepping into the frame must not score). Everything that
        # protects the person (fall detection, escalation) and the step counter keep running either way.
        self.counting = True
        self._cal: dict[str, deque[tuple[float, float]]] = {}      # standby: (time, angle) of each limb's joint, newest last
        self._last_t = 0.0
        self.last: dict = {}

    @property
    def exercise(self) -> str:
        return self.reps.exercise

    def set_exercise(self, exercise: str) -> None:
        self.reps = RepCounter(exercise)
        self.ema.reset()
        self._cal.clear()               # those angles belonged to the previous exercise's joints

    def _sample_standby(self, angles: dict[str, float], t: float) -> None:
        """While counting is off, remember the active exercise's joint angles for the last CAL_WINDOW_S seconds."""
        for side, key in self.reps.joints.items():
            if key in angles:
                buf = self._cal.setdefault(side, deque())
                buf.append((t, angles[key]))
                while buf and buf[0][0] < t - CAL_WINDOW_S:
                    buf.popleft()

    def _measured_rest(self, side: str) -> float | None:
        """The person's own starting angle for this limb, from the last second and a half of standby, or None if that cannot be
        trusted: too few samples, they stopped (the person left the view), or the angle was moving about."""
        window = [(t, a) for t, a in self._cal.get(side, ()) if t >= self._last_t - CAL_WINDOW_S]
        if len(window) < CAL_MIN_SAMPLES:
            return None
        first, last = window[0][0], window[-1][0]
        if last - first < CAL_MIN_SPAN_S or self._last_t - last > CAL_MAX_STALE_S:
            return None
        angles = [a for _, a in window]
        if max(angles) - min(angles) > CAL_STABLE_DEG:
            return None
        return statistics.median(angles)

    def calibrate(self) -> bool:
        """Call when counting is about to start (standby -> counting). Each limb's `rest` becomes the median angle the person held in
        standby, and its `peak` follows from the table's range and direction (rep_counter.calibrated_limits), so someone starting
        seated (a hip at 90 degrees, not 170) or lying gets a baseline that fits them. Anything doubtful (too few samples, moving,
        out of view, no room between rest and peak) keeps the table's standing values for that limb. True if any limb was calibrated."""
        cfg = EXERCISES[self.exercise]
        found = {}
        for side in self.reps.limbs if self.calibrating else ():
            measured = self._measured_rest(side)
            limits = None if measured is None else calibrated_limits(cfg["rest"], cfg["peak"], measured)
            if limits is not None:
                found[side] = limits
        # Both sides start from the same posture: if one limb could not be measured (out of view, held mid-movement) or the two
        # disagree, calibrating just one would count one side against the person's baseline and the other against the standing
        # table (one arm counts, the other never does). Then every limb keeps the table values.
        rests = [r for r, _ in found.values()]
        if len(self.reps.limbs) > 1 and (len(found) < len(self.reps.limbs) or max(rests) - min(rests) > CAL_SIDE_DIFF_DEG):
            found = {}
        # A posture close to the table's (standing) needs no calibration, and shifting the peak by sensor noise only makes it harder to
        # reach: calibrate only when the person really starts elsewhere (seated, lying).
        if all(abs(r - cfg["rest"]) < CAL_MIN_SHIFT_DEG for r, _ in found.values()):
            found = {}
        for side, limb in self.reps.limbs.items():
            if side in found:
                limb.set_limits(*found[side], calibrated=True)
            else:
                limb.use_table()
        self._cal.clear()
        return bool(found)

    def reset(self) -> None:
        self.reps.reset()
        self.mobility.reset()
        self.ema.reset()

    def process(self, res: PoseResult | None, t: float) -> dict:
        """t in seconds (wall clock live, frame_index / fps on upload)."""
        angles: dict[str, float] = {}
        self._last_t = t
        if res is not None:
            angles = self.ema.update(joint_angles(res, use_3d=self.angles_3d), t)
            if self.counting:
                self.reps.update(angles, t)
            else:
                self._sample_standby(angles, t)         # not counting yet: learn where this person starts from
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
