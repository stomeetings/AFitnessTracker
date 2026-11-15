"""Per-limb rep FSM with hysteresis and ROM tracking.

Progress p = (angle - rest) / (peak - rest): 0 at rest, 1 at full peak, works for both flexion
(angle decreasing) and raise/extension (angle increasing) exercises.
IDLE -> FLEXING -> PEAK -> RETURNING -> (rep complete) -> IDLE
"""
from __future__ import annotations

EXERCISES = {
    "Elbow Curl":     {"joints": {"L": "l_elbow", "R": "r_elbow"},       "rest": 160, "peak": 50},
    "Shoulder Raise": {"joints": {"L": "l_shoulder", "R": "r_shoulder"}, "rest": 20,  "peak": 90},
    "Knee Flexion":   {"joints": {"L": "l_knee", "R": "r_knee"},         "rest": 170, "peak": 100},
    "Hip Flexion":    {"joints": {"L": "l_hip", "R": "r_hip"},           "rest": 170, "peak": 110},
    # ponytail: mean knee angle only; add hip-height-vs-knee gating if seated leg swings false-count.
    "Sit-to-Stand":   {"joints": {"B": "knee_mean"},                     "rest": 100, "peak": 165},
    # Added in the library expansion. Angles are first-pass values pending physio review.
    "Overhead Reach":        {"joints": {"L": "l_shoulder", "R": "r_shoulder"}, "rest": 20,  "peak": 150},
    "Seated Knee Extension": {"joints": {"L": "l_knee", "R": "r_knee"},         "rest": 95,  "peak": 165},
    "Mini Squat":            {"joints": {"B": "knee_mean"},                     "rest": 170, "peak": 115},
    "Side Leg Raise":        {"joints": {"L": "l_hip_abd", "R": "r_hip_abd"},   "rest": 92,  "peak": 120},
    "Wall Push-up":          {"joints": {"B": "elbow_mean"},                    "rest": 165, "peak": 95},
}

IDLE, FLEXING, PEAK, RETURNING = "IDLE", "FLEXING", "PEAK", "RETURNING"
# hysteresis bands on progress p
START, ABORT = 0.20, 0.10     # leave IDLE above START; FLEXING falls back to IDLE below ABORT
PEAK_IN, PEAK_OUT = 0.80, 0.60
DONE = 0.15                   # RETURNING completes the rep below DONE
MIN_REP_S, MAX_REP_S = 0.4, 10.0


class LimbFSM:
    def __init__(self, rest: float, peak: float):
        self.rest, self.peak = rest, peak
        self.reset()

    def reset(self) -> None:
        self.state = IDLE
        self.reps = 0
        self.last_rom = 0.0
        self.best_rom = 0.0
        self.angle: float | None = None
        self._t_start = 0.0
        self._amin = self._amax = 0.0
        self._idle_min = self._idle_max = None  # angle extremes seen while IDLE, so ROM includes the rest position

    def update(self, angle: float, t: float) -> bool:
        """Returns True on the frame a rep completes."""
        self.angle = angle
        p = (angle - self.rest) / (self.peak - self.rest)
        if self.state == IDLE:
            if self._idle_min is None:
                self._idle_min = self._idle_max = angle
            self._idle_min, self._idle_max = min(self._idle_min, angle), max(self._idle_max, angle)
            if p > START:
                self.state, self._t_start = FLEXING, t
                self._amin, self._amax = self._idle_min, self._idle_max
                self._idle_min = self._idle_max = None
            return False
        self._amin, self._amax = min(self._amin, angle), max(self._amax, angle)
        if self.state == FLEXING:
            if p < ABORT:
                self.state = IDLE
            elif p > PEAK_IN:
                self.state = PEAK
        elif self.state == PEAK:
            if p < PEAK_OUT:
                self.state = RETURNING
        elif self.state == RETURNING:
            if p > PEAK_IN:
                self.state = PEAK
            elif p < DONE:
                self.state = IDLE
                if MIN_REP_S <= t - self._t_start <= MAX_REP_S:
                    self.reps += 1
                    self.last_rom = self._amax - self._amin
                    self.best_rom = max(self.best_rom, self.last_rom)
                    return True
        return False

    def snapshot(self) -> dict:
        return {
            "reps": self.reps, "state": self.state,
            "angle": None if self.angle is None else round(self.angle, 1),
            "last_rom": round(self.last_rom, 1), "best_rom": round(self.best_rom, 1),
        }


class RepCounter:
    def __init__(self, exercise: str):
        cfg = EXERCISES[exercise]
        self.exercise = exercise
        self.joints: dict[str, str] = cfg["joints"]
        self.limbs = {side: LimbFSM(cfg["rest"], cfg["peak"]) for side in self.joints}

    def update(self, angles: dict[str, float], t: float) -> None:
        for side, key in self.joints.items():
            if key in angles:
                self.limbs[side].update(angles[key], t)

    def active(self) -> bool:
        return any(l.state != IDLE for l in self.limbs.values())

    def reset(self) -> None:
        for l in self.limbs.values():
            l.reset()

    def snapshot(self) -> dict:
        return {side: l.snapshot() for side, l in self.limbs.items()}
