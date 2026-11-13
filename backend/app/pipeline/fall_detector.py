"""3-stage fall detection: NORMAL -> SUSPECTED -> CONFIRMED (-> NORMAL on recovery).

Signals (2D image space, normalised by torso length):
- hip-centre downward velocity (torso lengths / s)
- hip drop below the recent standing baseline (torso lengths), or hips at ankle height
- torso tilt from vertical (degrees) and bounding-box aspect ratio (w / h)
- pose lost shortly after a rapid drop

A fall = hips LOW and body HORIZONTAL. Tilt or a wide box alone (bending, reaching, sitting) never
leaves NORMAL; a drop only counts when it is both fast and large.
"""
from __future__ import annotations

import math
from collections import deque

from app.engines import PoseResult

NORMAL, SUSPECTED, CONFIRMED = "NORMAL", "SUSPECTED", "CONFIRMED"

TILT_SUSPECT = 55.0    # degrees from vertical
ASPECT_WIDE = 1.2
DROP_V = 2.5           # torso lengths / s downward: a fall is ~4+, sitting down fast ~1.5-2
HIP_DROP = 1.2         # hip below standing baseline by this many torso lengths = floor level
                       # (sitting on a chair drops the hip ~0.8 torso)
HIP_NEAR_ANKLE = 0.4   # hip within this many torso lengths of the ankles (vertical) = on the floor
VEL_WINDOW_S = 0.3
BASELINE_S = 30.0      # standing baseline = highest hip seen in this window
CONFIRM_S = 1.5        # fallen + still this long -> CONFIRMED
RECOVER_S = 1.0        # not fallen this long clears SUSPECTED or CONFIRMED -> NORMAL
STILL_V = 0.4
SUSPECT_TIMEOUT_S = 5.0
LOST_AFTER_DROP_S = 3.0  # pose lost within this many seconds of a drop counts as fallen
# ponytail: baseline is image-space over 30 s, so walking toward the camera inflates torso_ref and
# a slow lie-down on the floor still confirms. Add frame-height floor band / 3D hips if either bites.


class FallDetector:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.status = NORMAL
        self.tilt: float | None = None
        self.vel = 0.0
        self.aspect: float | None = None
        self.hip_drop: float | None = None
        self._hist: deque[tuple[float, float, float]] = deque()   # (t, hip_y px, torso px)
        self._last_drop_t = -1e9
        self._since = 0.0          # time the current status began
        self._still_since: float | None = None
        self._clear_since: float | None = None

    def _measure(self, res: PoseResult, t: float) -> tuple[bool, bool]:
        """Updates tilt/vel/aspect/hip_drop. Returns (horizontal, low)."""
        ls, rs, lh, rh = (res.get(n) for n in ("l_shoulder", "r_shoulder", "l_hip", "r_hip"))
        if not all((ls, rs, lh, rh)):
            return False, False
        sh = ((ls.x + rs.x) / 2, (ls.y + rs.y) / 2)
        hip = ((lh.x + rh.x) / 2, (lh.y + rh.y) / 2)
        dx, dy = sh[0] - hip[0], sh[1] - hip[1]
        torso = max(math.hypot(dx, dy), 1e-6)
        self.tilt = math.degrees(math.atan2(abs(dx), abs(dy)))
        self.aspect = res.bbox.w / max(res.bbox.h, 1e-6)

        self._hist.append((t, hip[1], torso))
        while len(self._hist) > 1 and t - self._hist[0][0] > BASELINE_S:
            self._hist.popleft()
        torso_ref = max(s[2] for s in self._hist)          # upright torso is the longest in image space
        base_hip = min(s[1] for s in self._hist)           # highest hip = standing
        self.hip_drop = (hip[1] - base_hip) / torso_ref

        t0, y0, _ = next(s for s in self._hist if t - s[0] <= VEL_WINDOW_S)
        self.vel = ((hip[1] - y0) / torso_ref) / (t - t0) if t > t0 else 0.0   # +ve = moving down the image
        if self.vel > DROP_V and self.hip_drop > HIP_DROP / 2:
            self._last_drop_t = t

        la, ra = res.get("l_ankle"), res.get("r_ankle")
        near_ankle = bool(la and ra) and ((la.y + ra.y) / 2 - hip[1]) < HIP_NEAR_ANKLE * torso_ref
        low = self.hip_drop > HIP_DROP or near_ankle
        horizontal = self.tilt > TILT_SUSPECT or self.aspect > ASPECT_WIDE
        return horizontal, low

    def update(self, res: PoseResult | None, t: float) -> dict:
        if res is not None:
            horizontal, low = self._measure(res, t)
            fallen = horizontal and low
            still = abs(self.vel) < STILL_V
            dropping = self._last_drop_t == t
        else:   # pose lost: fallen only if it vanished right after a drop
            fallen, still, dropping = (t - self._last_drop_t) < LOST_AFTER_DROP_S, True, False
            self.vel = 0.0

        prev = self.status
        if self.status == NORMAL:
            if dropping or fallen:
                self.status = SUSPECTED
        else:
            if fallen:
                self._clear_since = None
                if self.status == SUSPECTED and still:
                    self._still_since = self._still_since if self._still_since is not None else t
                    if t - self._still_since >= CONFIRM_S:
                        self.status = CONFIRMED
                elif self.status == SUSPECTED:
                    self._still_since = None
            else:   # a single clear frame mid-fall must not reset: require sustained not-fallen
                self._still_since = None
                self._clear_since = self._clear_since if self._clear_since is not None else t
                if t - self._clear_since >= RECOVER_S:
                    self.status = NORMAL
            if self.status == SUSPECTED and t - self._since > SUSPECT_TIMEOUT_S:
                self.status = NORMAL

        if self.status != prev:
            self._since = t
            self._still_since = self._clear_since = None
        return self.snapshot(t)

    def snapshot(self, t: float | None = None) -> dict:
        return {
            "status": self.status,
            "tilt_deg": None if self.tilt is None else round(self.tilt, 1),
            "drop_velocity": round(self.vel, 2),
            "hip_drop": None if self.hip_drop is None else round(self.hip_drop, 2),
            "aspect_ratio": None if self.aspect is None else round(self.aspect, 2),
            "seconds_in_state": None if t is None else round(t - self._since, 1),
        }
