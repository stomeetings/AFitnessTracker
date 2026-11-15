"""3-stage fall detection: NORMAL -> SUSPECTED -> CONFIRMED (-> NORMAL on recovery).

SUSPECTED as soon as a floor posture (or a fast drop) is seen; CONFIRMED only after DWELL_S continuously
down; back to NORMAL from either state as soon as the person is upright again for RECOVER_S.

Signals (2D image space, normalised by torso length):
- hip-centre downward velocity (torso lengths / s)
- hip drop below the recent standing baseline (torso lengths), or hips at ankle height
- torso tilt from vertical (degrees) and bounding-box aspect ratio (w / h)
- pose lost shortly after a rapid drop

A fall = hips LOW and body HORIZONTAL. Tilt or a wide box alone (bending, reaching, sitting) never
leaves NORMAL; a drop only counts when it is both fast and large.

Lying along the camera's line of sight (feet or head toward the camera) is neither horizontal nor wide in
the image: the skeleton stays upright, just shorter. Two extra cues cover it:
- COMPRESSED: shoulder-to-ankle height far below the standing height seen earlier at the same floor spot
- INVERTED: shoulders lower in the image than the ankles (head toward the camera)
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
FLAT_EXTENT = 0.8      # shoulders-to-legs vertical extent under this many torso lengths = body flat on the floor
                       # (all-fours / kneeling keeps the hips ~0.9 torso above the knees, so it stays above this)
ASPECT_LYING = 1.6     # box this much wider than tall = lying; used on its own only when the torso is not tracked
COMPRESSED = 0.55      # shoulder-to-ankle height under this fraction of the standing height at the same spot = lying
                       # toward the camera (measured ~0.4 on docs/sample/image_9379a9.jpg; sitting on a chair is ~0.65)
LEGS_SHORT = 0.27      # ... and hip-to-ankle height under this fraction of it (image: ~0.22; seated on a chair: ~0.31)
ANKLE_BIN_PX = 40     # "same spot" = ankles within one bin of this many pixels (image y)
INVERTED = 0.3         # shoulders below the ankles by this many torso lengths = lying head toward the camera
# ponytail: the standing reference is learned from this session only (cleared on camera switch), so a person
# already down when tracking starts, or lying where they never stood, is not caught by COMPRESSED. A one-time
# floor calibration or 3D pose would close that; a camera looking almost straight down can false-trigger it.
DWELL_S = 5.0          # down on the floor this long without getting up -> CONFIRMED (starts the alert countdown)
RECOVER_S = 0.5        # upright this long clears SUSPECTED or CONFIRMED -> NORMAL (debounce for stray frames)
VEL_WINDOW_S = 0.3
BASELINE_S = 30.0      # standing baseline = highest hip seen in this window
LOST_AFTER_DOWN_S = 6.0  # pose lost within this long of a drop / floor posture still counts as down (> DWELL_S)
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
        self.on_floor = False      # this frame looks like a person down (horizontal and low); shown in telemetry
        self.impact = False        # this fall episode began with a fast drop (vs. a slow descent to the floor)
        self._hist: deque[tuple[float, float, float]] = deque()   # (t, hip_y px, torso px)
        self._stand: dict[int, float] = {}   # ankle-y bin -> tallest shoulder-to-ankle height seen there (px)
        self._last_drop_t = self._last_down_t = -1e9   # last fast drop / last frame seen down on the floor
        self._since = 0.0          # time the current status began
        self._clear_since: float | None = None

    def _measure(self, res: PoseResult, t: float) -> bool:
        """Updates tilt/vel/aspect/hip_drop. Returns True when this frame looks like a person down on the floor."""
        ls, rs, lh, rh = (res.get(n) for n in ("l_shoulder", "r_shoulder", "l_hip", "r_hip"))
        self.aspect = res.bbox.w / max(res.bbox.h, 1e-6)
        if not all((ls, rs, lh, rh)):
            # Torso not tracked, which is common for a person lying down: judge by the box shape alone.
            # A head-and-shoulders close-up also gives a wide box, but there the shoulders span most of it.
            self.tilt, self.hip_drop, self.vel = None, None, 0.0
            closeup = bool(ls and rs) and abs(ls.x - rs.x) > 0.5 * res.bbox.w
            return self.aspect > ASPECT_LYING and not closeup
        sh = ((ls.x + rs.x) / 2, (ls.y + rs.y) / 2)
        hip = ((lh.x + rh.x) / 2, (lh.y + rh.y) / 2)
        dx, dy = sh[0] - hip[0], sh[1] - hip[1]
        torso = max(math.hypot(dx, dy), 1e-6)
        self.tilt = math.degrees(math.atan2(abs(dx), abs(dy)))

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
        # flat: whole body at one height. Needs no standing baseline (it expires after BASELINE_S on the floor,
        # or never exists if tracking starts with the person already down) and no visible ankles.
        legs = [k for k in (res.get(n) for n in ("l_knee", "r_knee", "l_ankle", "r_ankle")) if k]
        ys = [k.y for k in (ls, rs, lh, rh, *legs)]
        flat = bool(legs) and (max(ys) - min(ys)) < FLAT_EXTENT * torso_ref
        low = self.hip_drop > HIP_DROP or near_ankle or flat
        horizontal = self.tilt > TILT_SUSPECT or self.aspect > ASPECT_WIDE

        # lying along the camera's line of sight: the skeleton stays upright in the image, only shorter
        compressed = inverted = False
        if la and ra:
            ankle_y = (la.y + ra.y) / 2
            body = ankle_y - sh[1]                              # shoulder-to-ankle height; negative = upside down
            b = int(ankle_y // ANKLE_BIN_PX)
            expected = max(self._stand.get(i, 0.0) for i in (b - 1, b, b + 1))
            self._stand[b] = max(self._stand.get(b, 0.0), body)
            # both the whole body and the hip-to-ankle part must be short: a seated person is short overall but
            # their shins are still full length, however far they lean forward
            compressed = expected > 0 and body < COMPRESSED * expected and (ankle_y - hip[1]) < LEGS_SHORT * expected
            inverted = body < -INVERTED * torso_ref
        return (horizontal and low) or compressed or inverted

    def update(self, res: PoseResult | None, t: float) -> dict:
        if res is not None:
            fallen = self._measure(res, t)
            dropping = self._last_drop_t == t
            if fallen:
                self._last_down_t = t
        else:   # pose lost: still down only if it vanished shortly after a drop or while on the floor
            fallen = (t - max(self._last_drop_t, self._last_down_t)) < LOST_AFTER_DOWN_S
            dropping = False
            self.vel = 0.0

        self.on_floor = fallen
        prev = self.status
        if self.status == NORMAL:
            if dropping or fallen:
                self.status = SUSPECTED         # shown at once; the alarm path needs the dwell below
        elif fallen:
            self._clear_since = None
            if self.status == SUSPECTED and t - self._since >= DWELL_S:
                self.status = CONFIRMED         # down for the whole dwell time, moving or not
        else:   # upright again. Debounced: one stray upright frame must not clear a real fall
            self._clear_since = self._clear_since if self._clear_since is not None else t
            if t - self._clear_since >= RECOVER_S:
                self.status = NORMAL

        if self.status != prev:
            self._since = t
            self._clear_since = None
        # impact is decided per episode (not at confirmation time, which can be seconds after the drop)
        if self.status == NORMAL:
            self.impact = False
        elif dropping or (prev == NORMAL and (t - self._last_drop_t) < LOST_AFTER_DOWN_S):
            self.impact = True
        return self.snapshot(t)

    def snapshot(self, t: float | None = None) -> dict:
        return {
            "status": self.status,
            "impact": self.impact,
            "on_floor": self.on_floor,
            "tilt_deg": None if self.tilt is None else round(self.tilt, 1),
            "drop_velocity": round(self.vel, 2),
            "hip_drop": None if self.hip_drop is None else round(self.hip_drop, 2),
            "aspect_ratio": None if self.aspect is None else round(self.aspect, 2),
            "seconds_in_state": None if t is None else round(t - self._since, 1),
        }
