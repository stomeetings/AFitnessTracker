"""Fall escalation: detector CONFIRMED -> "Are you okay?" countdown -> ALERTED unless acknowledged or recovered.

IDLE -> COUNTDOWN -> ALERTED            (countdown ran out with no answer)
        COUNTDOWN -> IDLE               (person got up: detector back to NORMAL)
        COUNTDOWN -> ACKNOWLEDGED       ("I'm OK"), cleared to IDLE once the detector is NORMAL again
        ALERTED   -> IDLE               (person got up), or ACKNOWLEDGED ("I'm OK" while still down)

Pure and time-based like the rest of the pipeline. Only update() changes state, and it runs on the worker
thread; ack() just raises a flag, so it is safe to call from an API thread.
"""
from __future__ import annotations

from app.pipeline.fall_detector import CONFIRMED, NORMAL

IDLE, COUNTDOWN, ALERTED, ACKNOWLEDGED = "IDLE", "COUNTDOWN", "ALERTED", "ACKNOWLEDGED"

COUNTDOWN_S = 10.0          # time to press "I'm OK" or get up before the alarm; FallCheck.tsx draws its ring to this


class Escalation:
    def __init__(self):
        self.state = IDLE
        self.impact = False
        self.fired = False          # True for the single update() in which ALERTED is entered (notifier hook)
        self._deadline = 0.0
        self._ack = False
        self._test = False          # TEMP (Test Fall button): a simulated episode is running

    def ack(self) -> None:
        self._ack = True

    def simulate(self) -> None:
        """TEMP (Test Fall button): start the countdown on the next frame without a real fall."""
        self._test = True

    def update(self, fall: dict, t: float) -> dict:
        """fall: FallDetector snapshot (needs "status" and "impact")."""
        ack, self._ack = self._ack, False       # consumed every frame: a stale ack cannot pre-answer a later fall
        status = fall["status"]
        self.fired = False
        if self.state == IDLE:
            if status == CONFIRMED or self._test:
                self.state, self.impact = COUNTDOWN, self._test or bool(fall.get("impact"))
                self._deadline = t + COUNTDOWN_S
        elif self.state == COUNTDOWN:
            if ack:
                self.state = ACKNOWLEDGED
            elif status == NORMAL and not self._test:   # a simulated fall has no detector state to recover from
                self.state = IDLE
            elif t >= self._deadline:
                self.state, self.fired = ALERTED, True
        elif self.state == ALERTED:
            if ack or (status == NORMAL and not self._test):    # answered, or the person is back on their feet
                self.state = IDLE if status == NORMAL else ACKNOWLEDGED
        elif status == NORMAL:                  # ACKNOWLEDGED: stay quiet until this episode is over
            self.state = IDLE
        if self.state not in (COUNTDOWN, ALERTED):
            self._test = False                  # simulated episode over (acknowledged)
        left = max(0.0, self._deadline - t) if self.state == COUNTDOWN else 0.0
        return {"state": self.state, "seconds_left": round(left, 1), "impact": self.impact}
