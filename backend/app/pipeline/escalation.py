"""Fall escalation: detector CONFIRMED -> "Are you okay?" countdown -> ALERTED unless acknowledged or recovered.

IDLE -> COUNTDOWN -> ALERTED            (countdown ran out with no answer)
        COUNTDOWN -> ALERTED            (escalate(): voice "help" or the Trigger Emergency button)
        COUNTDOWN -> IDLE               (person got up: detector back to NORMAL)
        COUNTDOWN -> ACKNOWLEDGED       ("I'm OK"), cleared to IDLE once the detector is NORMAL again
        ALERTED   -> IDLE               (person got up), or ACKNOWLEDGED ("I'm OK" while still down)
        ACKNOWLEDGED -> COUNTDOWN       (still down RECHECK_S after the answer: ask again)

Pure and time-based like the rest of the pipeline. Only update() changes state, and it runs on the worker
thread; ack() and escalate() just raise flags, so they are safe to call from an API or speech-reader thread.
"""
from __future__ import annotations

from app.pipeline.fall_detector import CONFIRMED, NORMAL

IDLE, COUNTDOWN, ALERTED, ACKNOWLEDGED = "IDLE", "COUNTDOWN", "ALERTED", "ACKNOWLEDGED"

COUNTDOWN_S = 10.0          # time to press "I'm OK" or get up before the alarm; FallCheck.tsx draws its ring to this
RECHECK_S = 120.0           # after "I'm OK", if the person is still down this long, ask again


class Escalation:
    def __init__(self):
        self.state = IDLE
        self.impact = False
        self.fired = False          # True for the single update() in which ALERTED is entered (notifier hook)
        self.rechecks = 0           # how many times this episode has been re-asked after an acknowledgement
        self._deadline = 0.0
        self._acked_at = 0.0
        self._ack = False
        self._escalate = False
        self._test = False          # TEMP (Test Fall button): a simulated episode is running

    def ack(self) -> None:
        self._ack = True

    def escalate(self) -> None:
        """Skip the rest of the countdown (voice "help", Trigger Emergency button). Only acts during COUNTDOWN."""
        self._escalate = True

    def simulate(self) -> None:
        """TEMP (Test Fall button): start the countdown on the next frame without a real fall."""
        self._test = True

    def update(self, fall: dict, t: float) -> dict:
        """fall: FallDetector snapshot (needs "status" and "impact")."""
        ack, self._ack = self._ack, False       # consumed every frame: a stale ack cannot pre-answer a later fall
        esc, self._escalate = self._escalate, False     # same for escalate(): it never pre-arms a later fall
        status = fall["status"]
        self.fired = False
        if self.state == IDLE:
            if status == CONFIRMED or self._test:
                self.state, self.impact = COUNTDOWN, self._test or bool(fall.get("impact"))
                self._deadline = t + COUNTDOWN_S
        elif self.state == COUNTDOWN:
            if ack:
                self.state, self._acked_at = ACKNOWLEDGED, t
            elif esc:
                self.state, self.fired = ALERTED, True
            elif status == NORMAL and not self._test:   # a simulated fall has no detector state to recover from
                self.state = IDLE
            elif t >= self._deadline:
                self.state, self.fired = ALERTED, True
        elif self.state == ALERTED:
            if ack or (status == NORMAL and not self._test):    # answered, or the person is back on their feet
                if status == NORMAL:
                    self.state = IDLE
                else:
                    self.state, self._acked_at = ACKNOWLEDGED, t
        elif status == NORMAL:                  # ACKNOWLEDGED: stay quiet until this episode is over
            self.state = IDLE
        elif t - self._acked_at >= RECHECK_S:   # answered "OK" but still down: do not stay silent forever
            self.state, self._deadline = COUNTDOWN, t + COUNTDOWN_S
            self.rechecks += 1
        if self.state not in (COUNTDOWN, ALERTED):
            self._test = False                  # simulated episode over (acknowledged)
        if self.state == IDLE:
            self.rechecks = 0
        left = max(0.0, self._deadline - t) if self.state == COUNTDOWN else 0.0
        return {"state": self.state, "seconds_left": round(left, 1), "impact": self.impact, "rechecks": self.rechecks}
