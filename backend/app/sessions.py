"""Per-patient session state, and the registry that owns it.

A Session is everything that belongs to ONE patient and nothing that belongs to a video source or a pose engine:
the analytics pipeline (exercise, reps, mobility, fall detection, escalation), the voice check-in listener, and the
commands that change them. Frame sources (the desktop camera today, a browser WebSocket later) and the engine feed a
session through process(); they never reach into its pipeline. Two sessions share no state at all.

Threading contract (the one the old Worker already followed):
- API threads only call the command methods (request_exercise, request_reset, request_baseline_reset, ack, escalate,
  simulate_fall, listen). Those set flags or call flag-raisers; they never touch the pipeline.
- ONE feeder thread calls apply_pending() and process(), and is the only thread that mutates the pipeline. Since
  Phase 2 that is the InferenceService thread, for every session (desktop camera and remote sessions alike).
"""
from __future__ import annotations

import threading
import time
import uuid

from app.engines import BBox, Keypoint, PoseResult
from app.pipeline.core import Pipeline
from app.pipeline.rep_counter import EXERCISES
from app.voice.listener import VoiceListener

EXERCISE_NAMES = list(EXERCISES)
REFERENCE_LONG_SIDE = 640                       # fall_detector.py's pixel constants (ANKLE_BIN_PX) are tuned for the 640 x 480 desktop webcam (a 640 px LONG side)
ACTIVE_ALERT_STATES = ("COUNTDOWN", "ALERTED")      # while the fall dialog is open


def to_reference(res: PoseResult | None, frame_width: float, frame_height: float | None = None,
                 reference: int = REFERENCE_LONG_SIDE) -> PoseResult | None:
    """The pose in the coordinates of a frame whose LONG side is `reference` px, so every source is judged at the scale
    of the 640 x 480 desktop webcam. The long side (not the width) is used so a phone held in portrait (480 x 640) and
    in landscape (640 x 480) get the same scale: rotating a phone does not change how large a person is in pixels, and
    scaling by width alone would inflate the portrait frame by a third. Without a height, the width is the long side.
    Returns `res` itself when nothing needs scaling. The input is not modified; `world` coordinates (metres) are not
    image pixels and are left alone."""
    long_side = max(frame_width, frame_height or 0)
    if res is None or long_side <= 0 or long_side == reference:
        return res
    k = reference / long_side
    keypoints = {name: Keypoint(p.x * k, p.y * k, p.conf, p.world) for name, p in res.keypoints.items()}
    b = res.bbox
    return PoseResult(keypoints, BBox(b.x * k, b.y * k, b.w * k, b.h * k, b.conf), res.timestamp_ms)


class Session:
    def __init__(self, session_id: str, exercise: str | None = None, *, local_voice: bool = False, standby: bool = False):
        """local_voice: give this session the server-side microphone listener (the desktop kiosk). Sessions that
        record voice in the browser leave it off, so two sessions never compete for one microphone.
        standby: start with rep counting paused until request_start() (a phone user is still getting into position)."""
        exercise = exercise or EXERCISE_NAMES[0]
        if exercise not in EXERCISES:
            raise ValueError(f"exercise must be one of {EXERCISE_NAMES}")
        self.id = session_id
        self.created_at = time.time()
        self.closed = False
        self.pipeline = Pipeline(exercise)
        self.pipeline.counting = not standby
        self.voice: VoiceListener | None = None
        if local_voice:
            self.voice = VoiceListener(on_ok=self.pipeline.escalation.ack, on_help=self.pipeline.escalation.escalate)
        self.pose_visible = False
        self.latency_ms = 0.0               # inference time of the last frame, as reported by the source
        self.pending_exercise: str | None = None
        self.pending_reset = False
        self.pending_baseline_reset = False
        self.pending_counting: bool | None = None

    # ---- lifecycle ----------------------------------------------------------------------------------------------

    def open(self) -> None:
        """Call once when the session goes live: reports early whether voice can work (is the Groq key there?)."""
        if self.voice is not None:
            self.voice.init()

    def close(self) -> None:
        """Releases what the session holds (the microphone). Safe to call twice."""
        if self.closed:
            return
        self.closed = True
        if self.voice is not None:
            self.voice.quit()

    # ---- commands: safe from any thread (they set flags only) ---------------------------------------------------

    def request_exercise(self, name: str) -> None:
        if name not in EXERCISES:
            raise ValueError(f"exercise must be one of {EXERCISE_NAMES}")
        self.pending_exercise = name

    def request_reset(self) -> None:
        self.pending_reset = True

    def request_start(self) -> None:
        """"Let's go" / Start: counters back to 0 and rep counting on."""
        self.pending_reset = True
        self.pending_counting = True

    def request_stop(self) -> None:
        """Pause rep counting and go back to standby (the counts stay on screen until the next start)."""
        self.pending_counting = False

    def request_baseline_reset(self) -> None:
        """The camera moved or was switched: relearn the standing baseline (applied by the feeder thread)."""
        self.pending_baseline_reset = True

    def ack(self) -> None:
        """"I'm OK": answers the fall countdown or silences a raised alert."""
        self.pipeline.escalation.ack()

    def escalate(self) -> None:
        """"Trigger Emergency": skips the rest of the countdown."""
        self.pipeline.escalation.escalate()

    def simulate_fall(self) -> None:
        """TEMP (Test Fall button): start the countdown without a real fall."""
        self.pipeline.escalation.simulate()

    def listen(self) -> dict:
        """The dialog's spoken prompt has finished: open the microphone, but only during a fall check."""
        if self.voice is None:
            return {"status": "unavailable", "error": "voice check-in is not enabled for this session"}
        state = self.pipeline.last.get("fall", {}).get("alert", {}).get("state")
        if state in ACTIVE_ALERT_STATES:
            self.voice.start()
        return {"status": self.voice.status, "error": self.voice.error}

    # ---- feeder thread only -------------------------------------------------------------------------------------

    def apply_pending(self) -> None:
        """Applies what the API asked for. Call between frames."""
        if self.pending_exercise and self.pending_exercise != self.pipeline.exercise:
            self.pipeline.set_exercise(self.pending_exercise)
        self.pending_exercise = None
        if self.pending_reset:
            self.pipeline.reset()
            self.pending_reset = False
        if self.pending_counting is not None:                # after the reset, so "start" = zero counts AND counting
            self.pipeline.counting = self.pending_counting
            self.pending_counting = None
        if self.pending_baseline_reset:
            self.pipeline.fall.reset()
            self.pending_baseline_reset = False

    def process(self, res: PoseResult | None, t: float, infer_ms: float = 0.0, frame_width: float | None = None,
                frame_height: float | None = None) -> dict:
        """One frame's pose (or None: nobody found) at time t seconds. With the frame's size, the pose is first
        rescaled to the reference scale (see to_reference). Returns the pipeline's latest result."""
        if frame_width or frame_height:
            res = to_reference(res, frame_width or 0, frame_height)
        self.latency_ms = infer_ms
        self.pose_visible = res is not None
        last = self.pipeline.process(res, t)
        if self.voice is not None and self.voice.active and last["fall"]["alert"]["state"] not in ACTIVE_ALERT_STATES:
            self.voice.stop()               # the fall dialog is gone: close the microphone (a bool check per frame)
        return last

    # ---- read by any thread -------------------------------------------------------------------------------------

    def telemetry(self) -> dict:
        """What this patient's UI shows. Source-specific fields (camera, fps, engine) are added by the source."""
        out = {"latency_ms": round(self.latency_ms, 1), "pose_visible": self.pose_visible,
               "exercise": self.pipeline.exercise, "counting": self.pipeline.counting, **self.pipeline.last}
        fall = out.get("fall")
        if fall is not None and self.voice is not None:
            out["fall"] = {**fall, "voice": self.voice.snapshot()}      # a copy: pipeline.last belongs to the feeder
        return out


class SessionLimitError(RuntimeError):
    """The manager already holds its maximum number of sessions."""


class SessionManager:
    """Creates, stores and closes sessions. Thread-safe."""

    def __init__(self, max_sessions: int | None = None):
        self.max_sessions = max_sessions
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self, session_id: str | None = None, exercise: str | None = None, *, local_voice: bool = False,
               standby: bool = False) -> Session:
        """A new session under `session_id` (a random one if omitted). Raises ValueError for a duplicate id or an
        unknown exercise, SessionLimitError when full; nothing is registered in those cases."""
        with self._lock:
            sid = session_id or uuid.uuid4().hex
            if sid in self._sessions:
                raise ValueError(f"session {sid!r} already exists")
            if self.max_sessions is not None and len(self._sessions) >= self.max_sessions:
                raise SessionLimitError(f"at most {self.max_sessions} sessions at a time")
            session = Session(sid, exercise, local_voice=local_voice, standby=standby)
            self._sessions[sid] = session
            return session

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            return self._sessions.get(session_id)

    def close(self, session_id: str) -> bool:
        """Removes and closes the session. False if there was none."""
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        session.close()
        return True

    def close_all(self) -> None:
        with self._lock:
            sessions, self._sessions = list(self._sessions.values()), {}
        for session in sessions:
            session.close()

    def ids(self) -> list[str]:
        with self._lock:
            return list(self._sessions)

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)

    def __contains__(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._sessions
