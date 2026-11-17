"""Multi-session pose inference: one thread, one engine, many patients.

Frame sources (the desktop camera Worker, the remote /ws/session gateway) hand frames to submit() / infer() together
with the Session they belong to. The service's single inference thread then, every cycle:
  1. applies an engine change, forgets closed sessions, and applies every registered session's pending commands. It is
     the ONLY feeder thread of every session, which is what Session's threading contract asks for;
  2. takes the newest frame of each session that has one waiting (an older frame of the same session is dropped: live
     pose never queues, and one fast sender cannot crowd the others out);
  3. runs the poses: ONE batch call for a stateless engine (YOLO), or one frame at a time on the session's own engine
     instance for a stateful one (MediaPipe tracks between frames and must never see two patients);
  4. feeds each pose to its own session, rescaled to the reference scale by the frame's size (see
     sessions.to_reference), and resolves the frame's future.
A failure for one session is contained: its frame gets an error result and its pipeline is NOT fed (an engine hiccup is
not "nobody in view"), while the other sessions in the same cycle are processed normally.
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass

import numpy as np

from app.engines import PoseEngine, PoseResult
from app.sessions import Session

log = logging.getLogger("inference")


@dataclass
class FrameResult:
    seq: int
    pose: PoseResult | None = None      # in the frame's own pixel coordinates (what a source draws); None = nobody found
    infer_ms: float = 0.0               # engine time of the batch this frame was part of
    pipeline_ms: float = 0.0            # the session's analytics for this frame
    batch_size: int = 0
    dropped: bool = False               # replaced by a newer frame of the same session, or the service stopped
    error: str | None = None            # the engine or the session failed: the session was NOT fed this frame


@dataclass
class _Slot:
    session: Session
    frame: np.ndarray | None            # None for a precomputed pose (the browser ran the model itself)
    t: float
    seq: int
    future: Future
    precomputed: bool = False           # `given` is the pose: no engine is involved
    given: PoseResult | None = None     # None here means "nobody in view", as a result of detection would
    size: tuple[int, int] | None = None  # (width, height) of the frame a precomputed pose belongs to


class InferenceService:
    def __init__(self, create_engine: Callable[[str], PoseEngine], engine_name: str = "yolo",
                 fallback_engine: str | None = "mediapipe", max_batch: int = 8):
        """create_engine(name) must return a LOADED engine, or raise."""
        self._create = create_engine
        self._engine_name = engine_name
        self._fallback = fallback_engine
        self.max_batch = max_batch
        self.ready = threading.Event()              # set once the first engine is loaded, or every candidate failed
        self.stats = {"batches": 0, "frames": 0, "dropped": 0}
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._slots: dict[str, _Slot] = {}          # session id -> its newest waiting frame; order = who waited longest
        self._sessions: dict[str, Session] = {}     # every session that has submitted a frame (swept for commands)
        self._pending_engine: str | None = None
        self._to_forget: set[str] = set()
        self._loaded = False
        self._shared: PoseEngine | None = None      # the one instance of a stateless engine
        self._spare: PoseEngine | None = None       # a freshly loaded stateful engine, waiting for its first session
        self._per_session: dict[str, PoseEngine] = {}
        self._seq = 0
        self._thread: threading.Thread | None = None

    # ---- public API: any thread -----------------------------------------------------------------------------------

    @property
    def engine_name(self) -> str:
        return self._engine_name

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="inference")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=10)

    def wait_ready(self, timeout: float | None = None) -> bool:
        return self.ready.wait(timeout)

    def request_engine(self, name: str) -> None:
        """Switch engines between batches. If the new one fails to load, the current one stays."""
        with self._lock:
            self._pending_engine = name
        self._wake.set()

    def forget(self, session_id: str) -> None:
        """A session has ended: drop its waiting frame and release its engine instance."""
        with self._lock:
            self._to_forget.add(session_id)
        self._wake.set()

    def submit(self, session: Session, frame: np.ndarray, t: float) -> Future:
        """Queue a frame (t = its capture time in seconds). The future resolves to a FrameResult once the frame has been
        inferred and the session fed. Never blocks. A newer frame of the same session replaces a waiting one."""
        return self._enqueue(session, lambda seq, future: _Slot(session, frame, t, seq, future))

    def submit_pose(self, session: Session, pose: PoseResult | None, t: float, width: int, height: int) -> Future:
        """Queue a pose that was ALREADY computed (the browser ran the model on the phone): the session's analytics (reps, falls,
        escalation) run on it exactly as on a server-inferred one, with no engine involved, so it also works when no engine is
        loaded. `pose` is in pixels of a width x height frame; None = nobody in view. Same latest-wins rule as submit()."""
        return self._enqueue(session, lambda seq, future: _Slot(session, None, t, seq, future, precomputed=True, given=pose,
                                                                size=(width, height)))

    def _enqueue(self, session: Session, make_slot) -> Future:
        future: Future = Future()
        with self._lock:
            self._seq += 1
            old = self._slots.get(session.id)
            self._slots[session.id] = make_slot(self._seq, future)      # keeps its place in the order
            self._sessions[session.id] = session
        if old is not None:
            self.stats["dropped"] += 1
            old.future.set_result(FrameResult(old.seq, dropped=True))
        self._wake.set()
        return future

    def infer(self, session: Session, frame: np.ndarray, t: float, timeout: float = 5.0) -> FrameResult | None:
        """submit() and wait. None on timeout (the session is not fed a made-up "nobody found" frame; the frame may
        still be processed later)."""
        try:
            return self.submit(session, frame, t).result(timeout)
        except FutureTimeout:
            return None

    # ---- the inference thread -------------------------------------------------------------------------------------

    def _run(self) -> None:
        try:
            for name in dict.fromkeys([self._engine_name, self._fallback]):
                if not name:
                    continue
                try:
                    self._activate(name)
                    break
                except Exception:  # noqa: BLE001  no GPU / missing weights must not stop the service
                    log.exception("engine %s failed to load%s", name, ", trying the fallback" if name == self._engine_name else "")
        finally:
            self.ready.set()                        # sources must never wait forever, even if nothing loaded
        while not self._stop.is_set():
            self._wake.wait(0.1)                    # the timeout also sweeps pending commands when no frames arrive
            self._wake.clear()                      # slots are inserted before wake is set, so no frame is lost
            try:
                self._housekeeping()
                batch = self._take()
                if batch:
                    self._process(batch)
            except Exception:  # noqa: BLE001  this one thread serves every session: it must outlive any single failure
                log.exception("inference loop error (continuing)")
        self._shutdown()

    def _activate(self, name: str) -> None:
        engine = self._create(name)                 # may raise: the current engine stays untouched in that case
        self._close_engines()
        self._engine_name, self._loaded = name, True
        if getattr(engine, "stateless", True):
            self._shared, self._spare = engine, None
        else:
            self._shared, self._spare = None, engine

    def _switch_engine(self, name: str) -> None:
        try:
            self._activate(name)
            log.info("engine switched to %s", name)
        except Exception:  # noqa: BLE001
            log.exception("could not switch to engine %s; staying on %s", name, self._engine_name)

    def _housekeeping(self) -> None:
        with self._lock:
            name, self._pending_engine = self._pending_engine, None
            forgotten, self._to_forget = self._to_forget, set()
            dropped = [self._slots.pop(sid) for sid in forgotten if sid in self._slots]
            for sid in forgotten:
                self._sessions.pop(sid, None)
            sessions = list(self._sessions.values())
        for slot in dropped:
            slot.future.set_result(FrameResult(slot.seq, dropped=True))
        for sid in forgotten:
            engine = self._per_session.pop(sid, None)
            if engine is not None:
                self._close(engine)
        if name and name != self._engine_name:
            self._switch_engine(name)
        for session in sessions:
            try:
                session.apply_pending()
            except Exception:  # noqa: BLE001
                log.exception("session %s: applying its pending commands failed", session.id)

    def _take(self) -> list[_Slot]:
        with self._lock:
            batch = [self._slots.pop(sid) for sid in list(self._slots)[: self.max_batch]]
            more = bool(self._slots)
        if more:
            self._wake.set()                        # more than one batch is waiting: do not sleep
        return batch

    def _engine_for(self, session: Session) -> PoseEngine:
        if not self._loaded:
            raise RuntimeError("no pose engine is loaded")
        if self._shared is not None:
            return self._shared
        engine = self._per_session.get(session.id)
        if engine is None:
            engine, self._spare = self._spare or self._create(self._engine_name), None
            self._per_session[session.id] = engine
        return engine

    def _infer(self, batch: list[_Slot]) -> tuple[list[PoseResult | None], list[str | None]]:
        poses: list[PoseResult | None] = [None] * len(batch)
        errors: list[str | None] = [None] * len(batch)
        groups: dict[int, tuple[PoseEngine, list[int]]] = {}
        for i, slot in enumerate(batch):
            if slot.precomputed:                    # the pose came with the message: nothing to infer
                poses[i] = slot.given
                continue
            try:
                engine = self._engine_for(slot.session)
            except Exception as e:  # noqa: BLE001
                log.warning("session %s: no engine (%s)", slot.session.id, e)
                errors[i] = f"engine unavailable: {e}"
                continue
            groups.setdefault(id(engine), (engine, []))[1].append(i)
        for engine, indices in groups.values():
            frames = [batch[i].frame for i in indices]
            stamps = [int(batch[i].t * 1000) for i in indices]
            try:
                for i, pose in zip(indices, engine.infer_batch(frames, stamps)):
                    poses[i] = pose
            except Exception:  # noqa: BLE001
                log.exception("batch of %d failed; retrying the frames one by one", len(indices))
                for i in indices:                   # one bad frame must not take the others down with it
                    try:
                        poses[i] = engine.infer(batch[i].frame, int(batch[i].t * 1000))
                    except Exception as e:  # noqa: BLE001
                        errors[i] = f"inference failed: {type(e).__name__}"
        return poses, errors

    def _process(self, batch: list[_Slot]) -> None:
        started = time.perf_counter()
        poses, errors = self._infer(batch)
        infer_ms = (time.perf_counter() - started) * 1000
        self.stats["batches"] += 1
        self.stats["frames"] += len(batch)
        for slot, pose, error in zip(batch, poses, errors):
            own_ms = 0.0 if slot.precomputed else infer_ms      # no server time was spent on a pose the phone computed
            result = FrameResult(slot.seq, pose, own_ms, batch_size=len(batch), error=error)
            if error is None:
                began = time.perf_counter()
                width, height = slot.size if slot.size is not None else (slot.frame.shape[1], slot.frame.shape[0])
                try:
                    slot.session.apply_pending()
                    slot.session.process(pose, slot.t, infer_ms=own_ms, frame_width=width, frame_height=height)
                except Exception as e:  # noqa: BLE001  one session's failure stays with that session
                    log.exception("session %s: processing a frame failed", slot.session.id)
                    result.error = f"session error: {type(e).__name__}"
                result.pipeline_ms = (time.perf_counter() - began) * 1000
            slot.future.set_result(result)

    def _close(self, engine: PoseEngine) -> None:
        try:
            engine.close()
        except Exception:  # noqa: BLE001
            log.exception("closing an engine failed")

    def _close_engines(self) -> None:
        for engine in [self._shared, self._spare, *self._per_session.values()]:
            if engine is not None:
                self._close(engine)
        self._shared = self._spare = None
        self._per_session = {}

    def _shutdown(self) -> None:
        with self._lock:
            waiting, self._slots = list(self._slots.values()), {}
        for slot in waiting:                        # wake anyone blocked on a frame that will never be inferred
            slot.future.set_result(FrameResult(slot.seq, dropped=True))
        self._close_engines()
        self._loaded = False
