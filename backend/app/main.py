"""FastAPI app. Frame sources feed patient Sessions through one InferenceService; MJPEG stream, WS telemetry.

- app/sessions.py   per-patient state: analytics, fall escalation, voice check-in
- app/inference.py  ONE thread, ONE engine, many sessions: batches waiting frames, feeds each pose to its own session
- Worker (below)    the desktop frame source: camera + MJPEG stream; submits frames for the "local" session
- app/remote.py     the other frame source: mobile browsers over /ws/session (off unless REMOTE_SESSIONS=1)

Run from backend/:  .venv\\Scripts\\uvicorn app.main:app --port 8000
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from contextlib import asynccontextmanager

os.environ.setdefault("OPENCV_VIDEOIO_DEBUG", "1")   # must precede `import cv2`: backend open/fail reasons on stderr
os.environ.setdefault("OPENCV_LOG_LEVEL", "INFO")
import cv2  # noqa: E402
from collections import deque
from pathlib import Path

from starlette.concurrency import run_in_threadpool
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.engines import make_engine
from app.inference import InferenceService
from app.pipeline.annotate import draw
from app.pipeline.rep_counter import EXERCISES
from app.remote import create_router
from app.security import TunnelGuard, is_tunneled
from app.sessions import EXERCISE_NAMES, Session, SessionManager
from app.storage import Storage
from app.sync_api import create_sync_router
from app.voice import settings
from app.voicecmd import MAX_AUDIO_BYTES, process_audio

ENGINES = {
    "mediapipe": ("mediapipe", {"variant": "full"}),
    "mediapipe_lite": ("mediapipe", {"variant": "lite"}),      # faster, less accurate: pick it on the badge to compare live
    "yolo": ("yolo", {"device": "cuda:0"}),
}
LOCAL_SESSION_ID = "local"     # the one patient the desktop camera belongs to
CAMERA_INDEX = 0
CAMERA_MAX = 4                 # probe indices 0..4
CAMERA_BACKENDS = [("dshow", cv2.CAP_DSHOW), ("msmf", cv2.CAP_MSMF)]
CAMERA_WARMUP_S = 2.0          # virtual cams (DroidCam, OBS) can take >1 s to deliver the first frame
STREAM_FPS_CAP = 25
WS_HZ = 20

log = logging.getLogger("camera")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(name)s: %(message)s")


def open_camera(index: int, warmup_s: float = CAMERA_WARMUP_S):
    """Tries DSHOW then MSMF, waits up to warmup_s for a real frame. Returns (cap, info) or (None, reason)."""
    reasons = []
    for name, be in CAMERA_BACKENDS:
        cap = cv2.VideoCapture(index, be)
        if not cap.isOpened():
            reasons.append(f"{name}: open failed")
            log.warning("camera %d: %s backend refused to open", index, name)
            cap.release()
            continue
        deadline = time.perf_counter() + warmup_s
        while time.perf_counter() < deadline:
            ok, frame = cap.read()
            if ok and frame is not None and frame.size:
                info = {"backend": name, "width": frame.shape[1], "height": frame.shape[0]}
                log.info("camera %d: opened via %s at %dx%d", index, name, frame.shape[1], frame.shape[0])
                return cap, info
            time.sleep(0.05)
        reasons.append(f"{name}: opened but no frame in {warmup_s:.0f}s")
        log.warning("camera %d: %s opened but delivered no frame within %.1fs (device busy or virtual cam not streaming?)",
                    index, name, warmup_s)
        cap.release()
    return None, "; ".join(reasons)


def list_video_devices() -> list[str]:
    """DirectShow capture device names; list position == OpenCV CAP_DSHOW index. [] if enumeration fails."""
    # ponytail: assumes MSMF enumerates in the same order as DirectShow (true on typical 1-2 camera setups);
    # if a label ever disagrees with the picture, enumerate MSMF separately for cameras opened via msmf.
    try:
        import comtypes
        from pygrabber.dshow_graph import FilterGraph
        try:
            comtypes.CoInitialize()     # called from a threadpool thread; COM init is per-thread
        except OSError:
            pass                        # already initialised in another mode
        return FilterGraph().get_input_devices()
    except Exception as e:  # noqa: BLE001
        log.warning("device name listing failed: %s", e)
        return []


def camera_label(name: str) -> str:
    n = name.lower()
    if "droidcam" in n:
        return "Mobile Cam"
    if "integrated" in n:
        return "Laptop Webcam"
    return name


def default_camera_index() -> int:
    """Index of the built-in webcam according to the DirectShow device list, else CAMERA_INDEX."""
    names = list_video_devices()[:CAMERA_MAX + 1]
    return next((i for i, n in enumerate(names) if camera_label(n) == "Laptop Webcam"), CAMERA_INDEX)


def _engine_kwargs(name: str) -> dict:
    """The engine's fixed settings plus the tuning knobs from the environment / backend/.env (read at every load, so a
    change applies the next time the engine is switched to):
      YOLO_IMGSZ  network input size, a multiple of 32 (default 640). Phone frames are at most 480 px on the long side, so
                  480 does about 45% less work with little loss; desktop frames are 640 wide.
      YOLO_HALF   0 / false / off turns FP16 off (it is on by default and only ever used on a CUDA device)."""
    kind, base = ENGINES[name]
    kwargs = dict(base)
    if kind == "yolo":
        size = int(settings.get_float("YOLO_IMGSZ", 640))
        kwargs["imgsz"] = size if 160 <= size <= 1280 and size % 32 == 0 else 640
        kwargs["half"] = (settings.get("YOLO_HALF") or "1").lower() not in ("0", "false", "no", "off")
    return kwargs


def _create_engine(name: str):
    """A loaded pose engine for the inference service (YOLO's load includes a dummy-frame warm-up inference)."""
    kind = ENGINES[name][0]
    engine = make_engine(kind, **_engine_kwargs(name))
    engine.load()
    return engine


class Worker(threading.Thread):
    """The desktop frame source: owns the camera and the MJPEG stream (the latest JPEG only, never a queue). Each frame
    goes to the inference service for one Session; the pose comes back for drawing. Everything per-patient lives in the
    session, and the pose engine belongs to the service."""

    def __init__(self, session: Session, service: InferenceService):
        super().__init__(daemon=True)
        self.session = session
        self.service = service
        self.camera_index = CAMERA_INDEX
        self.camera_info: dict | None = None
        self.pending_camera: int | None = None
        self.camera_error: str | None = None
        self.jpeg: bytes | None = None
        self.seq = 0
        self.fps = 0.0
        # Smoothed per-stage cost in ms. capture_ms is mostly time spent WAITING for the camera's next frame
        # (the camera sets the pace), not CPU work, so do not try to optimise it. infer_ms is the engine time of the
        # batch this frame was in, pipeline_ms the session's analytics for it; both are measured inside the service.
        self.timings = {"capture_ms": 0.0, "infer_ms": 0.0, "pipeline_ms": 0.0, "draw_ms": 0.0, "encode_ms": 0.0}
        self.privacy = False            # requested via POST /api/config; plain bool, read once per frame
        self.jpeg_private = False       # the published jpeg was encoded with privacy on (guards the stream)
        self.error: str | None = None
        self.new_frame = threading.Condition()
        self.stop_flag = threading.Event()

    def _apply_camera(self, cap):
        idx, self.pending_camera = self.pending_camera, None
        if idx is None or (idx == self.camera_index and cap is not None):
            return cap
        # requested index first, then its neighbours (DroidCam/OBS often shift indices), never the current one
        candidates = [idx] + [j for j in sorted(range(CAMERA_MAX + 1), key=lambda j: abs(j - idx)) if j != idx]
        candidates = [j for j in candidates if not (j == self.camera_index and cap is not None)]
        failures = []
        for j in candidates:
            new, info = open_camera(j)
            if new is None:
                failures.append(f"camera {j}: {info}")
                continue
            if cap is not None:
                cap.release()
            self.camera_index, self.camera_info, self.error = j, info, None
            self.camera_error = None if j == idx else f"camera {idx} not available, using camera {j} instead"
            self.session.request_baseline_reset()   # perspective / distance changed: relearn the standing baseline
            return new
        self.camera_error = f"no camera found near index {idx}; kept camera {self.camera_index}"   # old stream stays alive
        log.error("camera switch failed: %s", " | ".join(failures))
        return cap

    def run(self):
        cap, info = open_camera(self.camera_index)
        if cap is None:
            self.error = f"cannot open camera {self.camera_index} ({info})"   # keep running so the user can switch
        else:
            self.camera_info = info
        self.service.wait_ready()           # the engine is loaded (or every candidate failed): same wait as before
        t0 = prev = time.perf_counter()
        while not self.stop_flag.is_set():
            cap = self._apply_camera(cap)
            tc = time.perf_counter()
            ok, frame = cap.read() if cap is not None else (False, None)
            if not ok:
                time.sleep(0.01)
                continue
            t1 = time.perf_counter()
            # one hop through the service: pose, analytics, fall escalation and the voice follow-up all happen there
            result = self.service.infer(self.session, frame, t1 - t0)
            if result is None or result.dropped or result.error:
                continue                    # busy, stopping or failed: nothing to draw or publish for this frame
            res = result.pose
            t3 = time.perf_counter()
            private = self.privacy         # read once: the blanking and the published flag must agree
            if private:
                frame[:] = 0                # in place, after inference and analytics: raw video is never encoded
            if res:
                draw(frame, res)
            t4 = time.perf_counter()
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            now = time.perf_counter()
            self.fps = 0.9 * self.fps + 0.1 / max(now - prev, 1e-3)
            prev = now
            for key, ms in (("capture_ms", (t1 - tc) * 1000), ("infer_ms", result.infer_ms),
                            ("pipeline_ms", result.pipeline_ms), ("draw_ms", (t4 - t3) * 1000),
                            ("encode_ms", (now - t4) * 1000)):      # blanking is part of draw_ms
                self.timings[key] = 0.9 * self.timings[key] + 0.1 * ms
            with self.new_frame:
                self.jpeg = buf.tobytes()
                self.jpeg_private = private
                self.seq += 1
                self.new_frame.notify_all()
        if cap is not None:
            cap.release()

    def telemetry(self) -> dict:
        """The source's own fields (camera, engine, speed, stream) plus the patient's, from the session."""
        engine = self.service.engine_name
        return {
            "camera_index": self.camera_index,
            "camera_backend": (self.camera_info or {}).get("backend"),
            "camera_error": self.camera_error,
            "engine": engine,
            "device": ENGINES[engine][1].get("device", "cpu"),
            "fps": round(self.fps, 1),
            "timings": {k: round(v, 2) for k, v in self.timings.items()},
            # True only once a blanked frame has actually been published, so the UI can trust it as confirmation
            "privacy": self.privacy and self.jpeg_private,
            "error": self.error,
            **self.session.telemetry(),
        }


def _remote_enabled() -> bool:
    return (settings.get("REMOTE_SESSIONS") or "").lower() in ("1", "true", "yes", "on")


def _remote_limit() -> int:
    return int(settings.get_float("REMOTE_MAX_SESSIONS", 4))


def _remote_token() -> str:
    return settings.get("REMOTE_ACCESS_TOKEN", "") or ""


def _remote_origins() -> list[str]:
    return [o.strip() for o in (settings.get("REMOTE_ALLOWED_ORIGINS") or "").split(",") if o.strip()]


DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"     # `npm run build` output, served on :8000 when present


sessions = SessionManager()
local_session = sessions.create(LOCAL_SESSION_ID, local_voice=True)
service = InferenceService(_create_engine, engine_name="yolo", fallback_engine="mediapipe")
worker = Worker(local_session, service)


@asynccontextmanager
async def lifespan(app: FastAPI):
    worker.camera_index = default_camera_index()    # enumerated here, not on the worker thread that owns the capture
    if _remote_enabled() and not _remote_token():
        log.warning("REMOTE_SESSIONS is on but REMOTE_ACCESS_TOKEN is empty: anyone who can reach this server can start a "
                    "remote session. Set a token before exposing it through a tunnel.")
    local_session.open()                   # checks for the Groq key; the microphone opens only during a fall check
    service.start()                        # loads the engine; sessions' frames are inferred (and fed) on its thread
    worker.start()
    yield
    worker.stop_flag.set()
    service.stop()
    sessions.close_all()
    if _storage is not None:
        _storage.close()


app = FastAPI(title="AI Rehab Tracker", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(TunnelGuard)      # requests through a tunnel/proxy cannot reach the desktop routes (app/security.py)
app.include_router(create_router(sessions, service, enabled=_remote_enabled, max_remote=_remote_limit,
                                 token=_remote_token, allowed_origins=_remote_origins, engine_names=lambda: list(ENGINES)))
_storage: Storage | None = None


def get_storage() -> Storage:
    """data/rehab.db, opened on first use (importing the app must not create files)."""
    global _storage
    if _storage is None:
        _storage = Storage()
    return _storage


app.include_router(create_sync_router(get_storage, _remote_token))     # the phones' upload and the history (app/sync_api.py)


class Config(BaseModel):
    engine: str | None = None
    exercise: str | None = None
    privacy: bool | None = None


@app.get("/api/health")
def health(request: Request):
    public = {"remote_sessions": _remote_enabled(), "remote_auth": bool(_remote_token())}
    if is_tunneled(request.scope["headers"]):       # a tunnel visitor learns only what the connect screen needs
        return {"status": "ok" if worker.error is None else "error", **public}
    return {"status": "ok" if worker.error is None else "error", **public, **worker.telemetry()}


@app.get("/api/exercises")
def exercises():
    return EXERCISE_NAMES


@app.get("/api/exercise_info")
def exercise_info():
    """Rest/peak target angles and tracked sides (L/R, or B = bilateral) per exercise, for the frontend guide."""
    return {name: {"rest": cfg["rest"], "peak": cfg["peak"], "sides": list(cfg["joints"])} for name, cfg in EXERCISES.items()}


@app.get("/api/telemetry")
def telemetry():
    return worker.telemetry()


@app.post("/api/config")
def config(cfg: Config):
    if cfg.engine is not None:
        if cfg.engine not in ENGINES:
            raise HTTPException(400, f"engine must be one of {list(ENGINES)}")
        service.request_engine(cfg.engine)      # applied between batches; if it fails to load the current one stays
    if cfg.exercise is not None:
        try:
            local_session.request_exercise(cfg.exercise)
        except ValueError as e:
            raise HTTPException(400, str(e))
    if cfg.privacy is not None:
        worker.privacy = cfg.privacy    # applied by the worker on its next frame
    return worker.telemetry()


class Camera(BaseModel):
    index: int


@app.get("/api/cameras")
def cameras():
    # ponytail: a full probe can take several seconds (2 backends x warm-up per dead index); the active
    # camera is in use by the worker and is reported without reprobing. Cache if this gets called often.
    cams = []
    for i in range(CAMERA_MAX + 1):
        if i == worker.camera_index and worker.error is None:
            cams.append({"index": i, "status": "active", **(worker.camera_info or {})})
            continue
        cap, info = open_camera(i, warmup_s=1.0)
        if cap is None:
            cams.append({"index": i, "status": "unavailable", "reason": info})
        else:
            cap.release()
            cams.append({"index": i, "status": "ok", **info})
    available = [c["index"] for c in cams if c["status"] != "unavailable"]
    devices = list_video_devices()
    for c, name in zip(cams, devices):      # cams is in index order, so position == index
        c["name"], c["label"] = name, camera_label(name)
    return {"active": worker.camera_index, "available": available, "cameras": cams,
            "devices": devices, "error": worker.camera_error}


@app.post("/api/camera")
def camera(c: Camera):
    if not 0 <= c.index <= CAMERA_MAX:
        raise HTTPException(400, f"index must be 0..{CAMERA_MAX}")
    worker.camera_error = None
    worker.pending_camera = c.index
    return {"ok": True}


@app.post("/api/fall/ack")
def fall_ack():
    """"I'm OK": answers the fall countdown or silences a raised alert."""
    local_session.ack()    # only raises a flag; the inference thread applies it on the next frame
    return {"ok": True}


@app.post("/api/fall/listen")
def fall_listen():
    """The dialog's spoken prompt has finished: open the microphone (only during a fall check)."""
    return {"ok": True, **local_session.listen()}


@app.post("/api/fall/escalate")
def fall_escalate():
    """"Trigger Emergency": skips the rest of the countdown."""
    local_session.escalate()
    return {"ok": True}


@app.post("/api/fall/test")
def fall_test():
    """TEMP (Test Fall button): simulate a confirmed fall so the countdown dialog can be tried without falling."""
    local_session.simulate_fall()
    return {"ok": True}


_local_voice_calls: deque[float] = deque()      # upload times for the desktop page's rate limit


@app.post("/api/voice")
async def voice(request: Request):
    """One spoken utterance from the desktop page's microphone ("Server camera" mode): the body is the audio, the
    Content-Type its format. Same processing as the phone's `audio` socket message (app/voicecmd.py). A desktop route:
    refused through a tunnel by TunnelGuard."""
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_AUDIO_BYTES:
        raise HTTPException(413, "audio too large")
    raw = b""
    async for chunk in request.stream():
        raw += chunk
        if len(raw) > MAX_AUDIO_BYTES:
            raise HTTPException(413, "audio too large")
    return await run_in_threadpool(process_audio, local_session, local_session.voice, raw,
                                   request.headers.get("content-type", ""), _local_voice_calls)


@app.post("/api/session/start")
def session_start():
    """HUD Start (Server camera): counters to 0 and rep counting on, same as saying "let's go"."""
    local_session.request_start()
    return {"ok": True}


@app.post("/api/session/stop")
def session_stop():
    """HUD Pause (Server camera): rep counting back to standby."""
    local_session.request_stop()
    return {"ok": True}


@app.post("/api/reset")
def reset():
    local_session.request_reset()
    return {"ok": True}


@app.websocket("/ws/telemetry")
async def ws_telemetry(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            await ws.send_json(worker.telemetry())
            await asyncio.sleep(1 / WS_HZ)
    except WebSocketDisconnect:
        pass


def mjpeg():
    last, min_dt = -1, 1.0 / STREAM_FPS_CAP
    while True:
        with worker.new_frame:
            worker.new_frame.wait_for(lambda: worker.seq != last, timeout=1.0)
            if worker.jpeg is None:
                continue
            last, jpeg = worker.seq, worker.jpeg
            if worker.privacy and not worker.jpeg_private:
                continue    # privacy was just switched on: the stored frame is raw, wait for a blanked one
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(min_dt)


@app.get("/api/video_feed")
def video_feed():
    return StreamingResponse(mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")


BROWSER_MODELS = {"pose_landmarker_lite.task", "pose_landmarker_full.task"}      # the only files served, by exact name
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


@app.get("/models/{name}")
def browser_model(name: str):
    """The MediaPipe pose model for the browser's on-device inference, served from this machine so a phone loads it once from
    here (and the browser caches it for a week) instead of depending on a third-party CDN. Only the two files above."""
    if name not in BROWSER_MODELS:
        raise HTTPException(404, "no such model")
    path = MODELS_DIR / name
    if not path.is_file():
        raise HTTPException(404, "model not downloaded: run backend/scripts/download_models.py")
    return FileResponse(path, media_type="application/octet-stream", headers={"Cache-Control": "public, max-age=604800"})


# Last, so every API and WebSocket route above wins: the built frontend, for a single-port deployment on :8000.
if DIST.is_dir():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="frontend")
