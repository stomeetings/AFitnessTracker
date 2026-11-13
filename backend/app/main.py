"""FastAPI app: webcam capture + pose inference + analytics worker thread, MJPEG stream, WS telemetry.

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
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.engines import make_engine
from app.pipeline.annotate import draw
from app.pipeline.core import Pipeline
from app.pipeline.rep_counter import EXERCISES

ENGINES = {
    "mediapipe": ("mediapipe", {"variant": "full"}),
    "yolo": ("yolo", {"device": "cuda:0"}),
}
EXERCISE_NAMES = list(EXERCISES)
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


class Worker(threading.Thread):
    """Owns the camera, the engine and the analytics pipeline. Keeps only the latest JPEG (never queues)."""

    def __init__(self):
        super().__init__(daemon=True)
        self.engine_name = "yolo"
        self.pipeline = Pipeline(EXERCISE_NAMES[0])
        self.pending_engine: str | None = None
        self.pending_exercise: str | None = None
        self.pending_reset = False
        self.camera_index = CAMERA_INDEX
        self.camera_info: dict | None = None
        self.pending_camera: int | None = None
        self.camera_error: str | None = None
        self.jpeg: bytes | None = None
        self.seq = 0
        self.fps = 0.0
        self.latency_ms = 0.0
        self.pose_visible = False
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
            self.pipeline.fall.reset()      # perspective / distance changed: relearn the standing baseline
            return new
        self.camera_error = f"no camera found near index {idx}; kept camera {self.camera_index}"   # old stream stays alive
        log.error("camera switch failed: %s", " | ".join(failures))
        return cap

    def _apply_pending(self, engine):
        if self.pending_engine and self.pending_engine != self.engine_name:
            engine.close()
            self.engine_name = self.pending_engine
            engine = make_engine(ENGINES[self.engine_name][0], **ENGINES[self.engine_name][1])
            engine.load()
        self.pending_engine = None
        if self.pending_exercise and self.pending_exercise != self.pipeline.exercise:
            self.pipeline.set_exercise(self.pending_exercise)
        self.pending_exercise = None
        if self.pending_reset:
            self.pipeline.reset()
            self.pending_reset = False
        return engine

    def run(self):
        cap, info = open_camera(self.camera_index)
        if cap is None:
            self.error = f"cannot open camera {self.camera_index} ({info})"   # keep running so the user can switch
        else:
            self.camera_info = info
        try:
            engine = make_engine(ENGINES[self.engine_name][0], **ENGINES[self.engine_name][1])
            engine.load()               # YOLO load includes a dummy-frame warm-up inference
        except Exception:  # noqa: BLE001  no GPU / missing weights must not kill the stream
            log.exception("default engine %s failed to load, falling back to mediapipe", self.engine_name)
            self.engine_name = "mediapipe"
            engine = make_engine(ENGINES[self.engine_name][0], **ENGINES[self.engine_name][1])
            engine.load()
        t0 = prev = time.perf_counter()
        while not self.stop_flag.is_set():
            engine = self._apply_pending(engine)
            cap = self._apply_camera(cap)
            ok, frame = cap.read() if cap is not None else (False, None)
            if not ok:
                time.sleep(0.01)
                continue
            t1 = time.perf_counter()
            res = engine.infer(frame, int((t1 - t0) * 1000))
            self.latency_ms = (time.perf_counter() - t1) * 1000
            self.pose_visible = res is not None
            self.pipeline.process(res, t1 - t0)
            if res:
                draw(frame, res)
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            now = time.perf_counter()
            self.fps = 0.9 * self.fps + 0.1 / max(now - prev, 1e-3)
            prev = now
            with self.new_frame:
                self.jpeg = buf.tobytes()
                self.seq += 1
                self.new_frame.notify_all()
        engine.close()
        if cap is not None:
            cap.release()

    def telemetry(self) -> dict:
        return {
            "camera_index": self.camera_index,
            "camera_backend": (self.camera_info or {}).get("backend"),
            "camera_error": self.camera_error,
            "engine": self.engine_name,
            "device": ENGINES[self.engine_name][1].get("device", "cpu"),
            "fps": round(self.fps, 1),
            "latency_ms": round(self.latency_ms, 1),
            "pose_visible": self.pose_visible,
            "error": self.error,
            "exercise": self.pipeline.exercise,
            **self.pipeline.last,
        }


worker = Worker()


@asynccontextmanager
async def lifespan(app: FastAPI):
    worker.camera_index = default_camera_index()    # enumerated here, not on the worker thread that owns the capture
    worker.start()
    yield
    worker.stop_flag.set()


app = FastAPI(title="AI Rehab Tracker", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Config(BaseModel):
    engine: str | None = None
    exercise: str | None = None


@app.get("/api/health")
def health():
    return {"status": "ok" if worker.error is None else "error", **worker.telemetry()}


@app.get("/api/exercises")
def exercises():
    return EXERCISE_NAMES


@app.get("/api/telemetry")
def telemetry():
    return worker.telemetry()


@app.post("/api/config")
def config(cfg: Config):
    if cfg.engine is not None:
        if cfg.engine not in ENGINES:
            raise HTTPException(400, f"engine must be one of {list(ENGINES)}")
        worker.pending_engine = cfg.engine
    if cfg.exercise is not None:
        if cfg.exercise not in EXERCISES:
            raise HTTPException(400, f"exercise must be one of {EXERCISE_NAMES}")
        worker.pending_exercise = cfg.exercise
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


@app.post("/api/reset")
def reset():
    worker.pending_reset = True
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
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(min_dt)


@app.get("/api/video_feed")
def video_feed():
    return StreamingResponse(mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")
