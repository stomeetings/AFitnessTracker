"""Remote (mobile browser) sessions over a WebSocket: frames in, telemetry and keypoints out. No video comes back: the
phone draws the skeleton over its own camera preview.

Wire protocol (one socket = one session; see docs/cloud_mobile_blueprint.md section 2):
  server -> client  {"type":"hello","session":id,"reference_long_side":640,"exercises":[...],"engine":"yolo","engines":[...]}   once, first
                    (every "t" reply also carries "engine", the pose model in use; cmd "engine" with a name from "engines"
                    switches it for the whole server, and an empty list means switching is not offered)
  client -> server  binary: [u32 frame number][f64 capture time, ms, client clock][JPEG bytes]    big-endian
  server -> client  {"type":"t","ack":n,"frame":{"w","h"},"pose":{"kp":{name:[x,y,conf]},"bbox":[x,y,w,h]}|null,
                     ...the session's telemetry...}                      one per processed frame ("error" if it failed)
  client -> server  {"type":"cmd","name":"exercise","value":"Elbow Curl"} | "reset" | "ack_fall" | "escalate" | "test_fall"
  server -> client  {"type":"cmd_result","name":...,"ok":true|false,"error":...}
  client -> server  {"type":"audio","mime":"audio/webm;codecs=opus","data":"<base64>"}      one spoken utterance (<= 400 kB)
  server -> client  {"type":"voice","ok":true,"text":"...","intent":...}   or {"type":"voice","ok":false,"error":...,"status"?}
      intent while a fall check is open: "ok" (-> acknowledged) | "help" (-> escalated) | null
      intent otherwise: "start_session" (-> counters reset) | "stop_session" | "show_preview" | "hide_preview" | null
      (the last three are for the page to carry out). The audio goes to Groq's Whisper and is never stored.

The client should send frame N+1 only after the ack for frame N. If it does not, only the newest frame is ever
processed (the reader keeps one slot, and the inference service keeps one frame per session): latency stays bounded.

Access control (see app/security.py): the endpoint is switched off unless REMOTE_SESSIONS=1; a browser's Origin must be
this site (or listed in REMOTE_ALLOWED_ORIGINS); the number of simultaneous remote sessions is capped
(REMOTE_MAX_SESSIONS, default 4). If REMOTE_ACCESS_TOKEN is set, the client's FIRST message must be
  client -> server  {"type":"auth","token":"..."}                    within 5 s, else the socket is closed with code 4401
(a message, not a URL parameter, so the token never appears in server or proxy logs). Wrong tokens are rate limited.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import struct
import uuid
from collections import deque
from collections.abc import Callable

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket

from app.engines import PoseResult
from app.inference import FrameResult, InferenceService
from app.security import FailureLimiter, origin_allowed, token_ok
from app.sessions import EXERCISE_NAMES, REFERENCE_LONG_SIDE, Session, SessionLimitError, SessionManager
from app.voice.listener import VoiceListener
from app.voicecmd import MAX_AUDIO_BYTES, process_audio

log = logging.getLogger("remote")

HEADER = struct.Struct(">Id")       # frame number, capture time in ms: the byte order a JS DataView writes by default
MAX_FRAME_BYTES = 2_000_000
MAX_TEXT_BYTES = 4096               # commands and the auth message are tiny
AUDIO_PREFIX = '{"type":"audio"'    # the client writes the type first, so the (large) audio messages are told apart cheaply
MAX_AUDIO_TEXT = MAX_AUDIO_BYTES * 4 // 3 + 200      # an utterance base64 encoded, plus the JSON around it
MAX_PENDING = 16                    # sockets accepted but not yet authenticated, so they cannot pile up
CLOSE_UNAUTHORIZED = 4401           # application close code: token missing or wrong
REMOTE_PREFIX = "remote-"


def decode_frame(jpeg: bytes) -> np.ndarray | None:
    """JPEG bytes -> BGR image, or None when it is not a decodable image (or absurdly large)."""
    if not jpeg or len(jpeg) > MAX_FRAME_BYTES:
        return None
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)


def pose_payload(pose: PoseResult | None) -> dict | None:
    """Keypoints in the frame's own pixel coordinates, so the phone can scale them onto its preview."""
    if pose is None:
        return None
    b = pose.bbox
    return {"kp": {name: [round(p.x, 1), round(p.y, 1), round(p.conf, 2)] for name, p in pose.keypoints.items()},
            "bbox": [round(v, 1) for v in (b.x, b.y, b.w, b.h)]}


def reply_for(session: Session, seq: int, result: FrameResult, width: int, height: int, engine: str | None = None) -> dict:
    msg = {"type": "t", "ack": seq, "frame": {"w": width, "h": height}, "infer_ms": round(result.infer_ms, 1),
           "pose": pose_payload(result.pose), **session.telemetry()}
    if engine is not None:
        msg["engine"] = engine                  # the pose model in use right now, so the phone can show it
    if result.error:
        msg["error"] = result.error
    return msg


def run_command(session: Session, text: str, service: InferenceService | None = None, engines: list[str] | None = None) -> dict | None:
    """Handles one {"type":"cmd",...} message. Commands only set flags; the inference thread applies them.
    A stray {"type":"auth"} (the client sends one even when no token is required) is ignored: returns None.
    "engine" switches the pose model for the WHOLE server (every session, the desktop included): it needs `service` and a
    non-empty list of allowed names, and the change happens between batches (a failed load keeps the old model)."""
    try:
        msg = json.loads(text)
        if msg.get("type") == "auth":
            return None
        name = msg["name"]
    except (ValueError, KeyError, TypeError, AttributeError):
        return {"type": "cmd_result", "ok": False, "error": "bad command"}
    try:
        if name == "exercise":
            session.request_exercise(str(msg.get("value", "")))
        elif name == "reset":
            session.request_reset()
        elif name == "start_session":           # the HUD's Start button: same as saying "let's go"
            session.request_start()
        elif name == "stop_session":            # the HUD's Pause button: back to standby, the camera keeps running
            session.request_stop()
        elif name == "ack_fall":
            session.ack()
        elif name == "escalate":
            session.escalate()
        elif name == "test_fall":               # the dashboard's temporary Test Fall button, kept working on a phone
            session.simulate_fall()
        elif name == "engine":
            value = str(msg.get("value", ""))
            if service is None or value not in (engines or []):
                raise ValueError(f"engine must be one of {engines or []}")
            service.request_engine(value)
        else:
            return {"type": "cmd_result", "name": name, "ok": False, "error": f"unknown command {name!r}"}
    except ValueError as e:
        return {"type": "cmd_result", "name": name, "ok": False, "error": str(e)}
    return {"type": "cmd_result", "name": name, "ok": True}


async def _close(ws: WebSocket, code: int) -> None:
    try:
        await ws.close(code=code)
    except Exception:  # noqa: BLE001  already closed by the peer
        pass


def create_router(sessions: SessionManager, service: InferenceService, *, enabled: Callable[[], bool] = lambda: True,
                  max_remote: Callable[[], int] = lambda: 4, token: Callable[[], str] = lambda: "",
                  allowed_origins: Callable[[], list[str]] = lambda: [], engine_names: Callable[[], list[str]] = lambda: [],
                  auth_timeout: float = 5.0,
                  limiter: FailureLimiter | None = None) -> APIRouter:
    router = APIRouter()
    limiter = limiter or FailureLimiter()
    pending = [0]                                   # accepted sockets that have not finished authenticating

    def remote_count() -> int:
        return sum(1 for sid in sessions.ids() if sid.startswith(REMOTE_PREFIX))

    async def authenticate(ws: WebSocket) -> bool:
        if limiter.blocked():
            await _close(ws, 1013)                  # too many wrong tokens lately: try again in a minute
            return False
        supplied: str | None = None
        try:
            text = await asyncio.wait_for(ws.receive_text(), auth_timeout)
            if len(text) <= MAX_TEXT_BYTES:
                msg = json.loads(text)
                if msg.get("type") == "auth":
                    supplied = str(msg.get("token", ""))
        except asyncio.TimeoutError:
            pass                                    # said nothing (a client with no token): refused, but not a guess
        except Exception:  # noqa: BLE001  disconnect, not text, not JSON
            supplied = ""
        if supplied is not None and token_ok(supplied, token()):
            return True
        if supplied is not None:
            limiter.record()
            log.warning("remote session refused: wrong access token")
        await _close(ws, CLOSE_UNAUTHORIZED)
        return False

    @router.websocket("/ws/session")
    async def ws_session(ws: WebSocket) -> None:
        if not enabled():
            await ws.close(code=1008)               # policy violation: remote sessions are switched off
            return
        hosts = [ws.headers.get("host", ""), ws.headers.get("x-forwarded-host", "").split(",")[0]]
        if not origin_allowed(ws.headers.get("origin"), hosts, allowed_origins()):
            log.warning("remote session refused: origin %r is not this site (REMOTE_ALLOWED_ORIGINS)", ws.headers.get("origin"))
            await ws.close(code=1008)
            return
        if remote_count() >= max_remote() or pending[0] >= MAX_PENDING:
            await ws.close(code=1013)               # try again later
            return
        pending[0] += 1
        try:
            await ws.accept()
            if token() and not await authenticate(ws):
                return
        finally:
            pending[0] -= 1
        if remote_count() >= max_remote():          # others may have got in while this one was authenticating
            await _close(ws, 1013)
            return
        try:
            # standby: a phone user is still placing the phone and stepping into the frame; reps count after "let's go"
            session = sessions.create(REMOTE_PREFIX + uuid.uuid4().hex[:12], standby=True)
        except SessionLimitError:
            await ws.close(code=1013)
            return
        log.info("remote session %s connected", session.id)
        loop = asyncio.get_running_loop()
        send_lock = asyncio.Lock()
        newest: list[bytes | None] = [None]         # the newest frame not yet taken; a newer one replaces it
        wake, closed = asyncio.Event(), asyncio.Event()

        async def send(obj: dict) -> None:
            async with send_lock:                   # the reader (command replies) and the processor both send
                await ws.send_json(obj)

        async def handle_frame(data: bytes) -> None:
            if len(data) <= HEADER.size:
                await send({"type": "t", "ack": 0, "error": "frame too short"})
                return
            seq, capture_ms = HEADER.unpack_from(data)
            frame = await loop.run_in_executor(None, decode_frame, data[HEADER.size:])     # off the event loop
            if frame is None:
                await send({"type": "t", "ack": seq, "error": "not a JPEG image"})
                return
            height, width = frame.shape[:2]
            result = await asyncio.wrap_future(service.submit(session, frame, capture_ms / 1000.0))
            if not result.dropped:                  # dropped: replaced by a newer frame, or the service is stopping
                await send(reply_for(session, seq, result, width, height, service.engine_name))

        listener = VoiceListener(on_ok=session.ack, on_help=session.escalate)    # only its sentence logic; no microphone
        voice_calls: deque[float] = deque()
        voice_task: list[asyncio.Task | None] = [None]

        async def voice_fail(error: str) -> None:
            await send({"type": "voice", "ok": False, "error": error})

        async def handle_audio(text: str) -> None:
            try:
                msg = json.loads(text)
                mime = str(msg.get("mime", ""))
                raw = base64.b64decode(msg.get("data", ""), validate=True)
            except (ValueError, TypeError, AttributeError):
                await voice_fail("bad audio message")
                return
            # the same path the desktop page's POST /api/voice takes: validation, Groq, intent
            await send(await loop.run_in_executor(None, process_audio, session, listener, raw, mime, voice_calls))

        async def reader() -> None:
            try:
                while True:
                    msg = await ws.receive()
                    if msg["type"] == "websocket.disconnect":
                        return
                    if msg.get("bytes") is not None:
                        if len(msg["bytes"]) <= MAX_FRAME_BYTES + HEADER.size:      # an oversized frame is just ignored
                            newest[0] = msg["bytes"]
                            wake.set()
                    elif msg.get("text") is not None:
                        text = msg["text"]
                        if text.startswith(AUDIO_PREFIX):
                            busy = voice_task[0] is not None and not voice_task[0].done()
                            if len(text) > MAX_AUDIO_TEXT:
                                await voice_fail("audio too large")
                            elif busy:
                                await voice_fail("still processing the last sentence")
                            else:                                       # off the reader, so frames and commands keep flowing
                                voice_task[0] = asyncio.create_task(handle_audio(text))
                        elif len(text) <= MAX_TEXT_BYTES:
                            reply = run_command(session, text, service, engine_names())
                            if reply is not None:
                                await send(reply)
            finally:
                closed.set()
                wake.set()

        async def processor() -> None:
            while True:
                await wake.wait()
                wake.clear()
                data, newest[0] = newest[0], None
                if data is None:
                    if closed.is_set():
                        return
                    continue
                await handle_frame(data)

        await send({"type": "hello", "session": session.id, "reference_long_side": REFERENCE_LONG_SIDE,
                    "exercises": EXERCISE_NAMES, "engine": service.engine_name, "engines": engine_names()})
        tasks = [asyncio.create_task(reader()), asyncio.create_task(processor())]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            if voice_task[0] is not None:
                tasks.append(voice_task[0])
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            service.forget(session.id)
            sessions.close(session.id)
            log.info("remote session %s closed", session.id)

    return router
