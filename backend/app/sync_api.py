"""The phones' upload and the dashboard's history (Phase 4 of the offline-first plan).

  POST /api/sync/sessions   {"deviceId": "...", "sessions": [{id, exercise, startedAt, endedAt, limbs, calibration, workouts, appVersion,
                            events: [{t, type, payload}]}, ...]}  ->  {"stored": [ids], "duplicate": [ids]}
  GET  /api/sessions        ?limit=50&offset=0  ->  {"sessions": [...], "total", "summary": {sessions, reps, fall_sessions}}

Both need the access token (`Authorization: Bearer <token>` or `X-Access-Token`) whenever REMOTE_ACCESS_TOKEN is set, and through a
tunnel they are refused outright when no token is configured: workout history is patient data. Uploads are size limited (body bytes,
sessions per batch, events per session, field shapes) and idempotent by session id (see storage.py), so a retry is always safe.
Validation is plain Python on the decoded JSON: nothing is trusted, nothing is stored until the whole batch has been checked.
"""
from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable

from fastapi import APIRouter, Request
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

from app.security import FailureLimiter, is_tunneled, token_ok
from app.storage import Storage

log = logging.getLogger("sync")

MAX_BODY_BYTES = 1_000_000          # one request: 20 sessions with their events are well under this
MAX_SESSIONS = 20
MAX_EVENTS = 20_000                 # per session
MAX_WORKOUTS = 500
EVENT_TYPES = frozenset({"start", "stop", "pause", "resume", "exercise", "rep", "fall", "alert", "end"})
_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


class Invalid(ValueError):
    """The upload does not have the expected shape (answered with 400 and the reason)."""


def _int(v, name: str, lo: int = 0) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v < lo or v > 10**15:
        raise Invalid(f"{name} must be a whole number")
    return v


def _text(v, name: str, limit: int) -> str:
    if not isinstance(v, str) or not v or len(v) > limit:
        raise Invalid(f"{name} must be text of 1 to {limit} characters")
    return v


def validate_batch(msg) -> tuple[str, list[dict]]:
    """(device id, sessions) of a decoded upload, or Invalid."""
    if not isinstance(msg, dict):
        raise Invalid("expected a JSON object")
    device = _text(msg.get("deviceId"), "deviceId", 64)
    sessions = msg.get("sessions")
    if not isinstance(sessions, list) or not 1 <= len(sessions) <= MAX_SESSIONS:
        raise Invalid(f"sessions must be a list of 1 to {MAX_SESSIONS}")
    seen: set[str] = set()
    for s in sessions:
        if not isinstance(s, dict):
            raise Invalid("a session must be an object")
        sid = s.get("id")
        if not isinstance(sid, str) or not _ID.match(sid):
            raise Invalid("a session needs an id of 8 to 64 letters, digits, - or _")
        if sid in seen:
            raise Invalid("the same session twice in one upload")
        seen.add(sid)
        _text(s.get("exercise"), "exercise", 64)
        _int(s.get("startedAt"), "startedAt")
        if s.get("endedAt") is not None:
            _int(s["endedAt"], "endedAt")
        if "appVersion" in s:
            _text(s["appVersion"], "appVersion", 32)
        for key, kind in (("limbs", dict), ("calibration", dict), ("workouts", list)):
            if key in s and not isinstance(s[key], kind):
                raise Invalid(f"{key} has the wrong shape")
        if len(s.get("workouts") or []) > MAX_WORKOUTS:
            raise Invalid("too many workouts in one session")
        for w in s.get("workouts") or []:
            if not isinstance(w, dict) or not isinstance(w.get("limbs") or {}, dict):
                raise Invalid("a workout has the wrong shape")
        for limbs in [s.get("limbs") or {}] + [w.get("limbs") or {} for w in s.get("workouts") or []]:
            for limb in limbs.values():
                if not isinstance(limb, dict) or isinstance(limb.get("reps", 0), bool) or not isinstance(limb.get("reps", 0), int):
                    raise Invalid("limb counts must hold whole numbers")
        events = s.get("events", [])
        if not isinstance(events, list) or len(events) > MAX_EVENTS:
            raise Invalid(f"events must be a list of at most {MAX_EVENTS}")
        for e in events:
            if (not isinstance(e, dict) or e.get("type") not in EVENT_TYPES or isinstance(e.get("t"), bool)
                    or not isinstance(e.get("t"), (int, float)) or not isinstance(e.get("payload", {}), dict)):
                raise Invalid("an event needs t (a number), a known type and a payload object")
    return device, sessions


def create_sync_router(storage: Callable[[], Storage], token: Callable[[], str] = lambda: "", limiter: FailureLimiter | None = None) -> APIRouter:
    router = APIRouter()
    limiter = limiter or FailureLimiter()

    def refused(request: Request) -> JSONResponse | None:
        """None when the caller may continue, else the answer to send."""
        if limiter.blocked():
            return JSONResponse({"detail": "too many wrong tokens, try again in a minute"}, status_code=429)
        expected = token()
        auth = request.headers.get("authorization", "")
        supplied = auth[7:].strip() if auth.lower().startswith("bearer ") else request.headers.get("x-access-token", "").strip()
        if supplied:
            if expected and token_ok(supplied, expected):
                return None
            limiter.record()
            log.warning("sync refused: wrong access token")
            return JSONResponse({"detail": "wrong access token"}, status_code=401)
        if expected:
            return JSONResponse({"detail": "access token required"}, status_code=401)
        if is_tunneled(request.scope["headers"]):
            return JSONResponse({"detail": "set REMOTE_ACCESS_TOKEN before using this through a tunnel"}, status_code=403)
        return None                                              # no token configured, and a direct (localhost / LAN) caller

    @router.post("/api/sync/sessions")
    async def sync_sessions(request: Request):
        denied = refused(request)
        if denied:
            return denied
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > MAX_BODY_BYTES:
            return JSONResponse({"detail": "upload too large"}, status_code=413)
        body = bytearray()
        async for chunk in request.stream():                     # counted as it arrives: a missing or false Content-Length cannot get around it
            body += chunk
            if len(body) > MAX_BODY_BYTES:
                return JSONResponse({"detail": "upload too large"}, status_code=413)
        try:
            device, sessions = validate_batch(json.loads(bytes(body)))
        except (ValueError, UnicodeDecodeError) as e:           # Invalid is a ValueError; so is bad JSON
            return JSONResponse({"detail": str(e) if isinstance(e, Invalid) else "not valid JSON"}, status_code=400)
        stored, duplicate = await run_in_threadpool(storage().insert_batch, device, sessions)     # SQLite blocks: not on the event loop
        log.info("sync: %d stored, %d already known (device %s)", len(stored), len(duplicate), device[:8])
        return {"stored": stored, "duplicate": duplicate}

    @router.get("/api/sessions")
    def list_sessions(request: Request, limit: int = 50, offset: int = 0):
        denied = refused(request)
        if denied:
            return denied
        return storage().list_sessions(limit, offset)

    return router
