"""Spoken commands from a BROWSER microphone, for either source: the desktop page ("Server camera", POST /api/voice) and a phone
("This device", the `audio` message on /ws/session). Both end up in `process_audio`, so they cannot behave differently:

  audio bytes -> Groq Whisper -> `interpret` (with a fall check open: "I am fine" / "help"; otherwise: the session commands)

The audio is never stored. Blocking (network call): run it in a thread, not on the event loop.
"""
from __future__ import annotations

import logging
import time
from collections import deque

from app.sessions import ACTIVE_ALERT_STATES, Session
from app.voice import groq_stt
from app.voice.intent import classify_command
from app.voice.listener import NO_KEY, VoiceListener

log = logging.getLogger("voicecmd")

MAX_AUDIO_BYTES = 400_000           # one utterance: ~8 s of browser-compressed audio is 50-100 kB
VOICE_PER_MIN = 20                  # uploads per source per minute: a cap on Groq cost and rate limit
AUDIO_TYPES = {"audio/webm": "webm", "audio/mp4": "m4a", "audio/x-m4a": "m4a", "audio/aac": "aac", "audio/ogg": "ogg",
               "audio/mpeg": "mp3", "audio/wav": "wav", "audio/x-wav": "wav"}


def audio_name(mime: str) -> tuple[str, str] | None:
    """("speech.webm", "audio/webm") for a browser's "audio/webm;codecs=opus"; None for a type Whisper cannot read."""
    base = mime.split(";")[0].strip().lower()
    ext = AUDIO_TYPES.get(base)
    return (f"speech.{ext}", base) if ext else None


def interpret(session: Session, listener: VoiceListener | None, text: str) -> str | None:
    """What a spoken sentence means right now. While the fall dialog is open only the safety answers count ("I am fine"
    acknowledges, "help" escalates, through the same listener the desktop microphone uses, so sentences cut by a pause are
    joined). Otherwise only the session commands count, so ordinary talk can never cancel or raise an emergency.
    "start session" starts counting from zero and "stop session" pauses counting here; showing / hiding the guide and stopping
    the camera are for the page."""
    alert = ((session.telemetry().get("fall") or {}).get("alert") or {}).get("state")
    if alert in ACTIVE_ALERT_STATES:
        return listener.handle(text) if listener is not None else None
    intent = classify_command(text)
    if intent == "start_session":
        session.request_start()             # counters to 0 and rep counting on (it is paused in standby)
    elif intent == "stop_session":
        session.request_stop()              # rep counting back to standby; a phone also stops its camera (the page does that)
    return intent


def _fail(error: str, status: str | None = None) -> dict:
    return {"type": "voice", "ok": False, "error": error, **({"status": status} if status else {})}


def process_audio(session: Session, listener: VoiceListener | None, raw: bytes, mime: str, calls: deque) -> dict:
    """One uploaded utterance -> the reply for the page: {"type":"voice","ok":true,"text","intent"} or {"ok":false,"error"}.
    `calls` is the caller's own deque of recent upload times (the rate limit is per source)."""
    named = audio_name(mime)
    if named is None:
        return _fail(f"unsupported audio type {mime[:40]!r}")
    if not raw or len(raw) > MAX_AUDIO_BYTES:
        return _fail("audio too large" if raw else "no audio")
    key = groq_stt.api_key()
    if key is None:
        return _fail(NO_KEY, "unavailable")
    now = time.monotonic()
    while calls and now - calls[0] > 60:
        calls.popleft()
    if len(calls) >= VOICE_PER_MIN:
        return _fail("voice rate limit reached, try again in a minute")
    calls.append(now)
    try:
        heard = groq_stt.transcribe(raw, key, None, named[0], named[1])
    except groq_stt.SttError as e:
        return _fail(str(e))
    intent = interpret(session, listener, heard)
    log.info("session %s heard %r -> %s", session.id, heard, intent)
    return {"type": "voice", "ok": True, "text": heard, "intent": intent}
