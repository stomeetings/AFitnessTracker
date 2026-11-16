"""Voice check-in during a fall alert: records speech from the microphone, has Groq's Whisper transcribe it, and acts on
the text through intent.classify ("ok" -> on_ok(), "help" -> on_help()).

It all runs on one daemon thread that exists only while the fall dialog is open, so the camera / inference loop never
waits on audio or the network. Every public method returns immediately. If the key is missing, the microphone cannot be
opened or delivers no audio at all, or the network fails, the status becomes "unavailable" (with the reason in `error`)
and the dialog falls back to its buttons; nothing else is affected.

status: "idle" (ready) | "starting" (microphone being opened) | "listening" (audio is really flowing) |
        "transcribing" | "unavailable"
`warning` is set (status stays "listening") when the microphone is open and delivering audio but only digital silence:
many headsets send exact zeros until you speak, so that is NOT treated as a failure.
The browser records nothing: the microphone is the backend machine's, and the dialog only shows this status.
The "voice:" log lines trace each step (device, first audio, speech found, upload, answer) so a stall is easy to find.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable

from app.voice import groq_stt, recorder
from app.voice.intent import classify

log = logging.getLogger("voice")

MAX_CALLS = 8       # transcriptions per fall check: a safety cap on cost and on Groq's rate limit (a normal check uses 1-3)
NO_KEY = "GROQ_API_KEY is not set (put it in backend/.env)"
CARRY_S = 4.0       # a piece of speech that was not an answer is joined to the next piece heard within this long
WATCHDOG_S = 3.0    # the microphone must deliver audio blocks within this long; digital silence this long only warns
SOUND_FLOOR = 1e-4  # a live microphone is never quieter than this once it hears anything


class VoiceListener:
    def __init__(self, on_ok: Callable[[], None], on_help: Callable[[], None]):
        self.on_ok, self.on_help = on_ok, on_help
        self.status = "idle"
        self.transcript = ""
        self.intent: str | None = None
        self.error: str | None = None
        self.warning: str | None = None
        self._carry = ""            # earlier speech that was not an answer yet
        self._carry_t = 0.0
        self._lock = threading.Lock()
        self._stop: threading.Event | None = None
        self._thread: threading.Thread | None = None

    @property
    def active(self) -> bool:
        return self.status in ("starting", "listening", "transcribing")

    def snapshot(self) -> dict:
        return {"status": self.status, "transcript": self.transcript, "intent": self.intent,
                "error": self.error, "warning": self.warning}

    def init(self) -> None:
        """Call once at backend start: reports early whether voice can work at all (the key may still be added later)."""
        if groq_stt.api_key() is None:
            self._unavailable(NO_KEY)

    def start(self) -> None:
        """Opens the microphone and starts listening. Ignored while a session is already running."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive() and self._stop is not None and not self._stop.is_set():
                log.info("voice: start ignored, a session is already running")
                return
            key = groq_stt.api_key()
            if key is None:
                self._unavailable(NO_KEY)
                return
            self.transcript, self.intent, self.error, self.warning, self._carry = "", None, None, None, ""
            self.status = "starting"        # "listening" only once audio has really arrived (see _session)
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._session, args=(key, self._stop), daemon=True, name="voice")
            self._thread.start()
            log.info("voice: session started")

    def stop(self) -> None:
        with self._lock:
            if self._stop is not None and not self._stop.is_set():
                self._stop.set()
                log.info("voice: session stopped (the fall dialog is gone)")
            if self.active:
                self.status = "idle"

    def quit(self) -> None:
        self.stop()

    def handle(self, text: str, now: float | None = None) -> str | None:
        """Acts on what was heard. Returns the intent. A short pause cuts one sentence into two utterances ("I am" /
        "fine"), so a piece that was not an answer on its own is joined to the next piece heard within CARRY_S."""
        self.transcript = text
        if not text.strip():
            self.intent = None
            return None
        now = time.monotonic() if now is None else now
        joined = f"{self._carry} {text}" if self._carry and now - self._carry_t <= CARRY_S else text
        self.intent = classify(joined)
        if self.intent is None:
            self._carry, self._carry_t = joined[-120:], now
        else:
            self._carry = ""
        if self.intent == "ok":
            self.on_ok()
        elif self.intent == "help":
            self.on_help()
        return self.intent

    def _set(self, stop: threading.Event, status: str) -> None:
        with self._lock:
            if not stop.is_set():
                self.status = status

    def _unavailable(self, message: str) -> None:
        self.status, self.error = "unavailable", message
        log.warning("voice unavailable: %s", message)

    def _fail(self, stop: threading.Event, message: str) -> None:
        if not stop.is_set():
            self._unavailable(message)

    @staticmethod
    def _devices_text(sd) -> str:
        try:
            if sd is None:
                import sounddevice as sd
            return recorder.describe_inputs(sd)
        except Exception:  # noqa: BLE001
            return "  (could not list the input devices)"

    def _session(self, key: str, stop: threading.Event) -> None:
        sd = None
        try:
            import sounddevice as sd
            with recorder.open_stream(sd, recorder.device_preferences()) as (blocks, rate, device_name):
                segmenter = recorder.Segmenter(rate)
                opened = time.monotonic()
                got_audio = warned = False
                calls = 0
                log.info("voice: microphone open: %s at %d Hz, speech threshold %.4f", device_name, rate, segmenter.threshold)
                while not stop.is_set() and calls < MAX_CALLS:
                    waited = time.monotonic() - opened
                    try:
                        block = blocks.get(timeout=0.1)
                    except queue.Empty:
                        if not got_audio and waited > WATCHDOG_S:
                            log.warning("voice: %s opened but delivered no audio at all. Input devices:\n%s",
                                        device_name, self._devices_text(sd))
                            self._fail(stop, f"the microphone ({device_name}) delivered no audio "
                                             "(busy, blocked, or the wrong input device)")
                            return
                        continue
                    if not got_audio:
                        got_audio = True
                        self._set(stop, "listening")
                        log.info("voice: audio is flowing (first block: %d samples)", len(block))
                    utterance = segmenter.feed(block)
                    if segmenter.loudest >= SOUND_FLOOR and self.warning:
                        self.warning = None                  # sound has appeared: the microphone works
                    if utterance is None:
                        if not warned and waited > WATCHDOG_S and segmenter.loudest < SOUND_FLOOR:
                            warned = True
                            if not stop.is_set():
                                self.warning = (f"No sound from {device_name} yet. If you are speaking, this is probably "
                                                "the wrong input device (see the backend log).")
                            log.warning("voice: no sound at all from %s after %.1f s (a headset that is silent until you "
                                        "speak, a muted microphone, or the wrong input device). Input devices:\n%s",
                                        device_name, waited, self._devices_text(sd))
                        continue
                    calls += 1
                    wav = recorder.to_wav_bytes(utterance, rate)
                    log.info("voice: speech of %.2f s found -> sending %d bytes to Groq", len(utterance) / rate, len(wav))
                    self._set(stop, "transcribing")     # audio keeps queueing meanwhile and is processed afterwards
                    sent = time.monotonic()
                    text = groq_stt.transcribe(wav, key)
                    if stop.is_set():
                        return                           # the dialog closed while we waited: never act on late text
                    log.info("voice: Groq answered in %.2f s: %r -> %s", time.monotonic() - sent, text, self.handle(text))
                    self._set(stop, "listening")
                if calls >= MAX_CALLS:
                    log.info("voice: send limit of %d reached for this check", MAX_CALLS)
        except groq_stt.SttError as e:
            self._fail(stop, str(e))
        except Exception as e:  # noqa: BLE001  no sounddevice, device not found / busy, no usable sample rate ...
            if not stop.is_set():
                log.warning("voice: could not use the microphone (%s: %s). Input devices:\n%s",
                            type(e).__name__, e, self._devices_text(sd))
            self._fail(stop, f"{type(e).__name__}: {e}".splitlines()[0][:200])
        finally:
            if self.status in ("starting", "listening", "transcribing"):
                self._set(stop, "idle")
