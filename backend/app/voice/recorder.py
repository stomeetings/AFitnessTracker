"""Microphone capture for the voice check-in: picks and opens the input device, cuts speech out of the audio with a
simple energy detector, and packages one utterance as a small 16 kHz mono WAV held in memory (nothing is written to disk).
"""
from __future__ import annotations

import io
import logging
import math
import queue
import re
import wave
from collections import deque
from contextlib import contextmanager

import numpy as np

from app.voice import settings

log = logging.getLogger("voice")

FRAME_S = 0.02
PRE_S = 0.4          # kept from before speech was noticed, so a soft first sound (the "h" of "help") is not cut off
MIN_ON_S = 0.06      # voiced audio needed to count as speech: 3 frames of 20 ms (VOICE_MIN_ON_S overrides). Very short on
                     # purpose: a sharp "help" or "OK" can have only 0.05-0.1 s above the threshold on a quiet microphone
TRAIL_S = 0.4        # silence that ends an utterance (was 0.8, then 0.5): a one-word answer is sent 0.4 s after it ends.
                     # VOICE_TRAIL_S overrides. Shorter splits slow speech at its pauses; listener.handle() rejoins those.
MAX_S = 5.0          # hard cap per utterance, so a noisy room cannot record forever
MIN_RMS = 0.008      # a frame this loud (RMS, 0..1) counts as voiced; VOICE_MIN_RMS overrides, see .env.example
TONE_SHARE = 0.65    # a frame with this much of its energy in one 150 Hz band is a pure tone (the countdown beep)
WAV_RATE = 16000     # what is uploaded: speech needs no more, and it keeps the upload small


def _frames(seconds: float) -> int:
    return round(seconds / FRAME_S)


def _float_env(name: str, default: float) -> float:
    return settings.get_float(name, default)        # environment first, then backend/.env


def min_rms_from_env() -> float:
    return _float_env("VOICE_MIN_RMS", MIN_RMS)


def tone_share_from_env() -> float:
    return _float_env("VOICE_TONE_SHARE", TONE_SHARE)


# ---- choosing the microphone ------------------------------------------------------------------------------------

def parse_device_list(value) -> list:
    """STT_INPUT_DEVICE: one or several choices separated by commas or semicolons, tried in this order. Each is a device
    number or (part of) a device name; spaces and surrounding quotes are ignored. No commas inside a name."""
    out: list = []
    for part in re.split(r"[;,]", value or ""):
        part = part.strip().strip("\"'").strip()
        if part:
            out.append(int(part) if part.isdigit() else part)
    return out


def device_preferences() -> list:
    return parse_device_list(settings.get("STT_INPUT_DEVICE"))


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())     # "Microphone (2- KORP THALLIUM)" -> "microphone2korpthallium"


def _inputs(sd) -> list:
    return [(i, d) for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]


def describe_inputs(sd) -> str:
    """One line per input device: [index] name (audio API, channels, rate), the system default marked."""
    try:
        default = sd.query_devices(kind="input")["index"]
    except Exception:  # noqa: BLE001
        default = None
    lines = []
    for i, d in _inputs(sd):
        try:
            api = sd.query_hostapis(d["hostapi"])["name"]
        except Exception:  # noqa: BLE001
            api = f"audio API {d['hostapi']}"
        mark = "   <- Windows default" if i == default else ""
        lines.append(f"  [{i}] {d['name']}  ({api}, {d['max_input_channels']} ch, {int(d.get('default_samplerate', 0))} Hz){mark}")
    return "\n".join(lines) or "  (no input devices found)"


def resolve_device(sd, device):
    """None -> None (the Windows default input). A number -> itself, if that input exists. Text -> an input device: an
    exact name first, then any device whose name contains the text ignoring case, spaces and punctuation, looking in the
    default audio API first and then the others (one microphone is listed once per audio API). Raises ValueError, with
    the list of input devices, when nothing fits."""
    if device is None:
        return None
    inputs = _inputs(sd)
    hint = "list them with: backend/.venv/Scripts/python backend/scripts/stt_smoke.py --devices"
    if isinstance(device, int):
        if device not in dict(inputs):
            raise ValueError(f"input device {device} does not exist or has no microphone input; the input devices are:\n"
                             f"{describe_inputs(sd)}\n({hint})")
        return device
    default_api = sd.query_devices(kind="input")["hostapi"]
    wanted, loose = device.lower(), _norm(device)
    for only_default_api in (True, False):
        for matches in (lambda name: name.lower() == wanted, lambda name: bool(loose) and loose in _norm(name)):
            for i, d in inputs:
                if (not only_default_api or d["hostapi"] == default_api) and matches(d["name"]):
                    return i
    raise ValueError(f"no input device matches {device!r}; the input devices are:\n{describe_inputs(sd)}\n({hint})")


def pick_device(sd, preferences):
    """The first of the preferences that exists -> (index, name). No preferences -> (None, the system default's name).
    Raises ValueError (listing the devices) when none of them exists."""
    if not preferences:
        return None, sd.query_devices(kind="input")["name"]
    for preference in preferences:
        try:
            index = resolve_device(sd, preference)
        except ValueError as e:
            log.warning("voice: %s", e)
            continue
        return index, sd.query_devices(index)["name"]
    raise ValueError("none of the configured input devices (" + ", ".join(str(p) for p in preferences)
                     + ") was found; the input devices are:\n" + describe_inputs(sd))


@contextmanager
def open_stream(sd, preferences):
    """Yields (queue of float32 mono blocks, sample rate, device name) while the microphone is open.

    Opens the first of `preferences` that exists (see pick_device). The device's own sample rate is tried first, then
    common ones, and mono before stereo, because some USB headsets refuse one of them on Windows; to_wav_bytes()
    resamples. The caller's own errors inside the `with` block are not caught here."""
    index, name = pick_device(sd, preferences)
    info = sd.query_devices(index, "input")
    rates = list(dict.fromkeys([int(info["default_samplerate"]), 48000, 44100, 16000]))
    channel_options = [1] + ([2] if info.get("max_input_channels", 1) >= 2 else [])
    blocks: queue.Queue = queue.Queue()

    def on_audio(indata, frames, time_info, status):    # PortAudio's thread: only copy and enqueue
        blocks.put(indata[:, 0].copy())

    stream, rate, last_error = None, 0, None
    for channels in channel_options:
        for candidate in rates:
            attempt = None
            try:
                attempt = sd.InputStream(device=index, channels=channels, dtype="float32", samplerate=candidate,
                                         callback=on_audio)
                attempt.start()
            except Exception as e:  # noqa: BLE001  PortAudioError: invalid sample rate / channels, device busy ...
                last_error = e
                log.warning("voice: could not open %s at %d Hz, %d channel(s): %s", name, candidate, channels, e)
                try:
                    if attempt is not None:
                        attempt.close()
                except Exception:  # noqa: BLE001
                    pass
                continue
            stream, rate = attempt, candidate
            break
        if stream is not None:
            break
    if stream is None:
        raise last_error if last_error is not None else RuntimeError(f"could not open {name}")
    log.info("voice: input device [%s] %s at %d Hz", index if index is not None else "default", name, rate)
    try:
        yield blocks, rate, name
    finally:
        try:
            stream.stop()
            stream.close()
        except Exception:  # noqa: BLE001
            pass


# ---- finding speech in the audio --------------------------------------------------------------------------------

def tonal_share(frame) -> float:
    """Share of a frame's energy that sits in its 3 strongest adjacent FFT bins (a 150 Hz band): about 0.8-1.0 for a
    beep, well below 0.6 for speech, whose energy is spread over several harmonics and a lot of noise."""
    spectrum = np.abs(np.fft.rfft(frame * np.hanning(len(frame)))) ** 2
    total = float(spectrum.sum())
    if total <= 0.0 or len(spectrum) < 3:
        return 0.0
    return float((spectrum[:-2] + spectrum[1:-1] + spectrum[2:]).max() / total)


class Segmenter:
    """Feed it microphone blocks; feed() returns a float32 array when an utterance has just ended, otherwise None.

    Speech starts after MIN_ON_S of voiced audio (a leaky counter, so the dips inside a word do not reset it) and ends
    after TRAIL_S of silence. Quiet rooms and clicks never reach the cloud. Pure tones are treated as silence: the
    0.15 s countdown beeps are as long as a quick "help", so duration alone cannot tell them apart, and they must
    not raise the ambient estimate either.

    Diagnostics for the smoke test: floor, loudest, tone_frames, and ignored (short sounds that never became speech)."""

    def __init__(self, rate: int, min_rms: float | None = None, tone_share: float | None = None,
                 trail_s: float | None = None, min_on_s: float | None = None):
        self.n = max(3, int(rate * FRAME_S))
        self.trail_s = trail_s if trail_s is not None else _float_env("VOICE_TRAIL_S", TRAIL_S)
        self.min_on_s = min_on_s if min_on_s is not None else _float_env("VOICE_MIN_ON_S", MIN_ON_S)
        self._min_on = max(1, math.ceil(self.min_on_s / FRAME_S - 1e-9))     # whole frames, rounded up
        self.min_rms = min_rms if min_rms is not None else min_rms_from_env()
        self.tone_share = tone_share if tone_share is not None else tone_share_from_env()
        self.floor = 0.004          # ambient level estimate; follows the quiet frames
        self.loudest = 0.0
        self.tone_frames = 0
        self.ignored: list[dict] = []
        self._buf = np.zeros(0, dtype=np.float32)
        self._pre: deque = deque(maxlen=_frames(PRE_S))
        self._frames: list = []
        self._on = 0
        self._quiet = 0
        self._speaking = False
        self._blip = 0
        self._blip_peak = 0.0

    @property
    def threshold(self) -> float:
        return max(self.min_rms, 4.0 * self.floor)

    def feed(self, block):
        self._buf = np.concatenate((self._buf, block.astype(np.float32, copy=False)))
        while len(self._buf) >= self.n:
            frame, self._buf = self._buf[:self.n], self._buf[self.n:]
            done = self._frame(frame)
            if done is not None:
                return done                 # the rest of the buffer is handled by the next feed()
        return None

    def _frame(self, frame):
        rms = float(np.sqrt(np.mean(frame * frame)))
        self.loudest = max(self.loudest, rms)
        voiced = rms > self.threshold
        if voiced and tonal_share(frame) > self.tone_share:     # only loud frames pay for the FFT
            voiced = False
            self.tone_frames += 1
        elif not voiced:
            self.floor += 0.05 * (rms - self.floor)             # tone frames are excluded: a beep is not "ambient"
        if not self._speaking:
            self._pre.append(frame)
            if voiced:
                self._blip += 1
                self._blip_peak = max(self._blip_peak, rms)
            self._on = self._on + 1 if voiced else max(0, self._on - 1)
            if self._on == 0 and self._blip:                    # a sound that never became speech: keep for diagnostics
                self.ignored = (self.ignored + [{"s": self._blip * FRAME_S, "peak": self._blip_peak}])[-20:]
                self._blip, self._blip_peak = 0, 0.0
            if self._on >= self._min_on:
                self._speaking, self._frames, self._quiet = True, list(self._pre), 0
                self._blip, self._blip_peak = 0, 0.0
            return None
        self._frames.append(frame)
        self._quiet = 0 if voiced else self._quiet + 1
        if self._quiet >= _frames(self.trail_s) or len(self._frames) >= _frames(MAX_S):
            utterance = np.concatenate(self._frames)
            self._speaking, self._frames, self._on, self._quiet = False, [], 0, 0
            self._pre.clear()
            return utterance
        return None


def to_wav_bytes(samples, rate: int) -> bytes:
    """16 kHz mono 16-bit WAV, in memory. The utterance is peak-normalised (at most 8x) so a quiet laptop microphone
    reaches a clear level; one gain for the whole utterance, so there is no pumping and no clipping."""
    x = np.asarray(samples, dtype=np.float32)
    if rate != WAV_RATE and len(x) > 1:
        k = max(1, round(rate / WAV_RATE))
        if k > 1:
            x = np.convolve(x, np.full(k, 1.0 / k, dtype=np.float32), mode="same")    # crude low-pass before decimating
        n = int(len(x) * WAV_RATE / rate)
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
    peak = float(np.abs(x).max()) if len(x) else 0.0
    if peak > 1e-6:
        x = x * min(8.0, 0.7 / peak)
    pcm = (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(WAV_RATE)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()
