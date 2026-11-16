"""Speech to text through Groq's hosted Whisper (OpenAI-compatible endpoint).

This is the only place audio leaves the machine, and only the speech segment recorded while the fall dialog is open.
The API key comes from the GROQ_API_KEY environment variable or backend/.env and is never logged or put in an error.
"""
from __future__ import annotations

from app.voice import settings

URL = "https://api.groq.com/openai/v1/audio/transcriptions"
MODEL = "whisper-large-v3"
TIMEOUT_S = (3.0, 6.0)      # connect, read: the fall countdown is only 10 s long


class SttError(Exception):
    """Transcription failed (no key, network, rejected, rate limit). The message is safe to show and to log."""


def api_key(env_file=None) -> str | None:
    """GROQ_API_KEY from the environment, else from backend/.env. Read on every call, so editing .env needs no restart."""
    return settings.get("GROQ_API_KEY", None, env_file)


def _speech_text(body: dict) -> str:
    """Whisper invents text for audio that holds no speech ("Thank you." ...). With verbose_json every segment carries
    the model's own no-speech probability and confidence; segments it doubts are dropped (Whisper's own rule)."""
    segments = body.get("segments")
    if isinstance(segments, list) and segments:
        kept = [s.get("text", "") for s in segments
                if not (s.get("no_speech_prob", 0.0) > 0.6 and s.get("avg_logprob", 0.0) < -1.0)]
        return " ".join(t.strip() for t in kept if t.strip())
    return str(body.get("text", "")).strip()


def transcribe(wav: bytes, key: str, post=None) -> str:
    """Returns the recognised text ("" when it is judged not to be speech). Raises SttError. `post` is injectable
    for tests; by default it is requests.post.

    Deliberately no `prompt`: a prompt such as "I am okay" can be echoed back by Whisper on near-silent audio, which
    would cancel a real emergency."""
    if post is None:
        try:
            import requests
        except ImportError:
            raise SttError("the 'requests' package is not installed") from None
        post = requests.post
    try:
        r = post(URL, headers={"Authorization": f"Bearer {key}"},
                 files={"file": ("speech.wav", wav, "audio/wav")},
                 data={"model": MODEL, "language": "en", "temperature": "0", "response_format": "verbose_json"},
                 timeout=TIMEOUT_S)
    except Exception as e:  # noqa: BLE001  requests.RequestException, timeouts, DNS ...
        raise SttError(f"network error ({type(e).__name__})") from None
    code = getattr(r, "status_code", 0)
    if code in (401, 403):
        raise SttError("Groq rejected the API key")
    if code == 429:
        raise SttError("Groq rate limit reached")
    if code != 200:
        raise SttError(f"Groq returned HTTP {code}")
    try:
        body = r.json()
    except Exception:  # noqa: BLE001
        raise SttError("unreadable response from Groq") from None
    return _speech_text(body)
