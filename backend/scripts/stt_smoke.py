"""Manual check of the voice check-in: microphone -> speech detection -> Groq Whisper -> intent. No backend needed.

  backend/.venv/Scripts/python backend/scripts/stt_smoke.py             listen for 25 s; each utterance goes to Groq
  backend/.venv/Scripts/python backend/scripts/stt_smoke.py --beeps     the same, with the dialog's countdown tick played
                                                                        every second, to prove beeps are not sent
  backend/.venv/Scripts/python backend/scripts/stt_smoke.py --classify help helpi aelp bachao hello
                                                                        what intent each text maps to (no mic, no network)
  backend/.venv/Scripts/python backend/scripts/stt_smoke.py --devices   which input device actually hears you?

Needs GROQ_API_KEY (environment or backend/.env). Without it only the speech detection is tested and nothing is sent.
Run the listening modes in a normal terminal window, NOT through Claude Code's `!` prefix: that only shows output when
the command has finished, so you cannot see when to speak.
"""
import queue
import sys
import time
from contextlib import ExitStack
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.voice import groq_stt, recorder  # noqa: E402
from app.voice.intent import classify  # noqa: E402

LISTEN_S = 25


def classify_texts(texts) -> int:
    for t in texts:
        print(f"{t!r:30} -> {classify(t)}")
    return 0


def list_devices() -> int:
    """--devices: opens every input device of the default audio API for 1 s and prints its loudest sample.
    Keep talking (or play music) while it runs; the device with a clearly non-zero peak is the one to use."""
    import sounddevice as sd
    default = sd.query_devices(kind="input")
    print(f"Default input device: [{default['index']}] {default['name']}")
    print("Measuring each input device for 1 s: keep talking until this finishes.\n")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] < 1 or d["hostapi"] != default["hostapi"]:
            continue                                    # other audio APIs list the same microphones again
        rate = int(d["default_samplerate"])
        try:
            rec = sd.rec(rate, samplerate=rate, channels=1, dtype="float32", device=i)
            sd.wait()
        except Exception as e:  # noqa: BLE001
            print(f"[{i}] {d['name']}: cannot open ({type(e).__name__})")
            continue
        mark = "   <- current default" if i == default["index"] else ""
        print(f"[{i}] {d['name']}: peak {float(abs(rec).max()):.3f}{mark}")
    print("\nUse the best one: put its number or a part of its name in backend/.env, e.g.  STT_INPUT_DEVICE=KORP THALLIUM")
    print("(several choices separated by commas are tried in order: STT_INPUT_DEVICE=KORP THALLIUM, Realtek).")
    print("The backend reads it at the next fall check, no restart needed.")
    return 0


def _tick(rate: int = 44100):
    """The dialog's countdown tick: a 0.15 s, 880 Hz square wave fading out, at gain 0.2 (see FallCheck.tsx)."""
    import numpy as np
    n = int(0.15 * rate)
    envelope = 0.2 * np.exp(np.linspace(0.0, np.log(0.001 / 0.2), n))
    return (np.sign(np.sin(2 * np.pi * 880 * np.arange(n) / rate)) * envelope).astype(np.float32)


def listen(beeps: bool) -> int:
    import sounddevice as sd
    key = groq_stt.api_key()
    print("GROQ_API_KEY found: every utterance is sent to Groq for transcription." if key else
          "No GROQ_API_KEY (backend/.env): only the speech detection is tested, nothing is sent.")
    preferences = recorder.device_preferences()
    heard = 0
    totals = []                                         # seconds from the end of speech to the answer, per utterance
    tick, next_tick = _tick(), 0.0
    with ExitStack() as stack:
        try:
            blocks, rate, device_name = stack.enter_context(recorder.open_stream(sd, preferences))
        except Exception as e:  # noqa: BLE001  device not found or busy: say so, and show the choices
            print(f"Could not open the microphone: {e}")
            return 1
        segmenter = recorder.Segmenter(rate)
        print(f"input {device_name} (STT_INPUT_DEVICE: {', '.join(map(str, preferences)) or 'not set: the Windows default'})"
              f" at {rate} Hz, speech threshold "
              f"{segmenter.threshold:.4f} (RMS), speech starts after {segmenter.min_on_s:g} s, "
              f"an utterance ends after {segmenter.trail_s:g} s of silence")
        print(f"speak now for {LISTEN_S} s: \"help\", \"I am fine\", \"I am okay\", \"I am not okay\" ..."
              + ("  (countdown ticks are playing)" if beeps else "") + "\n")
        end = time.time() + LISTEN_S
        while time.time() < end:
            if beeps and time.time() >= next_tick:
                sd.play(tick, 44100)                    # non-blocking, like the dialog's tick
                next_tick = time.time() + 1.0
            try:
                block = blocks.get(timeout=0.2)
            except queue.Empty:
                continue
            utterance = segmenter.feed(block)
            if utterance is None:
                continue
            heard += 1
            print(f"utterance {heard}: {len(utterance) / rate:.1f} s of audio")
            if not key:
                continue
            started = time.time()
            wav = recorder.to_wav_bytes(utterance, rate)
            encoded = time.time() - started
            try:
                text = groq_stt.transcribe(wav, key)
            except groq_stt.SttError as e:
                print(f"  transcription failed: {e}")
                continue
            groq_s = time.time() - started - encoded
            total = segmenter.trail_s + encoded + groq_s
            totals.append(total)
            print(f"  {text!r} -> {classify(text)}")
            print(f"  about {total:.2f} s after you stopped speaking: {segmenter.trail_s:g} s of silence waited"
                  f" + {encoded * 1000:.0f} ms encoding + {groq_s:.2f} s Groq")
    print(f"\n{heard} utterance(s) sent. Ambient level {segmenter.floor:.4f}, loudest frame {segmenter.loudest:.4f}, "
          f"speech threshold {segmenter.threshold:.4f}.")
    if totals:
        print(f"Response time after you stop speaking: average {sum(totals) / len(totals):.2f} s, slowest {max(totals):.2f} s"
              f" ({segmenter.trail_s:g} s of that is the silence wait; set VOICE_TRAIL_S to change it).")
    print(f"{segmenter.tone_frames} tone-like frame(s) ignored as beeps.")
    if beeps and segmenter.tone_frames == 0:
        print("  The ticks were not recognised as beeps (or the microphone did not hear them): if the utterances above"
              " were just ticks, lower VOICE_TONE_SHARE (default 0.65) so more sounds count as tones.")
    if segmenter.ignored:
        print("Sounds that were too short to count as speech (a quick word that shows up here was NOT sent):")
        for s in segmenter.ignored[-10:]:
            print(f"  {s['s']:.2f} s, loudest frame {s['peak']:.3f}")
        print(f"If one of these was your word, lower VOICE_MIN_ON_S (now {segmenter.min_on_s:g} s; speech is counted in"
              f" 20 ms frames). A word that is not listed at all never rose above the threshold "
              f"({segmenter.threshold:.4f}): lower VOICE_MIN_RMS. Both can go in backend/.env.")
    if heard == 0:
        if segmenter.loudest < 0.002:
            print("The microphone delivered almost pure silence: check it is not muted, that Windows Settings > Privacy >"
                  " Microphone allows desktop apps, and run with --devices to pick the right input.")
        else:
            print("Sound reached the microphone but never counted as speech: speak closer / louder, or lower the"
                  " threshold with VOICE_MIN_RMS (see backend/.env.example).")
    return 0


if __name__ == "__main__":
    if "--classify" in sys.argv:
        sys.exit(classify_texts(sys.argv[sys.argv.index("--classify") + 1:]))
    sys.exit(list_devices() if "--devices" in sys.argv else listen("--beeps" in sys.argv))
