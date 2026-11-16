# Plan 02 — Offline voice check-in (Sherpa-ONNX) and acknowledgement re-check

Self-contained implementation plan. Read this whole file before editing anything.
Prerequisite: Plan 01 is implemented, because its `timings` telemetry is used here to prove nothing got slower.

## 0. Rules that override everything else

1. **Additive only.** These existing behaviours must work exactly as they do now:
   - A patient who stands up clears SUSPECTED, CONFIRMED, the countdown and a sounding alarm automatically.
   - The fall dialog speaks its prompt, beeps each second, shows the 10 s ring, and has the "I'm OK" button.
   - `POST /api/fall/ack` and the "Test Fall" button / `POST /api/fall/test` keep working.
   - Silence during the countdown ends in the alarm state.
2. **Voice is an extra way to answer, never a requirement.** If the speech package, the model or the microphone
   is missing or fails, the system reports voice as unavailable and everything else behaves as today.
3. **Speech recognition is 100% local.** No audio or text leaves the machine. English only.
4. **The worker frame loop must not get slower.** Recognition runs in a separate OS process. The worker thread
   may only read a flag or call a method that returns immediately.
5. Do not change `fall_detector.py`. Do not change the wording, timing or sounds already in `FallCheck.tsx`;
   only add to it.
6. All existing tests stay green (`21/21 passed` before this plan; more after).
7. Out of scope: database, charts, PDF, Twilio/WhatsApp/SMS, cloud services, WebRTC.

## 1. Project facts you need

- `backend/app/pipeline/escalation.py` — `Escalation` state machine, updated once per frame on the worker
  thread from `Pipeline.process()` (`backend/app/pipeline/core.py`). States: `IDLE`, `COUNTDOWN`, `ALERTED`,
  `ACKNOWLEDGED`. `COUNTDOWN_S = 10.0`. `ack()` and `simulate()` only raise flags that `update()` consumes, which
  is why they are safe to call from API threads. `fired` is True for the one frame in which `ALERTED` is entered.
  `_test` marks a simulated fall (Test Fall button): a simulated countdown is not cancelled by `status == NORMAL`.
- Telemetry carries `fall.alert = {state, seconds_left, impact}`.
- `frontend/src/FallCheck.tsx` — full-screen dialog rendered by `App.tsx` while `fall.alert.state` is `COUNTDOWN`
  or `ALERTED`. On mount and whenever `alerted` changes it calls `say(...)`:
  countdown prompt **"Are you okay? Press I'm OK, or stand up."**, alarm prompt
  **"No response. Please check on the patient now."**
- `backend/app/main.py` — `Worker` thread, FastAPI endpoints, `worker.telemetry()`.
- Models live in `backend/models/` (`MODELS_DIR` in the engine files); `backend/scripts/download_models.py`
  fetches them.
- Tests: `tests/test_analytics.py`, plain `test_*` functions with a runner at the bottom. Helpers there you
  should reuse: `_fall(p)`, `_hold(p, t, pose, seconds)`, `LYING`, `STANDING`.
- The assistant session may have no shell. Commands are for the user to run with the `!` prefix, one per prompt,
  through bash (forward slashes; `cd` persists).

## 2. Architecture

```
Browser (FallCheck.tsx)                      Backend process                         STT process (spawned)
  speaks prompt ── onend ──► POST /api/fall/listen ─► VoiceListener.start() ─ cmd ─►  open mic, decode
  shows badges  ◄── telemetry fall.voice ◄── reader thread ◄──────────── events ───  partial / final text
  buttons ──► /api/fall/ack, /api/fall/escalate      │ classify(final text)
                                                     ├─ "ok"   → escalation.ack()
                                                     └─ "help" → escalation.escalate()
Worker loop: when alert state is no longer COUNTDOWN/ALERTED → VoiceListener.stop()  (one comparison per frame)
```

- **Where the microphone is:** the backend machine's default input device, captured in the STT process with
  `sounddevice`. This assumes the backend runs on the computer in the patient's room, which is the current
  setup. (Streaming a remote browser's microphone to the backend is a later option, not part of this plan.)
- **Why a separate process:** Python threads share one interpreter lock, so decoding audio in a thread would
  take CPU time from the frame loop. A process has its own.
- **Lifecycle:** spawn the process once at backend start (`multiprocessing.get_context("spawn")`, daemon). It
  loads the model, then blocks on its command queue using no CPU. The microphone is opened only between
  `start` and `stop`. Reason for not spawning per fall: model load takes a second or more, and the countdown is
  only 10 s.
- **Failure isolation:** if the process cannot import `sherpa_onnx`/`sounddevice`, find the model or open the
  microphone, it sends an `error` event and idles. The parent sets status `unavailable`. If the process dies,
  the reader thread notices (`process.is_alive()` false) and sets `unavailable`; do not auto-respawn in a loop.

## 3. Dependencies and model

- Add to `backend/requirements.txt`: `sherpa-onnx` and `sounddevice`. Pin to the versions pip resolves at
  install time; do not invent version numbers.
- Model: **`sherpa-onnx-streaming-zipformer-en-2023-06-26`** (streaming English transducer, the English model
  used in the official Sherpa-ONNX docs). Cache path:
  `backend/models/sherpa-onnx-streaming-zipformer-en-2023-06-26/` containing `tokens.txt` and the
  encoder / decoder / joiner `.onnx` files (prefer the `int8` encoder and joiner for CPU speed if provided).
- Extend `backend/scripts/download_models.py` to download and unpack the model archive from the official
  Sherpa-ONNX releases into that folder, skipping if it already exists.
- **Before coding, verify against the official sources** (file names and API differ between releases):
  - https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html
  - https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/speech-recognition-from-microphone-with-endpoint-detection.py
- Expected API shape (confirm in that example): `sherpa_onnx.OnlineRecognizer.from_transducer(tokens=...,
  encoder=..., decoder=..., joiner=..., num_threads=1, sample_rate=16000, feature_dim=80,
  decoding_method="greedy_search", enable_endpoint_detection=True, ...)`, then `create_stream()`,
  `stream.accept_waveform(16000, samples)`, `while recognizer.is_ready(stream): recognizer.decode_stream(stream)`,
  `recognizer.get_result(stream)`, `recognizer.is_endpoint(stream)`, `recognizer.reset(stream)`.
- Settings: `num_threads=1`, CPU provider, microphone at 16 kHz mono float32, blocks of about 100 ms.
  Optional env var `STT_INPUT_DEVICE` to pick a non-default microphone.

## 4. Backend changes

### 4.1 `backend/app/voice/intent.py` (new, pure, no imports beyond `re`)

`classify(text: str) -> str | None` returning `"help"`, `"ok"` or `None`.

1. Normalise: lowercase; replace `i'm` / `im` with `i am`; `can't` / `cant` with `cannot`; strip everything
   except letters and spaces; collapse spaces. (The model may output upper case without punctuation.)
2. **Help / negative first.** Return `"help"` if any of these appear as whole words or phrases:
   `help`, `emergency`, `ambulance`, `hurt`, `hurts`, `pain`, `cannot`, `not okay`, `not ok`, `not fine`,
   `not alright`, `not all right`, `no`.
3. Otherwise return `"ok"` only for a clear multi-word affirmative:
   `i am (okay|ok|fine|alright|all right|good)`.
4. Anything else returns `None`: a bare `okay`, `yes`, `fine`, unrelated speech, or empty text.

Consequences to keep (asymmetric on purpose): a mixed message such as "no, I am fine" returns `"help"`.
A false alarm is acceptable; a wrongly cancelled emergency is not.

### 4.2 `backend/app/voice/stt_process.py` (new)

`run(cmd_q, evt_q, model_dir, device)` — the child process entry point.
- Import `sherpa_onnx` and `sounddevice` **inside** the function; on any failure put
  `("error", "<short reason>")` and keep serving commands (replying `error` to `start`).
- Lower the process priority (Windows: below-normal via `psutil`, already installed with Ultralytics; wrap in
  `try/except`).
- Loop on `cmd_q.get()`:
  - `"start"`: open an `InputStream`; put `("status", "listening")`; feed audio to the recognizer; put
    `("partial", text)` when the text changes; on endpoint put `("final", text)` and reset the stream. Check
    `cmd_q` without blocking between audio blocks.
  - `"stop"`: close the stream, put `("status", "idle")`.
  - `"quit"`: exit.
- Never block the audio callback; push blocks into a local queue and decode in the loop.

### 4.3 `backend/app/voice/listener.py` (new)

`VoiceListener(on_ok, on_help)`:
- `start_process()`: spawn the child; start a daemon reader thread that drains `evt_q`.
- `start()` / `stop()`: put the command; both return immediately. `start()` is ignored if already listening or
  unavailable.
- State readable by telemetry: `status` in `"off" | "unavailable" | "idle" | "listening"`, `transcript` (latest
  partial or final text, cleared on `start()`), `intent` (last classified result).
- Reader thread: on `("final", text)` call `classify(text)`; `"ok"` → `on_ok()`, `"help"` → `on_help()`.
  **Act on final (endpointed) results only**, never on partials: partial hypotheses change as more audio
  arrives.
- `active` property: True while status is `listening`.

### 4.4 `backend/app/pipeline/escalation.py` (additions only)

- New constant `RECHECK_S = 120.0`.
- `escalate()`: sets a flag, like `ack()`. In `update()`: if the flag is set while `COUNTDOWN`, go to `ALERTED`
  with `fired = True`. Ignored in every other state. Consume the flag every frame, as `ack` is.
- **Re-check:** record the time `ACKNOWLEDGED` is entered. While `ACKNOWLEDGED`, if the detector status is still
  not `NORMAL` and `t - acknowledged_at >= RECHECK_S`, go back to `COUNTDOWN` with a fresh
  `_deadline = t + COUNTDOWN_S` and increment a `rechecks` counter. `status == NORMAL` still returns to `IDLE`
  first, exactly as now.
- Output dict gains `"rechecks": int` (0 for a first countdown). Existing keys unchanged.
- Order inside the `COUNTDOWN` branch: `ack` first, then `escalate`, then recovery, then deadline. Keep the
  existing `_test` handling so the Test Fall button still works; a simulated episode that is acknowledged ends
  as it does today (no re-check for simulated falls: `_test` is cleared when the state leaves
  `COUNTDOWN`/`ALERTED`, and detector status is `NORMAL`, so it returns to `IDLE`).

### 4.5 `backend/app/main.py`

- Create one `VoiceListener` next to `worker`, with
  `on_ok=worker.pipeline.escalation.ack` and `on_help=worker.pipeline.escalation.escalate`.
  (`Pipeline` is created once in `Worker.__init__` and only mutated afterwards, so these bound methods stay valid.)
  Start its process in `lifespan` before `worker.start()`; send `quit` on shutdown.
- `POST /api/fall/listen`: if the current alert state is `COUNTDOWN` or `ALERTED`, call `voice.start()`.
  Returns `{"ok": true, "status": voice.status}`. Otherwise does nothing.
- `POST /api/fall/escalate`: calls `worker.pipeline.escalation.escalate()` (manual "Trigger Emergency").
- In `Worker.run()`, after `self.pipeline.process(...)`, one line:
  `if voice.active and self.pipeline.last["fall"]["alert"]["state"] not in ("COUNTDOWN", "ALERTED"): voice.stop()`.
  This is the only addition to the frame loop.
- `Worker.telemetry()`: add `voice` inside the `fall` dict when it exists:
  `{"status": voice.status, "transcript": voice.transcript, "intent": voice.intent}`. Build a shallow copy of
  `fall` for this; do not mutate `pipeline.last` from the API thread.

## 5. Frontend changes (`frontend/src/FallCheck.tsx`, additions only)

- **Start listening after the prompt ends.** In the existing `say()` call path, attach `onend` (and `onerror`)
  to the utterance: 300 ms after it fires, `POST /api/fall/listen`. Fallback: if neither fires within 6 s
  (speech blocked or unsupported), post anyway. Do this for both the countdown prompt and each re-check prompt.
  **Why this matters: the existing prompt contains the words "I'm OK". If the microphone were open while it
  plays, the system could cancel a real emergency by hearing itself.**
- **Status badge** under the paragraph text, driven by `t.fall.voice`:
  - `listening`, empty transcript → "Listening…" with a small pulsing dot
  - `listening`, transcript present → `Heard: "…"`
  - `unavailable` or `off` → "Voice unavailable — use the buttons"
  - nothing rendered if `voice` is absent (older backend)
- **Buttons:** keep the existing emerald "I'm OK" button unchanged. Add a second, red-outlined
  "Trigger Emergency" button beside it that posts `/api/fall/escalate`; show it only during `COUNTDOWN`.
- **Re-check wording:** when `alert.rechecks > 0`, the heading may read "Still down — are you okay?"; the spoken
  prompt, ring, beeps and buttons are the same as for a first countdown.
- **Beeps stay as they are.** They will reach the microphone; acceptance check 7 verifies they do not produce a
  false intent. Only if that check fails, lower the beep gain while `voice.status === "listening"`.
- `frontend/src/useTelemetry.ts`: add `rechecks?: number` to `FallAlert` and
  `voice?: { status: "off" | "unavailable" | "idle" | "listening"; transcript: string; intent: string | null }`
  to the `fall` type.
- `App.tsx` passes `voice` to `FallCheck` alongside `alert`. No other dashboard change.

## 6. Unit tests (add to `tests/test_analytics.py`)

**Intent parsing** — table-driven, `classify` imported from `app.voice.intent`:

| Input | Expected |
|---|---|
| `"help"`, `"HELP ME"`, `"I need help"`, `"call an ambulance"` | `"help"` |
| `"I am not okay"`, `"I'm not ok"`, `"no"`, `"I can't get up"`, `"it hurts"` | `"help"` |
| `"no I am fine"` (mixed) | `"help"` |
| `"I am okay"`, `"I'M OKAY"`, `"yes I am fine"`, `"im alright"`, `"i am all right"` | `"ok"` |
| `"okay"`, `"yes"`, `"fine"`, `""`, `"are you okay press i am ok or stand up"`* | `None` / see note |

\* The echoed prompt contains "i am ok" and would classify as `"ok"`. Keep this row as a test that documents the
risk: assert it returns `"ok"`, with a comment that the defence is the listen-after-prompt rule in section 5,
not the parser.

**Escalation** — use `Pipeline` with the existing helpers:

- `escalate()` during `COUNTDOWN` → `ALERTED` on the next frame, `fired` True exactly once.
- `escalate()` while `IDLE` does nothing, and does not pre-arm a later fall.
- Acknowledge during the countdown, stay `LYING` for 121 s → back to `COUNTDOWN` with `rechecks == 1` and
  `seconds_left` near 10; stay down unanswered 10 s more → `ALERTED`.
- Acknowledge, then stand up within the 2 minutes → `IDLE`, and no re-check ever fires.
- Acknowledge twice across two re-checks → `rechecks == 2`.
- All existing escalation tests still pass unmodified. If one needs changing, stop and report why instead of
  editing it.

**Manual smoke script** `backend/scripts/stt_smoke.py`: starts the STT process, listens for 15 s, prints each
partial, final and its `classify` result. Used for acceptance check 5.

## 7. Acceptance checks (all must pass)

1. Tests: previous count plus the new ones, all passing. `npx tsc --noEmit` clean.
2. **Without the model or packages installed:** backend starts normally; "Test Fall" shows the dialog with
   "Voice unavailable — use the buttons"; prompt, beeps, countdown, "I'm OK" and alarm all work as before.
3. **Stand-up recovery unchanged:** real floor posture → countdown → stand up → dialog closes by itself.
4. **Listen starts after the prompt:** press "Test Fall"; the badge changes to "Listening…" only after the
   spoken prompt has finished. With speakers at normal volume, 10 trials in silence: the countdown is never
   cancelled by the prompt itself.
5. `stt_smoke.py` at the patient's real distance from the microphone: "I am okay" and "help" are transcribed
   and classified correctly in most attempts. Record the hit rate in the activity log; it is expected to drop
   with distance.
6. **Voice paths:** say "I am okay" → countdown cancels (state `ACKNOWLEDGED`, or `IDLE` for a simulated fall);
   say "help" → alarm state immediately; say "okay" alone → countdown continues; say nothing → alarm at 0.
7. **Beeps do not trigger intents:** 20 "Test Fall" countdowns left to run in silence: none cancelled or
   escalated early.
8. **Re-check:** lie down, answer "I'm OK" (button or voice), stay down: the dialog returns after 2 minutes.
9. **Performance:** using the `timings` telemetry from Plan 01, compare 60 s idle with a "Test Fall" countdown
   while listening: `infer_ms`, `pipeline_ms`, `draw_ms`, `encode_ms` and `fps` stay within normal noise of the
   Plan 01 reference. With the dialog closed, the STT process uses no measurable CPU.
10. "Trigger Emergency" moves a countdown to the alarm state at once.

## 8. Known limits to state in the UI help or handover notes

- A laptop microphone a few metres from a person on the floor will miss quiet speech. Silence still escalates.
- Another person or a television saying "I am okay" can cancel the countdown; the 2-minute re-check is the
  backstop.
- English only. Accented or slurred speech lowers recognition; ambiguous results never cancel.

## 9. Files touched

| File | Change |
|---|---|
| `backend/requirements.txt` | `sherpa-onnx`, `sounddevice` |
| `backend/scripts/download_models.py` | fetch the STT model |
| `backend/scripts/stt_smoke.py` (new) | manual microphone check |
| `backend/app/voice/__init__.py`, `intent.py`, `stt_process.py`, `listener.py` (new) | voice subsystem |
| `backend/app/pipeline/escalation.py` | `escalate()`, 2-minute re-check, `rechecks` in output |
| `backend/app/main.py` | listener wiring, two endpoints, one stop check in the loop, `fall.voice` telemetry |
| `frontend/src/FallCheck.tsx` | listen-after-prompt, status badge, "Trigger Emergency" |
| `frontend/src/useTelemetry.ts`, `frontend/src/App.tsx` | types, pass `voice` through |
| `tests/test_analytics.py` | intent and escalation tests |
| `docs/activity_log.md` | results of the acceptance checks |

Suggested order: 4.1 + tests → 4.4 + tests → 4.2/4.3 + smoke script → 4.5 → frontend → acceptance checks.
Commit after each step that leaves the tests green.
