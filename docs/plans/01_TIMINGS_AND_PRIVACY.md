# Plan 01 — Stage timings and Privacy Mode

Self-contained implementation plan. Read this whole file before editing anything.

## 0. Rules that override everything else

1. **Additive only.** Do not remove, rename or change the behaviour of any existing feature: rep counting, the
   ten exercises, the Exercise Guide, fall detection (5 s dwell, stand-up auto-recovery), the fall dialog
   (`FallCheck.tsx`: spoken prompt, beeps, 10 s countdown, "I'm OK"), the "Test Fall" button and
   `POST /api/fall/test`, the camera picker, the engine toggle.
2. **Do not modify** `backend/app/pipeline/annotate.py`, `fall_detector.py`, `escalation.py`, `rep_counter.py`,
   `kinematics.py`, or `frontend/src/FallCheck.tsx` / `ExerciseGuide.tsx` in this plan.
3. **The worker frame loop must not get slower.** The only additions allowed inside it are
   `time.perf_counter()` calls, a few float operations, and one in-place frame fill when privacy is on.
4. Existing tests must stay green: `tests/test_analytics.py` currently reports `21/21 passed`.
5. Out of scope: database, charts, PDF, Twilio/WhatsApp, speech recognition (that is Plan 02).

## 1. Project facts you need

- Backend: FastAPI, one `Worker` thread in `backend/app/main.py` that owns camera, engine and pipeline.
  Per frame it does: `cap.read()` → `engine.infer()` → `self.pipeline.process()` → `draw()` (only if a pose was
  found) → `cv2.imencode()` → publish `self.jpeg`.
- Telemetry: `Worker.telemetry()` returns a dict sent 20 times a second over `/ws/telemetry` and by
  `GET /api/telemetry`. It already has `fps` and `latency_ms` (inference time only).
- Settings from the UI go through `POST /api/config` (`Config` model: `engine`, `exercise`).
- Frontend: `frontend/src/App.tsx` (dashboard), `frontend/src/useTelemetry.ts` (the `Telemetry` type).
  A `post(url, body)` helper exists at the top of `App.tsx`.
- The assistant session may have no shell. Commands below are for the user to run with the `!` prefix, one per
  prompt, through bash: use forward slashes; `cd` persists between commands.
  - Tests (from repo root): `backend/.venv/Scripts/python tests/test_analytics.py`
  - Type-check: `cd frontend && npx tsc --noEmit; cd ..`

## 2. Part A — per-stage timing telemetry

### Backend (`backend/app/main.py`)

In `Worker.__init__` add:

```python
self.timings = {"capture_ms": 0.0, "infer_ms": 0.0, "pipeline_ms": 0.0, "draw_ms": 0.0, "encode_ms": 0.0}
```

In `Worker.run()`, take `time.perf_counter()` marks at the stage boundaries and smooth each stage the same way
`fps` is smoothed (`new = 0.9 * old + 0.1 * sample`). Sketch of the loop body, keeping every existing line:

```python
tc = time.perf_counter()                       # NEW: before cap.read()
ok, frame = cap.read() if cap is not None else (False, None)
if not ok:
    time.sleep(0.01)
    continue
t1 = time.perf_counter()                       # existing
res = engine.infer(frame, int((t1 - t0) * 1000))
t2 = time.perf_counter()                       # NEW
self.latency_ms = (t2 - t1) * 1000             # same value as before, reuse t2
self.pose_visible = res is not None
self.pipeline.process(res, t1 - t0)
t3 = time.perf_counter()                       # NEW
# (privacy blanking goes here, see Part B)
if res:
    draw(frame, res)
t4 = time.perf_counter()                       # NEW
ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
now = time.perf_counter()                      # existing
for key, dt in (("capture_ms", t1 - tc), ("infer_ms", t2 - t1), ("pipeline_ms", t3 - t2),
                ("draw_ms", t4 - t3), ("encode_ms", now - t4)):
    self.timings[key] = 0.9 * self.timings[key] + 0.1 * dt * 1000
```

Notes:
- `capture_ms` is mostly **waiting for the camera's next frame**, not CPU work. Document this in a comment so
  nobody tries to "optimise" it. The other four are real processing cost.
- Privacy blanking (Part B) is counted inside `draw_ms`, which is the honest place for it.
- `pipeline.process()` must keep receiving `t1 - t0` exactly as now.

In `Worker.telemetry()` add one key: `"timings": {k: round(v, 2) for k, v in self.timings.items()}`.
Do not rename or remove `fps` or `latency_ms`.

### Frontend

- `frontend/src/useTelemetry.ts`: add to `Telemetry`
  `timings?: { capture_ms: number; infer_ms: number; pipeline_ms: number; draw_ms: number; encode_ms: number };`
- `frontend/src/App.tsx`: directly under the existing 2×2 grid of cards (Mobility, Fall status, FPS, Latency),
  add one small line, same muted style as other helper text (`text-[10px] text-slate-500 tabular-nums`):
  `capture 31.2 · infer 18.4 · pipeline 0.3 · draw 0.4 · encode 2.1 ms`. Render it only when `t?.timings`
  exists. Do not change the existing cards.

## 3. Part B — Privacy Mode

### Behaviour

When privacy is on, the stream shows the skeleton and box on a black background. The raw camera image is never
JPEG-encoded and never leaves the backend. Inference, rep counting, fall detection, telemetry and every on-screen
metric behave exactly as with privacy off, because the frame is blanked **after** inference and
`pipeline.process()`.

### Backend (`backend/app/main.py`)

- `Worker.__init__`: `self.privacy = False`.
- `Config` model: add `privacy: bool | None = None`.
- `config()` handler: `if cfg.privacy is not None: worker.privacy = cfg.privacy` (a plain bool read once per
  frame by the worker; no lock needed, same as other flags).
- `Worker.telemetry()`: add `"privacy": self.privacy`.
- In `Worker.run()`, between the `t3` mark and `if res: draw(frame, res)`:

```python
if self.privacy:
    frame[:] = 0        # in place: nothing of the camera image is encoded. Must run even when res is None.
```

`annotate.draw()` is unchanged, so the skeleton and box look exactly as they do today, just on black.
Do not create a new array per frame (no `np.zeros_like`); the in-place fill is the cheapest option.

### Frontend (`frontend/src/App.tsx`)

- **Toggle placement:** in the header, immediately to the left of the existing amber "Test Fall" button. Move
  the `ml-auto` class from "Test Fall" to the new switch so the group stays right-aligned. Style: a pill button
  with `role="switch"` and `aria-checked`, label "Privacy", a small track-and-knob that is emerald when on and
  slate when off (same visual language as the rest of the header).
- **State:** a boolean `privacyPref` initialised from `localStorage` key `privacy` (`"1"`/`"0"`, default off),
  wrapped in `try/catch` so blocked storage falls back to in-memory. Clicking toggles it and saves it.
- **Sync with the backend** (the backend is the one that applies it):

```tsx
const backendPrivacy = t?.privacy;
useEffect(() => {
  if (backendPrivacy !== undefined && backendPrivacy !== privacyPref) post("/api/config", { privacy: privacyPref });
}, [backendPrivacy, privacyPref]);
```

  This posts once per change, and again automatically after a backend restart, not on every telemetry tick.
- **No leak on restart:** while `privacyPref` is true but `t?.privacy` is not yet true, hide the video element
  (add the `invisible` class to the existing `<img src="/api/video_feed">`). Do not unmount it or change its
  `src`. The "LIVE" badge and everything else stay as they are.
- `frontend/src/useTelemetry.ts`: add `privacy?: boolean;` to `Telemetry`.

### What must not change

- With privacy off, the stream is pixel-for-pixel what it is today.
- The fall dialog, Exercise Guide, camera picker and engine toggle are untouched.
- No raw frames are written to disk by this plan (none are today either).

## 4. Tests

Add to `tests/test_analytics.py` (plain functions named `test_*`; the file has its own runner at the bottom):

- Nothing in the analytics pipeline changes, so no pipeline test is required. Do **not** import
  `app.main` in tests (it opens the camera and loads the engine on start).
- The privacy guarantee is one line in the worker; verify it live (below) rather than with a mock.

## 5. Acceptance checks (all must pass)

1. `backend/.venv/Scripts/python tests/test_analytics.py` → `21/21 passed`.
2. `cd frontend && npx tsc --noEmit; cd ..` → no output.
3. **Baseline before any code change:** with the backend running on the current code, note `fps` and
   `latency_ms` from the dashboard for about 60 s of normal use (standing, a few reps). Record them in
   `docs/activity_log.md`.
4. **After Part A, privacy off:** `fps` and `latency_ms` match the baseline within normal run-to-run noise
   (a few percent). Record the five `timings` values for 60 s in the activity log; this is the reference for
   Plan 02.
5. **Privacy on:** the stream shows only skeleton and box on black; `encode_ms` is not higher than with privacy
   off (it should be lower); `fps` is not lower.
6. **Behaviour parity, privacy on vs off:** count 5 reps of one exercise in each mode: same count. Press
   "Test Fall": the "Are you okay?" dialog, beeps, countdown and "I'm OK" all work in both modes.
7. **Restart safety:** turn privacy on, restart the backend, reload nothing. The video stays hidden until the
   stream is black again; no raw frame is visible at any point.
8. `GET http://localhost:8000/api/telemetry` contains `timings` (five keys) and `privacy`.

## 6. Files touched

| File | Change |
|---|---|
| `backend/app/main.py` | timing marks, `timings` and `privacy` in telemetry, `Config.privacy`, one blanking line |
| `frontend/src/useTelemetry.ts` | `timings` and `privacy` in the `Telemetry` type |
| `frontend/src/App.tsx` | Privacy switch, sync effect, hidden-until-confirmed image, timings line |
| `docs/activity_log.md` | baseline numbers and what was done |

Commit message suggestion: `Add per-stage timing telemetry and Privacy Mode (skeleton-only stream)`.
