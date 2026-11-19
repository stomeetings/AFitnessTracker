# Activity Log & Session Handoff

## Session 1 — 2026-11-13 (office PC)

### Completed

- **Workspace scaffolding:** empty folders `docs/`, `backend/`, `frontend/`, `tests/` exist. No application code, no virtual environment, no git repository yet.
- **R&D model evaluation:** compared MediaPipe Pose Landmarker with Ultralytics YOLO11n-pose (and `solutions.AIGym`) for CPU speed, bounding-box stability, and fall/walking flexibility. The comparison is from published figures; **no benchmark has been run**.
- **Architecture lock:** threaded anti-lag pipeline, MJPEG + WebSocket transport, separate-process upload jobs, H.264 output, SQLite schema, API surface, and P0–P3 roadmap agreed.
- **Blueprint written:** `docs/plan.md` holds the full plan and all finalized decisions.

### Finalized decisions (details in `docs/plan.md`)

- Primary engine MediaPipe Pose Landmarker behind an abstract `PoseEngine` interface; Ultralytics YOLO-Pose as a swappable, CUDA-capable backend.
- Live webcam mode and video-upload demo/validation mode share one pipeline.
- UI exercise selector: L/R Elbow Curl, Shoulder Raise, Knee Flexion/Extension, Hip Flexion, plus Sit-to-Stand. Mobility tracking and fall detection (`NORMAL -> SUSPECTED -> CONFIRMED`) are always on.
- Recording Mode A (annotated H.264 mp4 + SQLite/JSON) and Mode B (privacy: no video saved, uploaded source deleted after processing).
- Frontend: Vite + React + TypeScript + Tailwind CSS + Recharts.

### Hardware context shift

| | Previous (office PC) | Target (personal PC) |
|---|---|---|
| CPU | Intel i5-9500, 6 cores / 6 threads | Not yet recorded |
| GPU | Intel UHD 630 (integrated only) | Dedicated 4 GB GPU, CUDA (model not yet recorded) |
| Camera | None | Dedicated webcam |
| Software seen | Python 3.12.10, Node 24.18, no ffmpeg | To be checked |

Consequences:

- Live webcam testing becomes possible for the first time.
- The YOLO backend can run on CUDA. MediaPipe's Windows Python package still runs on CPU.
- All speed estimates in the plan were made with the office CPU in mind and must be re-measured.

### Next step for the new session: execute Phase P0

1. **Check the machine:** CPU, GPU model, NVIDIA driver and CUDA version (`nvidia-smi`), Python and Node versions, webcam index.
2. **Create the Python virtual environment** in `backend/`.
3. **Install dependencies:** MediaPipe, the CUDA-enabled PyTorch build matching the driver, Ultralytics, OpenCV, FastAPI, Uvicorn, NumPy, `imageio-ffmpeg`. Verify `torch.cuda.is_available()`.
4. **Implement the `PoseEngine` interface** with MediaPipe and YOLO backends.
5. **Benchmark the engines:** MediaPipe lite/full on CPU, YOLO-Pose on CPU and on CUDA. Save the FPS/latency table in `docs/`.
6. **Run the baseline webcam test:** live skeleton + bounding box on screen.
7. **Confirm or revise the engine choice** from the measurements, then scaffold FastAPI and the Vite frontend with the MJPEG stream.

## Session 2 — 2026-11-13 (personal PC)

### Completed

- **Hardware recorded** (see `docs/plan.md` section 12): i5-7300HQ, 16 GB, GTX 1050 4 GB (Pascal sm_61, driver 582.78), integrated webcam, Python 3.13.5, Node 22.16.0.
- **Six open questions answered** and written into `docs/plan.md` section 12.
- **P0 step 1 done:** `backend/.venv` on Python 3.13.5 (MediaPipe 1.2.3 has a 3.13 wheel, so no 3.12 fallback was needed). `backend/requirements.txt` pinned from the installed set; torch 2.14.1+cu126 / torchvision 0.29.1+cu126 from the cu126 index.
- **Verified** with `backend/scripts/verify_env.py`: MediaPipe imports, `torch.cuda.is_available()` True, GPU "NVIDIA GeForce GTX 1050" capability (6, 1), CUDA matmul OK, webcam index 0 opens at 640x480.

### Notes

- The first torch download dropped with `error: connection-failed`; the retry with `--resume-retries` succeeded. Install torch before ultralytics so pip does not pull the CPU build from PyPI.
- Both `opencv-python` and `opencv-contrib-python` 5.0.0.93 are installed (MediaPipe wants contrib, Ultralytics wants plain). Same version, harmless; only `opencv-python` is listed in requirements.
- Ultralytics also installed `polars`, `psutil`, `nvidia-ml-py` as its own dependencies.

### P0 step 2 done

- `backend/app/engines/`: `base.py` (`PoseEngine`, `PoseResult`, `Keypoint`, `BBox`, `Capabilities`, common joint schema, `SKELETON`), `mediapipe_engine.py` (lite/full, VIDEO mode, 3D world landmarks, bbox derived from landmarks), `yolo_engine.py` (yolo11n-pose, CUDA or CPU, native box), `make_engine()` factory.
- `backend/scripts/download_models.py`, `benchmark_engines.py`, `webcam_baseline.py`. Root `.gitignore` excludes `.venv`, `models/`, `data/`.
- MediaPipe 1.2.3 keeps the 0.10 Tasks API (`PoseLandmarker.detect_for_video`), so no API changes were needed.
- **Benchmark** (100 webcam frames, 640x480, inference only) saved to `docs/benchmark_results.md`:

| Engine | FPS | Median ms | p95 ms | Hit rate % | Peak VRAM MB |
|---|---|---|---|---|---|
| mediapipe-lite | 57.7 | 16.3 | 21.6 | 59 | 0 |
| mediapipe-full | 48.2 | 20.8 | 26.9 | 59 | 0 |
| yolo11n-pose-cuda | 45.6 | 20.4 | 29.2 | 76 | 82 |
| yolo11n-pose-cpu | 16.1 | 60.5 | 71.9 | 76 | 0 |

- Reading: MediaPipe full on CPU is as fast as YOLO on the GTX 1050 and well above the 25 FPS target, so the primary-engine choice stands. The hit-rate gap (59 vs 76) came from a partially framed subject during capture; YOLO detects partial bodies more readily, MediaPipe wants most of the body visible. Re-check with a full-body side view before P1.
- Workarounds: the venv's `requests` fails SSL verification against github.com (`CERTIFICATE_VERIFY_FAILED`), so Ultralytics' auto-download failed; the weights were fetched with Windows `curl`. `download_models.py` still handles the MediaPipe files; use curl for YOLO if it fails again. MediaPipe prints many harmless "Sparsity tensors" warnings on load.

### P0 step 3 done (webcam baseline confirmed by user first)

- `backend/app/main.py`: FastAPI with CORS; one `Worker` thread owns the camera and engine, keeps only the latest JPEG (no queue), supports hot engine swap between frames. Endpoints: `GET /api/health`, `GET /api/exercises`, `GET /api/telemetry` (polled JSON for P0; WebSocket arrives in P1 with fall alerts), `POST /api/config` (`engine`, `exercise`), `GET /api/video_feed` (MJPEG, capped 25 FPS).
- `backend/app/pipeline/annotate.py`: shared `draw()` (bbox, lines, dots only); `webcam_baseline.py` now imports it.
- `frontend/`: hand-written Vite 7 + React 19 + TypeScript + Tailwind 4 scaffold (no interactive `create-vite`), `lucide-react` and `recharts` installed. Dev server proxies `/api` to port 8000. `src/App.tsx` is the 2-column dashboard: MJPEG stage with status badge; sidebar with engine toggle, exercise select, FPS / latency / reps / fall cards. `tsc -b` and `vite build` pass.
- Smoke test: telemetry at ~30 FPS (camera-bound), 2 MB of MJPEG in 2 s, live switch to `yolo` reports `device cuda:0` with pose tracked.
- Note: the first smoke run reported `cannot open camera 0` while another process held the webcam. Only one process can own the camera; close any `webcam_baseline.py` window before starting the server.

### Launch

```
cd backend && .venv\Scripts\uvicorn app.main:app --port 8000
cd frontend && npm run dev        # http://localhost:5173
```

## Session 2 (continued) — Phase P1 done

### Modules (`backend/app/pipeline/`)

- `kinematics.py`: `angle_deg` (2D or 3D), `joint_angles(PoseResult)` for L/R elbow, shoulder, knee, hip plus `knee_mean` / `hip_mean`; uses MediaPipe world coords when present, image coords otherwise. `EMA(alpha=0.35)` per key; absent keys are not returned so FSMs freeze on occlusion.
- `rep_counter.py`: `LimbFSM` on normalised progress p = (angle - rest) / (peak - rest), so one FSM covers flexion (angle falls) and raises / Sit-to-Stand (angle rises). Hysteresis bands: start 0.20 / abort 0.10, peak in 0.80 / out 0.60, done 0.15; rep duration 0.4–10 s. ROM = max - min over the rep including the rest extreme seen while IDLE. Per-exercise rest/peak table in `EXERCISES`. Sit-to-Stand uses `knee_mean` (bilateral).
- `mobility.py`: ankle separation / torso length with 0.45 / 0.25 hysteresis counts steps (min 0.3 s apart); cadence over a 10 s window; `Walking` when hip moved > 0.5 torso in 2 s with a recent step, else `Active Exercise` (rep FSM busy) or `Stationary`.
- `fall_detector.py`: tilt (shoulder-centre to hip-centre vs vertical), hip downward velocity in torso lengths / s over 0.3 s, bbox w/h. `NORMAL -> SUSPECTED` on drop > 1.2 or tilt > 55° or aspect > 1.2; `-> CONFIRMED` after 1.5 s horizontal and still (or pose lost within 3 s of a drop); any sustained-upright 1.0 s returns to `NORMAL`; 5 s timeout clears a stale `SUSPECTED`. Known trade-off: a slow lie-down also confirms (see `ponytail:` note).
- `core.py`: `Pipeline.process(res, t_seconds)` runs EMA, reps, mobility, fall and posture (`standing` / `sitting` / `lying` / `unknown`) and returns one dict; same core will serve upload mode in P2.

### Server and UI

- `main.py`: worker runs the pipeline per frame; `POST /api/reset`; `WS /ws/telemetry` pushes the full telemetry at 20 Hz. Exercise switch and reset are applied by the worker thread between frames. YOLO now picks the largest person (single-patient rule); MediaPipe is `num_poses=1`.
- Frontend: `useTelemetry.ts` WebSocket hook with 1 s auto-reconnect; `App.tsx` shows per-side rep cards (count, FSM state, live angle bar, last / best ROM), mobility and fall cards, Reset button, and a top banner: amber for `SUSPECTED`, flashing red for `CONFIRMED`. Vite proxies `/ws` with `ws: true`.

### Verification

- `tests/test_analytics.py` (plain asserts, no framework): 8/8 pass. Covers 90° / 180° angles, 5 clean and 5 jittered (±4°) elbow reps per side with ROM, partial-range and too-fast rejection, Sit-to-Stand and Shoulder Raise direction, `NORMAL -> SUSPECTED -> CONFIRMED -> NORMAL` on a synthetic drop, confirm-on-pose-lost, recovery after a bend, and `Pipeline` posture / angles.
- Frontend `tsc -b` and `vite build` clean. WS probe on a scratch instance: 40 packets in 2 s with `angles`, `reps`, `mobility`, `fall`, `posture` keys; `/api/reset` returns `{"ok": true}`.
- Two bugs found by the tests and fixed: ROM excluded the rest angle; a single upright frame mid-fall cleared `SUSPECTED`.

Run tests: `backend\.venv\Scripts\python tests\test_analytics.py` from the repo root.

### P1.1 — fall detector false positives fixed (sitting / bending)

- Problem: tilt > 55° or a wide bbox alone raised `SUSPECTED`, so sitting down or bending to pick something up triggered alerts.
- `fall_detector.py` rewritten around one rule: **fallen = horizontal AND low**. `low` = hip-centre more than 1.2 torso lengths below the standing baseline (highest hip seen in the last 30 s, torso normalised by the longest torso in that window), or hips within 0.4 torso of the ankles. Sitting on a chair drops the hip ~0.8 torso, so it never counts as low.
- A drop only counts when fast (> 2.5 torso/s, was 1.2) **and** already past half the floor distance; a quick sit (~1.5–2 torso/s) stays `NORMAL`.
- `SUSPECTED`/`CONFIRMED` clear after 1 s of not-fallen (was "upright", which a seated slouch never satisfied). Telemetry gains `fall.hip_drop`.
- Tests now 10/10: bend to 70° with wide box stays `NORMAL`; slow and 0.5 s sit stay `NORMAL`; fall off a chair after 20 s seated still confirms; original drop / pose-lost / recovery sequences unchanged.
- Remaining known trade-off: a slow lie-down on the floor still confirms (floor exercises). Baseline is image-space, so walking toward the camera for 30 s could shift it; add a frame-height floor band or 3D hips if seen live.

### P1.2 — camera source switcher

- `main.py`: `GET /api/cameras` probes indices 0–3 with `CAP_DSHOW` (active index reported without reprobing) and returns `{active, available, error}`; `POST /api/camera {"index"}` queues a switch. The worker applies it between frames: opens the new index with DSHOW and reads one test frame, releases the old capture only on success, resets the fall detector standing baseline (perspective changed), otherwise sets `camera_error` and keeps streaming from the old camera. Startup camera failure no longer kills the worker, so a different index can be picked from the UI. Telemetry carries `camera_index` and `camera_error`.
- Frontend: `CameraPicker` pills (Laptop Webcam 0 / Mobile Cam 1 / Cam 2) above the engine toggle; spinner and "switching to camera N…" until telemetry reports the new index, red error line if it fails, indices not detected at startup are dimmed but still clickable (DroidCam can be started later).
- Smoke test (scratch server on 8001, probe script): bad index → `camera 3 not available`, index unchanged, no crash; `/api/cameras` and telemetry fields present. The success path could not be exercised because the webcam was held by the port-8000 server; test live. 10/10 tests, `tsc` + `vite build` clean.
- DroidCam: start the DroidCam client first so its virtual camera exists; it usually shows up as index 1 or 2.

### P1.3 — camera probing hardened (DroidCam fix)

- **Root cause found by smoke test:** on this machine index 1 (DroidCam virtual camera) refuses `CAP_DSHOW` but opens via `CAP_MSMF` at 640x480; the previous DSHOW-only, single-`read()` probe therefore reported `camera 1 not available`.
- `main.py` `open_camera(index)`: tries DSHOW then MSMF, waits up to 2 s for a real first frame (virtual cams are slow to start), returns `(cap, {backend, width, height})` or a reason string. `OPENCV_VIDEOIO_DEBUG=1` / `OPENCV_LOG_LEVEL=INFO` are set before `import cv2`, and a `camera` logger prints per-backend open/no-frame reasons to the terminal.
- `GET /api/cameras`: probes 0..4, returns `cameras: [{index, status: active|ok|unavailable, backend, width, height, reason}]`, `available`, `devices` (Windows PnP camera names via PowerShell; order is not index order) and the current error.
- `POST /api/camera`: requested index first, then nearest neighbours in 0..4 (never the current camera). On fallback telemetry says `camera N not available, using camera M instead`; if nothing opens the old stream stays alive and the terminal logs every reason.
- Telemetry adds `camera_backend`. UI: pills for 0–2 plus any detected index, green dot on detected ones, hover shows backend/resolution or failure reason, rescan button, device-name hint line, amber message on fallback.
- Smoke test (scratch 8001, laptop cam held by the port-8000 server): scan 4.6 s → index 1 `ok msmf 640x480`, 2–4 unavailable; `POST index 4` fell back to camera 1 and streamed at 18.6 FPS. Tests 10/10, `tsc` + `vite build` clean.
- `core.py` untouched: it holds analytics only; camera handling lives in `main.py`.

## Session 3 — 2026-11-13 (personal PC)

### Theme pass reverted

- A frontend theme/drawer pass (`frontend/index.html`, `App.tsx`, `index.css`) plus a switch to `yolo11m` crashed the system. Everything uncommitted was discarded with `git restore .`; the tree went back to checkpoint `0991d26`. Do not retry `yolo11m` on the GTX 1050 without measuring VRAM first.
- `docs/sample/mage_d3495f.jpg` is untracked and was left in place.

### P1.4 — camera labels from real device names

- `main.py` `list_video_devices()`: now uses pygrabber's DirectShow list (position == OpenCV `CAP_DSHOW` index) instead of the PowerShell PnP lookup. `camera_label(name)`: contains "droidcam" → "Mobile Cam", contains "integrated" → "Laptop Webcam", otherwise the real name. `GET /api/cameras` adds `name` and `label` to each camera entry.
- `App.tsx`: hardcoded `CAMERA_LABELS` (`0: Laptop Webcam`, `1: Mobile Cam`) removed. Pills show the backend label and exist only for the active camera, cameras that opened in the scan, or devices Windows lists by name. Hover shows the full device name. No styling or layout changes. A device started after page load needs the rescan button.
- `requirements.txt`: `pygrabber==0.2` added (version pinned from memory; drop the pin if pip rejects it). Install: `backend\.venv\Scripts\pip install pygrabber==0.2`. Without it pills fall back to "Cam N".
- Assumption to verify live: MSMF numbers cameras in the same order as DirectShow. DroidCam opens only via MSMF on this machine (see P1.3), so if the "Mobile Cam" pill shows the laptop picture, MSMF needs its own enumeration.

### P1.5 — startup defaults

- Default engine is now `yolo` (YOLO11n-pose, `cuda:0`); MediaPipe stays in the UI toggle. If YOLO fails to load at startup the worker logs it and falls back to MediaPipe instead of dying.
- `yolo_engine.py` `load()`: one dummy-frame inference (blank 640x480) so CUDA init is not paid on the first live frame. It also runs when switching to YOLO from the toggle. Startup takes a few seconds longer.
- Default camera: `default_camera_index()` picks the first device labelled "Laptop Webcam", else index 0. It runs in `lifespan` before the worker starts, not on the capture thread.
- Frame loop, overlay drawing and 20 Hz telemetry untouched.

### Status at end of session 3

- **Nothing in P1.4 / P1.5 has been run**: the assistant session had no shell, so no pip install, `tsc`, tests or live check. First job next session: install pygrabber, start the backend, confirm pills, default camera and YOLO startup, run the 10 tests and `tsc` + `vite build`.
- Commit and push were handed to the user as commands; check `git log` / `git status` to see whether they were run. Files changed: `backend/app/main.py`, `backend/app/engines/yolo_engine.py`, `backend/requirements.txt`, `frontend/src/App.tsx`, `docs/activity_log.md`.

## Session 4 — 2026-11-15 (personal PC)

### Audio-visual form correction: built, then fully reverted

- A form-rule checker (`pipeline/form.py`), red/green skeleton, voice cues and a "Form check" switch were built and then removed at the user's request (the Elbow Curl rule false-fired on 2D YOLO angles). `form.py` is deleted; nothing references it. Do not assume any of it exists.

### Exercise Guide (commit `93168df`, pushed on `feature/exercise-guide`)

- `frontend/src/ExerciseGuide.tsx`: modal opened from the info button next to the exercise dropdown. Full-body skeleton with glowing joints, dashed angle arcs and degree labels, left/right alternating demo, Left and Right stat cards, big angle readout.
- `GET /api/exercise_info` returns `{name: {rest, peak, sides}}` from `rep_counter.EXERCISES`.
- Later reworked (uncommitted, see below): one config row per exercise in `GUIDES`, plus a tab row to browse every exercise without changing the tracked one. Arm exercises are drawn front-on, leg exercises in a 3/4 view. The figure faces the viewer, so its right side is on the viewer's left.

### Fall escalation (Phase 1) and fall-check dialog (Phase 2)

- `backend/app/pipeline/escalation.py` (new): `IDLE -> COUNTDOWN -> ALERTED`, or `ACKNOWLEDGED`. `COUNTDOWN_S = 10.0` for every fall. Getting up clears `COUNTDOWN` and `ALERTED`; "I'm OK" (`POST /api/fall/ack`) answers either. `fired` is true for one frame when `ALERTED` is entered (hook for a future notifier).
- Telemetry: `fall.alert = {state, seconds_left, impact}`; `fall` also carries `impact` and `on_floor`.
- `frontend/src/FallCheck.tsx` (new): full-screen dialog while state is `COUNTDOWN` or `ALERTED`. Seconds ring, "I'm OK" button, beep each second, spoken "Are you okay?", red screen and two-tone alarm when unanswered. Esc is disabled. Shows a hint if the browser is blocking sound.
- **Temporary test hook, kept on purpose by the user:** amber "Test Fall" button in the header, `POST /api/fall/test`, and `Escalation.simulate()` / `_test`. All three are marked `TEMP`. A simulated countdown is not cancelled by standing up; only "I'm OK" ends it.

### Fall detector changes (`backend/app/pipeline/fall_detector.py`)

- **Dwell:** a floor posture shows `SUSPECTED` at once and becomes `CONFIRMED` only after `DWELL_S = 5.0` continuously down. The old "1.5 s still" rule, `STILL_V` and `SUSPECT_TIMEOUT_S` are gone. Alarm therefore sounds 15 s after going down.
- **Recovery:** upright for `RECOVER_S = 0.5` returns to `NORMAL` from any state.
- **Pose lost:** counts as still down for `LOST_AFTER_DOWN_S = 6.0` after a drop or a floor posture.
- **New floor cues**, added because lying on the floor read as NORMAL:
  - torso keypoints missing -> box wider than `ASPECT_LYING = 1.6` (unless shoulders span over half the box: close-up);
  - `flat`: shoulders-to-legs vertical extent under `FLAT_EXTENT = 0.8` torso (needs no standing baseline or ankles);
  - `compressed`: lying feet-toward-camera (reference `docs/sample/image_9379a9.jpg`, skeleton upright on screen, tilt 3 deg). Shoulder-to-ankle height under `COMPRESSED = 0.55` and hip-to-ankle under `LEGS_SHORT = 0.27` of the standing height seen earlier at the same ankle-y bin (`_stand`, 40 px bins, session-long);
  - `inverted`: shoulders below ankles in the image (head toward camera).
- Known limits: `compressed` needs the person to have stood near that spot earlier in the session; keypoints for the reference test were read off the screenshot by eye; a deep squat or floor sitting held 5 s, floor exercises, or standing under a very steep camera can start the countdown.
- `core.py`: posture is "lying" when `on_floor`.

### Exercise library (Phase 4)

- `rep_counter.EXERCISES` gained Overhead Reach (shoulder 20->150), Seated Knee Extension (knee 95->165), Mini Squat (`knee_mean` 170->115), Side Leg Raise (`l/r_hip_abd` 92->120), Wall Push-up (`elbow_mean` 165->95). Angles are first-pass values pending physio review.
- `kinematics.py`: `{side}_hip_abd` (other hip - hip - knee) and `elbow_mean`.
- Hamstring Curl and Lateral Arm Raise were not added: they duplicate Knee Flexion and Shoulder Raise. Heel-Toe Balance needs a hold timer and is deferred.

### Deferred by the user

- WhatsApp emergency alerts. Design was: self-hosted Evolution API in Docker, `backend/app/notify.py` on a daemon thread triggered by `Escalation.fired`, text then optional snapshot. Risks noted: unofficial API (use a dedicated SIM), patient image leaves the machine.

### Status at end of session 4

- **Tests:** `backend/.venv/Scripts/python tests/test_analytics.py` -> 21/21 passed on the last run. That run was before the Test Fall hook was removed and then restored; re-run to confirm. `npx tsc --noEmit` in `frontend` was clean before the fall-detector and dialog timing changes; re-run too.
- **The assistant session has no shell.** The user runs commands with the `!` prefix, one per prompt, through bash: use forward slashes, and `cd` persists between commands.
- **Git, as last seen:** branch `feature/exercise-guide`, last commit `93168df` (Exercise Guide), pushed. Everything after it was uncommitted: escalation, FallCheck, new exercises, guide refactor, fall detector changes. `git checkout main` was refused because of those uncommitted changes. Check `git status` and `git log` first thing.
- **Commands still to run, in order** (from the repo root):
  1. `git add backend/app frontend/src tests` then `git status` (`docs/sample/` should stay untracked: it holds a photo of a person at home and a third-party UI screenshot)
  2. `git commit -m "Add robust floor fall detection with 5s dwell, auto-recovery, and 10s countdown (keeping test button)"`
  3. `git push`
  4. `git checkout main`, `git merge feature/exercise-guide`, `git push origin main` (`main` is still at `0991d26`, so this fast-forwards and brings the whole branch across)
- **Never run live by the assistant:** camera pills and pygrabber (`backend/.venv/Scripts/pip install pygrabber==0.2`), YOLO default with warm-up, the look of all ten guide animations (Wall Push-up, Overhead Reach, Seated Knee Extension most likely to need tweaks), the fall dialog layout and sound, and the real lying-on-floor case. If a real fall still shows NORMAL, capture the `fall` section of `GET /api/telemetry` while down.
- This log entry itself is uncommitted; `docs/activity_log.md` is not in the `git add` line above, so add it explicitly.

## Session 5 — 2026-11-16 (personal PC)

### Roadmap review (no code changed)

- Reviewed four upgrade areas: clinical biomechanics (tempo, symmetry, compensation), voice check-in with
  emergency alerts, persistence and reporting, privacy mode and low-light handling.
- Main findings: everything is feasible without frame-loop cost except CLAHE, which is per-frame work. Python
  threads share one interpreter lock, so CPU-heavy background work needs a separate process, not a thread.
  The system has no login and stores nothing yet. After "I'm OK" the system never asks again if the person
  stays down (fixed by the re-check in plan 02).
- Browser speech recognition was rejected for voice check-in: Chrome sends audio to Google by default and
  on-device support is experimental. Decision: local Sherpa-ONNX on the backend machine instead.

### Two implementation plans written (`docs/plans/`), to be implemented in order

1. `01_TIMINGS_AND_PRIVACY.md` — per-stage timing telemetry (`capture`, `infer`, `pipeline`, `draw`, `encode`)
   to set the FPS/latency baseline, and Privacy Mode (backend blanks the frame before `draw()`; header switch;
   video hidden until the backend confirms, so a restart cannot leak raw frames).
2. `02_SHERPA_STT_AND_RECHECK.md` — offline English speech recognition in a separate process, microphone open
   only while the fall dialog is showing and only after the spoken prompt ends; intent rules (help/negative
   escalates, clear multi-word affirmative cancels, anything else lets the countdown run); "Trigger Emergency"
   button; 2-minute re-check after an acknowledgement if the person is still down.

Each plan is self-contained (file paths, schema changes, tests, acceptance checks) for a fresh session.

### Standing rule from the user

**Never remove or break an existing working feature.** Stand-up auto-recovery, the spoken "Are you okay?"
prompt, the manual "I'm OK" button and the "Test Fall" button all stay. Voice input and Privacy Mode are purely
additive.

### Explicitly excluded for now

SQLite/PostgreSQL logging, trend charts, PDF reports, Twilio/WhatsApp/SMS alerts, WebRTC caregiver view.

### Still open from session 4 (unchanged)

- Check `git status` / `git log` first: at last sight everything after commit `93168df` was uncommitted and
  `main` was still at `0991d26`. `docs/plans/` and this log entry are new and uncommitted too.
- Never verified live: camera pills and pygrabber, YOLO warm-up, the ten guide animations, the fall dialog's
  look and sound, the real lying-on-floor case.
- The speech model named in plan 02 and its Python API must be confirmed against the Sherpa-ONNX docs before
  coding; they were taken from documentation search results, not from a working install.

### Plan 01 (timings and Privacy Mode) — implemented, NOT yet verified

Implemented by a Sonnet 5.5 session that had no shell, so none of the checks below have been run. Plan 02 was not
touched. No existing feature was removed (Test Fall button and endpoint, fall dialog, stand-up recovery, rep
counting all unchanged); `annotate.py`, `fall_detector.py`, `escalation.py`, `FallCheck.tsx` and
`ExerciseGuide.tsx` were not edited.

- `backend/app/main.py`: `time.perf_counter()` marks around capture / infer / pipeline / draw / encode, smoothed
  0.9/0.1 into `Worker.timings`, reported as `timings` in telemetry (`fps` and `latency_ms` unchanged).
  `Worker.privacy` set through `POST /api/config {"privacy": bool}`. When on, the frame is blanked in place
  (`frame[:] = 0`) after inference and `pipeline.process()` and before `draw()`, so the skeleton and box are drawn
  on black and raw video is never JPEG-encoded.
- Two additions beyond the plan, both to make "raw video never leaves the backend" hold:
  - Telemetry `privacy` is true only after a blanked frame has actually been published (`jpeg_private`), not
    merely when the flag is set, so the UI's confirmation is truthful.
  - `mjpeg()` skips a stored frame that was encoded before privacy was switched on, so a client that connects
    in that moment cannot receive one stale raw frame.
- `frontend/src/useTelemetry.ts`: `timings` and `privacy` in the `Telemetry` type.
- `frontend/src/App.tsx`: "Privacy" switch in the header (left of Test Fall; `ml-auto` moved to it); preference in
  `localStorage` key `privacy`; re-posted to the backend when it disagrees with telemetry (also after a reconnect);
  the `<img>` gets `invisible` until the backend is online and reports `privacy: true`; a small timings line under
  the FPS/Latency cards.
- No new unit tests, as the plan specifies (the analytics pipeline is unchanged; `app.main` must not be imported
  in tests).

**To verify, in order** (acceptance checks in `docs/plans/01_TIMINGS_AND_PRIVACY.md` section 5):
1. `backend/.venv/Scripts/python tests/test_analytics.py` -> expect `21/21 passed`.
2. `cd frontend && npx tsc --noEmit; cd ..` -> expect no output.
3. Live: the timings line appears; with privacy on the stream is skeleton-on-black, `encode_ms` is not higher
   and `fps` not lower than with it off; 5 reps count the same in both modes; Test Fall dialog works in both;
   after turning privacy on and restarting the backend the video stays hidden until the stream is black.
4. Record the five `timings` values (60 s, privacy off) here as the reference for Plan 02. The baseline `fps` and
   `latency_ms` from before the change were not captured, since no shell was available to run the old code;
   compare against what the dashboard showed before this session if you remember it.
5. Known limit: after a backend restart the MJPEG `<img>` does not reconnect by itself, so the page needs a
   reload (this was already the case before; hidden-until-confirmed still protects the reload).

Once 1-4 pass, mark this section verified and start `02_SHERPA_STT_AND_RECHECK.md`.

### Plan 02 (offline voice check-in + 2-minute re-check) — implemented, NOT yet verified

Implemented by a Sonnet 5.5 session with no shell: nothing below has been run, installed or downloaded. Existing
behaviour was kept (stand-up recovery, spoken prompt, beeps, "I'm OK", Test Fall button/endpoint, `fall_detector.py`
untouched, existing tests unmodified).

- `backend/app/voice/` (new): `intent.py` (asymmetric `classify`: help/negative first, only "I am okay/fine/..."
  cancels), `stt_process.py` (Sherpa-ONNX streaming English, CPU, own process, idle until "start", mic open only
  while listening; finds model files by prefix, int8 encoder/joiner), `listener.py` (`VoiceListener`: reader thread,
  acts on final sentences only).
- `backend/app/pipeline/escalation.py`: `escalate()` (COUNTDOWN -> ALERTED at once), `RECHECK_S = 120` (ACKNOWLEDGED
  while still down -> new COUNTDOWN), `rechecks` in `fall.alert`. Simulated (Test Fall) episodes never re-check.
- `backend/app/main.py`: `voice` listener started in `lifespan`; `POST /api/fall/listen`, `POST /api/fall/escalate`;
  one `voice.active` check per frame (stops the mic when the dialog is gone); `fall.voice` in telemetry.
- `frontend/src/FallCheck.tsx`: microphone is requested (`/api/fall/listen`) only after the spoken prompt ends + 300 ms
  (6 s fallback), with a guard against the end event of a cancelled prompt; status line ("Listening…", `Heard: "…"`,
  "Voice unavailable — use the buttons"); red "Trigger Emergency" button beside "I'm OK" (countdown only); heading
  "Still down — are you okay?" on re-checks. Layout: dialog scrolls, content `min-h-full`.
- `backend/requirements.txt`: `sherpa-onnx`, `sounddevice` (unpinned: pin after installing).
  `backend/scripts/download_models.py --stt` fetches the model; `backend/scripts/stt_smoke.py` is the mic check.
- Tests added to `tests/test_analytics.py`: intent table (incl. the documented echo risk), model-file picking,
  escalate, escalate-while-idle, re-check, no re-check after standing, two re-check rounds, Test Fall unchanged.

**Assumptions to confirm** (taken from docs/search results, not from a working install):
1. ~~`sherpa-onnx` and `sounddevice` have wheels for Python 3.13 on this machine.~~ CONFIRMED 2026-11-16: pip installed
   `sherpa-onnx 2.0.1` (cp313 win_amd64 wheel) + `sherpa-onnx-core 2.0.1`; `sounddevice 0.5.6` was already present.
   Pin them in `requirements.txt` (`sherpa-onnx==2.0.1`, `sounddevice==0.5.6`).
2. The model archive URL and file names (`...-epoch-99-avg-1-chunk-16-left-128[.int8].onnx`); a subagent read them
   from a summary of the docs page, not verbatim. Still unconfirmed until `download_models.py --stt` has run.
3. ~~`OnlineRecognizer.from_transducer(...)` arguments~~ CONFIRMED against the installed 2.0.1 source
   (`sherpa_onnx/online_recognizer.py`): `from_transducer` accepts every keyword the code passes (`tokens, encoder,
   decoder, joiner, num_threads, sample_rate, feature_dim, enable_endpoint_detection, rule1/2/3_*, decoding_method,
   provider`); `get_result` returns `str`; `create_stream / decode_stream / is_ready / is_endpoint / reset` exist.
   `OnlineStream.accept_waveform` lives in the compiled extension and was not checked, only seen in the official example.
4. `multiprocessing` "spawn" works under the way uvicorn is launched here (child imports only `app.voice.stt_process`).

**Verified so far (user ran these, 2026-11-16):** `pip install -r backend/requirements.txt` OK (sherpa-onnx 2.0.1);
`download_models.py --stt` OK and the unpacked file names match `find_model_files` exactly (int8 encoder/joiner, fp32
decoder, `tokens.txt`); first `stt_smoke.py` run: the speech process spawned, the model loaded, the microphone opened
and audio reached the recogniser (default device 44.1 kHz, resampled to 16 kHz). No sentence was printed, but that run
went through `!`, which shows output only after the command ends, so the speaking window could not be timed.
`stt_smoke.py` now also reports the loudest microphone sample (silent mic vs. no words recognised); the child sends a
`("level", peak)` event on stop and the backend logs it. Recognition itself is still unverified: re-run the smoke
script in a normal terminal window.
Second run (normal terminal): 0 sentences, loudest sample 0.000 = the default input device delivers pure silence
(suspects: DroidCam's virtual microphone as Windows default, Windows microphone privacy for desktop apps, muted or
wrong device). Added `stt_smoke.py --devices` (measures every input device for 1 s); the chosen device is passed to
the backend with the `STT_INPUT_DEVICE` environment variable (number or part of the name).
`--devices` result: Windows default input = "Microphone (DroidCam Audio)" (peak 0.000, silent unless the phone streams
audio); "Microphone (Realtek Audio)" = laptop mic, peak 0.086 (chosen); "Microphone (2- KORP THALLIUM)" = headset,
peak 1.000 (clipping, not on the patient). A name in `STT_INPUT_DEVICE` is resolved by `resolve_device` against the
default audio API only (sounddevice's own lookup can match one microphone once per audio API). Persist with
`setx STT_INPUT_DEVICE "Realtek"` (new windows only). New test: `test_resolve_device_uses_default_audio_api_only`
(31 tests expected in total).
Third smoke run (laptop mic, loudest sample 0.108): recognition works end to end (partials and finals stream, `classify`
runs), but three attempts at "I am okay" were heard as "I AM", "I AM LOOKING", "I AM ODY" -> all `None` (safe: the
countdown continues). Likely causes: low input level, accent, small model. "help" and "I am fine" not yet tested.
Next: raise the Windows input level, retry with "I am fine" / "help" / "I need help" / "I am not okay"; if "fine" and
"help" are reliable add a hint to the Listening line ("say I am fine or help"); if recall stays poor with a good level,
consider a larger Sherpa model (bigger download, slower decode).
Fixes after the fourth smoke run (loudest sample 0.152; "I am okay" heard as "I AM WICKED"; "I am fine" split into
finals 'I' + 'AM FINE'): (1) `stt_process.py` `RULE1_S = 2.4`, `RULE2_S = 1.5` (the code already used sherpa's defaults
2.4 / 1.2, so the real change is rule2 1.2 -> 1.5 s; an answer is acted on 1.5 s after the last word). (2) software gain
`GAIN = 3.0` (env `STT_GAIN`, 1 = off) via `boost()`: it backs off so recent peaks stay under 0.95 and never clips
or attenuates; the diagnostic peak stays the RAW level. (3) `intent.py`: leading "I" optional ("am fine", "am okay",
"i'm ok", "all good", "doing fine/okay"); bare "okay"/"yes"/"fine"/"good" still `None`; "not" and any `n't` form now
escalate like "help"/"no". Tests added: `test_intent_accepts_sentences_cut_by_endpointing`,
`test_gain_boosts_quiet_audio_without_clipping` (33 tests expected). Not fixed by any of this: accent / the small
model; a pause longer than 1.5 s inside a sentence still splits it (joining consecutive finals would cover that).
Result of the gain fix: loudest sample 0.302 (was ~0.15). Next tweak: `stt_process.finished()` ends a sentence as soon as
the text classifies as a clear affirmative ("I am fine", "am okay", ...) instead of waiting RULE2_S, so repeats and
trailing words ("I am fine I am fine I ...") are not glued on. Chosen without a settle delay because the model's text
only changes once per ~0.6 s chunk. Help / negative text is never finished early (a partial "NO" can become "NOW").
Trade-off: "I am fine, but I can't get up" is cut at "fine" if the "but" is not in the same decode chunk; the 2-minute
re-check is the backstop. Test added: `test_clear_affirmative_finishes_the_sentence_at_once` (34 tests expected).
Next round (user reported "fine" heard as KIND / FIGHT / FIVE and "okay" as OCHING): the 0.302 was the RAW level, so 3x
put loud syllables near/over full scale (the limiter only reacts after a loud sound). Changes: `GAIN` 3.0 -> 1.5,
`RULE2_S` 1.5 -> 0.8; `finished()` now also ends a sentence on a STRONG distress word (help, emergency, ambulance,
hurt(s), pain, cannot, stuck, fall/fallen/fell, sick, dizzy) - weak ones (no, not, bad) still wait for the end of the
sentence because a partial "NO" is often the start of "NOW" and a likely noise artefact (they still escalate, <= 0.8 s
later); `intent.py`: those distress words added, and after am / i am / i'm the near-misses oky, okey, five, kind, fight
(plus well) count as "ok" (bare words still `None`, any negative still wins). The function is `classify` (not
`parse_intent`; kept so nothing breaks). Known consequence: "I am five years old" classifies as "ok". Tests: finishing
test updated (help / cannot now finish early on purpose), `test_intent_accepts_phonetic_near_misses_only_after_i_am`,
`test_voice_settings_stay_gentle` (36 tests expected).
Live testing then showed a soft "help" transcribed as "HELL" / "HELD" (-> `None`): both added to the help words
(weak tier: they wait for the end of the sentence, because a partial "HELL" can become "HELLO"); "oke" and "ache"
(= "okay" in some accents) added to the near-misses accepted right after am / i am / i'm. `test_intent_near_misses_seen_in_live_testing`
(37 tests expected). Note: "ache" is NOT a distress word on its own, so "my back ache" is `None`.

**To verify, in order** (acceptance checks 1-10 in `docs/plans/02_SHERPA_STT_AND_RECHECK.md` section 7):
1. `backend/.venv/Scripts/python tests/test_analytics.py` -> previous 21 plus 9 new = expect `30/30 passed`.
2. `cd frontend && npx tsc --noEmit; cd ..` -> no output.
3. Without the packages/model installed: backend starts; Test Fall dialog shows "Voice unavailable — use the
   buttons"; everything else works as before.
4. `backend/.venv/Scripts/pip install -r backend/requirements.txt`, then
   `backend/.venv/Scripts/python backend/scripts/download_models.py --stt`, then
   `backend/.venv/Scripts/python backend/scripts/stt_smoke.py` at the patient's distance; record the hit rate here.
5. Live: listening starts only after the prompt ends; "I am okay" cancels, "help" alarms at once, bare "okay" and
   silence let the countdown run; 20 silent Test Fall runs with no false cancel; re-check after 2 min down;
   compare the Plan 01 `timings` and `fps` idle vs. while listening (no degradation).

### Voice switched from Sherpa-ONNX to Groq Whisper (2026-11-16) — implemented, NOT yet run

Why: the local streaming model kept mishearing short words ("okay" -> ODY / WICKED / OCHING, "fine" -> KIND / FIVE,
"help" -> HELL) even after gain, endpointing and phonetic-tolerance tuning. User decision: use Groq's `whisper-large-v3`.
**Privacy change:** audio of the speech segment recorded while the fall dialog is open now leaves the machine (HTTPS to
`api.groq.com`). Nothing is recorded outside a fall check, silence/noise is not sent, nothing is written to disk
(the WAV is built in memory), and the key is never logged.

- New in `backend/app/voice/`: `groq_stt.py` (key from `GROQ_API_KEY` env or `backend/.env`, read on every start;
  POST to `/openai/v1/audio/transcriptions`, model `whisper-large-v3`, `language=en`, `temperature=0`, `verbose_json`;
  segments Whisper doubts (`no_speech_prob > 0.6` and `avg_logprob < -1`) are dropped; **no `prompt`**, because a prompt
  like "I am okay" can be echoed back on near-silent audio and would cancel a real emergency; every failure becomes
  `SttError`), `recorder.py` (microphone via sounddevice, energy-based `Segmenter`: speech starts after 0.2 s voiced,
  ends after 0.8 s silence, 5 s cap, 0.15 s countdown beeps ignored, threshold `VOICE_MIN_RMS` default 0.012; 16 kHz
  mono WAV, peak-normalised up to 8x), `listener.py` (one daemon thread per fall check, max 4 transcriptions per check;
  status idle / listening / transcribing / unavailable). `intent.py` is unchanged in behaviour (`classify`), minus the
  streaming-only `urgent` helper.
- No key or any network/API error -> status "unavailable" -> the dialog shows "Voice unavailable — use the buttons";
  the buttons, countdown, alarm and re-check are unaffected. The dialog also shows "Processing…" while Groq works and
  hints "say I am fine or help" while listening. The "listen only after the prompt ends" rule is kept.
- Removed: `sherpa-onnx` from `requirements.txt` (added `requests==2.34.2`; `sounddevice==0.5.6` stays),
  the `--stt` model download in `download_models.py`, gain / endpointing / partial-trigger code. `stt_process.py` is
  now an empty stub (no delete tool was available): **delete it**, and the downloaded model and package:
  `rm backend/app/voice/stt_process.py`, `rm -rf backend/models/sherpa-onnx-streaming-zipformer-en-2023-06-26`,
  `backend/.venv/Scripts/pip uninstall -y sherpa-onnx sherpa-onnx-core`.
- Secrets: `.gitignore` now ignores `.env` / `.env.*` (it did not before) except `backend/.env.example`. The key is
  only in `backend/.env`. It was pasted into a chat message, so consider rotating it in the Groq console.
- Tests: Sherpa-only tests removed (model files, finishing, gain/settings); added Groq request/response with a mocked
  `post`, hallucination filter, failures-without-leaking-the-key, key from env then `.env`, listener acting on text,
  no-key behaviour, segmenter (utterance, beeps/quiet, cap), WAV format, audio -> mocked Whisper -> action.
  44 tests expected.
- Untested assumptions: the Groq request format and rate limits (no network access from the assistant session);
  the speech-detection threshold on this laptop's microphone (calibrate with `stt_smoke.py`, which prints the ambient
  level, loudest frame and threshold); sounddevice recording from the Realtek device at its default rate.
- To verify, in order: `backend/.venv/Scripts/python tests/test_analytics.py` (44/44); `cd frontend && npx tsc --noEmit`;
  `backend/.venv/Scripts/python backend/scripts/stt_smoke.py` in a normal terminal (say "I am fine", "help", "I am not
  okay"; check each is transcribed and classified, and that silence and the beeps send nothing); then the live dialog
  (Test Fall -> speak after the prompt). Plan 02's Sherpa sections are obsolete (banner added to the plan file).

### Short-word "help" with Groq Whisper (2026-11-16) — implemented, NOT yet run

Observed: phrases ("I am okay", "I am not okay") work through Groq, but a single quick "help" is sometimes missed or
mapped wrongly. Two separate causes, two fixes:
- `intent.py` (parse step): "help" is now matched loosely: any word containing "elp" (help, helping, helpi, aelp, yelp)
  or one letter from "help" (halp, hep, hell, held); "hello" / "hey" / "hold" do not match. Added Urdu/Hindi bachao
  (+ spellings) and madad. The function is still `classify`; `parse_intent` is now an alias (the name the notes use).
  Side effects: "helpful" / "Delphi" also count (false alarms are the accepted direction).
- `recorder.py` (hearing step): `MIN_ON_S` 0.2 -> 0.12 s, `MIN_RMS` 0.012 -> 0.008, `PRE_S` 0.3 -> 0.4 s (soft "h").
  Because a 0.15 s countdown beep is as long as a quick "help", duration alone cannot separate them (and beeps would
  be uploaded and use up the per-check cap), so frames that are pure tones (`tonal_share` > `TONE_SHARE` 0.65: >65 %
  of the frame's energy in one 150 Hz band) count as silence and do not raise the ambient estimate. `MAX_CALLS` 4 -> 8.
  Env knobs: `VOICE_MIN_RMS`, `VOICE_TONE_SHARE` (documented in `backend/.env.example`).
- `stt_smoke.py`: new `--classify TEXT...` (offline mapping check), `--beeps` (plays the dialog's tick every second
  while listening), and it now prints ambient level, threshold, tone frames ignored, and short sounds that were NOT sent.
- Tests added: loose help matching, tonal_share, a quick single word is kept, beeps ignored without hiding the next
  word (48 tests expected).
- UNVERIFIED: tone threshold 0.65 and the beep rejection are based on spectral arithmetic (square wave ~0.8, speech
  <= ~0.6), not measurements through this laptop's speakers and microphone; run `stt_smoke.py --beeps` to check.

### Faster hand-off to Groq (2026-11-16) — implemented, NOT yet run

Observed: recognition works (check-in phrases and "help") but the utterance waited too long before being sent.
- `recorder.py`: `TRAIL_S` 0.8 -> 0.5 s (silence that ends an utterance; `VOICE_TRAIL_S` overrides, also a
  `trail_s` argument on `Segmenter`). Speech sensitivity is unchanged on purpose: `MIN_RMS` 0.008, `MIN_ON_S` 0.12,
  `PRE_S` 0.4 s (no clipping of a soft first sound).
- Cost of a shorter pause: slow speech is cut at its pauses more often ("I am" / "fine"). `listener.handle()` now
  joins a piece that was not an answer with the next piece heard within `CARRY_S` = 4 s, so "I am" + "fine" -> ok;
  a distress word still acts at once, a bare word still never counts, blank text changes nothing.
- `stt_smoke.py` prints, per utterance, "about X s after you stopped speaking: 0.5 s silence + encode ms + Groq s",
  and an average / slowest at the end.
- Not done (possible next step if the measured Groq time dominates): reuse one HTTPS connection / warm it at the start
  of the check; each request currently opens a new connection.
- Tests: trailing-silence timing, joined sentences; two existing duration bounds adjusted (50 tests expected).

### Very short words and a bug in the settings (2026-11-16) — implemented, NOT yet run

Observed: smoke test listed 0.04-0.08 s sounds as "too short to count as speech". Cause: the minimum speech length
(`MIN_ON_S`, 0.12 s since the "help" round), not the silence timeout. A quick "help" has only ~40-80 ms above the
threshold on this laptop microphone.
- `recorder.py`: `MIN_ON_S` 0.12 -> 0.06 s (3 frames of 20 ms; rounded up to whole frames, so "0.05" also means 0.06),
  `TRAIL_S` 0.5 -> 0.4 s. New env `VOICE_MIN_ON_S` and `Segmenter(min_on_s=...)`. Cost: a click or tap of 60 ms or
  more is now sent (Whisper returns nothing for it; capped at 8 sends per check). A 2-frame (40 ms) sound is still
  ignored by default; `VOICE_MIN_ON_S=0.04` lets that through too.
- BUG fixed: only `GROQ_API_KEY` was ever read from `backend/.env`; `VOICE_TRAIL_S`, `VOICE_MIN_RMS`,
  `VOICE_TONE_SHARE` and `STT_INPUT_DEVICE` were read from the process environment only, so putting them in `.env`
  (as `.env.example` says) did nothing. New `backend/app/voice/settings.py` reads environment first, then `.env`, on
  every use; `groq_stt.api_key` and the recorder use it.
- `stt_smoke.py` prints the effective `speech starts after` value and, under the list of ignored short sounds, which
  setting to change.
- Tests: silence timing now explicit per setting, very short word vs click, settings from env then `.env` (tests ignore
  the developer's real `.env`): 52 tests expected.

### "Listening…" shown but no audio captured (2026-11-16) — implemented, NOT yet run

Report: the dialog says "Listening…" but nothing is captured or sent; no console errors. The report assumed a browser
recorder (getUserMedia / MediaRecorder / WebSocket audio). **There is none**: the frontend never touches the microphone;
the backend machine's microphone is captured by `sounddevice` in `listener.py`. The dialog only shows the status the
backend reports, and the backend used to report "listening" the instant the check started, before the microphone was
opened or any audio had arrived, so the label proved nothing.
- `listener.py`: status is now "starting" until the first audio block arrives, then "listening". Watchdog
  (`WATCHDOG_S` = 3 s): if the stream delivers no audio, or only digital silence (loudest sample < 1e-4, e.g. a muted
  or virtual microphone such as DroidCam Audio), the status becomes "unavailable" with the reason in `error`, which the
  dialog shows (tooltip) and telemetry carries. Trace log lines `voice: ...` (backend terminal): session started,
  input device [index] name, microphone open at N Hz + threshold, first audio block, speech of X s found -> sending N
  bytes, Groq answered in X s: text -> intent, session stopped, send limit.
- `recorder.py` logs the chosen input device; `/api/fall/listen` returns `{ok, status, error}`.
- Frontend (temporary `[voice] DEBUG` console logs in `FallCheck.tsx`): prompt finished -> request to open the
  microphone -> backend answer -> each change of the backend's voice status / transcript. Search the console for
  `[voice]`. The dialog shows "Starting microphone…" while "starting".
- Most likely real cause on this machine: the Windows default input is "Microphone (DroidCam Audio)", which is silent.
  `STT_INPUT_DEVICE` was only ever set in one PowerShell window; it can now live in `backend/.env`
  (`STT_INPUT_DEVICE=Realtek`), which persists. Also: restart the backend after code changes.
- Tests: fake microphone that delivers nothing, and one that delivers digital silence (53 tests expected).

### Binding the KORP THALLIUM headset (2026-11-16) — implemented, NOT yet run

Report: "Voice unavailable"; the headset "KORP THALLIUM" (device 3) is not locked onto. Findings: `backend/.env` had no
`STT_INPUT_DEVICE`, so the backend used the Windows default input (DroidCam Audio, silent). And my silence watchdog
(previous round) treated digital silence as a dead microphone; many headsets send exact zeros until you speak, so a
working headset could be failed as "unavailable" during the quiet seconds after the prompt.
- `recorder.py`: `STT_INPUT_DEVICE` is now a list (`device_preferences()` / `parse_device_list`): a number, or part of a
  name (case, spaces and punctuation ignored: "korp thallium", "Korp-Thallium", "2- korp" all match "Microphone (2- KORP
  THALLIUM)"), or several choices separated by commas / semicolons, tried in order; quotes are ignored.
  `resolve_device`: a number must be an existing input; text matches an exact name first, then loosely, in the default
  audio API first and then the others (one microphone is listed once per API); a failure lists the input devices.
  `pick_device` skips choices that do not exist (headset unplugged -> next choice). `open_stream` tries the device's
  rate, then 48000 / 44100 / 16000, mono then stereo, and raises the real PortAudio error; it yields
  `(blocks, rate, device name)`. `describe_inputs` prints `[index] name (audio API, channels, rate)` with the default marked.
- `listener.py`: digital silence for 3 s no longer fails the session. Status stays "listening", `warning` is set
  ("No sound from <device> yet ... probably the wrong input device") and shown in the dialog (amber), cleared as soon as
  sound appears; the log lists the input devices. Still a failure ("unavailable"): the stream delivers no audio blocks at
  all, the device cannot be found or opened. Every failure logs the input device list.
- `backend/.env`: added `STT_INPUT_DEVICE=KORP THALLIUM` (you can add a fallback: `KORP THALLIUM, Realtek`).
  `stt_smoke.py` shows the chosen device and, when it cannot open one, the list.
- Tests: list parsing, KORP matching in every spelling, fall-through of choices, rate fallback / real error, a silent
  device warns and keeps listening (57 tests expected).

### Cloud + mobile architecture blueprint (2026-11-16) — plan only, nothing implemented

Saved as `docs/cloud_mobile_blueprint.md`. Work on it starts next session; no application code was changed for it.
- **Central point:** a cloud server has no camera or microphone, so capture must move into the patient's browser and
  the single global `Worker` must become one pipeline per connected patient.
- **Recommendations:** pose on the phone (MediaPipe in the browser) as the production default with server YOLO as
  fallback; WSS with ack-paced JPEG frames (one frame in flight) instead of WebRTC; no video sent back (keypoints +
  telemetry, skeleton drawn on a canvas over the local preview); browser microphone + speech detection, utterances
  uploaded to a new `/api/voice` that reuses `groq_stt` and `intent.classify`; PWA for exercise sessions, a fixed
  tablet on a charger for fall monitoring (native only for background monitoring / guaranteed alerts).
- **Phases:** 0 close out MVP (commit/push, rotate the Groq key, live voice test) -> 1 sessions in the backend ->
  2 browser camera to cloud inference -> 3 browser voice -> 4 auth, TLS, CORS, deployment -> 5 mobile UI / PWA ->
  6 on-device pose.
- **Decisions the user must make before Phase 1:** on-device pose as default or server-only; GPU or CPU hosting and
  region; primary use case (own phone vs fixed tablet); Ultralytics AGPL licence if YOLO is served from the cloud.
- Bandwidth and latency figures in the blueprint are estimates, to be replaced by measurements in Phase 2.

### Where things stand at the end of this session (resume here)

- Tests: 57/57 passed and `npx tsc --noEmit` clean on the last runs (after the KORP THALLIUM device work).
- **Not yet verified live:** the fall dialog with the headset microphone through Groq (click Test Fall, wait for the
  prompt, say "help"; read the `voice:` lines in the backend terminal). `backend/.env` has
  `STT_INPUT_DEVICE=KORP THALLIUM`.
- **Git:** on `main`; last commit seen was `d37caf3` ("checkpoint MVP and add implementation plans"). Everything since
  (timings, Privacy Mode, Groq voice, escalation re-check, docs) was uncommitted when last checked. Commit steps
  given to the user: delete the stub `backend/app/voice/stt_process.py`, `git add .gitignore backend frontend tests
  docs/activity_log.md docs/plans docs/cloud_mobile_blueprint.md`, confirm no `.env` or `docs/sample/` is staged,
  commit, `git push`. Run `git status` first thing.
- Housekeeping still open: rotate the Groq key (it was pasted into chat); remove the temporary `[voice] DEBUG`
  console logs in `FallCheck.tsx`; uninstall `sherpa-onnx` and delete its model folder under `backend/models/`.
- The assistant session has no shell: the user runs commands with the `!` prefix (bash: forward slashes, `!` must be
  the first character) or in a normal PowerShell window.

### Phase 1 of the cloud/mobile blueprint: sessions extracted (2026-11-17) — implemented, NOT yet run

Done by a Sonnet 5.5 session with no shell: nothing below has been executed.
- New `backend/app/sessions.py`:
  - `Session` = everything per patient and nothing about cameras or engines: `Pipeline` (exercise, reps, mobility,
    fall detector, escalation), the voice listener (optional: `local_voice=True` only for the desktop kiosk, so two
    sessions never compete for one microphone), `pose_visible`, `latency_ms`, and the pending-command flags.
    Commands safe from any thread (only set flags): `request_exercise` (raises `ValueError` with the old message),
    `request_reset`, `ack`, `escalate`, `simulate_fall` (TEMP), `listen`. Feeder-thread-only: `apply_pending`,
    `reset_fall_baseline`, `process(res, t, infer_ms, frame_width)`, which also closes the microphone once the fall
    dialog is gone. `telemetry()` returns the patient's fields; `open()` / `close()` for lifecycle (idempotent).
  - `SessionManager`: `create` (random id if none; refuses duplicate ids, unknown exercises, and a full house via
    `SessionLimitError`; nothing is registered on failure), `get`, `close`, `close_all`, `ids`, `len`, `in`.
  - `to_reference(res, frame_width)`: rescales a pose to the 640-wide frame `fall_detector.py`'s pixel constants assume.
- `backend/app/main.py`: `Worker(session)` keeps the camera, engine, MJPEG stream, privacy flag, fps and timings;
  per-patient work goes through `local_session` (created by `SessionManager` as id "local"). The global `voice`, and
  `worker.pipeline` / `pending_exercise` / `pending_reset` are gone. Routes keep their URLs, JSON and error text:
  `/api/config`, `/api/reset`, `/api/fall/{ack,listen,escalate,test}`, `/api/telemetry`, `/api/health`, `/ws/telemetry`,
  `/api/video_feed`, `/api/cameras`, `/api/camera`. Telemetry has exactly the same keys (source fields from the worker,
  patient fields from the session).
- Deliberately NOT done (later phases): the desktop worker does not pass `frame_width`, so desktop behaviour is
  unchanged; no idle-session expiry, no HTTP session ids/auth (all routes still address the one local session), engine
  and inference are still owned by the worker (batching across sessions is Phase 2).
- Tests added to `tests/test_analytics.py` (63 expected): two sessions share no state; commands only set flags until
  `apply_pending`; voice start/stop follows the fall dialog and voice-less sessions work; manager lifecycle and limits;
  `to_reference` (and a 320 / 1280 px frame gives the same fall verdict as 640); and a route test with FastAPI's
  TestClient (no `with`, so the lifespan, camera, engine and microphone never start) that checks every desktop route
  and the telemetry keys the frontend reads.
- To verify: `backend/.venv/Scripts/python tests/test_analytics.py` (63/63), then start the backend and use the app
  as usual (camera pills, exercise switch, Reset, Test Fall dialog, Privacy, engine switch); `/api/telemetry` should look
  the same as before. Next phase: browser camera to cloud inference (`/ws/session`).

### Phase 2 of the cloud/mobile blueprint, backend half: multi-session inference (2026-11-17) — implemented, NOT yet run

Done by a Sonnet 5.5 session with no shell: nothing below has been executed, and Phase 1 itself had not been run
either when this was written (run the tests for both before trusting either).
- New `backend/app/inference.py` `InferenceService`: ONE thread, one engine, many sessions. Each cycle it applies an
  engine change, forgets closed sessions and applies every registered session's pending commands (so it is the only
  feeder thread of every session, as `sessions.py`'s threading contract needs), takes the NEWEST frame of each session
  that has one waiting (older ones resolve as `dropped`; at most `max_batch`=8 per cycle, longest-waiting first), runs
  the poses, feeds each pose to its own session with the frame's size, and resolves the frame's Future with a
  `FrameResult` (pose in the frame's own pixels for drawing, `infer_ms`, `pipeline_ms`, `batch_size`, `dropped`, `error`).
  `submit()` never blocks; `infer()` waits with a timeout and returns None on timeout (never feeds a made-up empty pose).
  Failures are contained per session: an engine error gives that frame an error result and that session is NOT fed
  ("engine hiccup" is not "nobody in view"); a failed batch is retried frame by frame. A failed engine switch keeps the
  current engine; if the default engine fails to load the fallback ("mediapipe") is used, as before; if nothing loads,
  sources get error results instead of hanging.
- Engines: `PoseEngine.stateless` (default True) and `infer_batch()` (default: loop). `YoloEngine.infer_batch` sends the
  whole batch through the GPU in one `predict` call (`infer` now shares `_to_pose` with it; same behaviour as before).
  `MediaPipeEngine.stateless = False`: it tracks between frames, so the service gives each session its OWN instance
  (and closes it when the session is forgotten) instead of sharing one across patients.
- `sessions.py`: `to_reference(res, width, height)` now scales by the frame's LONG side (640 px) so portrait and landscape
  phone frames and the 640 x 480 webcam are judged at the same scale (the blueprint said "640-wide"; width alone would
  inflate a 480 x 640 portrait frame by a third). `Session.process(..., frame_width, frame_height)`; the service always
  passes the real frame size. Desktop frames at 640 x 480 are unchanged (identity); other webcam sizes are now normalised.
  `reset_fall_baseline()` became the flag `request_baseline_reset()` applied in `apply_pending()` (the camera thread no
  longer touches the pipeline).
- `main.py`: the `Worker` is only camera + MJPEG stream; it submits each frame with `service.infer(local_session, ...)`
  and draws the returned pose. Engine ownership and switching moved to the service (`/api/config` engine ->
  `service.request_engine`). Telemetry keys unchanged; `infer_ms` / `pipeline_ms` now come from the service.
- New `backend/app/remote.py`: `/ws/session` gateway. Protocol (docstring): binary `[u32 seq][f64 capture ms][JPEG]` in,
  `{"type":"t","ack":seq,"frame","pose":{kp,bbox},...telemetry}` out, `{"type":"cmd",...}` for exercise / reset / ack_fall
  / escalate. Decoding runs off the event loop, only the newest unprocessed frame is kept, remote ids are `remote-...`,
  the session is closed and forgotten on disconnect. **OFF unless `REMOTE_SESSIONS=1`** (no authentication yet; Phase 4),
  capped by `REMOTE_MAX_SESSIONS` (default 4). Documented in `backend/.env.example`.
- NOT done (still Phase 2 in the blueprint): the frontend (`useCamera`, capture pump, canvas skeleton overlay, dropping
  `/api/video_feed`), and `?session=` scoping of the REST config / fall endpoints. No auth, TLS, or limits beyond the cap.
- Tests (16 new, 79 expected): reference scale by long side and the baseline flag; routing each frame to its own
  session; batching (gate-held engine, 4 sessions in one call, `max_batch` cap with no starvation); newest frame wins;
  failure containment; stateful engine per session and release on forget; engine switch and failed switch; fallback and
  no-engine; five resolutions / orientations give the identical fall verdict; timeout feeds nothing; pending commands
  swept without frames; desktop worker wiring (and `/ws/session` refused by default); remote session end to end with
  TestClient; refusal when disabled or full.
- To verify: `backend/.venv/Scripts/python tests/test_analytics.py` (79/79), then use the desktop app as usual (video,
  skeleton, reps, Test Fall, engine switch YOLO <-> MediaPipe, Privacy, camera switch): behaviour should be unchanged.
  Watch `fps` and the timings line: the frame now makes one hop through the service thread.

### Browser camera mode + PWA basics (2026-11-17) — implemented, NOT yet run

Requested as "Phase 3 of the blueprint"; in the blueprint this is the frontend half of Phase 2 plus the PWA part of
Phase 5 (its Phase 3, browser voice, is untouched). Written without a shell: nothing was type-checked, built or run.
- Header toggle "Server camera | This device" (saved in localStorage). Server camera = the old page, default, untouched
  behaviour. This device = `frontend/src/remote/`:
  - `useRemoteSession.ts`: `getUserMedia` (front/rear via `facingMode: ideal`, camera switch keeps the old one on failure),
    `/ws/session` client with backoff reconnects (re-sends the chosen exercise on a new session), ack-paced JPEG pump
    (one frame in flight, 3 s ack timeout, 480 px / q0.6 stepping down to 320 / 0.4 when the round trip passes 250 ms),
    commands (exercise, reset, ack_fall, escalate, test_fall), screen wake lock, camera-interrupted handling. Nothing
    starts until the "Start camera" tap.
  - `RemoteView.tsx`: `<video playsInline muted>` preview with a canvas skeleton overlay (colour follows fall status),
    rep chips, status chip, start/error states; mirrored by CSS only for the front camera (sent frames never mirrored);
    `RemotePanel`: switch camera, stop, fps / round trip / frame size.
  - `logic.ts` (+ `logic.test.ts`, 14 vitest cases): wire format (matches `struct ">Id"`), capture size by long side,
    adaptive quality, backoff, letterbox mapping, pose drawing.
- Privacy in this mode hides the preview image immediately; frames still go to the server (inference needs them).
- `FallCheck` takes optional `onAck` / `onEscalate` / `canListen` (defaults = old behaviour); in device mode the dialog
  answers over the socket and does not ask the backend for a microphone (no voice for remote patients until blueprint
  Phase 3). Telemetry type: the six server-camera fields became optional. Test Fall works in both modes.
- Backend (small): socket command `test_fall`; `/api/health` now has `remote_sessions` so the UI can say "switched off".
  One test line added (79 tests unchanged in count).
- PWA: `manifest.webmanifest`, SVG `icon.svg`, `sw.js` (network first, never touches `/api` or `/ws`; registered only in
  production builds), `index.html` meta tags, `viewport-fit=cover`; layout padding/wrapping for phones.
- Needs HTTPS for the camera on a phone (a plain `http://192.168...` page has no camera API); localhost is exempt. The
  existing `.loca.lt` tunnel gives HTTPS. The backend needs `REMOTE_SESSIONS=1`.
- Known limits: skeleton lags the preview by one round trip; a backgrounded tab throttles timers (frames slow down);
  a reconnect starts a fresh server session (rep counts reset); no auth yet (Phase 4).
- To verify: `cd frontend; npm install; npx tsc --noEmit; npm test`, then the manual steps in the chat summary.

### Tunnel readiness and access control (2026-11-17) — implemented, NOT yet run

Phase 4 (first slice) of the blueprint, for trying "This device" from a real phone over cloudflared / ngrok. Written without a
shell: not type-checked, built or run. Runbook with exact commands: `docs/cloud_mobile_blueprint.md`.
- Found while planning: a tunnel on :5173 would have published the DESKTOP routes too (the laptop's camera stream
  `/api/video_feed`, camera switch, config, fall buttons, telemetry). New `backend/app/security.py` `TunnelGuard` refuses those
  for any request that arrived through a proxy (forwarding headers the proxy adds, or a public-looking Host name); direct
  localhost / LAN use is unchanged. `/api/health` shows a tunnel visitor only status + `remote_sessions` + `remote_auth`.
- `/ws/session`: `REMOTE_ACCESS_TOKEN` (first message `{"type":"auth","token"}` within 5 s, else close 4401; constant-time
  compare; global limiter of 10 wrong tokens / minute; token never in a URL, so never in uvicorn or tunnel logs), browser
  Origin must match the Host (or X-Forwarded-Host) or be in `REMOTE_ALLOWED_ORIGINS`, cap of 16 unauthenticated sockets,
  session cap re-checked after auth, text messages capped at 4 KB and frames at 2 MB, startup warning when remote sessions
  are on without a token. Teardown on disconnect was already in place (`finally`: forget + close session). Setting name stays
  `REMOTE_MAX_SESSIONS` (not `MAX_REMOTE_SESSIONS`).
- Frontend: `?token=` is read, saved and removed from the address bar; a token field appears when the server wants one
  (checked via `/api/health` before the camera prompt) or rejects it; the token is sent as the first socket message. Any address
  other than localhost starts in "This device" mode (a saved choice still wins); the server-camera telemetry socket is no longer
  opened while that mode is not showing.
- Vite: `allowedHosts` now covers `.trycloudflare.com`, ngrok domains and `.loca.lt` (a tunnel name would otherwise get "Blocked
  request"), for dev and preview; `/api` and `/ws` (ws: true) proxying was already correct and the Host header is not rewritten.
- FastAPI serves `frontend/dist` on :8000 when it exists (single-port option; mounted last so APIs win).
- Tests: 5 new Python tests (token/origin/limiter/host basics, token flow incl. stray auth, lockout, silent client; origin checks;
  tunnel guard; health through a tunnel) and 4 new vitest cases (`extractToken`, `isLocalHost`). Two older tests now also pass
  when `REMOTE_SESSIONS=1` is set in `backend/.env`. Expected: 84 Python tests, 18 vitest.
- Limits: the token is one shared secret, not per-user; REST CORS is still `*`; nothing terminates HTTPS itself (the tunnel does).
- To verify: `backend/.venv/Scripts/python tests/test_analytics.py`; `cd frontend; npm install; npx tsc --noEmit; npm test; npm run build`.

### Voice in "This device" mode (2026-11-17) — implemented, NOT yet run

Audit: the voice code was only ever the FALL CHECK-IN on the server's own microphone (`VoiceListener` + `recorder`, opened while
the fall dialog is up, understanding just "ok" / "help"). The browser never recorded anything, remote sessions have no
`local_voice`, and the page passed `canListen={false}` in device mode. The commands "start my session", "show / hide preview" and
"stop session" did not exist anywhere (only listed as a plan in the blueprint), so they could not have worked in Server Camera
mode either.
- Backend: `intent.classify_command` (verb + noun anywhere in the sentence, one letter of slack, any negative or ambiguity gives
  None); `groq_stt.transcribe(..., filename, content_type)` so browser formats (webm, mp4/m4a, aac, ogg ...) are named correctly;
  `/ws/session` accepts `{"type":"audio","mime","data":base64}` (<= 400 kB, one at a time, 20 per minute per session), runs Whisper in
  an executor, and answers `{"type":"voice","ok","text","intent"}`. `remote.interpret`: with a fall check open only the safety
  answers count (through a `VoiceListener` used for its sentence-joining logic only, no microphone), otherwise only commands;
  "start session" resets the counters server-side. Not stored; `NO_KEY` -> status "unavailable".
- Frontend (`frontend/src/remote/voice.ts`, hook, view): `getUserMedia` asks camera and microphone in one prompt on the Start tap
  (a refused or missing microphone leaves the camera running and shows "Mic blocked"); MediaRecorder type chosen from webm/opus,
  webm, mp4, aac, else the browser default; an always-running recorder is cut at the end of each sentence so the first word is
  not lost; speech detector with an adaptive threshold; paused while the page speaks or ticks. Mic chip (Listening / Processing /
  Mic blocked / Voice unavailable) and the last sentence heard on the video, status line in the panel, and the fall dialog shows
  the same status it shows for the desktop microphone. Stop session stops camera and microphone (one tap restarts); hide / show
  preview = Privacy switch.
- Why on the socket, not `/api/voice`: the blueprint sketched a REST upload; the socket is already authenticated, origin-checked,
  capped and allowed through the tunnel guard, so no new public route and no token header to forward.
- Tests: 4 new Python (commands, transcribe format, voice over the gateway incl. fall answers and context rules, rejects and rate
  limit) -> 88 expected; 10 new vitest (MIME choice, detector, quiet hold) -> 28 expected.
- Limits: the continuous microphone costs one Whisper call per utterance, background talk included (capped at 20 / min); the
  alarm sound during "No response" is not filtered out of the microphone (the phone's echo cancellation has to handle it); no
  voice outside a fall check in Server Camera mode.
- To verify: Python and frontend checks as before, then on the phone say each command, and during Test Fall say "I am fine".

### Voice commands unified across "Server camera" and "This device" (2026-11-17) — implemented, NOT yet run

Before this, commands existed only for phones. Now one path serves both:
- Backend: new `app/voicecmd.py` (`audio_name`, `interpret`, `process_audio`) used by BOTH `/ws/session`'s `audio` message and the
  new `POST /api/voice` (body = audio, Content-Type = format; 413 over 400 kB; a desktop route, so refused through a tunnel).
  `interpret` (moved out of `remote.py`): fall check open -> only "I am fine" / "help"; otherwise only commands; "start session"
  resets the counters. New phrases: "let's go" / "lets go" / "let us go" (<= 6 words, any negative cancels it).
- Frontend: `useVoiceInput` (capture, status, last sentence heard; the caller supplies `upload`: socket on a phone, `uploadToServer`
  on the desktop page), `VoiceChip` (`Heard: "..."`, same in both modes), `intentEffects` in `voice.ts` = the single mapping from a
  command to page changes, applied in one `App` effect for both modes, 700 ms after the chip shows the sentence.
- Rules as built: show preview -> visible; hide preview -> hidden; start session / let's go -> counters reset (server) AND preview
  hidden; stop session -> preview hidden, and a phone also stops its camera (the server's camera keeps running, so in Server camera
  mode "stop session" only hides the preview). "Preview" = `showPreview` state (session-only; the Privacy switch is separate and
  saved; either hides the image). In Server camera mode hiding also switches the backend's blanking on, so the skeleton stays on black.
- FIX (same day): "display preview" was heard but did nothing when the saved Privacy switch was on (an earlier build's voice "hide
  preview" had saved it), because Privacy won over the new session-only preview state. Now "show preview" also switches Privacy off
  (`intentEffects().clearPrivacy`); "hide preview" never turns it on. Also: a later utterance that is not a command no longer cancels
  a command still waiting for its 700 ms delay. vitest 35 expected.
- Session completion auto-hide: a phone session that ends (stop, interrupted camera) hides the preview, and the next Start shows it
  again. ASSUMPTION: Server camera mode has no session end of its own, so nothing else triggers it there.
- Server camera: the browser microphone is opt-in (header "Voice" button, remembered, restarted by itself when the browser already
  granted it) and is released while the fall dialog is open, when the backend's own microphone answers it, so the two never fight over
  one device.
- Tests: Python +1 (`/api/voice`: same answers incl. "let's go", mp4/webm names, 413, tunnel 403) and the command test extended;
  vitest +6 (`intentEffects` identical preview change in both modes, `micErrorText`). Expected: 89 Python, 34 vitest.

### CORRECTION: "preview" means the Exercise Guide, not the camera image (2026-11-17)

I had read "show / hide preview" as hiding the camera image (several rounds of fixes, banners and diagnostics for a symptom that was
really a misreading). It means the Exercise Guide modal (Info button, animated skeleton). Now: `intentEffects` returns `showGuide`;
"show / display preview" opens the guide (the same `guideOpen` state the Info button sets), "hide / close preview" closes it, "start
session" / "let's go" and "stop session" close it, and a phone session that ends closes it. Everything that hid the camera image by voice
(`showPreview`, `previewEpoch`, the banners, `clearPrivacy`, the extra console state line) was removed; the Privacy switch is back to
exactly what it was. The `[voice] heard ... -> intent` and `[voice] applying` console lines stay. vitest 34 expected.
Known: the guide is a modal, so it covers the voice chip; the commands still work while it is open.

### Camera switch fix: "Could not start video source" (2026-11-17) — implemented, NOT yet run on a phone

Cause (likely, from the message): the switch opened the new camera BEFORE releasing the old one, and most phones cannot hold two
cameras at once (Android Chrome: NotReadableError "Could not start video source"). `switchCamera` now releases the current camera
first, waits 200 ms, then tries the other camera with `facingMode: ideal`, then `exact`, then goes back to the previous camera with
a message; if even that fails the session stops with "Tap Start to try again". Taps during a switch are ignored. The microphone is
untouched. No device-ID logic: `facingMode` is what mobile browsers expect, and device IDs change between visits.

### Pose-model badge and switcher on the phone (2026-11-17) — implemented, NOT yet run

- Backend: every `/ws/session` reply carries `engine` (the model in use), `hello` carries `engine` and `engines` (from `ENGINES`),
  and the socket command `{"type":"cmd","name":"engine","value":"yolo"}` calls `InferenceService.request_engine` (applied between batches; a failed
  load keeps the old model; a name not in the list, or no list, is refused). The switch is SERVER-WIDE: it also changes the model the
  laptop and any other phone are using. Reachable through a tunnel because it rides the authenticated socket (`/api/config` is not).
- Frontend: a badge under the status chip, `[chip] YOLO 21 ms tap to switch`: model, smoothed inference time from the server's
  `infer_ms`, tap = switch to the other one (spinner and `-> MediaPipe...` until the replies name the new model; gives up waiting after
  a minute). The panel line also shows `fps / round trip / frame size / model ms`. Pure helpers `engineLabel`, `nextEngine` in `logic.ts`.
- Tests: +1 Python (hello/replies name the model, bad name refused, switch applies, no list = refused), +3 vitest. Expected: 90 Python,
  37 vitest.
- Evaluating on the phone: the inference time is the server-side model time only; `fps` and `round trip` include the network. MediaPipe
  here keeps one model instance per session (it tracks between frames), YOLO shares one, so with several phones YOLO scales better.

### Standby until "let's go" + full-screen HUD (2026-11-17) — implemented, NOT yet run

- Standby is a SERVER-side switch, not just a display: `Pipeline.counting` skips `RepCounter.update` while false. Fall detection,
  escalation and the step counter keep running in standby (a person who falls while setting up is still caught). `Session(standby=True)`;
  phone sessions (`/ws/session`) start in standby, the desktop session still counts from the start (unchanged). `request_start()` =
  counters to 0 + counting on (applied by the feeder, reset first); `request_stop()` = back to standby (counts stay on screen).
  Telemetry carries `counting`.
- Triggers: voice "start my session" / "let's go" (`voicecmd.interpret`), socket commands `start_session` / `stop_session` (HUD
  buttons), and for the desktop `POST /api/session/start|stop` (desktop routes, refused through a tunnel). "Stop session" by voice on a
  phone still stops the camera (as before) and the next Start begins in standby again; on the desktop it pauses counting.
- HUD (`remote/Hud.tsx`, shared by both modes): status pill, `READY — Say "Let's go" to start` (amber) / `ACTIVE` (green), large rep
  counter (dimmed in standby), model badge/switcher, camera flip (phone), full-screen toggle, Start / Pause, fall pill, voice chip.
  Only its buttons take taps. Start also closes the Exercise Guide.
- Full screen (`remote/useFullscreen.ts`): Fullscreen API on the video wrapper, falling back to `fixed inset-0 z-40` where the
  browser has none (iPhone Safari); body scroll locked and Esc exits in the fallback. The HUD is inside the wrapper so it stays
  visible; the Exercise Guide and fall check are native modal dialogs (top layer) and sit above either kind.
- Tests: +3 Python (standby pauses reps but not falls and start/pause flags; phone session standby and HUD commands over the socket;
  desktop session routes + tunnel refusal). Expected 93 Python, 37 vitest (no new pure helper). The browser parts (HUD layout,
  Fullscreen API, fallback) have not run.

### Speed on phones: what was and was not changed (2026-11-17) — implemented, NOT yet measured

Premise check: the phone runs NO pose model (it sends JPEG frames; YOLO and MediaPipe run on the server, blueprint Phase 6 is the
on-device version), so a "mobile INT8 bundle", `modelComplexity` on the phone and "UI thread" inference do not exist yet.
- YOLO (server GPU): FP16 (`half=True`) on CUDA devices, never on CPU; `YOLO_HALF=0` turns it off. INT8 was NOT done: it needs a
  TensorRT / ONNX export with calibration data, costs keypoint accuracy (the rep counter depends on it), and cannot be validated here.
  `YOLO_IMGSZ` (multiple of 32, default 640 unchanged) lets the network input match the phone's <= 480 px frames (about 45% less work).
- MediaPipe: its three confidences were already 0.5. "Lite" (complexity 0) is now a selectable third model ("MediaPipe Lite" on the
  badge and the desktop picker) for live comparison; it is faster but LESS accurate than Full, so it is not the default.
- Resolution: the phone already sends at most 480 px on the long side (adaptive 320-480), under the requested 640x480; unchanged.
- Throttle: the phone sends at most 20 frames/s (`canSendFrame`: min 50 ms gap, one frame in flight, overdue ack = resend). The server
  already keeps only the newest frame per session, so a busy GPU skips frames instead of queueing.
- The real limit through a tunnel is usually the round trip (one frame in flight = fps ~ 1 / round trip). The panel shows `round trip`
  and the model's own `ms`: if `ms` is small and the round trip is large, no model change will help; the fixes are a closer server,
  allowing 2 frames in flight, or on-device inference (Phase 6).
- Tests: +2 Python (FP16 only on GPU and env settings; Lite selectable) -> 95 expected; +5 vitest -> 42 expected.

### Phase 6: pose model on the phone, only joints sent (2026-11-17) — implemented, NOT yet run

Decision (deviates from the request on one point): the model runs on the phone, but rep counting, fall detection and escalation STAY
on the server's pipeline and receive the phone's keypoints. Re-implementing them in TypeScript would have meant two copies of
safety-critical logic (falls) that can drift apart. The goal (no video over the tunnel, no round trip before the skeleton appears)
is met: the skeleton is drawn from the phone's own pass; telemetry (reps, ACTIVE/READY, fall state) arrives at ~10 per second and
nothing waits for it. The server is not "only synced at rep increments": it needs the pose stream to count reps and watch for falls.
- Phone: `@mediapipe/tasks-vision` 0.10.14 (exact pin: its WebAssembly is loaded from jsDelivr at the same version), imported on demand;
  Lite model (complexity 0), VIDEO mode, GPU delegate then CPU, confidences 0.5, model from `/models/...` on this server (cached a week)
  then Google's host; `requestAnimationFrame` loop on NEW video frames, max 30 per second; 33 landmarks -> the 17 joint names the server
  already uses (`LANDMARK_JOINTS`); skeleton drawn straight onto the canvas (not through React state). If it cannot start or the model
  throws, the page falls back to sending frames (message shown; not remembered).
- Wire: `{"type":"kp","seq","t","w","h","kp":{joint:[x,y,conf]}}` (about 0.6 kB, never queued: skipped while the socket's send buffer
  is above 64 kB). Server: `parse_keypoints` validates (joint names from `engines.base.KEYPOINT_NAMES`, finite numbers, sane size),
  `InferenceService.submit_pose` feeds the session through the same batch cycle with NO engine involved (works even if no server engine
  loaded), replies at most 10 per second and without `pose`. Standby, HUD, voice, falls: unchanged.
- Default: on-device on phones/tablets (`isMobileDevice`), server models elsewhere; the badge cycles `On-device MediaPipe -> MediaPipe
  -> MediaPipe Lite -> YOLO` and remembers the choice. The badge shows `fps · ms` of the local model.
- Server also serves `/models/pose_landmarker_{lite,full}.task` (exact names only) from `backend/models/`; the service worker skips it.
- Tests: +5 Python (keypoint validation; submit_pose with no engine; poses and frames in one cycle; keypoint messages over the socket
  with standby start/stop and a bad message; the model route) -> 100 expected; +7 vitest -> 49 expected.
- Limits: needs WebAssembly and (for speed) WebGL in the phone's browser; a hidden tab pauses `requestAnimationFrame`, so on-device pose
  (and falls) stop while the page is in the background, as for frames before; the full-precision model is not offered on the phone.
- To verify: `npm install` (new dependency), the usual checks, then on the phone: the badge should say `On-device MediaPipe`, 25-30 fps
  and a few ms to tens of ms; check the HUD counts reps after "let's go" and the Network tab shows only small text frames, no JPEGs.

### Client-side YOLO in the model switcher (2026-11-17) — REVERTED the same day, see "Client-side YOLO removed" below

- Third on-device option beside MediaPipe: **On-device YOLO** (YOLO11n-pose through ONNX Runtime Web; WebGPU, else WebGL, else
  WebAssembly, each tried with a warm-up run so a provider that cannot run every operator falls through to the next). The badge now
  cycles `On-device MediaPipe -> On-device YOLO -> MediaPipe -> MediaPipe Lite -> YOLO` (the last three on the server). Same wire
  format as MediaPipe (`kp` messages), same server pipeline; YOLO gives the 13 COCO joints (no feet), as the server's own YOLO does.
- Input is letterboxed to 320 x 320 before the network (`YOLO_INPUT`, `letterbox`), one reused buffer and canvas (no per-frame
  allocation), single-threaded WebAssembly (more threads need cross-origin isolation); the output is decoded by `decodeYoloPose`
  (best-scoring person, mapped back to the video frame's pixels, either output layout).
- Detector abstraction (`onDevicePose.ts` `Detector`, `createDetector`): the loop now supports asynchronous detectors, skips frames
  while a detection runs (never queues), and reports fps, ms and a `slow` verdict (below 12 fps, with hysteresis, after 2 s of
  warm-up). Slow shows an amber model badge and an amber "Slow on this device" pill; it does NOT switch models by itself, so the
  comparison stays yours. Load or run failures fall back to server frames with the reason shown (as for MediaPipe).
- The ONNX file is NOT in the repository: run `backend/.venv/Scripts/python backend/scripts/export_onnx.py` (320, FP32, opset 12,
  about 11 MB; `--size 480` for another). The backend serves `/models/yolo11n-pose-{320,480}.onnx` (exact names only); without the file
  the page says so and uses the server. INT8 / FP16 are not done: FP16 cannot run on onnxruntime-web's WebAssembly provider, INT8 needs
  calibration images and costs keypoint accuracy that the rep counter relies on.
- ONNX Runtime Web is loaded from jsDelivr at a pinned version (1.20.1; `ort.all.min.mjs`), not installed with npm, so the build does
  not bundle it; the file name and version were written from memory and are unverified.
- Tests: +7 vitest (letterbox, decoding in both layouts, joint names, threshold, bad shapes, slow verdict, mode mapping) -> 56 expected;
  the model-route test also covers the ONNX names. Python stays 100.

### Fixes found while exporting the ONNX model (2026-11-17)

- The export failed on this machine because Ultralytics' automatic install of `onnx` / `onnxruntime` / `onnxslim` (via `uv`) is refused
  by the network's HTTPS certificate ("invalid peer certificate: UnknownIssuer"). Install them yourself: `uv pip install --system-certs
  --python backend/.venv/Scripts/python.exe "onnx>=1.12,<2" onnxruntime onnxslim`, then rerun `export_onnx.py` (documented in the script).
- Real bug in the earlier FP16 change: Ultralytics 8.4 deprecates `half` (it maps to `quantize=16`) and prints "'half' is deprecated" on
  EVERY call, so the server's YOLO would have logged that warning on every frame. `YoloEngine._precision_args()` now passes
  `quantize=16` when the installed Ultralytics knows it, `half=True` for older ones. The export script no longer passes `half` either.

### Client-side YOLO removed (2026-11-17)

On the phone it lagged badly, so it is gone and the on-device pipeline is back to MediaPipe only.
- Removed: the `device_yolo` mode and badge entry, the ONNX Runtime Web loader, YOLO letterbox / output decoding, the slow-device
  warning (it was added together with YOLO), `/models/yolo11n-pose-*.onnx` from the model route (now only the two MediaPipe files),
  and the 7 vitest cases for them.
- Restored to their pre-YOLO form: `onDevicePose.ts` (the detector abstraction and the asynchronous loop with its `busy` gate and
  warm-up bookkeeping went too, so the loop is exactly the one that felt smooth), the on-device parts of `useRemoteSession.ts`,
  `RemoteView.tsx` and `Hud.tsx`, and the two-value `InferenceMode` ("device" | "server") in `logic.ts`.
- KEPT, because it has nothing to do with the browser: the server-side YOLO FP16 fix (`YoloEngine._precision_args()` passes
  `quantize=16`, or `half` on older Ultralytics, instead of a `half` that Ultralytics 8.4 warns about on every frame).
- Files to delete by hand (they could not be deleted from the assistant's session, and were emptied): `frontend/src/remote/onDeviceYolo.ts`,
  `backend/scripts/export_onnx.py`, and the generated `backend/models/yolo11n-pose-320.onnx` (git-ignored). The Python packages
  `onnx`, `onnxruntime`, `onnxslim` (+ `protobuf`, `ml-dtypes`, `colorama`) installed for the export can be removed too.
- Expected: 100 Python tests, 49 vitest.

### Mobile performance & seated/bed tracking roadmap (2026-11-17) — strategy only, no code changed

Saved as `docs/mobile-performance-roadmap.md`. Key findings from the code: phone poses arrive 2D only (world landmarks are not
sent, so angles depend on the camera angle); rep `rest`/`peak` thresholds assume a standing body; one hidden joint (often the hip
or ankle when seated) switches an angle off; per-frame EMA smoothing lags more on slow phones; and the fall detector's "flat body"
cue would raise an alarm for anyone lying in bed for 5 s. Recommended order: 3D world landmarks, standby rest calibration and
time-based smoothing first; then 640x480 camera input and fewer re-renders; posture profiles with physio review; bed fall logic
only with sign-off; Capacitor native only when background fall monitoring is required.

### Roadmap Phase A items 1, 2 and 5: 3D world landmarks, standby calibration, time-based smoothing (2026-11-17) — implemented, NOT yet run

- **A1, world landmarks.** The phone sends MediaPipe's metric 3D `worldLandmarks` in the `kp` message (`"world":{joint:[x,y,z]}` in
  metres from the hip centre, 3 decimals; `worldToJoints`, `keypointMessage`), `parse_keypoints` validates them (known joints, finite,
  within 3 m) and attaches them to `Keypoint.world`, and `joint_angles` (which already preferred 3D when every joint of an angle has it)
  now measures phone poses in 3D, so a knee bending toward the camera is no longer flat in the image. Optional on the wire (older
  clients unchanged); `to_reference` already preserved `world`.
- **A2, rest calibration.** While counting is off (READY) the pipeline keeps the last 1.5 s of each limb's exercise-joint angle. On
  standby -> counting (`Session.apply_pending`, after the reset) `Pipeline.calibrate()` sets each limb's `rest` to the MEDIAN of that
  window and its `peak` to rest +/- the table's range, in the table's direction (`rep_counter.calibrated_limits`, peak kept within
  15..178 degrees). It keeps the table's standing values for any limb whose standby cannot be trusted: fewer than 8 samples or under
  0.8 s of them, newest sample older than 0.5 s (person left the view), angle moved more than 20 degrees, or less than 25 degrees
  between rest and peak (e.g. Sit-to-Stand measured while already standing). The desktop session counts from the start, so it has no
  standby and is never calibrated (unchanged). Limb telemetry gains `rest`, `peak`, `calibrated`; the HUD shows "calibrated rest 92°"
  while ACTIVE.
- **A5, time-based smoothing.** `EMA(tau=0.08)` with `alpha = 1 - exp(-dt / tau)` replaces the per-frame 0.35 in the pipeline (0.08 s is
  what 0.35 amounted to at 30 fps, so desktop behaviour is about the same); `EMA(alpha)` still works for fixed use. A slow phone now
  reacts over the same time as a fast one.
- **Known limit of A2:** the peak keeps the table's range. For an exercise whose seated range of motion is smaller (e.g. Hip Flexion
  starting at 90 degrees), the peak may not be reachable and reps would still not complete; the table needs seated variants (roadmap A3)
  with a physiotherapist's angles.
- Tests: +6 Python (time-based smoothing, 3D angles, world in messages, calibration, calibration guards, session wiring) -> 106
  expected; +3 vitest -> 52 expected.

### Bug report: phone counts the left arm only after Phase A (2026-11-17) — calibration made all-or-nothing

- **Investigated, no asymmetry found in:** coordinate mirroring (nothing in `landmarksToKeypoints`/`worldToJoints` flips or swaps
  sides; the index table maps 11/13/15 to `l_*` and 12/14/16 to `r_*` for 2D and 3D alike, and a mirror would not change an angle
  anyway), the world-landmark parsing (`parse_keypoints` is side-blind) and `joint_angles` (the same triples for both sides, 3D only
  when all three joints have it, per side). The phone-only difference is the new standby calibration, which desktop "Server camera"
  sessions never run.
- **Most likely cause (not reproduced here, no phone data):** `Pipeline.calibrate()` decided PER LIMB. If one arm was measured and the
  other was not (out of the frame, moving, held at a odd angle), the first counted against the person's baseline and the other
  against the standing table, so one arm counted and the other never did.
- **Fix:** for two-sided exercises calibration is all-or-nothing: if either limb cannot be measured or the two measured rests differ
  by more than 30 degrees (`CAL_SIDE_DIFF_DEG`), BOTH keep the table values. Tests: +1 Python (`test_one_side_is_never_calibrated_alone`),
  calibration-guard test now fills every limb.
- **If it still happens:** the HUD pill and `reps.L/R` telemetry (`rest`, `peak`, `calibrated`, `angle`) show what each side measured;
  a right `angle` of null means the right joints are below 0.5 confidence in the phone's framing.

### Follow-up: standing worse on the phone after Phase A (2026-11-17)

- Cause (likely): calibrating a STANDING person moved `rest`/`peak` by sensor noise (e.g. elbow 160 -> 150 pulls the peak from 50 to 40,
  harder to reach). Fix: calibrate only when the measured rest differs from the table's by 25 degrees or more (`CAL_MIN_SHIFT_DEG`);
  standing keeps the table values exactly as on the laptop. Seated/lying still calibrate.
- New switch `ANGLES_3D=0` (backend/.env) measures angles in 2D again, to tell whether the phone's 3D world landmarks are what hurts
  standing; default is 3D on. Test `test_one_side_is_never_calibrated_alone` extended with the standing case.

### Bug report: high-end phone worse after Phase A; `ANGLES_3D=0` had no effect (2026-11-17)

- **Found:** `ANGLES_3D` was read with `os.getenv`, but nothing loads `backend/.env` into the process environment, so the earlier
  `ANGLES_3D=0` test changed nothing. The switches now go through `app/voice/settings.get` (environment, then `backend/.env`), read
  when a session starts (reload the phone page, no backend restart needed): `ANGLES_3D=0` (2D angles), `SMOOTH_TAU_S=0` (the old fixed
  per-frame alpha 0.35), `CALIBRATE=0` (table rest/peak). All three off = the pre-Phase-A pipeline. Test added.
- **Reviewed for frame rate, no defect found:** the phone loop is capped at 30 fps (a 60 fps phone still sends 30), timestamps are the
  phone's `performance.now()` and strictly increasing, `alpha = 1 - exp(-dt/tau)` is correct at any dt (at 30 fps 0.29 vs the old 0.35),
  calibration samples by time (1.5 s window), keypoint replies are rate-limited to 10/s and only the newest waiting frame per session is
  processed. So the fast phone is to be bisected with the three switches rather than guessed at.

### Native demo APK plan (2026-11-17) — written, NOT yet run

- `NATIVE_DEMO_PLAN.md`: Android build without Android Studio (JDK 21, command-line tools, sdkmanager, env vars), Capacitor
  WebView shell loading the tunnel URL (`server.url`), camera/mic permissions, Galaxy A52 setup (USB debugging, Auto Blocker off),
  demo-day runbook with scrcpy, troubleshooting table.
- `scripts/build_and_serve_apk.py`: writes `frontend/capacitor.config.json`, `cap add/sync android`, patches the manifest
  permissions, `gradlew assembleDebug`, serves only the APK on `0.0.0.0:8080` with a terminal QR code (`qrcode` package), or
  `--adb` installs over USB. Refuses URLs containing a token; warns on http (no camera outside a secure context).
- No pose-tracking or backend code changed. JDK 21 instead of the requested 17: current Capacitor needs it.

### Native demo APK: Section 2 done, rehearsal build in progress (2026-11-17/18)

- **Section 1** (JDK 21 Temurin 21.0.13, command-line tools, platform-tools, android-35, build-tools, Gradle Windows-ROOT
  config) done by the user.
- **Section 2 done:** `npm install @capacitor/core @capacitor/android` and `-D @capacitor/cli` in `frontend/` (npm audit reports 3
  vulnerabilities, 1 moderate, 2 critical; already present before the CLI; deliberately NOT `npm audit fix --force` before the demo,
  review after). `npm run build` ok. `qrcode` 8.2 installed in `backend/.venv`.
- **Section 3 rehearsal** with `--url https://showtimes-shore-review-bunch.trycloudflare.com/` (bare URL, no token):
  - `frontend/capacitor.config.json` written; `npx cap add android` created `frontend/android/`; the manifest got 4 permission
    lines (CAMERA, RECORD_AUDIO, MODIFY_AUDIO_SETTINGS, camera feature; INTERNET was already there); `cap sync` ok.
  - Fix 1: `JAVA_HOME is not set`: Claude Code's shell keeps the environment from its start. Workaround: export `JAVA_HOME` /
    `ANDROID_HOME` in the command (with `cygpath -w`), or restart Claude Code from a new terminal.
  - Fix 2: `'gradlew.bat' is not recognized`: Claude Code's shell does not search the current folder. The script now calls
    gradlew by its full path.
  - Fix 3: `PKIX path building failed` on the Gradle download: the wrapper reads only `GRADLE_OPTS`, not `gradle.properties`.
    The script sets `GRADLE_OPTS=-Djavax.net.ssl.trustStoreType=Windows-ROOT` on Windows (full certificate checks, Windows store).
    `NATIVE_DEMO_PLAN.md` step 1.5 updated.
  - Fix 4: Gradle download `timeout (10000ms)`: the script patches `android/gradle/wrapper/gradle-wrapper.properties` to the
    smaller `-bin` package and `networkTimeout=120000` (new `patch_wrapper()`).
  - Result: Gradle 8.14.3 downloaded, daemon started; the run was then moved to the background by Claude Code's 120 s limit and
    killed because Windows ran critically low on memory (not a build error).
- **Next:** stop the backend and leftover daemon (`frontend/android/gradlew.bat --stop`), run the script in a separate Git Bash
  window (not via `!`), expect `BUILD SUCCESSFUL` + QR code, then install on the Galaxy A52 and continue Section 4 (Auto Blocker
  off, scrcpy). If memory runs out again: cap Gradle memory (`org.gradle.jvmargs`) based on the laptop's RAM.
- Still open before any commit: remove the real `REMOTE_SESSIONS` / `REMOTE_ACCESS_TOKEN` lines from `backend/.env.example`,
  rotate the token; keep `docs/sample/` and `backend/.env` out of git.

### Start countdown and preview speech cues (2026-11-18)

- New `frontend/src/cues.ts`: `speak()` (speechSynthesis, silent if unsupported) and `runCountdown()` ("Ready...", 3, 2, 1, "Go!", one `setTimeout` per step, cancellable).
- `App.tsx`: the Start button and the voice command "start_session" show a non-blocking overlay (`pointer-events-none`, no blur) and speak each step; the Exercise Guide ("preview") opening/closing says "Displaying preview" / "Closing preview" (skipped while a countdown is speaking).
- Nothing touches the frame loop, the pose code or the server. Reps still start counting immediately on Start (the overlay is a cue only). Phone check pending; tsc/vitest/build not yet run.

### Canvas sized to the video picture, not the outer box (2026-11-18)

- `RemoteView.tsx`: a "stage" rectangle = `fitContain(frame ratio, outer box)` (centred, rounded px) now wraps the `<video>` and the `<canvas>`; both fill the stage only. The canvas backing store, the `paint()` mapping and `drawPose`'s box all use the stage size, so the canvas is exactly the rendered picture (previously it was the whole outer box and relied on `drawPose`'s letterbox maths to match the video's `object-contain` bars, which could drift when the ratio of `s.frame` and the real frame differed, e.g. during a camera-tier change). The mirror transform sits on the stage. The outer box (aspect-ratio / `78dvh` / full screen) is unchanged and is what the ResizeObserver measures.
- Mathematically the same picture when everything agrees; the difference is robustness. Frame loop untouched (paint still allocation-free: `stage` is one small object per render, not per pose). tsc/vitest/build and a phone check (20:9 phone, portrait + fullscreen) pending.

### Offline fix 4: local engine follows the server's ANGLES_3D=0 (2026-11-18)

Bug report: "This phone" counts worse than the server. Compared both paths: the keypoints, capture time, `to_reference` scaling, EMA (tau 0.08 s), rep state machine and calibration are identical (6 golden traces match). One real difference: `backend/.env` has `ANGLES_3D=0` (2D angles; 3D world landmarks counted worse on phones, see 2026-11-17), so the server measured in 2D, while `LocalEngine` used the Pipeline default (3D) because the offline engine cannot read `backend/.env`. `useRemoteSession.ts` `localEngineOptions()` now defaults the local engine to 2D too; `localStorage.angles3d="1"` or `VITE_ANGLES_3D=1` switches to 3D. Pipeline defaults and golden tests are untouched. Not yet checked on a phone.

### Offline fix 3: "This phone" forces on-device pose (2026-11-18)

Root cause of the banner still showing: `analyticsLocal = engine==="local" && inference==="device"`, and a desktop browser defaults to `inference="server"`, so Analytics = "This phone" still ran the server path (server counts, socket required, "Connection lost"). `start()` now sets inference to `device` (not saved) when the engine is `local`. If the model itself cannot load, the existing fallback to the server stays (and then the banner is true). Not yet checked on a phone.

### Offline fix 2: no "Connection lost" banner with local analytics (2026-11-18)

`RemoteView.tsx`: `lost` now also requires `!s.analyticsLocal`, so the red banner can never show while the reps are counted on the phone (only "Offline — caregiver will not be notified"). The hook already stayed `live` and skipped the socket offline. The banner is still right with Analytics = "Server", where the server is the counter. Not yet checked on a phone.

### Offline fix: no remote fetches in local/offline mode (2026-11-18)

Bug: offline (or "This device"), the page still fetched `/api/exercises` and `/api/exercise_info` and retried the `/ws/session` socket.
- `App.tsx`: the exercise list and the guide table come from the bundled `shared/exercises.json` (`BUNDLED_INFO`); the server's table is fetched only in "Server camera" mode and only when `navigator.onLine`. `/api/cameras` is only in the desktop server-camera picker.
- `useRemoteSession.ts` `connect()`: with local analytics and no network it does not open a socket; it sets `linkUp=false` and rechecks every 3 s (no traffic). `start()` already skipped `/api/health` and opens camera + IndexedDB first; the reconnecting banner never shows with local analytics (only the "Offline — caregiver will not be notified" banner).
- Server analytics offline still shows "Reconnecting" (the server is the counter there, so that is true).
Not yet checked on a phone.

### Offline-first, Phase 4: sync queue, SQLite store, offline notices (2026-11-18)

**Verified (2026-11-18):** Python 118/118 (one test of mine fixed: newer FastAPI has no `.path` on `app.routes` entries); vitest 124/124 (12 new sync tests); tsc clean; build ok; engine still 28 us per frame. `@capacitor/network` and `@capacitor/app` installed; an APK rebuild (`cap sync`) is needed for the native events. Not yet checked on a phone.
- **Backend:** `app/storage.py` (stdlib `sqlite3`, WAL, `data/rehab.db`, tables `sessions` + `events`; `insert_batch` is one transaction, `INSERT ... ON CONFLICT (id) DO NOTHING`, a known id is skipped with its events), `app/sync_api.py` (`POST /api/sync/sessions`, `GET /api/sessions?limit&offset` with `summary: {sessions, reps, fall_sessions}`), wired in `main.py` (storage opened on first use, closed in the lifespan), both paths added to `PUBLIC_API` in `security.py`.
- **Server-side rules:** the access token (`Authorization: Bearer` or `X-Access-Token`, constant-time, 10 wrong guesses per minute then 429) is required whenever `REMOTE_ACCESS_TOKEN` is set, for history too (patient data); with no token configured only a direct (localhost/LAN) caller is let in, a tunnel gets 403. Body capped at 1,000,000 bytes (by Content-Length and again while streaming), at most 20 sessions per upload, 20,000 events per session, plain-Python validation of every field (types, id format, event types), duplicate ids in one upload refused; nothing is stored unless the whole batch validates.
- **Frontend, `src/sync/syncQueue.ts` (+ `index.ts`):** triggers = app start, `online`, `@capacitor/network` `networkStatusChange` (connected), `@capacitor/app` `resume` (native only, loaded on demand), a 60 s timer that exists only while sessions wait, and the end of a recorded session. Each pass: list unsynced -> if a workout session runs, do nothing and ask again in 5 s -> `GET /api/health` with a 2 s abort (not `navigator.onLine`) -> wait for `requestIdleCallback` (2 s deadline; `setTimeout` fallback) before each batch -> up to 20 sessions per request, fall/alert sessions first (`flagged`, set by the recorder), body kept under 900 kB by halving -> `markSynced` only on a 2xx; 401/403 = "access token needed" (no retry loop); 429/5xx/network = backoff `reconnectDelay` x jitter; 400/413 splits the batch until one session alone is refused, which is set aside (`syncFailed`, kept on the device) -> synced sessions older than 30 days are pruned and `meta.lastSync` is written. Sessions left open by a crash count as finished after 24 h.
- **UI:** the camera card shows "All synced" / "N sessions waiting to sync" (+ offline / access token needed / will retry) through `useSyncExternalStore`, which re-renders only that card and only when the status object changes. In local-analytics mode, when the server link is down (or not answering within 4 s of the start), the HUD shows a solid red "Offline — caregiver will not be notified" banner, and the voice chip says "Voice needs a connection". The local fall check (dialog, beep, spoken question, I'm OK / Help) is untouched.
- **Deliberate choices:** uploads wait for the whole workout session (camera loop running), not only for counting: the loop runs in standby too. The first write of a session id wins (the phone only uploads finished sessions, so there is nothing to update).
- **Tests:** Python (`tests/test_analytics.py`): idempotent upload, mixed new/known batches, event counts not doubled, paging and totals, workouts summed for reps, token required / header variants / lock-out after wrong guesses, no token = direct only (tunnel 403) and with token through the tunnel guard, 413 by Content-Length and by streaming, 20+ sessions, bad JSON, 11 malformed shapes, atomic rejection, WAL file surviving a restart, routes registered and allow-listed. Frontend (`sync/syncQueue.test.ts`, fake store and network, fake timers): nothing to do, 2 s health timeout then backoff, busy deferral, batches of 20 with flagged first, marking only after 2xx, 5xx retry, 401 no hammering, split on 400, split on size, trigger folding, 60 s timer lifecycle, status notifications.

### Offline-first, Phase 3: local storage (IndexedDB) (2026-11-18)

Written without a shell; tsc / vitest / build to be run by the user.
- **`frontend/src/store/`:** `db.ts` (a ~110-line IndexedDB wrapper, no library: database `rehab` v1 with `sessions` (key id, index startedAt), `events` (auto id, index sessionId), `meta` (key); every write is ONE transaction; the connection is reopened if the WebView closed it; `commit(session, events)`, `getSession`, `listSessions`, `listEvents`, `listUnsynced`, `markSynced`, `getMeta`/`setMeta`, `deviceId()` (made once, kept in `meta`; `lastSync` is for Phase 4)), `types.ts` (SessionRecord: id uuid, exercise, startedAt, endedAt, reps/ROM per limb, calibration per limb, workouts, appVersion, `synced:false`; EventRecord: sessionId, t, type, payload), `recorder.ts`, `index.ts` (`requestPersistence`, the shared `getStore`, `engineSummary`, `recordEngine`).
- **Persistence:** `main.tsx` calls `navigator.storage.persist()` at start-up (best effort, logged).
- **`SessionRecorder`:** collects events in memory; writes the session record plus the new events in one transaction at `begin()`, every 5 s, on `visibilitychange: hidden` and `pagehide`, and at `end()` (with `endedAt` and an `end` event). Writes run one after another; a failed write keeps its events (cap 5000, oldest dropped) for the next try, logs one warning per 30 s and never throws; no storage (IndexedDB blocked) = it quietly does nothing. A session is one camera session (Start camera .. Stop); because Start resets the counts, each Start..Stop stretch is kept as an entry in `workouts` with its own reps/ROM. If the app dies mid-session the record has `endedAt: null` and the events up to the last write.
- **Engine events (`engine/localEngine.ts`):** an optional `listener` gets `start`, `stop`, `pause`, `resume`, `exercise`, `rep` (every completed rep, with the per-limb snapshot), `fall` (status change), `alert` (state change). Created only when something happens; the per-frame check is the same two compares as before. A throwing listener is ignored.
- **Wiring (`useRemoteSession.ts`):** a local-analytics session creates the recorder next to the engine (`recordEngine`) and ends it in `stop()`, when the analytics move to the server, or when the badge switches to a server model. Server-analytics sessions are not recorded here (the server will own them).
- **Tests:** `store/recorder.test.ts` (in-memory store + fake timers: begin write, 5 s batching, on-demand and final write, workouts per Start..Stop, failed write retried without throwing, no storage, engine events incl. a throwing listener). The IndexedDB wrapper itself is not unit-tested (no IndexedDB in the test environment): it needs one look on a phone (Chrome `chrome://inspect` > Application > IndexedDB > rehab).
- **Phase 4 notes:** `listUnsynced()` only returns sessions with an `endedAt`; sessions left open by a crash (endedAt null) need a rule (e.g. treat as ended after a day). A session's final write failing is not retried after `end()`.

### Offline-first, Phase 2: TypeScript pipeline + golden parity (2026-11-18)

**Verified (2026-11-18):** `make_golden.py` wrote 6 traces (2,940 frames); Python 111/111; vitest 103/103 (golden parity on all 6 traces: reps, rep states, calibration, fall, alert, steps, angles within 0.2 deg); tsc clean; build ok. Engine speed measured at 26-38 us per frame (budget 100 us). One test bug fixed on the way (the golden test took the limb sides from frame 0, which is wrong across an exercise switch). Not yet checked on a phone.
- **Shared exercise table:** `frontend/src/shared/exercises.json`, generated from `rep_counter.EXERCISES` (still the master) by `scripts/export_exercises.py` (`--check` verifies). Python test `test_the_phone_engine_reads_the_same_exercise_table_as_the_server` fails when the two differ. The engine imports it through `engine/exercises.ts`.
- **Engine, `frontend/src/engine/`:** `kinematics.ts` (PoseFrame, joint angles 2D/3D, time-based EMA), `repCounter.ts` (LimbFSM, calibrated_limits), `fallDetector.ts`, `escalation.ts`, `mobility.ts`, `pipeline.ts` (bbox from keypoints, to-reference scaling, standby calibration with the CAL_* guards), `localEngine.ts` (a Session on the phone: start/stop/pause/resume/exercise/reset/ack/escalate/test fall + `telemetry()` in the server's shape + a `version` counter). Faithful ports; the joints are copied once per frame into a preallocated Float64Array and everything is addressed by index; history is in ring buffers (30 s hip history capacity 4096, mobility 512/128, calibration 256). No object, array or closure is created per frame; `telemetry()` allocates and runs only on the 250 ms UI tick or when `version` changes (rep, fall status, alert state, counting, exercise).
- **Parity tests:** `scripts/make_golden.py` builds 6 synthetic traces (elbow curls with 3D joints at 320x240, seated knee flexion with calibration + pause/resume, fall + countdown + alert + "I'm OK", Test Fall + lost pose + emergency, walking, exercise switch + reset), replays each through the real server path (`parse_keypoints` -> `Session.apply_pending` -> `Session.process`) and writes inputs + Python results to `tests/golden/*.json` (recorded phone traces: put them in `tests/golden_input/`). `frontend/src/engine/golden.test.ts` runs the same frames through `LocalEngine` and compares reps, rep states, calibrated flags, fall status, on-floor, alert state, rechecks, counting, steps (same value sequence, each change within 1 frame) and the angles (same joints present, within 0.2 degrees). `engine.test.ts` has golden-free unit tests, and a speed check (bound 0.5 ms per frame; the printed `us per frame` is the real number). Python test `test_golden_traces_replay_the_same_way_twice...` checks the generator is deterministic and the traces really count/calibrate/fall/step.
- **Engine switch + UI:** `EngineMode` "local" | "server" (`logic.ts`, `defaultEngineMode`: local in the bundled APK, server otherwise; remembered in `localStorage.engine`; a Server / This phone toggle in the camera panel, disabled while running). Local analytics need the on-device pose (`analyticsLocal`). In `useRemoteSession.ts`: a local session starts with no network (no health check), creates the `LocalEngine`, and the on-device loop's `on.pose` callback feeds it (`eng.process`) instead of sending a kp message; commands (`send`) apply to the engine at once; an optional WebSocket stays as a voice link (exercise / start / stop / reset are mirrored to the server so its interpreter's pause/resume state matches; a dropped link does not touch the session and retries quietly). If on-device pose fails, or the badge switches to a server model, the analytics move to the server.
- **Voice:** `App.tsx` applies `pause_session` / `resume_session` / `start_session` from the server's interpretation to the local engine; the HUD, FallCheck, cues are unchanged. Voice needs the link (Groq runs on the server), so offline the mic chip reports "Not connected to the server" and the buttons do the same things.
- **Known behaviours kept from the server:** resume re-measures the rest angle from the posture held during the pause (as `apply_pending` does); the Test Fall/ack flow is the same state machine.
- **Not done / needs sign-off:** an "offline, caregiver not notified" notice in the HUD (the safety decision in `docs/offline-first-architecture.md` section 3); the local session is not stored yet (Phase 3).

### Offline-first, Phase 1: bundled APK (2026-11-18)

Plan and phases: `docs/offline-first-architecture.md` (Phases 2-4 open; Phase 2 needs a sign-off on the offline fall behaviour).
- New `frontend/src/net.ts` (`apiUrl`, `wsUrl`, `isBundled`, `cleanBase`, `toWsUrl`; base from `localStorage.apiBase` or `VITE_API_BASE`, "" = same origin so browsers are unchanged) + `net.test.ts`. Every `fetch("/api/...")`, the two WebSockets and the video_feed `<img>` now use it (`App.tsx`, `FallCheck.tsx`, `useTelemetry.ts`, `useRemoteSession.ts`, `useVoiceInput.ts`).
- `onDevicePose.ts`: `WASM_BASE` is `/mediapipe/wasm` when `VITE_BUNDLED=1`, the CDN otherwise. New `frontend/scripts/copy-assets.mjs` (copy / `--clean` the WASM and Lite model into git-ignored `public/mediapipe`, `public/models`).
- `scripts/build_and_serve_apk.py --bundled --url <backend>`: builds with `VITE_BUNDLED=1` + `VITE_API_BASE`, config without `server.url` (`androidScheme: https`), cleans the copied files afterwards. Default (tunnel-shell) mode unchanged.
- `App.tsx`: the bundled app starts in "This device" mode (its host `localhost` would otherwise select Server camera).
- Backend unchanged; `.env.example` documents `REMOTE_ALLOWED_ORIGINS=https://localhost` for the bundled app (needed or the WebSocket is refused).
- Counting still runs on the backend (through the tunnel) until Phase 2. tsc/vitest/build and an APK build not yet run.

### Session-end cue: speak before tearing down (2026-11-18)

- `speak()` / `cue()` in `cues.ts` now return a Promise that resolves when the phrase is finished (native: when `TextToSpeech.speak` resolves; browser: utterance `onend`/`onerror`; also on failure or no speech support; it never rejects). New `cueThen(name, maxMs)` races it against a timeout.
- `App.tsx` `endSession()` awaits `cueThen("end", 4000)` and only then calls `remote.stop()`, with an `ending` ref so a second tap/voice command while it speaks does nothing. Cost: the camera runs about 2-3 s longer after Stop/"stop session" (the Paused card and countdown are cleared at once). Other callers ignore the promise, so their behaviour is unchanged.
- Cause not proven: the previous code spoke and stopped in the same tick; I could not reproduce it without a device. If it is still silent, the speech itself is not playing (TTS engine / volume), which the console warning `[cues] ...` would show. tsc/vitest/build and phone check pending.

### Session-end cue and tall-screen layout (2026-11-18)

- `cues.ts`: new phrase `end` = "OK, closing session. Thank you!". `App.tsx` `endSession()` (speaks it, clears Paused/countdown, then `remote.stop()`) is used by the sidebar Stop button (`RemotePanel` got an `onStop` prop) and the voice "stop session". It is not called when the session ends by itself (camera interrupted, switch to Server camera), so nothing speaks unprompted.
- Layout: the camera video was ALREADY `object-contain` (never stretched, letterboxed on 20:9 / 19.5:9 screens) and the skeleton maths (`drawPose` / fitContain) assumes exactly that, so `object-cover` was NOT applied: it would crop the video while the overlay stayed uncropped (misaligned skeleton) and cut off a patient at the edges. Only change: the preview's max height uses `78dvh` instead of `78vh` (the mobile browser bar no longer makes it taller than the visible screen). A screenshot of the distortion is needed to find anything else.

### Tunnel over HTTP/2 and skeleton flicker (2026-11-18)

- Tunnel: no script launches cloudflared, so the command is changed where it is documented (`NATIVE_DEMO_PLAN.md`, `docs/cloud_mobile_blueprint.md`): `cloudflared tunnel --url http://localhost:8000 --protocol http2` (TCP instead of the default QUIC/UDP). The tunnel must be restarted by hand with it; the URL changes, so the APK needs a rebuild.
- Canvas (`RemoteView.tsx`): the existing code already sized the backing store with `devicePixelRatio` and cleared every frame; kept, and the clear now covers the whole backing store in device pixels (identity transform) before the CSS-pixel transform. Canvas class gained `pointer-events-none block`. Likely real cause of the "glitter": a one-frame dropout (`pose === null`) blanked the skeleton for a frame; a null pose within 150 ms of the last drawn one is now ignored (the canvas keeps the last drawing), after which it clears as before. Display only; counting and fall logic are server-side and unaffected. Zero allocation per frame (a ref and `performance.now()`).
- tsc/vitest/build and the phone to be checked by the user.

### Model preloading and the phantom "Closing preview" (2026-11-18)

- Phantom cue cause: the preview cue was an effect on `guideOpen` that skipped the first run with a ref. `main.tsx` uses React StrictMode, which runs mount effects twice in development, so the second run spoke "Closing preview" at launch (it also spoke when a session start/end closed the guide). Fix in `App.tsx`: the effect is gone; `showGuide(open)` (Info button, the guide's close button, voice "show/hide preview") is the only caller of the cue, and says nothing when the state would not change. Start/resume/stop that close the guide do it silently.
- Model manager (`onDevicePose.ts`): `getLandmarker()` is a page-wide singleton promise (WASM + model + GPU delegate), followed by `warmUp()`: one detection on a blank 256x256 canvas (timestamp 1 ms) so shader compilation is paid in the background; a failed load is not cached (retries on Start). `preloadLandmarker()` is called from an `App.tsx` effect when the device camera mode is active (not in Server camera mode, which never needs it). `useRemoteSession` now awaits `getLandmarker()` (instant when ready) and no longer closes the model on unmount (StrictMode would otherwise re-use a closed instance). The YOLO weights are server-side and unaffected.
- Not in the frame loop. tsc/vitest/build to be run by the user; phone timing of Start not yet measured.

### Desktop speech: no cancel() directly before speak() (2026-11-18)

- `cues.ts` browser path: an urgent phrase calls `cancel()` and speaks 30 ms later (Windows/Chrome drops a `speak()` issued in the same tick as `cancel()`); a non-urgent phrase that arrives inside that window is deferred by 30 ms too so the order holds. `resume()` still runs before every speak; utterance `Set` + `onend`/`onerror` cleanup and the throttled touch/click/keydown unlock are unchanged (the function is `unlockSpeech`). Native path: errors (sync or promise) are caught and `console.warn`ed, never thrown into the UI.
- tsc/vitest/build to be re-run by the user. Frame loop untouched.

### Native Capacitor TTS in the APK (2026-11-18)

- `cues.ts`: `speak()` uses `TextToSpeech.speak({ text, lang: "en-US", rate 1, pitch 1, volume 1, category: "ambient" })` from `@capacitor-community/text-to-speech` when `Capacitor.isNativePlatform()`; browsers keep the `speechSynthesis` path (utterance Set, resume, touch unlock). `unlockSpeech()` does nothing on native. Same phrases for countdown, preview, Paused, Resuming.
- Plugin version 8.0.2 (peer `@capacitor/core >=8`, the project has ^8.6.2) is the one to install: `cd frontend && npm install @capacitor-community/text-to-speech`, then the APK script runs `cap sync` (rebuild needed). Not installed by me (no shell), so tsc/build fail until it is. Option names/queue strategy come from memory of the plugin API, not from its README (could not be fetched): tsc will flag any wrong field. Phone check pending.

### Back to spoken cues (2026-11-18)

- On request, Web Audio chimes were replaced by `speechSynthesis` again (`cues.ts` rewritten; chime code removed). Phrases: "Ready", "3", "2", "1", "Go!"; "OK, displaying preview" / "OK, closing preview"; pause/resume say "Paused" / "Resuming" (urgent: they replace what is playing).
- Android/WebView fixes: module-level `Set` keeps live utterances (GC cut-off); `lang = "en-US"`; `if (paused) resume()` before every speak; `unlockSpeech()` keeps capture-phase listeners (pointerdown, touchstart, click, keydown) that send a zero-volume utterance on a touch (at most once per 5 s) to wake the TTS engine. Queued phrases do not cancel each other; only the countdown's first word and pause/resume replace playing speech.
- `App.tsx` calls changed (`cue`, `unlockSpeech`). Frame loop untouched. Phone check pending; the Capacitor WebView may have no TTS engine at all (see the reply).

### Chimes inaudible on mobile after voice commands: fix (2026-11-18)

- Cause: `chime()` called `ctx.resume()` and returned at once when the state was not yet "running" (resume is asynchronous), and the unlock listener removed itself after the first tap, so a context the system suspended later (microphone opened for voice) stayed silent for voice-triggered chimes.
- Fix in `cues.ts` only: `chime()` is async and awaits `resume()` before playing; `unlockAudio()` now keeps capture-phase listeners (pointerdown, touchstart, click, keydown) that create/resume the context and play a 1-sample silent buffer (the iOS unlock) on every touch, and returns a cleanup function (`App.tsx` effect). Peak gains 0.4-0.5 (were 0.14-0.25), notes moved to 523-880 Hz, scheduled 10 ms ahead.
- No frame-loop code touched. Phone check pending (still silent -> the phone's media volume, silent switch on iPhone, or the microphone routing audio to the quiet call stream).

### Web Audio chimes and voice pause/resume (2026-11-18)

- `cues.ts` rewritten: speech synthesis removed. `chime(name)` plays short sine tones (ready, tick, go, previewOpen, previewClose, pause) with a 8 ms attack and exponential decay, scheduled on `AudioContext.currentTime` (no files). The context is created/resumed on the first tap (`unlockAudio`, needed by iOS Safari); silent if audio is blocked. Countdown: ready, tick x3, go chime. Preview open/close: soft two-note tones.
- Voice pause/resume (server, same Groq path as the other commands): bare "pause" / "hold" / "hold on" / "hold please" -> `pause_session`; bare "resume" -> `resume_session`. `Session.request_pause()` (only while counting; keeps the counts) / `request_resume()` (no reset); `Session.paused`. While paused, "let's go" or a spoken "start session" is turned into `resume_session`. The HUD Start button and `request_start` still reset. "pause the session" is still `stop_session`.
- `App.tsx`: `pause_session` / `resume_session` act immediately (no 700 ms delay): a "Paused" card (centred, pointer-events-none) with a pause chime; resume plays the go chime. All timers/UI state, nothing in the frame loop.
- Test `test_voice_pause_keeps_the_counts_and_lets_go_resumes_without_a_reset`. Not run yet; phone check pending. The browser's own speech recognition is not used: commands come from the existing server-side voice path (it needs the Groq key and the mic button).

### Countdown overlay and speech fixes (2026-11-18)

- Overlay moved INSIDE the video `<section>` (`absolute inset-0`, flex-centred): the old `fixed` one outside it was invisible in full screen (only the fullscreen element is drawn). Large `text-6xl/7xl` bold number on a dark translucent bordered card, `pointer-events-none`. No `backdrop-blur` (rule: no blur over live video); `bg-slate-950/70` gives the same look cheaply.
- `cues.ts` speech: utterances are held in a module `Set` until they end (Android Chrome drops unreferenced ones); phrases are queued, not cancelled (only the countdown's first word clears older speech); a paused queue is resumed; `unlockSpeech()` speaks one silent utterance on the first tap so voice-command/timer speech is allowed on mobile.
- Preview phrases now queue instead of interrupting. Nothing touches the frame loop. tsc/vitest/build and phone check pending.

### Pre-commit sanitization (2026-11-18)

- `frontend/android/gradle.properties`: `org.gradle.jvmargs=-Xmx2048m -Dfile.encoding=UTF-8` (was -Xmx1536m; the first Gradle run had been
  killed for low system memory). Note the repo's Android project is `frontend/android`, not `android/`.
- **Secret removed:** `backend/.env.example` had the live `REMOTE_SESSIONS=1` / `REMOTE_ACCESS_TOKEN=...` lines (and `ANGLES_3D=0`) appended
  at the end. Replaced by commented, documented placeholders (`ANGLES_3D`, `SMOOTH_TAU_S`, `CALIBRATE`, `TUNNEL_URL`). The real values
  already live in the git-ignored `backend/.env`. A repo-wide search found no other token, `gsk_` key or API key in tracked files.
- `.gitignore`: added `docs/sample/` (people on the floor, never to be committed; the folder was empty/missing on disk when checked)
  and `frontend/capacitor.config.json` (holds the temporary tunnel address; the APK script rewrites it).
- Commit/push was NOT run by the assistant (no shell): commands were given to the user.

### Adaptive camera resolution (2026-11-18) — frontend checks pass (tsc clean, vitest 74/74, build ok); phone check pending (roadmap B1 + B3, rules from the cv-fitness-pipeline skill)

- **Why:** `openMedia` asked every device for 1280x720, but MediaPipe reads about a 256 px crop, so the extra pixels only cost
  texture upload and conversion on every frame.
- **Tiers** (`logic.ts`): `high` 1280x720, `mid` 640x480, `low` 480x360, requested as `ideal` (never `exact`: the browser picks its nearest
  mode instead of failing). Start (`initialTier`): a computer = high (unchanged for the laptop); a phone = mid, or low when
  `navigator.deviceMemory <= 3` or `hardwareConcurrency <= 4` (unknown values, e.g. iOS, do not count against a phone). A Galaxy A52 class
  phone therefore starts at 640x480. Ceiling (`tierCeiling`): phones never climb above mid, computers up to high.
- **Automatic fallback / recovery** (`createTierMonitor`, pure and unit-tested): fed every 0.5 s with the loop's fps and smoothed model ms.
  Over budget (ms > 33, or fps < 15 with ms > 20) for 3 s = one tier down; fps >= 24 with ms < 16 for 20 s = one tier up (not above the
  ceiling); the first 2 s after any change are ignored (the camera is restarting). A slow camera in a dim room with a cheap model (low
  fps, low ms) does NOT step down: it is not the phone's fault and fewer pixels would not help.
- **Applied without reopening the camera:** `applyTier` calls `track.applyConstraints(...)` on the running video track (reopening is what
  gave "Could not start video source" on camera switches); if the camera refuses, it keeps its size. A tier an earlier session settled on
  carries over to the next start and to a camera switch (`openMedia(..., tier)`).
- **Aspect ratio and alignment:** nothing is hard-coded to the camera size. Every `kp` message carries its own `w`/`h` (the server rescales
  by it), the loop passes `video.videoWidth/Height` of each frame to the listener and `drawPose` letterboxes by those, the canvas follows the
  CSS box, and the preview is `object-contain`; so a size change between frames cannot misalign the skeleton. The preview box's aspect
  ratio follows the telemetry's frame size (a computer stepping 16:9 -> 4:3 changes it; phones step 4:3 -> 4:3, so no layout jump).
- **Server-inference mode:** the camera starts at its tier but is not stepped (that mode already adapts the size of the frames it sends,
  `nextQuality`).
- HUD: the panel line now ends "camera 640×480 · only joint positions are sent".
- **To verify on the phone (cannot be tested here):** (1) the HUD panel shows 640×480 on the A52; (2) a forced slow period steps it to
  480×360 and the skeleton stays aligned through the change; (3) MediaPipe keeps running through a size change in VIDEO mode (if the
  loop ever stopped with "the server is doing it instead" right after a tier change, that is the cause: the fix would be to recreate
  the landmarker on a change). Tests: +4 vitest. Verify: `npx tsc --noEmit`, `npm test`, `npm run build`.

### Zero-allocation hot loop for the on-device pose path (2026-11-18) — frontend checks pass (tsc clean, vitest 69/69, build ok); phone check pending

Rules: `.claude/skills/cv-fitness-pipeline/SKILL.md`. No logic changed (same numbers, same message text, same strokes in the same order).

- **Allocations found per frame (30/s) and removed:**
  - `landmarksToKeypoints`: a new object + 17 new `[x, y, c]` arrays, plus `Object.entries` + iterator, per frame.
    -> `landmarksToKeypointsInto(out, ...)` writes into one persistent object whose 17 joint arrays are reused (a missing landmark
    is deleted from it, so "absent" still means absent); precomputed `JOINT_INDEX` / `JOINT_NAME` arrays walked by index.
  - `worldToJoints`: same -> `worldToJointsInto`.
  - `bboxFromKeypoints`: `Object.values` + `filter` + 2 `map` + spread into `Math.min/max` + result array -> `bboxFromKeypointsInto(out, kp)`:
    one pass, scalar min/max, writes a persistent 4-number array, returns false when fewer than 4 confident joints.
  - Hook callback (`useRemoteSession.ts`): a `{kp, bbox}` object, a `{w, h}` frame object and (when nobody is in view) a `{}` per frame
    -> one `shown` pose, one `frame` and a frozen shared `NO_JOINTS`, created once per loop start and mutated.
  - `RemoteView.tsx`: a `{pose, frame}` object per pose in the listener -> one mutated ref object; `getContext("2d")` per frame -> cached.
  - `drawPose`: a `fitContain` result object, two closures, array destructuring of the bbox and of every bone, `Object.values` + destructuring
    of every joint -> inlined letterbox maths, indexed loops, `for...in` over the pose. (`fitContain` stays exported.)
- **Safety of the shared references:** `onDevicePose` documents that `kp`/`world` are the loop's buffers, valid only inside the `pose`
  callback. Every consumer is synchronous: the canvas listener draws immediately, `keypointMessage` serialises before returning, and the
  repaint-after-resize keeps a reference to the SAME buffers, which always hold the latest frame (never a stale copy). The old allocating
  functions remain as thin wrappers (`landmarksToKeypoints`, `worldToJoints`) for tests and one-off use; the loop uses only the `Into` ones.
- **Not removable here (be honest):** the result object MediaPipe itself returns from `detectForVideo` (its landmark arrays are allocated
  inside the library in 0.10.14); the one JSON string per SENT frame in `keypointMessage` (plus its small envelope object); the 2 Hz
  `setLocal` stats. These are the minimum; verify on the phone with Chrome DevTools > Memory > Allocation sampling / Performance heap
  graph that the sawtooth is now small and the GC bars are rare.
- Tests: +5 vitest (buffers reused with identical numbers/key order, joint removal and return, world joints, box into a reused array,
  drawing calls from a reused pose). Verify: `npx tsc --noEmit`, `npm test`, `npm run build`.

### Audit fixes: telemetry re-render storm, dead-tunnel watchdog + banner, safety queue, HUD GPU effects (2026-11-18) — frontend checks pass (tsc clean, vitest 69/69, build ok); phone check pending

- **Re-render storm (`useRemoteSession.ts`).** Server replies (up to 10/s, every frame in server mode) no longer call `setMsg` /
  `setStats`. They are stored in refs (`msgRef`, pump state) and reach React on a 250 ms UI tick (`UI_TICK_MS`), or at once when
  `urgentChange` says so (first message, fall status or alert state, standby<->counting, exercise, model, error, a completed rep).
  Stats only call `setStats` when a rounded value changed. In server-inference mode the skeleton is now drawn from the reply
  straight through the pose listener (like on-device), not through `msg.pose` state; `RemoteView` registers the listener in both
  modes and no longer repaints from state (`pose` removed from the hook's return).
- **Dead-tunnel watchdog.** `lastRecv` / `lastSent` refs; the UI tick calls `connectionLost` (we sent something after the last
  reply and nothing came back for 4 s, not while the tab is hidden; both reset on becoming visible and on hello) and then drops the
  socket itself (a dead tunnel never closes it) and reconnects with the usual backoff. The socket-close path is now one function
  (`closed`) that also ignores stale sockets, clears the telemetry (so a dead session's reps can no longer look live) and wipes the
  skeleton. HUD: red pulsing "Connection lost — reconnecting… Reps and fall monitoring are paused" banner (`Hud` prop `lost`,
  `RemoteView` passes it while reconnecting with the camera running).
- **Safety queue.** `send()` for `ack_fall` / `escalate` while the socket is not ready is queued (max 8, newest kept, 120 s max
  age), re-sent right after the next hello, cleared on stop; the user is told "No connection. Your answer will be sent as soon as
  it is back." Other commands are still dropped while offline (they belong to the old session: reset, start/stop).
- **GPU effects.** All `backdrop-blur` removed from the HUD (opaque-ish `bg-slate-900/90` instead), rep-counter text shadow reduced
  from an 8 px blur to 2 px. There were no width/left animations in the HUD itself; the two in the side panel were fixed too:
  `App.tsx` angle bar now `scaleX` + `transition-transform`, privacy toggle knob `translate-x` instead of `left` + `transition-all`.
  Not changed: the guide's progress bar (`ExerciseGuide.tsx:238`).
- Tests: +vitest for `connectionLost`, `urgentChange`, `SAFETY_COMMANDS`. To verify: `npx tsc --noEmit`, `npm test`, `npm run build`;
  then on the phone: kill the tunnel (Ctrl+C in the cloudflared terminal) and expect the banner within about 4-5 s, reps hidden,
  and recovery by itself when the tunnel (same URL) is back.

### Skill added: api-tunnel-architecture + read-only network audit (2026-11-18)

- `.claude/skills/api-tunnel-architecture/SKILL.md`: lightweight-JSON-only on the live link (with the honest note that the phone
  cannot send "final rep counts" while counting and fall detection run on the server: it sends keypoints, capped at 4096 bytes),
  non-blocking bounded send paths, timeouts/backoff/heartbeat, offline-first caching, batching and idempotent uploads. No app code changed.
- Audit gaps (priority order): no half-open connection detection (no heartbeat/watchdog/online events: HUD can say LIVE while the
  tunnel is dead); offline means no counting and no fall monitoring with no clear message and stale reps shown; commands silently
  dropped when the socket is not open (incl. ack/escalate); no fetch timeouts (`App.tsx` post, `/api/exercises`, `/api/voice`, etc.);
  no startup retry/cache for exercise lists; no local persistence/batching of results; backoff without jitter; server-mode frame
  path allocates per frame; voice audio base64 over the socket; Capacitor shell has no offline fallback page.

### Skill added: react-capacitor-mobile + read-only UI audit (2026-11-18)

- `.claude/skills/react-capacitor-mobile/SKILL.md`: no React state for per-frame tracking data (refs + canvas/DOM writes, memoised
  leaves, 2-4 Hz for visible numbers), compositor-only animation (transform/opacity, no blur over live video), `dvh` + safe-area
  insets + `touch-manipulation` + instant `active:` feedback + 44 px targets. No app code changed.
- Audit findings (listed in the skill, priority order): server telemetry `setMsg`/`setStats` per reply re-renders the tree (and
  `useTelemetry` `setT` per message on desktop); width/left transitions (`App.tsx:53,249`, `ExerciseGuide.tsx:238`);
  `backdrop-blur` on the HUD over the video; `vh`/`screen` units (5 places); no safe-area padding anywhere; no
  `touch-action: manipulation` or press feedback on the HUD buttons, HUD buttons 40 px.

### Skill added: cv-fitness-pipeline (2026-11-18)

- `.claude/skills/cv-fitness-pipeline/SKILL.md`: guardrails for per-frame computer-vision code: 15-24 fps clock-based throttling
  with latest-frame-wins, no allocation in the hot loop (preallocated buffers, no React state per frame, borrowed MediaPipe
  results), O(1) smoothing (time-based EMA or One-Euro, applied once), and release of streams, landmarker, canvases, listeners
  on unmount/hidden. It names the files that already follow each rule and lists the known gaps (30 fps cap, per-frame allocations
  in `landmarksToKeypoints`/`worldToJoints`, 1280x720 request, unmount cleanup to verify) without changing any code.

### Session 2026-11-19: fullscreen, voice, token, reconnect loop (written WITHOUT a shell: nothing type-checked, built or run)

Read this first in the next session. Every item below is untested; run `npx tsc --noEmit`, `npx vitest run`, `npm run build` in `frontend/` first.

**Fullscreen exit (mobile).** `remote/useFullscreen.ts` `exit()` resets `fallback` AND `native` state, then `document.exitFullscreen()` in try/catch (only if `fullscreenElement`). `App.tsx`: when `remote.status` goes back to `idle`, it calls `fs.exit()`. `remote/Hud.tsx`: HUD container `z-50`, Exit button `min-h-11 min-w-11 touch-manipulation` (onClick only; adding onTouchEnd as well would toggle twice). Could not reproduce without a device.

**Voice: on-device speech was tried and REVERTED.** I added `@capacitor-community/speech-recognition` (`nativeSpeech.ts`, native route in `useVoiceInput.ts`, `VoiceChip` change, offline switching). It broke voice in BOTH modes (native path bypassed the server, which interprets the commands and holds the session state) and showed "Listening" without phrases on Android. Reverted by the user's request for the demo:
- `useVoiceInput.ts` is back to the original (record -> upload -> server classifies). `nativeSpeech.ts` is now an empty stub (delete it: `del frontend\src\remote\nativeSpeech.ts`). The plugin was removed from `package.json` (run `npm install`). `VoiceChip.tsx` is the original again (offline shows "Voice needs a connection").
- Offline "This phone" mode: no voice, use the HUD Start/Pause buttons.
- Kept, unused: `remote/intent.ts` + `intent.test.ts` (a TS copy of `backend/app/voice/intent.py` `classify_command`), in case on-device voice is retried. If it is: use the server path when online, the OS recognizer only offline, and forward commands to the server in Server mode; check `[speech]` logs via chrome://inspect; the plugin on Capacitor 8 needs a compatible version (7.0.1 was in package.json at one point).
- `AndroidManifest.xml` has `RECORD_AUDIO` (needed for the server voice path) and a `<queries>` RecognitionService block (no longer needed, harmless).

**First-time token prompt.** Cause: `start()` skipped `/api/health` when the analytics are local, so the prompt never appeared and the voice socket's 4401 was ignored. `useRemoteSession.ts`: with no saved token and a network, local-analytics start asks `/api/health` (2.5 s abort) and shows the prompt if `remote_auth`; a 4401 on the local voice link clears the token and sets an error; in server mode, after 3 failed reconnects it re-checks health and shows the token prompt instead of looping; close code 1008 gets a clear message.

**Reconnect hygiene.** `stop()` detaches the old socket's handlers before closing it and resets `opened`, `dropRef`, `lastRecv/lastSent`.

**Slowness.** `VoiceChip.tsx` still had `backdrop-blur` over the live video (the 2026-11-18 audit had removed all HUD blur). Removed. Not measured.

**"Connection lost" / reconnect loop on a tunnel (root cause).** MediaPipe is stateful: one PoseLandmarker per session (`engines/mediapipe_engine.py`, `stateless=False`). `InferenceService` kept ONE preloaded `_spare`; every later session loaded a new model on the inference thread (seconds on CPU, the LiteRT/XNNPACK `CpuAccelerator` log lines), stalling replies. The phone's dead-tunnel watchdog (`LOST_AFTER_MS` 4 s) then dropped the socket, the next session needed another load: a self-sustaining loop (connected -> accelerator -> closed -> DestroyAccelerator -> new remote-<id>).
- `backend/app/inference.py`: `_refill_spare()` loads the next engine on a side thread (`engine-spare`) after a session takes the spare; the spare is taken under the lock in `_engine_for`; new field `_refilling`.
- `useRemoteSession.ts`: before a socket's first answer the watchdog allows `FIRST_ANSWER_GRACE_MS` = 25 s (4 s afterwards).
- Verify: one `remote session ... connected` that stays open, one `engine-spare` load, no repeated `closed` / `DestroyAccelerator`; Stop/Start or a tunnel blip then answers within about a second. Alternative with no server model: "On-device MediaPipe" pose.

**Files touched this session:** `frontend/src/remote/{useFullscreen.ts,Hud.tsx,useRemoteSession.ts,useVoiceInput.ts,VoiceChip.tsx,nativeSpeech.ts,intent.ts,intent.test.ts}`, `frontend/src/App.tsx`, `frontend/package.json`, `backend/app/inference.py`, this log. Python tests were not run (`backend\.venv\Scripts\python tests\test_analytics.py`).

### Next: P2 persistence and upload mode

SQLite + JSON report, recording mode A/B, H.264 recorder via imageio-ffmpeg, upload job worker process, session history with Recharts.

### Notes for whoever picks this up

- The frontend folder on disk is named `frontend` (lowercase), although it has been referred to as `frontEnd`.
- Ultralytics is AGPL-3.0; decide on licensing before any closed-source client delivery that includes the YOLO backend.
- Assistant session memory is tied to the old machine and path; these two documents are the complete handoff.
