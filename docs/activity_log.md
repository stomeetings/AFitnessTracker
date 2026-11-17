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

### Next: P2 persistence and upload mode

SQLite + JSON report, recording mode A/B, H.264 recorder via imageio-ffmpeg, upload job worker process, session history with Recharts.

### Notes for whoever picks this up

- The frontend folder on disk is named `frontend` (lowercase), although it has been referred to as `frontEnd`.
- Ultralytics is AGPL-3.0; decide on licensing before any closed-source client delivery that includes the YOLO backend.
- Assistant session memory is tied to the old machine and path; these two documents are the complete handoff.
