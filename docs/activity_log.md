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

### Next: P2 persistence and upload mode

SQLite + JSON report, recording mode A/B, H.264 recorder via imageio-ffmpeg, upload job worker process, session history with Recharts.

### Notes for whoever picks this up

- The frontend folder on disk is named `frontend` (lowercase), although it has been referred to as `frontEnd`.
- Ultralytics is AGPL-3.0; decide on licensing before any closed-source client delivery that includes the YOLO backend.
- Assistant session memory is tied to the old machine and path; these two documents are the complete handoff.
