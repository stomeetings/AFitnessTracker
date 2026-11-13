# AI Patient Rehabilitation, Mobility & Fall Detection System — Master Blueprint

Status: **architecture locked, P0 decisions final, P0 step 1 (environment) in progress.** See sections 9 and 12.
Last updated: 2026-11-13 (session 2, personal PC).

---

## 1. Project goal and constraints

A real-time system that watches one patient through a camera (or an uploaded video), draws a bounding box and skeleton, counts left/right exercise repetitions with range of motion, tracks general mobility, and raises an alert on a fall.

- **100% local and offline.** No cloud LLMs, no external APIs at runtime.
- **Backend:** Python + FastAPI.
- **Frontend:** Vite + React + TypeScript + Tailwind CSS + Recharts.
- **Workspace layout:** `docs/`, `backend/`, `frontend/`, `tests/`.
- **Not a certified medical device.** Fall detection and step counts are estimates for monitoring and demo purposes.

## 2. Finalized decisions

| Area | Decision |
|---|---|
| Primary pose engine | **MediaPipe Pose Landmarker** (33 keypoints with 3D world landmarks, Apache 2.0) |
| Swappable backend | **Ultralytics YOLO-Pose** behind the same abstract interface, using CUDA on the target PC for live or demo-video validation |
| Input modes | Live webcam (MJPEG video + WebSocket telemetry) and pre-recorded video upload for demo/validation |
| Exercise selection | The user picks the active exercise in the UI; only that exercise's counters run, to avoid false triggers |
| Initial exercises | Left/Right Elbow Curl, Shoulder Raise, Knee Flexion/Extension, Hip Flexion, plus Sit-to-Stand (Chair Stands) |
| Always-on tracking | Walking/mobility tracking and emergency fall detection (`NORMAL -> SUSPECTED -> CONFIRMED`) |
| Recording Mode A | Annotated H.264 `.mp4` + SQLite/JSON telemetry |
| Recording Mode B | Privacy / summary-only: zero video saved; uploaded source videos are deleted after processing |
| Frontend stack | Vite + React + TypeScript + Tailwind CSS + Recharts |

## 3. R&D findings: pose engine comparison

**Important:** nothing has been benchmarked yet. The speed figures below are estimates from published numbers. Phase P0 replaces them with measurements on the target PC.

| Criterion | MediaPipe Pose Landmarker | YOLO11n-pose (ONNX / OpenVINO / CUDA) |
|---|---|---|
| CPU speed (estimate) | About 30+ FPS for `full`, more for `lite`; the detector only re-runs when tracking is lost | About 15–25 FPS at 640 px, 30+ at 320–416 px with OpenVINO; Ultralytics publishes ~52 ms/frame for ONNX on CPU |
| GPU | The Python package on Windows runs on CPU (no GPU delegate) | Runs on CUDA via PyTorch; expected to be well above real time on a 4 GB GPU |
| Keypoints | 33, including heels, toes and hands | 17 (COCO), ankles only |
| 3D | Yes, metric world landmarks; ROM angles are less camera-angle dependent | No, 2D only |
| Bounding box | None native; derived from landmarks, lost if the pose is lost | Real detector box with confidence every frame |
| Jitter | Built-in temporal smoothing | None; needs our own filter (One Euro) |
| Fallen / lying person | Weaker; trained mostly on upright poses and can lose the track | Stronger; the detector handles horizontal bodies |
| Multi-person | Weak | Strong, with built-in tracking |
| Licence | Apache 2.0 | **AGPL-3.0**; closed-source client delivery needs a paid Ultralytics licence |

### Why MediaPipe is primary

- Rehab is a single-patient problem where joint-angle accuracy matters most.
- Foot landmarks (heel, toe) make step counting more reliable.
- 3D world landmarks give better ROM values.
- The licence is safe for client demos.

### Why YOLO-Pose is kept as a backend

- A real detector box is more robust when the patient is lying on the floor.
- It can use the target PC's GPU, which is useful for fast processing of uploaded demo videos.
- It is the fallback if multi-person scenes ever become a requirement.
- **Licence caveat:** resolve the AGPL-3.0 question before shipping any build that includes it to a client.

### Other conclusions

- `ultralytics.solutions.AIGym` is not used. It counts one angle on one keypoint triplet with two thresholds and has no left/right independence, ROM, or dwell-time logic. We write our own FSMs.
- Running both engines on the same frame is rejected: double the cost for marginal gain.
- YOLO26n-pose (newer, claims faster CPU inference) may be added to the P0 benchmark as a third candidate. It has the same licence and 17-keypoint limits.
- MediaPipe's fall weakness is covered by a "rapid descent followed by pose lost" rule (section 6.4).

## 4. Pose-engine abstraction

Both engines implement one interface and return one normalised result, so analytics code never knows which engine is running.

- **Interface:** `PoseEngine` with `load()`, `infer(frame_bgr, timestamp_ms) -> PoseResult | None`, `close()`, and a `name` / `capabilities` descriptor.
- **`PoseResult`:**
  - `keypoints`: named joints in a common schema (shoulders, elbows, wrists, hips, knees, ankles, plus optional heels/toes), each with image x/y, visibility/confidence, and optional 3D world coordinates.
  - `bbox`: x, y, w, h and confidence. MediaPipe derives it from visible landmarks with padding and smoothing; YOLO uses the detector box.
  - `timestamp_ms`.
- **Capability flags:** `has_3d`, `has_feet`, `native_bbox`. Analytics degrade gracefully (for example, step counting falls back to ankles when feet are missing).
- **Selection:** a config value (`POSE_ENGINE=mediapipe|yolo`, `YOLO_DEVICE=cuda|cpu`), exposed read-only in the UI.

## 5. System architecture and anti-lag pipeline

```
Camera thread ──► latest-frame slot (size 1, old frames dropped)
                        │
                 Inference thread: pose → smoothing → angles → FSMs → mobility → fall → annotate
                        │
        ┌───────────────┼──────────────────┬───────────────────┐
   MJPEG endpoint   WebSocket telemetry   Recorder thread     SQLite writer
   (latest JPEG,    (10–15 Hz; fall       (Mode A only,       (batched)
    capped ~25 fps)  events immediate)     bounded queue)
```

### Rules

1. **Never queue frames.** If inference is slower than the camera, stale frames are dropped, so latency stays constant.
2. **Heavy work stays off the event loop.** Capture, inference and recording run in threads (native inference releases the GIL). FastAPI's async loop only serves bytes and JSON.
3. **Video over MJPEG, data over WebSocket.** The browser decodes MJPEG natively in an `<img>` tag. React re-renders only on small JSON messages.
4. **The backend owns the webcam.** No browser-to-backend frame round trip.
5. **All logic is time-based, not frame-based.** Velocities and dwell times use timestamps, so the same core gives the same results live (variable FPS) and on upload (every frame processed).
6. **Uploads run in a separate process.** The file is streamed to disk and processed as a job with progress pushed over WebSocket. The live loop is unaffected.
7. **Output video is H.264.** OpenCV's default `mp4v` does not play in a browser `<video>` tag. Frames are piped to the ffmpeg binary bundled with the `imageio-ffmpeg` pip package (works offline after install).
8. **Privacy mode never creates the recorder.** No video bytes are written in Mode B.

### One shared pipeline

A single `Pipeline.process(frame, timestamp)` core is used by both input modes:

| | Live mode | Upload mode |
|---|---|---|
| Frame source | Webcam thread, latest frame only | Video file, every frame |
| Timestamps | Wall clock | Frame index / source FPS |
| Runs in | Threads inside the API process | Separate worker process |
| Output | MJPEG stream + WebSocket telemetry | Annotated mp4 (Mode A) + report |

### Recording modes

| | Mode A (Video + Telemetry) | Mode B (Privacy / Summary-Only) |
|---|---|---|
| Annotated `.mp4` | Saved (H.264) | Not created |
| SQLite + JSON report | Saved | Saved |
| Fall snapshot image | Saved | Not created |
| Uploaded source video | Kept | Deleted after processing |

## 6. Detection logic

### 6.1 Joint angles

- Three-point angle from vectors with NumPy (`arctan2` / dot product), using 3D world landmarks when the engine provides them and 2D otherwise.
- Smoothed (One Euro or EMA) and gated on keypoint visibility (default > 0.5). Low-visibility frames freeze the FSM instead of counting.

### 6.2 Exercises (user-selected, one active at a time)

| Exercise | Sides | Angle / signal | Notes |
|---|---|---|---|
| Elbow Curl | L, R independent | shoulder–elbow–wrist | |
| Shoulder Raise | L, R independent | hip–shoulder–elbow | |
| Knee Flexion/Extension | L, R independent | hip–knee–ankle | |
| Hip Flexion | L, R independent | shoulder–hip–knee | |
| Sit-to-Stand (Chair Stands) | Bilateral | mean knee angle + hip height relative to knees | Counts full seated → standing → seated cycles; also reports time per stand |

- **FSM per limb:** `REST -> MOVING -> PEAK -> RETURNING -> REST`, one rep counted on return to rest.
- **Guards:** hysteresis thresholds (separate enter/exit angles), minimum dwell time at peak, minimum and maximum rep duration.
- **Recorded per rep:** start/end time, peak angle, ROM (max minus min angle).
- **Thresholds** live in a per-exercise config file and are tunable in P3.

### 6.3 Mobility (always on)

- **Posture:** standing, sitting or lying, from torso angle and hip/knee geometry.
- **Walking:** hip-centre displacement plus ankle oscillation above a threshold; accumulates walking duration.
- **Step count (estimate):** peaks in the inter-ankle (or heel) distance, normalised by torso length, with a minimum interval between steps. Most reliable in side view.

### 6.4 Fall detection (always on)

State machine: `NORMAL -> SUSPECTED -> CONFIRMED` (then back to `NORMAL` on recovery).

- **Triggers for SUSPECTED** (any strong combination):
  - Hip-centre downward velocity above threshold, normalised by torso length per second.
  - Bounding-box aspect ratio flips from tall to wide.
  - Torso tilt beyond about 60° from vertical.
  - Rapid descent followed by pose loss.
- **CONFIRMED:** the lying posture or pose loss persists for about 1 s.
- **On CONFIRMED:** immediate red on-screen warning, WebSocket alert pushed without waiting for the telemetry tick, and a `fall_events` row with the exact timestamp.
- **False-positive guard:** lying down slowly (floor exercises) is classed as posture `lying`, not a fall.

## 7. API endpoints

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Status, active engine, device (CPU/CUDA) |
| `GET /api/exercises` | List of selectable exercises |
| `POST /api/sessions/start` | Start a live session; body carries exercise and recording mode |
| `POST /api/sessions/{id}/exercise` | Switch the active exercise mid-session |
| `POST /api/sessions/{id}/stop` | Stop and finalise the session |
| `GET /api/stream/live` | MJPEG annotated video |
| `WS /ws/telemetry` | Live metrics, fall alerts, upload progress |
| `POST /api/uploads` | Upload `.mp4` / `.avi` with exercise and recording mode; returns a job id |
| `GET /api/jobs/{id}` | Upload job status and progress |
| `GET /api/sessions` | Session history |
| `GET /api/sessions/{id}` | Session report |
| `GET /api/sessions/{id}/report.json` | JSON export |
| `GET /api/sessions/{id}/video` | Annotated video (Mode A only; 404 in Mode B) |
| `DELETE /api/sessions/{id}` | Delete a session and its files |

**WebSocket message types:** `telemetry` (angles, rep counts, ROM, posture, steps, FPS), `fall_alert`, `fall_cleared`, `job_progress`, `session_ended`.

## 8. Database schema (SQLite, WAL mode)

SQLite is the single source of truth; the JSON report is generated from it.

| Table | Columns |
|---|---|
| `sessions` | `id`, `source` (`live` / `upload`), `recording_mode` (`video` / `summary`), `started_at`, `ended_at`, `status`, `video_path` (null in Mode B), `source_filename`, `model_backend`, `device`, `avg_fps` |
| `exercise_stats` | `id`, `session_id`, `exercise`, `side` (`L` / `R` / `B`), `rep_count`, `max_rom_deg`, `min_angle_deg`, `avg_rom_deg`, `active_seconds` |
| `rep_events` | `id`, `session_id`, `exercise`, `side`, `t_start`, `t_end`, `peak_angle_deg`, `rom_deg` |
| `mobility_summary` | `session_id`, `standing_s`, `walking_s`, `sitting_s`, `lying_s`, `step_count` |
| `fall_events` | `id`, `session_id`, `wall_clock_ts`, `video_offset_s`, `drop_velocity`, `aspect_ratio`, `torso_angle_deg`, `confirmed`, `snapshot_path` (null in Mode B) |
| `telemetry_samples` | `id`, `session_id`, `t`, `posture`, `angles_json`; downsampled to 1–2 Hz for report charts |
| `upload_jobs` | `id`, `session_id`, `status`, `progress`, `error`, `created_at`, `finished_at` |

All child tables reference `sessions.id` with cascade delete.

## 9. Roadmap

| Milestone | Scope | Exit criterion |
|---|---|---|
| **P0: Foundation and benchmark** | Python virtual environment; install MediaPipe, GPU-enabled PyTorch + Ultralytics, FastAPI; the `PoseEngine` interface with both backends; benchmark script (MediaPipe lite/full on CPU, YOLO-Pose on CPU and CUDA); baseline webcam test showing skeleton + bounding box; Vite/React scaffold showing the MJPEG stream | Measured FPS/latency table recorded in `docs/`; engine choice confirmed; live stream at 25+ FPS |
| **P1: Core analytics (live)** | Angle maths; per-limb FSMs with ROM; Sit-to-Stand; exercise selector; mobility and step estimate; fall detection with red alert; live dashboard over WebSocket | Left/right reps correct on test clips; a staged fall alerts within about 1.5 s |
| **P2: Persistence and upload mode** | SQLite and JSON reports; recording mode toggle (A/B); H.264 recorder; upload job pipeline; annotated playback; session history with Recharts | An uploaded video yields an annotated mp4 and report; Mode B writes no video and deletes the source |
| **P3: Hardening and demo polish** | Threshold configuration UI; unit tests for angles/FSMs/fall logic on synthetic keypoints; regression clips; fall snapshots; one-command launcher; AGPL decision for the YOLO backend | Tests pass; scripted demo runs end to end |

## 10. Planned repository layout

```
backend/
  app/
    main.py            FastAPI app, routes, WebSocket
    config.py          settings (engine, device, thresholds, paths)
    engines/           base.py (PoseEngine), mediapipe_engine.py, yolo_engine.py
    pipeline/          capture, pipeline core, annotate, recorder, mjpeg
    analytics/         angles, exercises (FSMs), mobility, fall
    storage/           SQLite schema, repository, JSON report
    jobs/              upload worker process
  scripts/             benchmark_engines.py, webcam_baseline.py
  requirements.txt
frontend/              Vite + React + TS + Tailwind + Recharts
tests/                 unit tests on synthetic keypoints, regression clips
docs/                  plan.md, activity_log.md, benchmark results
data/                  SQLite file, recordings, uploads (git-ignored)
```

## 11. Open items and risks

- **Benchmarks are unmeasured.** P0 must confirm the engine choice on the target PC.
- **MediaPipe will not use the GPU** in its Windows Python package; only the YOLO backend benefits from CUDA.
- **AGPL-3.0** on Ultralytics: decision deferred to P3. YOLO stays behind the `PoseEngine` interface so it can be removed from a client build.
- **Camera placement** strongly affects accuracy: full body in frame, side view for walking/steps and Sit-to-Stand, frontal for shoulder raises.
- **Pascal GPU:** PyTorch CUDA 12.8+ wheels dropped the GTX 1050 (sm_61). We pin the `cu126` wheel series; newer torch releases may not offer it, so the pin is verified at install time.

## 12. Target hardware baseline and finalized P0 decisions (2026-11-13)

| Item | Measured |
|---|---|
| CPU / RAM | Intel i5-7300HQ (4 cores / 4 threads), 16 GB |
| GPU | NVIDIA GeForce GTX 1050, 4 GB, Pascal (sm_61), driver 582.78, CUDA 13.0 capable |
| Camera | Integrated Webcam (index 0 expected) |
| Python | 3.13.5 (only interpreter installed) |
| Node / npm | 22.16.0 / 11.4.2 |
| ffmpeg | Not on PATH |

| Question | Decision |
|---|---|
| Python version | Try 3.13 first; fall back to 3.12 only if MediaPipe has no wheel |
| PyTorch build | Pin `cu126` inside `backend/.venv`; verify with `torch.cuda.is_available()` and a CUDA matmul |
| Benchmark candidates | MediaPipe Pose Lite and Full (CPU) vs YOLO11n-pose (CUDA and CPU); no third candidate |
| P0 scope | Smaller first pass: environment, benchmark script, OpenCV webcam skeleton window. FastAPI and Vite follow once the engine choice is confirmed |
| AGPL | Deferred to P3 |
| ffmpeg | Bundled via the `imageio-ffmpeg` pip package |
