# Offline-first architecture for the Android app

Status: **Phase 1 implemented (2026-11-18)**; Phases 2-4 open. Each phase ships alone and keeps the existing behaviour.

## Context

Until now the APK was only a shell: `capacitor.config.json` had `server.url` set to the trycloudflare tunnel, so the page loaded through
the tunnel, the MediaPipe WASM came from jsdelivr, and **rep counting, calibration, fall detection and escalation all ran on the
server** (`backend/app/pipeline/core.py` via `/ws/session`). The phone only computed keypoints. When the tunnel blinked the "Connection
lost" banner showed and counting stopped. Nothing was stored (no database, no session history).

Goal: a workout runs fully on the phone with no network; results are stored on the device and uploaded when a connection returns. The
30 fps pose loop and the current UI must not get slower or change behaviour.

## 1. Target architecture

```
 Phone (APK, assets bundled)                                   Backend (when reachable)
 ┌───────────────────────────────────────────────────────┐
 │ camera → MediaPipe (local WASM+model) → kp buffers     │
 │            │ same callback, allocation-free            │
 │            ▼                                           │
 │   LocalPipeline (TS port of app/pipeline)              │     optional live link (/ws/session)
 │   reps · calibration · fall · escalation ──────────────┼──►  caregiver dashboard, NOT required
 │            │ msgRef → existing 250 ms UI flush         │
 │            ▼                                           │
 │   SessionRecorder (events only: rep, fall, start/end)  │
 │            ▼                                           │
 │   IndexedDB: sessions · events · outbox                │
 │            ▼  (idle, online, app resume)               │
 │   SyncQueue ── POST /api/sync/sessions (idempotent) ───┼──►  SQLite data/rehab.db
 └───────────────────────────────────────────────────────┘
```

Principles:
- The local pipeline is the authority for a workout on the phone. The server is the long-term store and the live view.
- No I/O in the frame path. Storage is touched only on events (a rep, a fall-state change, start/end), never per frame.
- The UI is unchanged: the local pipeline emits the same `Telemetry` object the server sends, through the same refs and throttled flush.
- The server path stays (server inference, desktop "Server camera"), as the fallback.

## 2. Phases

### Phase 1: bundle everything into the APK (DONE)
- `frontend/src/net.ts`: `apiUrl()` / `wsUrl()` / `isBundled()`. Base from `localStorage.apiBase`, else build-time `VITE_API_BASE`, else
  "" (same origin: the browser behaves exactly as before). All `fetch("/api/...")` and WebSocket URLs go through it.
- `frontend/scripts/copy-assets.mjs`: copies the MediaPipe WASM and the Lite model into `public/` for a bundled build (`--clean` removes
  them; they are git-ignored). `onDevicePose.ts` uses `/mediapipe/wasm` when `VITE_BUNDLED=1`, the CDN otherwise.
- `scripts/build_and_serve_apk.py --bundled --url <backend>`: builds the page with `VITE_BUNDLED=1` and `VITE_API_BASE`, writes a
  Capacitor config without `server.url` (page comes from `https://localhost`), runs `cap sync` and Gradle as before.
- The bundled app always starts in "This device" mode (`App.tsx`), even though its hostname is `localhost`.
- Backend: set `REMOTE_ALLOWED_ORIGINS=https://localhost` (documented in `backend/.env.example`). CORS already allows all origins.
- Still needs the backend (through the tunnel) for counting until Phase 2; what changes is that the page, the WASM and the model no
  longer depend on it.

### Phase 2: TypeScript port of the pipeline (`frontend/src/engine/`) (IMPLEMENTED, awaiting the first run; see activity_log)
- `kinematics.ts`, `repCounter.ts`, `fallDetector.ts`, `escalation.ts`, `mobility.ts`, `pipeline.ts` (standby calibration, `CAL_*`).
- One source of truth for the exercise table: `scripts/export_exercises.py` -> `shared/exercises.json`; Python stays the master.
- Parity tests (the zero-regression guarantee): `scripts/make_golden.py` replays keypoint traces through the Python `Pipeline` into
  `tests/golden/*.json`; a vitest suite runs the TS engine on the same inputs (reps, state sequence, fall/alert transitions within one
  frame).
- Hot-loop rules from `.claude/skills/cv-fitness-pipeline`: preallocated buffers, no closures/arrays per frame, runs inside the existing
  `on.pose` callback in `useRemoteSession.ts` `startLocal`.
- Switch: `engine: "local" | "server"`; default `local` in the bundled APK, `server` elsewhere until parity is proven on phones.

### Phase 3: local storage (IndexedDB) (IMPLEMENTED, awaiting the first run; see activity_log)
- IndexedDB (no native plugin; a session is a few kB), `navigator.storage.persist()`. Stores: `sessions`, `events`, `meta`.
- `SessionRecorder` buffers events and writes in one transaction every 5 s, at session end and on `visibilitychange: hidden`/`pagehide`;
  write failures never reach the UI.

### Phase 4: sync queue + backend store (IMPLEMENTED, awaiting the first run; see activity_log)
- `sync/syncQueue.ts`: triggers (start, `online`, `@capacitor/network`, `@capacitor/app` resume, 60 s while pending); a 2 s
  `GET /api/health` check first (`navigator.onLine` lies); deferred while counting; batches of 20; backoff with jitter; `synced` only on 2xx.
- Backend: `app/storage.py` (SQLite, WAL), `POST /api/sync/sessions` (token, size limits, idempotent by session uuid),
  `GET /api/sessions`; the routes go on the tunnel allow-list in `security.py`.
- Sync while the app is closed (WorkManager / background-runner) is out of scope.

## 3. Safety decision (sign-off needed before Phase 2 ships)

Offline, a fall cannot reach a caregiver. The local engine still detects it and runs the full local escalation (FallCheck dialog,
beep and TTS, "I'm OK" / "Help"). The alert event is stored and synced later. Recommendation: show "Offline - caregiver will not be
notified" in the HUD while offline; send the fall event first when the connection returns; a real offline alert (SMS plugin) is a
separate decision. Voice commands use Groq via the server, so they are unavailable offline (the buttons do the same things).

## 4. Demo plan (no tunnel visible)

1. Install the bundled APK. 2. Airplane mode: start, countdown, live reps, pause/resume, Test Fall, session end ("1 session waiting to
sync"). 3. Wi-Fi back on: "All synced", the laptop dashboard lists the session. 4. The tunnel is only the sync target, and sync is
retry-tolerant, so a blink is invisible. 5. scrcpy as in `NATIVE_DEMO_PLAN.md`; the fps badge shows the numbers do not change offline.
(Phases 2-4 are needed for steps 2-3; with Phase 1 alone the app still counts through the backend.)

## 5. Risks

| Risk | Mitigation |
|---|---|
| TS and Python counters drift apart | Golden parity tests; Python stays the master; one shared `exercises.json` |
| Frame-rate regression | Allocation-free engine; storage on events only; sync deferred while counting; compare fps/ms before and after |
| APK size | WASM (~9 MB) + Lite model (~6 MB): acceptable; web visitors still lazy-load them |
| Data loss | `storage.persist()`, 5 s flushes plus `pagehide`, idempotent upload |
| Duplicate uploads | Session uuid primary key, `ON CONFLICT DO NOTHING` |
| Desktop / server mode breaks | Untouched; the local engine is a third route behind a setting |
