# Cloud + mobile blueprint for the rehab and fall-detection system

Status: architecture plan only (written 2026-11-16). Nothing in this document has been implemented.

## Context

The MVP runs on one PC: the backend opens the camera with OpenCV, opens the microphone with `sounddevice`, runs
YOLO11n-pose, and streams an annotated MJPEG picture to a browser on the same machine. The goal now is a backend on
a cloud server with patients connecting from phones over the public internet.

**The central fact:** a cloud server has no camera and no microphone. Today's design cannot be "deployed" as it is;
capture has to move into the patient's browser, and the single global `Worker` (one camera, one pipeline, one
patient) has to become one pipeline per connected patient. Everything else in this blueprint follows from that.

Evidence note: the web research found solid material on transport trade-offs and iOS limits, but no public
architecture write-ups from the large rehab vendors, so statements about "industry practice" below are marked where
they rest on smaller products or on engineering reasoning.

---

## 1. Verdict first

| Question | Recommendation |
|---|---|
| Where should pose inference run? | **On the phone, in the browser, as the production default; server YOLO as the fallback.** Start with server inference because it reuses today's code, but build the transport so either can feed it |
| WebSocket or WebRTC for video? | **WebSocket (WSS) with JPEG frames, one frame in flight at a time.** WebRTC only if measurements later demand it |
| PWA or native app? | **PWA is enough for exercise sessions. Not enough for always-on fall monitoring** (see section 3) |
| Voice | Record and detect speech in the browser, upload only utterances over HTTPS; Groq key stays on the server |

### Why "all inference on the server" should be questioned

| | Frames to server (YOLO in cloud) | Keypoints to server (pose on phone) |
|---|---|---|
| Uplink | about 0.7–1.6 Mbit/s as JPEG at 10 fps (estimate: 8–20 kB per frame) | about 2–3 kB/s |
| Mobile data per hour | roughly 300–700 MB | under 2 MB |
| Feedback delay on 4G | network round trip + inference, typically a few hundred ms | none for the skeleton; analytics lag is invisible |
| Works through a network dropout | no: tracking and fall detection stop | pose keeps running; local alarm can still sound |
| Server cost | GPU server, scales with patients | small CPU server |
| Privacy | patient video leaves the home | only joint coordinates leave |
| Licence | serving Ultralytics YOLO over a network triggers AGPL (open-source the service or buy their licence) | MediaPipe is Apache-2.0 |
| Phone load | light (camera + JPEG encode) | moderate; needs a frame-rate cap on older phones |

The products the search did surface (Physitrack's motion capture, smaller pose apps) state that pose runs on the
device and only numbers leave it. That matches the table. `Pipeline.process()` is already engine-agnostic (it takes
a `PoseResult`), so the Python analytics, fall escalation and all 57 tests carry over unchanged in either mode.

---

## 2. Streaming over the public internet

### Transport: WSS with ack-paced JPEG

- Works on every network (port 443, plain TLS). No STUN/TURN servers to run; relays are WebRTC's main operational
  cost, and one practitioner report in the research spent weeks on TURN before abandoning it.
- **Never queue frames.** The client sends frame N+1 only after the server acknowledges frame N. On a slow link
  the frame rate falls by itself and latency stays bounded, which is the correct behaviour for live pose.
- Adapt on the client: target 8–12 fps; if the measured round trip exceeds about 250 ms, step down resolution
  (480 → 320 px wide) and JPEG quality (0.6 → 0.4); step back up when it recovers.
- WebRTC's real advantages (video codec at 200–400 kbit/s, built-in congestion control) matter only if server
  inference must run above about 15 fps or a live caregiver video call is added. Revisit then, with a hosted
  service rather than self-run relays.

### What comes back

No video. The server returns the existing telemetry JSON plus keypoints; the phone draws the skeleton on a canvas
over its own local camera preview. That removes the MJPEG stream, makes Privacy Mode free, and keeps the preview
smooth regardless of network.

### Wire protocol (one socket per patient session)

```
client → server   binary: [u32 seq][f64 capture_ms][JPEG bytes]          (frames mode)
                  text:   {"type":"kp","seq":n,"t":ms,"w":640,"h":480,"kp":[[x,y,conf]×17]}   (keypoints mode)
                  text:   {"type":"cmd","name":"exercise","value":"Elbow Curl"} | {"type":"ack_fall"} ...
server → client   text:   {"type":"t","ack":seq, ...telemetry..., "kp":[...]}
```

### Client capture loop (sketch)

```ts
// one frame in flight; quality/size adapt to round-trip time
async function pump(video: HTMLVideoElement, ws: WebSocket, cfg = { w: 480, q: 0.6 }) {
  const canvas = new OffscreenCanvas(cfg.w, Math.round(cfg.w * video.videoHeight / video.videoWidth));
  const ctx = canvas.getContext("2d")!;
  let seq = 0, sentAt = 0, waiting = false;
  ws.addEventListener("message", (e) => {
    const m = JSON.parse(e.data);
    if (m.ack !== seq) return;
    waiting = false;
    const rtt = performance.now() - sentAt;
    if (rtt > 250) { cfg.w = Math.max(320, cfg.w - 80); cfg.q = Math.max(0.4, cfg.q - 0.1); }
    else if (rtt < 120) { cfg.w = Math.min(480, cfg.w + 40); cfg.q = Math.min(0.6, cfg.q + 0.05); }
  });
  const tick = async () => {
    if (!waiting && ws.readyState === WebSocket.OPEN) {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);        // never mirror what is sent
      const jpeg = await (await canvas.convertToBlob({ type: "image/jpeg", quality: cfg.q })).arrayBuffer();
      const head = new DataView(new ArrayBuffer(12));
      head.setUint32(0, ++seq); head.setFloat64(4, performance.now());
      waiting = true; sentAt = performance.now();
      ws.send(new Blob([head, jpeg]));
    }
    video.requestVideoFrameCallback(tick);
  };
  video.requestVideoFrameCallback(tick);
}
```

### Server side (sketch)

```python
@app.websocket("/ws/session")
async def session_ws(ws: WebSocket, token: str):
    user = verify_token(token)                 # reject before accept()
    await ws.accept()
    s = sessions.create(user)                  # owns its own Pipeline(exercise)
    try:
        while True:
            msg = await ws.receive()
            if msg.get("bytes"):
                seq, t_ms = struct.unpack_from(">Id", msg["bytes"])
                s.latest = (seq, t_ms, msg["bytes"][12:])      # overwrite: only the newest frame is ever inferred
                infer_wakeup.set()
            else:
                s.handle_json(json.loads(msg["text"]))          # keypoints, commands, fall ack
    finally:
        sessions.close(s)
```

One inference thread takes the newest frame of each waiting session, runs YOLO as a batch, calls that session's
`Pipeline.process(res, t)` and sends telemetry with the ack. A GPU that needs 5–10 ms per frame serves many
patients at 10 fps this way.

### Details that will bite if missed

- **Resolution-dependent constants:** `fall_detector.py` uses pixel bins (`ANKLE_BIN_PX = 40`) tuned for 640×480.
  Scale incoming keypoints so the frame's LONG side is 640 px before the pipeline. (Long side, not width: a portrait
  phone frame of 480 x 640 must get the same scale as a landscape one of 640 x 480; scaling by width alone would inflate
  the portrait frame by a third. Implemented as `sessions.to_reference`.)
- **Timestamps:** use the client's capture time for `t`, not arrival time; network jitter otherwise corrupts the
  velocity-based fall logic and rep tempo.
- **Mirroring:** mirror the selfie preview with CSS only. Sending mirrored pixels swaps left and right.
- **Orientation:** phones send portrait frames; the canvas draw already yields upright pixels.

---

## 3. PWA or native

**A PWA covers exercise sessions.** Camera, microphone, WSS, screen wake lock, install-to-home-screen and on-device
pose all work in current mobile browsers. Every browser on iOS is WebKit, so Safari is the constraint: the research
shows a history of standalone-mode camera bugs (repeated permission prompts, streams stalling after Control
Center). Mitigation: test on real iPhones early, and allow running in the Safari tab, not only as an installed app.

**A PWA does not cover these**, and they are where native (or a thin native wrapper around the same web app)
becomes necessary:

| Need | Why the web falls short |
|---|---|
| Fall monitoring while the screen is off or the app is in the background | browsers stop the camera and timers when a page is not visible |
| Guaranteed emergency notification or call | web push is best-effort; no critical alerts |
| Long sessions on old phones | native runtimes (Core ML, NNAPI) run pose cooler than WebAssembly/WebGL |
| Clinical procurement, app-store presence | organisational, not technical |

**A product point that matters more than the framework:** a phone in a pocket cannot film its owner falling. Fall
detection needs a fixed camera with a view of the room. The realistic device is a tablet or old phone on a stand,
on a charger, running the web app full-screen with the wake lock held. That works as a PWA today (most reliably on
Android), and is the configuration to design the fall feature for.

**"Zero-click" has a hard floor on the web.** Browsers require one tap before camera, microphone and audio output
may start. The achievable experience is one large "Start" tap, then hands-free by voice.

---

## 4. Voice over the cloud

```
Browser: getUserMedia(audio, echoCancellation+noiseSuppression) → speech detector → one utterance (≤5 s)
   → POST /api/voice (WAV or WebM, auth token) → server → Groq Whisper → intent.classify → session action
   ← {text, intent} in the response and in telemetry
```

- **Reused as is:** `groq_stt.transcribe`, `intent.classify`, the join-split-sentences logic in
  `VoiceListener.handle`, the escalation hooks. **Retired in cloud mode:** `recorder.py` and `sounddevice`
  (kept for the local kiosk mode).
- **Speech detection in the browser:** port the existing `Segmenter` logic (energy + tone rejection) to an
  AudioWorklet, or use a packaged Silero VAD. Only utterances are uploaded, never a continuous stream: lower cost,
  less data, less audio leaving the device.
- **Echo gets easier:** the phone's own echo cancellation removes the app's spoken prompt from the microphone
  signal, which the PC setup could not do. Keep the listen-after-prompt rule anyway.
- **New commands** ("start my session", "next exercise", "pause", "stop"): a second small table in `intent.py`.
  Safety words keep priority over commands.
- **The Groq key never reaches the browser.** The server rate-limits `/api/voice` per session.

```python
@app.post("/api/voice")
async def voice(file: UploadFile, user=Depends(current_user)):
    s = sessions.for_user(user)
    wav = await file.read()
    if len(wav) > 400_000: raise HTTPException(413)
    text = await run_in_threadpool(groq_stt.transcribe, wav, groq_stt.api_key())
    return {"text": text, "intent": s.voice.handle(text)}
```

---

## 5. Implementation plan

**Phase 0 — close out the MVP.** Commit and push the outstanding work; rotate the Groq key that was pasted in
chat; run the live voice test that is still open.

**Phase 1 — sessions in the backend (no visible change).** Extract what `Worker` does per frame into a `Session`
(pipeline, telemetry, voice handler) and a `SessionManager`. The local OpenCV camera becomes one frame source
feeding one session, so the desktop app and all tests keep working. Normalise keypoints to the 640-wide reference
here. Files: `backend/app/main.py` (split), new `backend/app/sessions.py`.

*Status (2026-11-17): implemented and awaiting its first live run. `Session`, `SessionManager` and `to_reference` are in
`backend/app/sessions.py`; the desktop `Worker` in `main.py` is now only the camera/engine/stream source and feeds
the "local" session. See the activity log for details and what was deliberately left for later phases.*

**Phase 2 — browser camera to cloud inference.** `/ws/session` endpoint and batched inference thread; frontend
`useCamera` hook (front/back switch via `facingMode`, `<video playsInline muted>`), capture pump, canvas skeleton
overlay; remove the page's dependence on `/api/video_feed`. Session-scope the config and fall endpoints.

*Status (2026-11-17): the BACKEND half is implemented and awaiting its first live run: `backend/app/inference.py`
(one inference thread for every session: batches waiting frames for a stateless engine, one engine instance per session
for a stateful one, newest frame per session wins, failures contained per session), `backend/app/remote.py` (the
`/ws/session` gateway, wire protocol in its docstring; OFF unless `REMOTE_SESSIONS=1` because there is no
authentication yet), and the desktop `Worker` now submits its frames to the same service. NOT done yet: the frontend
(`useCamera`, capture pump, canvas overlay), and session-scoped REST (`?session=`) for the config and fall endpoints
(remote sessions use WebSocket commands instead). See the activity log.*

*Status (2026-11-17), frontend half of Phase 2 plus the PWA basics of Phase 5 (requested as "Phase 3"; the blueprint's own
Phase 3 is browser voice and is NOT started): implemented, awaiting its first run on a real phone. A header toggle switches
between "Server camera" (the existing page, default, unchanged) and "This device" (`frontend/src/remote/`: `useRemoteSession`
hook with `getUserMedia`, front/rear switch, one-frame-in-flight JPEG pump with adaptive size/quality, reconnects with
backoff, commands over the socket; `RemoteView` canvas skeleton overlay over the local preview; pure helpers and their
vitest tests in `logic.ts` / `logic.test.ts`). Privacy mode hides the preview image only (frames must still reach the
server). `public/manifest.webmanifest`, `icon.svg` and a network-first `sw.js` (production builds only) make it installable.
NOT done: `/api/video_feed` is still used by the default mode; REST `?session=` scoping; PNG icons (iOS needs an
apple-touch-icon); the single "Start" tap that also unlocks microphone and sound (Phase 5); everything else in Phase 5 and 6.*

**Phase 3 — browser voice.** Microphone capture + speech detector in the browser, `/api/voice`, command intents,
dialog wiring. Delete the frontend debug logs added during the microphone investigation.

*Status (2026-11-17), first slice for the "This device" mode (implemented, awaiting a phone test): the browser listens
continuously (`frontend/src/remote/voice.ts`: MediaRecorder with a supported type chosen at run time, energy detector, the
page's own beeps and spoken prompt are ignored), and sends each utterance as a base64 `audio` message on the SAME authenticated
`/ws/session` socket instead of a separate `/api/voice` endpoint (no second public route; the token, Origin check, session cap
and TunnelGuard already apply). The server transcribes with Groq, then decides by context: while a fall check is open only
"I am fine" / "help" count; otherwise only the commands "start my session" (resets counters), "stop session", "show preview",
"hide preview" (the Privacy switch). 20 uploads per session per minute. NOT done: voice in the "Server camera" mode outside
a fall check (there the microphone is the server's and is only opened for the fall dialog), "next exercise" / "pause", a
voice-only wake-up before the first tap (browsers need one tap for camera and microphone), and the debug logs in
`FallCheck.tsx` (still there).*

**Phase 4 — security and deployment.**
- HTTPS everywhere (required for camera and microphone). Reverse proxy with automatic certificates:
  ```
  rehab.example.com {
      reverse_proxy backend:8000        # Caddy proxies WebSockets and renews TLS itself
  }
  ```
- Login and tokens (short-lived JWT; checked on REST and before `ws.accept()`); today there is no authentication
  at all. Replace `allow_origins=["*"]` with the real origin. Per-session rate limits and message-size caps.
- One backend process per GPU with in-memory sessions, so the proxy needs sticky routing if there is ever more
  than one. Docker image; health endpoint exists.
- No video stored. Patient data handling, hosting region and Groq as a sub-processor need a written decision.

*Status (2026-11-17), Phase 4 first slice (tunnel readiness; implemented, awaiting its first run on a phone): access token for
`/ws/session` (`REMOTE_ACCESS_TOKEN`, sent as the first socket message so it stays out of logs; constant-time compare; 10 wrong
tries per minute locks attempts out), Origin check (`REMOTE_ALLOWED_ORIGINS`), session cap and a cap on not-yet-authenticated
sockets, message size caps; `TunnelGuard` (`backend/app/security.py`): requests that arrive through a tunnel/proxy cannot use the
desktop routes (server camera stream, camera switch, config, fall buttons, telemetry). The frontend takes `?token=` or a typed
PIN; Vite allows the cloudflared / ngrok host names; the built frontend is served by FastAPI on :8000 when `frontend/dist`
exists. NOT done from this phase: HTTPS termination for a real deployment (Caddy), user accounts / JWT, per-user rate limits,
CORS narrowing (REST still answers `*`; the guard and same-origin proxying make that harmless through a tunnel), Docker.*

#### Runbook: try it on a phone through a tunnel

One-time: `winget install --id Cloudflare.cloudflared` (or install ngrok), and `cd frontend; npm install`.

1. `backend/.env`: set `REMOTE_SESSIONS=1` and `REMOTE_ACCESS_TOKEN=<long random value>` (generate one:
   `backend/.venv/Scripts/python -c "import secrets; print(secrets.token_urlsafe(16))"`). Read on every request, no restart needed.
2. Terminal 1, backend: `cd backend; .venv\Scripts\uvicorn app.main:app --port 8000`
3. Terminal 2, frontend: `cd frontend; npm run dev`
4. Terminal 3, tunnel: `cloudflared tunnel --url http://localhost:5173` and copy the printed `https://<words>.trycloudflare.com`.
   (ngrok instead: `ngrok http 5173`; its free plan shows a one-time "Visit site" page first.)
5. Phone browser: open `https://<words>.trycloudflare.com/?token=<your token>`. The page opens in "This device" mode on its own;
   tap Start camera, allow the camera. Skeleton, reps and the fall dialog work; "Use rear" switches cameras.
6. Install as an app (needs the production build, because the service worker is not registered in dev): stop terminal 2 and run
   `cd frontend; npm run build`, restart the backend (it then serves the app itself on :8000), and tunnel that instead:
   `cloudflared tunnel --url http://localhost:8000`. Open the new address on the phone, then Android Chrome: menu, "Install app";
   iPhone Safari: Share, "Add to Home Screen" (iOS wants a PNG icon; the SVG one shows a generic tile).
7. When finished close the tunnel (Ctrl+C in terminal 3): the address stops working at once. Quick tunnels get a new name each run.

Wrong token: the page shows a token field. "Remote sessions are switched off": step 1 was missed.

**Phase 5 — mobile UI and PWA.** Single-column layout (video on top, cards below), large touch targets, one
"Start" tap that unlocks camera, microphone and sound, wake lock during a session, camera-switch button,
manifest and service worker (`vite-plugin-pwa`). Test on a real mid-range Android and a real iPhone.

**Phase 6 — on-device pose (recommended production default).** MediaPipe Pose Landmarker in the browser sends the
keypoints message instead of frames; the server path stays as the fallback for weak devices. Map MediaPipe's 33
landmarks to the existing joint names (the mapping exists in `mediapipe_engine.py`). Cap at 15 fps; measure heat
and battery over a 20-minute session.

*Status (2026-11-17): first slice implemented, awaiting a phone test (see the activity log): MediaPipe Lite runs in the browser
(`frontend/src/remote/onDevicePose.ts`), the phone sends `kp` messages instead of frames, and the SERVER keeps rep counting, fall
detection and escalation (`InferenceService.submit_pose`), so there is one implementation of the safety logic. On-device is the
default on phones; the server models stay selectable. NOT done: the Full model on the phone, a battery/heat measurement over 20
minutes, an offline copy of the WebAssembly files (they come from jsDelivr), and running analytics on the phone for fully
offline use.*

### Verification per phase

- Phases 1–3: existing 57 tests stay green; new tests for session isolation (two sessions do not share reps or
  fall state), frame-drop behaviour, keypoint normalisation, command intents.
- Phase 2 onward: Chrome DevTools network throttling ("Fast 4G", "Slow 4G") and a real cellular connection.
  Record round-trip time, achieved fps, and megabytes per 10 minutes; these replace the estimates above.
- Phase 4: unauthenticated REST and WebSocket requests are refused; a second origin is refused by CORS.
- Phase 5–6: full session on both phone platforms, screen stays on, camera switch works, fall dialog and voice
  work after a single tap.

---

## Decisions needed before Phase 1

1. **On-device pose as the default (Phase 6), or stay server-only?** This decides server size, data usage and
   the licence question. Recommendation: on-device.
2. **Hosting:** GPU server (needed for server inference at any scale) or CPU-only (enough with on-device pose)?
   Which region?
3. **Primary use case:** guided exercise on the patient's own phone, or a fixed tablet that also watches for
   falls? It changes the UI and the device guidance.
4. **Ultralytics licence** if YOLO is served from the cloud: buy it, open-source the service, or avoid by Phase 6.

## Sources

- Physitrack motion capture (on-device pose, only numbers leave): https://www.physitrack.com/motion-capture
- WebRTC vs WebSockets overview: https://getstream.io/blog/webrtc-websockets/
- A team's report on replacing WebRTC with WebSockets and JPEG: https://blog.helix.ml/p/we-killed-webrtc-and-nobody-noticed
- The case for WebRTC on poor networks (vendor): https://livekit.com/blog/why-webrtc-beats-websockets-for-voice-ai-agents
- iOS PWA compatibility notes: https://firt.dev/notes/pwa-ios/
- iOS PWA limitations guide (secondary source): https://www.magicbell.com/blog/pwa-ios-limitations-safari-support-complete-guide
- WebKit bug, camera stream in standalone PWA: https://bugs.webkit.org/show_bug.cgi?id=273938
- Apple forum, repeated camera prompts in web apps: https://developer.apple.com/forums/thread/788518
