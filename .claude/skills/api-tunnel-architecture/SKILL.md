---
name: api-tunnel-architecture
description: Network standards for the phone <-> backend link (Cloudflare/ngrok tunnel, /ws/session WebSocket, REST). Lightweight JSON only (no video frames or base64 images in on-device mode), non-blocking send paths that never stall the UI or pose loop, timeouts, backoff, batching and offline-first caching. Use whenever writing or changing WebSocket/fetch code, the wire protocol, telemetry or voice upload, the service worker, or backend/app/remote.py.
---

# API and tunnel architecture standards

The phone reaches the backend through a quick tunnel (Cloudflare `trycloudflare` / ngrok): a shared, rate-limited, sometimes
slow or briefly dropping path, with a per-request round trip of 50-300 ms. Everything here is built so a slow or dropped
tunnel degrades the experience gracefully and **never** freezes the UI or the pose loop.

Related skills: `cv-fitness-pipeline` (per-frame loop), `react-capacitor-mobile` (UI and render performance).

## 0. How this app divides the work (read before applying the rules literally)

Phone: camera, on-device MediaPipe, skeleton overlay. Server: rep counting, rest calibration, fall detection, escalation, voice
understanding. So the phone cannot send "final rep counts": it sends the **inputs** of the server pipeline, and that is the
accepted minimum for now:

- **Allowed on the live link:** `kp` JSON (17 joints x [x, y, conf], optional 3D `world`; about 1-1.5 kB, hard cap
  `MAX_TEXT_BYTES = 4096`), small `cmd` JSON, the one-off auth message, and the base64 `audio` JSON for a single voice
  utterance (a few tens of kB, capped by `MAX_AUDIO_TEXT`, only on a voice command, never streamed).
- **Telemetry back:** `t` replies rate-limited (`KP_REPLY_GAP_S`), without the pose.
- The long-term direction (roadmap Phase C and the offline-first section below) is moving counting to the phone so the tunnel
  carries session summaries. Do not start that as a side effect of an unrelated change.

## 1. Zero heavy payloads

- **In on-device mode never send video frames, JPEGs, canvas data, base64 images, `ImageData`, blobs or bitmaps.** Compute pose on
  the phone, send joints. `"server"` inference (JPEG frames) exists only as an explicit fallback or a user choice; it must stay
  bounded (see "Server mode" below) and must never be the silent default on a phone.
- **Budget on the wire:** upstream well under 50 kB/s on average. A `kp` message is rounded (1 decimal pixels, 3 decimals metres,
  2 decimals confidence); send no field the server does not read; no pretty-printing; no per-message strings that can be
  derived from the previous one. Add a field only with a size test (`len(text) < 4096` in `tests/test_analytics.py`).
- **Never base64 anything large.** Binary frames use the WebSocket binary type (`packFrame`). Audio is the single exception
  (one short utterance); if utterances grow, send them as a binary frame instead.
- **Cap and validate on the server:** every message type has a maximum size, every field is validated (`parse_keypoints`),
  unknown fields are rejected or ignored, oversize input is dropped, not processed.
- **Tunnel-facing hygiene:** the access token goes in the first message, never in a URL or log (see `security.py`); nothing
  patient-identifying or free text in telemetry that is not needed.
- **Downloads:** the model (`/models/...`) and the app shell are fetched once and cached; never re-download per session.

## 2. Non-blocking, asynchronous, never on the hot path

- **Fire and forget, with bounded buffers.** Per-frame sends are lossy by design: send only if the socket is `OPEN` and
  `bufferedAmount` is below a threshold (64 kB today), otherwise **drop the frame** (latest wins). Never accumulate an unbounded
  array of pending frames or messages; never `await` a send inside the pose loop.
- **One frame in flight** in server mode, and an ack timeout so a lost reply cannot stall it (`canSendFrame`, `ACK_TIMEOUT_MS`).
- **Nothing synchronous on the main thread:** no sync XHR, no `JSON.parse` of a large payload per frame, no `await fetch` inside
  `requestAnimationFrame` or the inference tick. Network work happens in event handlers (`onmessage`, `onopen`) and timers.
- **Commands and uploads go through a small queue, off the hot path.** Telemetry/summary uploads and non-urgent commands are
  appended to an in-memory (then persisted, see section 3) queue and flushed by a timer or `online` event, in the background, with
  a cap on length and age; urgent safety messages (fall acknowledge, escalate, "I'm OK") are sent immediately and retried until
  acknowledged.
- **Never block on the answer.** The UI reacts optimistically and reconciles on the reply; an unanswered request shows a pending
  state and then times out (below), it does not freeze a screen.
- **Heavy work stays off the UI thread:** encoding, compression or base64 of anything big belongs in a Worker (or is avoided).
- **Backend mirror:** the WebSocket handler never runs inference on the event loop (`InferenceService` thread, latest frame wins);
  slow or hung work is cancelled with a timeout; one slow client must not delay another.

## 3. Resilient, offline-first

- **Every network call has a timeout.** `fetch` gets an `AbortController` (5-10 s for control calls, 15-20 s for audio upload);
  socket requests have a reply deadline (the voice waiter has 15 s). A promise that can hang forever is a bug.
- **Reconnect with backoff and jitter** (`reconnectDelay`: 0.5 s doubling to 10 s; add jitter so many phones do not retry in
  step), reset the counter on a successful `hello`, stop retrying on auth failure (4401) and say why once.
- **Detect half-open connections.** A tunnel can die while the socket still reads `OPEN` for minutes. Use an application heartbeat
  or a reply watchdog: no server message for ~5 s while sending (the server answers `kp` at up to 10 Hz) means "connection lost":
  show it, close the socket and reconnect. React to `online` / `offline` events and `visibilitychange` (reconnect on return).
- **State resync after reconnect:** the server creates a fresh session, so re-send exercise, privacy and engine choices (done for
  the exercise); the UI shows "reconnecting", never a frozen skeleton.
- **Local-first for what the phone can do alone:** the skeleton overlay, camera, voice capture and the UI keep working with the
  tunnel down; only server-owned results (reps, fall verdicts) pause, and the UI says so plainly ("Offline: reps are not being
  counted"). Never display stale reps or "all clear" as if they were live; mark them stale after a few seconds without telemetry.
- **Cache what can be cached:** the app shell and the model (service worker / HTTP cache); static lookups (`/api/exercises`,
  `/api/exercise_info`) in `localStorage` or the Cache API with revalidation, so a blip at startup does not leave an empty
  picker; retry failed startup requests with backoff.
- **Batch and persist session results:** completed reps and the session summary are queued (IndexedDB, bounded) and uploaded in
  one request when the link is healthy; uploads are idempotent (client-generated ids) so a retry never double-counts.
  Clear on success; drop the oldest first when the cap is hit.
- **Never crash on a bad network:** every `fetch`/socket handler has an error path (`.catch`), failures are surfaced as one calm
  message, not a stack trace or an alert; JSON from the server is parsed in try/catch and shape-checked.
- **Safety first:** any feature that protects the person (fall detection) must say clearly when it is NOT running because of the
  connection, and must not fail silently. Until fall detection can run on the phone, a lost tunnel means no monitoring.

## Server mode (frames) limits, when it is used

Frames are JPEG binary, not base64, at `MAX_QUALITY` 480 px / 0.6 quality (about 20-40 kB each) and at most 20 fps: roughly
0.4-0.8 MB/s upstream, so it is a fallback for phones that cannot run the model, not a default. It adapts quality down on slow
round trips (`nextQuality`). Keep it that way: never raise the size or rate to "improve accuracy" over the tunnel.

## Checklist before finishing a network change

1. Does any new payload exceed the budget or add per-frame fields? Is there a size test?
2. Is every send non-blocking, bounded and droppable (or queued with a cap)? Nothing awaited in the pose loop?
3. Does every fetch and every socket request have a timeout and an error path?
4. What does the user see, and what happens to a pending action, when the tunnel drops for 10 s, for 2 minutes?
5. Idempotent retries? No secrets or tokens in URLs or logs?
6. `npx tsc --noEmit`, `npm test`, `pytest`-style suite green.

## Audit of the current code (2026-11-18, read-only; nothing was changed)

Already compliant:
- On-device mode sends only `kp` JSON, rounded, optional `world`, capped at 4096 bytes and size-tested; drops frames when
  `bufferedAmount >= 64 kB` (`useRemoteSession.ts:191`); never awaits in the loop.
- Server mode is binary JPEG (no base64), one frame in flight, ack timeout 3 s, quality adapts (`canSendFrame`, `nextQuality`).
- Reconnect with exponential backoff (0.5-10 s), counter reset on `hello`, no retry on 4401, exercise re-sent after reconnect,
  one health hint on the first refusal; token only in the first message; server caps every message type; `kp` replies rate-limited;
  the service worker never touches `/api`, `/ws` or `/models` (live patient data), shell cached network-first; model served locally
  and cached by the browser.

Gaps (priority order):
1. **No half-open detection.** Nothing watches for silence: no heartbeat, no reply watchdog, no `online`/`offline` listener
   (`grep` finds only `visibilitychange`). A tunnel that dies quietly leaves the socket `OPEN`; the phone keeps "sending", the HUD
   says LIVE, and reps and the fall monitor are not running. Highest priority because it is a safety indication.
2. **Offline = no counting, no fall monitoring, no message to the user.** Counting, calibration and fall detection are server-side.
   While reconnecting, reps are lost and the HUD only changes its status text; stale reps stay visible as if current.
3. **Commands are silently dropped when the socket is not open** (`useRemoteSession.ts:375`
   `if (ws.current?.readyState === OPEN) send(...)`, no queue, no retry, no feedback), including "ack fall", "escalate", "start/stop
   session". Safety commands need queue-and-retry-until-acknowledged.
4. **No fetch timeouts.** `App.tsx` `post()`, `/api/exercises`, `/api/exercise_info`, `/api/cameras`, `FallCheck.tsx` fetches,
   `useVoiceInput.ts:15` `/api/voice` (audio upload), `useRemoteSession.ts` `/api/health` all lack `AbortController`; a stalled tunnel
   leaves them pending indefinitely. Only the socket voice waiter has a deadline (15 s).
5. **No startup retry or cache for static lookups:** `/api/exercises` and `/api/exercise_info` are fetched once with
   `.catch(() => {})`; a blip leaves an empty exercise picker and guide until reload.
6. **No persistence or batching of results:** no local queue or IndexedDB; session history/persistence (roadmap P2) does not exist,
   so a dropped tunnel loses the session's reps. Telemetry is not uploaded; it is only streamed live.
7. **Backoff has no jitter** (`reconnectDelay`), and there is no cap on total retry time or a "gave up, tap to retry" state;
   `useTelemetry.ts` (desktop) retries every fixed 1 s.
8. **Server-mode frame path allocates and encodes on the main thread per frame:** `canvas.toBlob` + `blob.arrayBuffer()` +
   `packFrame` (`useRemoteSession.ts:151-152`), up to 20 times a second, and the canvas is resized (`canvas.width = w`) every tick,
   reallocating its backing store. Acceptable as a fallback only; should not be reachable silently on phones.
9. **Voice audio over the socket is base64-in-JSON** (`FileReader.readAsDataURL`, `useRemoteSession.ts:114-120`, about +33 %);
   short and event-driven so acceptable, but a binary frame is cheaper and avoids the FileReader copy.
10. **Service worker:** shell only; `/api/exercises` and `/api/exercise_info` are not cached, and the model relies on the browser's
    HTTP cache (can be evicted). No `sw` strategy for offline start of the app inside the Capacitor WebView (which loads the tunnel
    URL, so with the tunnel down the app shows the WebView's error page; a bundled fallback page would be friendlier).
