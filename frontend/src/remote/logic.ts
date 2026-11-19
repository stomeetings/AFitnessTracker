/** Pure helpers for the browser-camera ("This device") mode: wire format, adaptive quality, skeleton drawing.
 *  No React and no browser globals, so they are unit-tested in logic.test.ts. Protocol: backend/app/remote.py. */

export type Pose = { kp: Record<string, [number, number, number]>; bbox: [number, number, number, number] };
export type Quality = { longSide: number; quality: number };
export type AlertLevel = "ok" | "suspected" | "confirmed";

/** Same bone list as SKELETON in backend/app/engines/base.py (the local stream draws exactly these). */
export const SKELETON: ReadonlyArray<readonly [string, string]> = [
  ["l_shoulder", "r_shoulder"], ["l_hip", "r_hip"],
  ["l_shoulder", "l_hip"], ["r_shoulder", "r_hip"],
  ["l_shoulder", "l_elbow"], ["l_elbow", "l_wrist"],
  ["r_shoulder", "r_elbow"], ["r_elbow", "r_wrist"],
  ["l_hip", "l_knee"], ["l_knee", "l_ankle"],
  ["r_hip", "r_knee"], ["r_knee", "r_ankle"],
  ["l_ankle", "l_heel"], ["l_heel", "l_foot_index"],
  ["r_ankle", "r_heel"], ["r_heel", "r_foot_index"],
];

export const MAX_QUALITY: Quality = { longSide: 480, quality: 0.6 };
export const MIN_QUALITY: Quality = { longSide: 320, quality: 0.4 };
export const SLOW_RTT_MS = 250;       // above this, send smaller / more compressed frames
export const FAST_RTT_MS = 120;       // below this, step back up
export const ACK_TIMEOUT_MS = 3000;   // an ack this late is assumed lost, so the pump sends again instead of stalling
export const MAX_SEND_FPS = 20;       // never send faster than this: saves battery, heat and the shared GPU
export const MIN_FRAME_GAP_MS = 1000 / MAX_SEND_FPS;

/** Whether the pump may start the next frame now. At most MAX_SEND_FPS (frames are skipped, never queued), and only one frame
 *  in flight (the next waits for the server's ack) unless that ack is overdue and is assumed lost. */
export function canSendFrame(now: number, sentAt: number, inflight: boolean): boolean {
  const since = now - sentAt;
  if (since < MIN_FRAME_GAP_MS) return false;
  return !inflight || since > ACK_TIMEOUT_MS;
}

// ---- Link health and UI pacing (the tunnel can die without the socket noticing; React must not render per reply) ----------------

/** No server message for this long while we are sending = the link is dead (a quick tunnel can drop and leave the socket "open"). */
export const LOST_AFTER_MS = 4000;
/** The UI is brought up to date, and the link checked, this often: server replies arrive up to 10 times a second, React renders 4. */
export const UI_TICK_MS = 250;
/** Commands that protect the person: kept and re-sent after a reconnect instead of being lost with a dropped socket. */
export const SAFETY_COMMANDS: ReadonlyArray<string> = ["ack_fall", "escalate"];
export const SAFETY_QUEUE_MAX = 8;
export const SAFETY_MAX_AGE_MS = 120_000;     // an answer older than this belongs to a situation that is long over

/** Whether the link looks dead: we sent something after the last server message and nothing has come back for `afterMs`.
 *  Nothing sent (hidden tab, model still loading) means nothing is owed, so no verdict. */
export const connectionLost = (now: number, lastRecv: number, lastSent: number, afterMs = LOST_AFTER_MS): boolean =>
  lastSent > lastRecv && now - lastRecv > afterMs;

type Snapshot = {
  exercise?: string; counting?: boolean; engine?: string; error?: string | null;
  fall?: { status?: string; alert?: { state?: string } }; reps?: Record<string, { reps: number }>;
};
const repTotal = (s: Snapshot) => Object.values(s.reps ?? {}).reduce((n, l) => n + l.reps, 0);

/** Telemetry that must reach the screen at once, not at the next UI tick: the first message, a fall or alert change, standby <->
 *  counting, a new exercise or model, an error, a completed rep. Everything else (angles, states, timings) waits for the tick. */
export function urgentChange(prev: Snapshot | null, next: Snapshot): boolean {
  if (!prev) return true;
  return prev.fall?.status !== next.fall?.status || prev.fall?.alert?.state !== next.fall?.alert?.state
    || prev.counting !== next.counting || prev.exercise !== next.exercise || prev.engine !== next.engine
    || (prev.error ?? null) !== (next.error ?? null) || repTotal(prev) !== repTotal(next);
}

/** `[u32 frame number][f64 capture time ms]` big-endian + JPEG: what `struct.Struct(">Id")` reads on the server. */
export function packFrame(seq: number, captureMs: number, jpeg: ArrayBuffer): ArrayBuffer {
  const out = new Uint8Array(12 + jpeg.byteLength);
  const head = new DataView(out.buffer);
  head.setUint32(0, seq >>> 0);          // DataView is big-endian unless told otherwise
  head.setFloat64(4, captureMs);
  out.set(new Uint8Array(jpeg), 12);
  return out.buffer;
}

/** Frame size to send: the video scaled down so its LONG side is at most `longSide` (never up). Aspect is kept, so a
 *  portrait phone sends portrait frames; the server normalises by long side, so both are judged at the same scale. */
export function captureSize(videoW: number, videoH: number, longSide: number): { w: number; h: number } {
  const s = Math.min(1, longSide / Math.max(videoW, videoH));
  return { w: Math.max(1, Math.round(videoW * s)), h: Math.max(1, Math.round(videoH * s)) };
}

/** Adapt to the measured round trip: slow link -> smaller, blurrier frames; fast link -> step back up. */
export function nextQuality(q: Quality, rttMs: number): Quality {
  if (rttMs > SLOW_RTT_MS) {
    return { longSide: Math.max(MIN_QUALITY.longSide, q.longSide - 80), quality: Math.max(MIN_QUALITY.quality, round2(q.quality - 0.1)) };
  }
  if (rttMs < FAST_RTT_MS) {
    return { longSide: Math.min(MAX_QUALITY.longSide, q.longSide + 40), quality: Math.min(MAX_QUALITY.quality, round2(q.quality + 0.05)) };
  }
  return q;
}
const round2 = (n: number) => Math.round(n * 100) / 100;

/** The access token from a page address like `https://host/?token=abc`: the value, and the search string without it, so
 *  the token can be stripped from the address bar (and from history and screenshots) once it has been saved. */
export function extractToken(search: string): { token: string | null; rest: string } {
  const params = new URLSearchParams(search);
  const token = params.get("token");
  if (token === null) return { token: null, rest: search };
  params.delete("token");
  const rest = params.toString();
  return { token: token.trim() || null, rest: rest ? `?${rest}` : "" };
}

/** Phones and tunnels open this page on a real host name; only the laptop itself uses localhost. */
export const isLocalHost = (hostname: string) => ["localhost", "127.0.0.1", "[::1]", "::1"].includes(hostname);

// ---- On-device pose (MediaPipe running in this browser; only joint positions are sent) ----------------------------------------

/** The pseudo-model name for "the phone runs the model": it sits beside the server models ("yolo" ...) in the badge's list. */
export const ON_DEVICE = "device";
export type Keypoints = Record<string, [number, number, number]>;
export type InferenceMode = "device" | "server";

export const MAX_LOCAL_FPS = 30;                                  // the camera delivers 30 at most; more would only burn battery
export const MIN_LOCAL_GAP_MS = 1000 / MAX_LOCAL_FPS - 3;         // a little slack: a 30 fps camera's frames are 33.3 ms apart

/** Whether the on-device loop may run another inference now (at most MAX_LOCAL_FPS; frames in between are skipped). */
export const canRunInference = (now: number, last: number): boolean => now - last >= MIN_LOCAL_GAP_MS;

/** MediaPipe landmark index -> the joint name the server's pipeline uses (the same table as the server's MediaPipe engine). */
export const LANDMARK_JOINTS: Readonly<Record<number, string>> = {
  0: "nose",
  11: "l_shoulder", 12: "r_shoulder", 13: "l_elbow", 14: "r_elbow", 15: "l_wrist", 16: "r_wrist",
  23: "l_hip", 24: "r_hip", 25: "l_knee", 26: "r_knee", 27: "l_ankle", 28: "r_ankle",
  29: "l_heel", 30: "r_heel", 31: "l_foot_index", 32: "r_foot_index",
};

// ---- Adaptive camera resolution --------------------------------------------------------------------------------------------------
// The pose model reads about a 256 px crop whatever the camera delivers, so a bigger camera image only costs texture upload and
// conversion on every frame. The camera therefore runs in tiers: a starting tier chosen from the device, stepped DOWN when the
// model's own time stays over budget, and back UP (never above the ceiling) after a long stretch of headroom.

export type CameraTier = "high" | "mid" | "low";
export const TIER_ORDER: ReadonlyArray<CameraTier> = ["high", "mid", "low"];
export const TIER_SIZE: Readonly<Record<CameraTier, { w: number; h: number }>> = {
  high: { w: 1280, h: 720 }, mid: { w: 640, h: 480 }, low: { w: 480, h: 360 },
};

/** Where to start: a computer gets 720p; a phone 640x480, or 480x360 when it reports little memory (navigator.deviceMemory, Chrome
 *  only) or few cores. Unknown values (iOS Safari reports neither) do not count against the phone. */
export function initialTier(mobile: boolean, cores?: number, memoryGb?: number): CameraTier {
  if (!mobile) return "high";
  if ((memoryGb !== undefined && memoryGb <= 3) || (cores !== undefined && cores <= 4)) return "low";
  return "mid";
}
/** The best tier the monitor may climb back to: phones stop at 640x480 (more pixels buy the pose model nothing). */
export const tierCeiling = (mobile: boolean): CameraTier => (mobile ? "mid" : "high");

/** `getUserMedia` / `applyConstraints` size for a tier. `ideal`, never `exact`: the camera picks its nearest mode instead of failing. */
export const tierConstraints = (tier: CameraTier) => ({ width: { ideal: TIER_SIZE[tier].w }, height: { ideal: TIER_SIZE[tier].h } });

const SLOW_MS = 33;            // the model alone takes longer than a 30 fps frame: over budget
const SLOW_FPS = 15;           // below this with a model that is not cheap: the phone cannot keep up
const FAST_MS = 16, FAST_FPS = 24;
const DOWN_AFTER = 6;          // samples (one per 0.5 s): 3 s of trouble before stepping down
const UP_AFTER = 40;           // 20 s of headroom before stepping up
const SETTLE = 4;              // samples ignored after a change: the camera is still restarting

/** Decides camera tier changes from the on-device loop's stats (call `sample` about every 0.5 s with the achieved fps and the
 *  smoothed model ms). Returns the new tier when it changes, else null. Pure state, no timers: unit-tested. */
export function createTierMonitor(start: CameraTier, ceiling: CameraTier = start) {
  let index = TIER_ORDER.indexOf(start), slow = 0, fast = 0, settle = SETTLE;
  const top = TIER_ORDER.indexOf(ceiling);
  return {
    get tier(): CameraTier { return TIER_ORDER[index]; },
    sample(fps: number, ms: number): CameraTier | null {
      if (settle > 0) { settle--; return null; }
      const isSlow = ms > SLOW_MS || (fps > 0 && fps < SLOW_FPS && ms > 20);     // fps 0 = not measured yet
      const isFast = ms > 0 && ms < FAST_MS && fps >= FAST_FPS;
      slow = isSlow ? slow + 1 : 0;
      fast = isFast ? fast + 1 : 0;
      let next = index;
      if (slow >= DOWN_AFTER && index < TIER_ORDER.length - 1) next = index + 1;
      else if (fast >= UP_AFTER && index > top) next = index - 1;
      if (next === index) return null;
      index = next; slow = fast = 0; settle = SETTLE;
      return TIER_ORDER[index];
    },
  };
}

// The landmark table as two plain arrays, built ONCE: the per-frame code walks them by index (no Object.entries, no iterator).
const JOINT_INDEX: number[] = Object.keys(LANDMARK_JOINTS).map(Number);
const JOINT_NAME: string[] = JOINT_INDEX.map((i) => LANDMARK_JOINTS[i]);

/** Hot path (runs for every camera frame): fills `out` IN PLACE with the 33 normalised landmarks (x, y in 0..1) as the 17 named
 *  joints in pixels of the w x h frame, rounded to keep the message small, and returns it. `out` keeps its joint arrays between
 *  calls, so a steady stream of frames allocates nothing; a landmark that is missing is removed from `out`. `visibility` is the
 *  confidence; a missing one counts as 0. The result is only valid until the next call: copy it if it must be kept. */
export function landmarksToKeypointsInto(out: Keypoints, landmarks: ReadonlyArray<{ x: number; y: number; visibility?: number }>, w: number, h: number): Keypoints {
  for (let k = 0; k < JOINT_NAME.length; k++) {
    const name = JOINT_NAME[k], lm = landmarks[JOINT_INDEX[k]];
    if (!lm) { delete out[name]; continue; }                 // never happens with MediaPipe's fixed 33; keeps "absent" meaning absent
    let p = out[name];
    if (!p) { p = [0, 0, 0]; out[name] = p; }                // the first frame only
    p[0] = Math.round(lm.x * w * 10) / 10;
    p[1] = Math.round(lm.y * h * 10) / 10;
    p[2] = Math.round((lm.visibility ?? 0) * 100) / 100;
  }
  return out;
}

/** The same, into a fresh object (tests, one-off use). Not for the per-frame loop. */
export const landmarksToKeypoints = (landmarks: ReadonlyArray<{ x: number; y: number; visibility?: number }>, w: number, h: number): Keypoints =>
  landmarksToKeypointsInto({}, landmarks, w, h);

/** Joint name -> [x, y, z] in METRES from the hip centre (MediaPipe's world landmarks), the same joint names as `Keypoints`. */
export type World = Record<string, [number, number, number]>;

/** MediaPipe's 33 world landmarks as the 17 named joints, millimetre precision (3 decimals, metres). Angles measured in these do
 *  not depend on the camera's viewing angle, unlike angles measured in the image (a knee bending toward the camera barely changes
 *  in 2D). The server uses them when every joint of an angle has one. */
export function worldToJointsInto(out: World, world: ReadonlyArray<{ x: number; y: number; z: number }>): World {
  for (let k = 0; k < JOINT_NAME.length; k++) {                // hot path: in place, see landmarksToKeypointsInto
    const name = JOINT_NAME[k], p = world[JOINT_INDEX[k]];
    if (!p) { delete out[name]; continue; }
    let q = out[name];
    if (!q) { q = [0, 0, 0]; out[name] = q; }
    q[0] = Math.round(p.x * 1000) / 1000;
    q[1] = Math.round(p.y * 1000) / 1000;
    q[2] = Math.round(p.z * 1000) / 1000;
  }
  return out;
}

/** The same, into a fresh object (tests, one-off use). Not for the per-frame loop. */
export const worldToJoints = (world: ReadonlyArray<{ x: number; y: number; z: number }>): World => worldToJointsInto({}, world);

/** What the HUD says about the rest angles measured in standby ("calibrated rest 92°"), or undefined when none was. */
export function calibrationNote(reps?: Record<string, { calibrated?: boolean; rest?: number }>): string | undefined {
  const done = Object.entries(reps ?? {}).filter(([, l]) => l.calibrated && l.rest !== undefined);
  if (!done.length) return undefined;
  return "calibrated rest " + done.map(([side, l]) => `${done.length > 1 ? `${side} ` : ""}${Math.round(l.rest!)}°`).join(" · ");
}

/** Bounding box [x, y, w, h] of the confident joints, padded; null with fewer than 4 (no person to outline). For drawing only:
 *  the server works out its own. */
export function bboxFromKeypoints(kp: Keypoints, minConf = 0.5, pad = 0.05): [number, number, number, number] | null {
  const pts = Object.values(kp).filter((p) => p[2] >= minConf);
  if (pts.length < 4) return null;
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const px = (x1 - x0) * pad, py = (y1 - y0) * pad;
  return [x0 - px, y0 - py, x1 - x0 + 2 * px, y1 - y0 + 2 * py];
}

/** Hot-path twin of `bboxFromKeypoints`: writes the padded box into `out` ([x, y, w, h], reused) and returns true, or returns
 *  false (out untouched) with fewer than 4 confident joints. No intermediate arrays. */
export function bboxFromKeypointsInto(out: [number, number, number, number], kp: Keypoints, minConf = 0.5, pad = 0.05): boolean {
  let n = 0, x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const name in kp) {
    const p = kp[name];
    if (p[2] < minConf) continue;
    n++;
    if (p[0] < x0) x0 = p[0];
    if (p[0] > x1) x1 = p[0];
    if (p[1] < y0) y0 = p[1];
    if (p[1] > y1) y1 = p[1];
  }
  if (n < 4) return false;
  const px = (x1 - x0) * pad, py = (y1 - y0) * pad;
  out[0] = x0 - px; out[1] = y0 - py; out[2] = x1 - x0 + 2 * px; out[3] = y1 - y0 + 2 * py;
  return true;
}

/** The `kp` message the server's gateway expects (backend/app/remote.py). `"type":"kp"` must come first: the server tells these
 *  messages apart by that prefix. */
export const keypointMessage = (seq: number, tMs: number, w: number, h: number, kp: Keypoints, world?: World): string =>
  JSON.stringify({ type: "kp", seq, t: Math.round(tMs * 10) / 10, w, h, kp, ...(world ? { world } : {}) });

/** A phone or tablet (iPadOS reports itself as a Mac with a touch screen). */
export const isMobileDevice = (userAgent: string, maxTouchPoints: number): boolean =>
  /Android|iPhone|iPad|iPod|Mobile/i.test(userAgent) || (/Macintosh/.test(userAgent) && maxTouchPoints > 1);

/** On-device pose is the default on phones (no video over the network), the server's models elsewhere. */
export const defaultInference = (mobile: boolean): InferenceMode => (mobile ? "device" : "server");

/** Who runs the analytics (reps, calibration, falls, escalation): "local" = this phone (works offline), "server" = the backend over
 *  /ws/session. The bundled (offline-first) APK defaults to local, everything else to the server until parity is proven on phones.
 *  Local needs the pose computed on this device too: with server-side pose the analytics stay on the server. */
export type EngineMode = "local" | "server";
export const defaultEngineMode = (bundled: boolean): EngineMode => (bundled ? "local" : "server");
export const analyticsLocal = (engine: EngineMode, inference: InferenceMode): boolean => engine === "local" && inference === "device";

/** Short name of a pose model for badges ("yolo" -> "YOLO"). Unknown names are shown as they are. */
export const engineLabel = (name: string): string =>
  ({ yolo: "YOLO", mediapipe: "MediaPipe", mediapipe_lite: "MediaPipe Lite", device: "On-device MediaPipe" } as Record<string, string>)[name] ?? name;

/** The model a tap on the badge switches to: the next one in the server's list after `current` (wrapping). Null when there is
 *  nothing to switch to (an empty or one-item list). */
export function nextEngine(current: string | null, engines: readonly string[]): string | null {
  if (engines.length < 2) return null;
  const i = current === null ? -1 : engines.indexOf(current);
  return engines[(i + 1) % engines.length];
}

/** Backoff for reconnects: 0.5 s, 1 s, 2 s ... capped at 10 s. */
export const reconnectDelay = (attempt: number) => Math.min(10_000, 500 * 2 ** attempt);

/** Where an image of size sw x sh lands when letterboxed (object-contain) inside a dw x dh box. */
export function fitContain(sw: number, sh: number, dw: number, dh: number) {
  const scale = Math.min(dw / sw, dh / sh);
  const w = sw * scale, h = sh * scale;
  return { x: (dw - w) / 2, y: (dh - h) / 2, w, h, scale };
}

const LEVEL_COLOR: Record<AlertLevel, { bone: string; box: string }> = {
  ok: { bone: "#22c55e", box: "#00c8ff" },
  suspected: { bone: "#f59e0b", box: "#f59e0b" },
  confirmed: { bone: "#ef4444", box: "#ef4444" },
};

/** Draws the pose (bbox, bones, joints) onto a canvas context of CSS size `box`, mapping frame pixels onto the letterboxed
 *  preview. Only the pose is drawn, never text, so a mirrored (selfie) canvas stays readable. Colour follows the fall level. */
export function drawPose(
  ctx: Pick<CanvasRenderingContext2D, "clearRect" | "beginPath" | "moveTo" | "lineTo" | "stroke" | "arc" | "fill" | "strokeRect"> &
    { strokeStyle: unknown; fillStyle: unknown; lineWidth: number },
  pose: Pose | null,
  frame: { w: number; h: number },
  box: { w: number; h: number },
  level: AlertLevel = "ok",
  minConf = 0.5,
): void {
  ctx.clearRect(0, 0, box.w, box.h);
  if (!pose || !frame.w || !frame.h) return;
  // Hot path (every pose): no helper objects, no closures, no destructuring, no Object.values. The letterbox maths of fitContain
  // is inlined (same formula), and everything is read by index or by key straight from the pose.
  const scale = Math.min(box.w / frame.w, box.h / frame.h);
  const ox = (box.w - frame.w * scale) / 2, oy = (box.h - frame.h * scale) / 2;
  const colors = LEVEL_COLOR[level], kp = pose.kp, bb = pose.bbox;
  ctx.lineWidth = 2;
  ctx.strokeStyle = colors.box;
  ctx.strokeRect(ox + bb[0] * scale, oy + bb[1] * scale, bb[2] * scale, bb[3] * scale);
  ctx.strokeStyle = colors.bone;
  for (let i = 0; i < SKELETON.length; i++) {
    const bone = SKELETON[i], ka = kp[bone[0]], kc = kp[bone[1]];
    if (!ka || !kc || ka[2] < minConf || kc[2] < minConf) continue;
    ctx.beginPath();
    ctx.moveTo(ox + ka[0] * scale, oy + ka[1] * scale);
    ctx.lineTo(ox + kc[0] * scale, oy + kc[1] * scale);
    ctx.stroke();
  }
  ctx.fillStyle = "#ff5000";
  for (const name in kp) {
    const p = kp[name];
    if (p[2] < minConf) continue;
    ctx.beginPath();
    ctx.arc(ox + p[0] * scale, oy + p[1] * scale, 4, 0, TWO_PI);
    ctx.fill();
  }
}
const TWO_PI = Math.PI * 2;
