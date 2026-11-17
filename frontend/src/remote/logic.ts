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

/** Short name of a pose model for badges ("yolo" -> "YOLO"). Unknown names are shown as they are. */
export const engineLabel = (name: string): string =>
  ({ yolo: "YOLO", mediapipe: "MediaPipe", mediapipe_lite: "MediaPipe Lite" } as Record<string, string>)[name] ?? name;

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
  const f = fitContain(frame.w, frame.h, box.w, box.h);
  const px = (x: number) => f.x + x * f.scale, py = (y: number) => f.y + y * f.scale;
  const colors = LEVEL_COLOR[level];
  const [bx, by, bw, bh] = pose.bbox;
  ctx.lineWidth = 2;
  ctx.strokeStyle = colors.box;
  ctx.strokeRect(px(bx), py(by), bw * f.scale, bh * f.scale);
  ctx.strokeStyle = colors.bone;
  for (const [a, c] of SKELETON) {
    const ka = pose.kp[a], kc = pose.kp[c];
    if (!ka || !kc || ka[2] < minConf || kc[2] < minConf) continue;
    ctx.beginPath();
    ctx.moveTo(px(ka[0]), py(ka[1]));
    ctx.lineTo(px(kc[0]), py(kc[1]));
    ctx.stroke();
  }
  ctx.fillStyle = "#ff5000";
  for (const [x, y, conf] of Object.values(pose.kp)) {
    if (conf < minConf) continue;
    ctx.beginPath();
    ctx.arc(px(x), py(y), 4, 0, Math.PI * 2);
    ctx.fill();
  }
}
