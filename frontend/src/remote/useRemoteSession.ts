import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Telemetry } from "../useTelemetry";
import {
  MAX_QUALITY, ON_DEVICE, SAFETY_COMMANDS, SAFETY_MAX_AGE_MS, SAFETY_QUEUE_MAX, UI_TICK_MS, bboxFromKeypoints, canSendFrame, captureSize,
  connectionLost, defaultInference, extractToken, isMobileDevice, keypointMessage, nextQuality, packFrame, reconnectDelay, urgentChange,
  type InferenceMode, type Pose, type Quality,
} from "./logic";
import { createLandmarker, runLocalLoop, type LoadedLandmarker } from "./onDevicePose";
import { MIC_CONSTRAINTS, micErrorText } from "./voice";
import { useVoiceInput, type VoiceReply } from "./useVoiceInput";

const TOKEN_KEY = "remoteToken";
const saveToken = (v: string) => { try { if (v) localStorage.setItem(TOKEN_KEY, v); else localStorage.removeItem(TOKEN_KEY); } catch { /* storage blocked: this tab only */ } };

/** The access token: from `?token=...` in the page address (saved, then removed from the address bar) or from an earlier
 *  visit. Safe to call twice (React StrictMode does): the second call finds it saved. */
function loadToken(): string {
  try {
    const { token, rest } = extractToken(location.search);
    if (token) {
      saveToken(token);
      history.replaceState(null, "", location.pathname + rest + location.hash);
    }
    return localStorage.getItem(TOKEN_KEY) ?? "";
  } catch { return ""; }
}

const INFERENCE_KEY = "inference";
/** Who computes the pose: this device (phones by default: no video leaves it) or the server (frames are sent). A choice made on
 *  the badge is remembered; the automatic fallback to the server when on-device pose cannot start is not. */
function loadInference(): InferenceMode {
  try {
    const saved = localStorage.getItem(INFERENCE_KEY);
    if (saved === "device" || saved === "server") return saved;
  } catch { /* no storage */ }
  return defaultInference(isMobileDevice(navigator.userAgent, navigator.maxTouchPoints ?? 0));
}
const saveInference = (m: InferenceMode) => { try { localStorage.setItem(INFERENCE_KEY, m); } catch { /* storage blocked */ } };

/** Browser camera -> /ws/session -> telemetry + keypoints back (backend/app/remote.py has the protocol).
 *  One frame is in flight at a time, so a slow link lowers the frame rate instead of building a queue.
 *  Nothing starts until start() is called from a click: browsers require that for camera access. */

export type RemoteStatus = "idle" | "starting" | "connecting" | "live" | "reconnecting" | "error";
export type Facing = "user" | "environment";

type ServerMsg = Partial<Telemetry> & {
  type: string; ack?: number; frame?: { w: number; h: number }; pose?: Pose | null; infer_ms?: number;
  session?: string; name?: string; ok?: boolean; error?: string; text?: string; intent?: string | null; status?: string;
  engines?: string[];
};

const HEALTH_HINT = "Remote sessions are switched off on the server. Start the backend with REMOTE_SESSIONS=1 (backend/.env).";

/** The camera, and (when asked) the microphone in the same permission prompt, split into two streams so switching the
 *  camera never touches the microphone. A refused or missing microphone does not stop the camera: it is reported in
 *  `micError` and the session runs without voice. Must run from a tap (browsers ask for permission only then). */
async function openMedia(facing: Facing, audio: boolean, exact = false): Promise<{ stream: MediaStream; mic: MediaStream | null; micError: string | null }> {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error(window.isSecureContext ? "This browser has no camera API." : "The camera needs HTTPS (or localhost). Open this page over https://.");
  }
  // `ideal` first: a laptop with one camera still starts when "rear" is asked for. `exact` is the retry for phones that
  // ignore `ideal` and hand back the same camera.
  const video = { facingMode: exact ? { exact: facing } : { ideal: facing }, width: { ideal: 1280 }, height: { ideal: 720 } };
  if (audio) {
    try {
      const all = await navigator.mediaDevices.getUserMedia({ video, audio: MIC_CONSTRAINTS });
      const tracks = all.getAudioTracks();
      return { stream: new MediaStream(all.getVideoTracks()), mic: tracks.length ? new MediaStream(tracks) : null, micError: tracks.length ? null : "No microphone found." };
    } catch (e) {
      const stream = await navigator.mediaDevices.getUserMedia({ video, audio: false });   // throws too if the camera was the problem
      return { stream, mic: null, micError: micErrorText(e) };
    }
  }
  return { stream: await navigator.mediaDevices.getUserMedia({ video, audio: false }), mic: null, micError: null };
}

export function useRemoteSession() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [status, setStatus] = useState<RemoteStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [facing, setFacing] = useState<Facing>("user");
  // Server replies arrive up to 10 times a second. They are kept in refs and reach React (`msg`, `stats`) at most every UI_TICK_MS,
  // or at once when something important changed (urgentChange): rendering the whole page per reply made phones stutter.
  const [msg, setMsg] = useState<ServerMsg | null>(null);
  const msgRef = useRef<ServerMsg | null>(null);          // the newest telemetry
  const shownMsg = useRef<ServerMsg | null>(null);        // what `msg` holds
  const [needsToken, setNeedsToken] = useState(false);   // the server wants an access token we do not have (or it was wrong)
  const token = useRef<string | null>(null);
  if (token.current === null) token.current = loadToken();
  const INITIAL_STATS = { fps: 0, rtt: 0, longSide: MAX_QUALITY.longSide, inferMs: 0 };
  const [stats, setStats] = useState(INITIAL_STATS);
  const shownStats = useRef(INITIAL_STATS);
  const lastRecv = useRef(0);                              // when the server last said anything, and when we last sent something
  const lastSent = useRef(0);                              // that wants an answer: together they tell a dead tunnel from a quiet one
  const safetyQueue = useRef<{ name: string; at: number }[]>([]);   // ack / escalate typed while the link was down
  const dropRef = useRef<(() => void) | null>(null);      // gives up on the current socket (set by connect)
  const lastFrame = useRef<{ w: number; h: number }>({ w: 0, h: 0 });
  const [engines, setEngines] = useState<string[]>([]);              // pose models the server offers to switch between
  const [engineWanted, setEngineWanted] = useState<string | null>(null);   // asked for, not yet confirmed by the server
  // On-device pose: the model runs in this browser and only joint positions are sent (no frames).
  const [inference, setInferenceState] = useState<InferenceMode>(loadInference);
  const inferenceRef = useRef(inference);
  const [local, setLocal] = useState({ fps: 0, ms: 0, status: "idle" as "idle" | "loading" | "running", delegate: "" });
  const landmarkerRef = useRef<LoadedLandmarker | null>(null);       // loaded once, kept across stops and restarts
  const loopRef = useRef<{ stop: () => void } | null>(null);
  const kpSeq = useRef(0);
  const poseListener = useRef<((pose: Pose | null, frame: { w: number; h: number }) => void) | null>(null);
  /** The view registers a function that draws a pose straight onto its canvas: 30 a second must not go through React state. */
  const setPoseListener = useCallback((fn: ((pose: Pose | null, frame: { w: number; h: number }) => void) | null) => { poseListener.current = fn; }, []);

  const want = useRef(false);                       // the user asked for a running session
  const ws = useRef<WebSocket | null>(null);
  const opened = useRef(false);                     // this connection attempt got as far as the server's hello
  const switching = useRef(false);                  // a camera switch is under way (taps meanwhile are ignored)
  const voiceWaiter = useRef<((r: VoiceReply) => void) | null>(null);

  /** One utterance over the session socket (base64 in JSON): the same authenticated socket the frames use, so the access
   *  token and the tunnel rules already apply. Resolves with the server's `voice` reply. */
  const uploadViaSocket = (blob: Blob) => new Promise<VoiceReply>((resolve) => {
    const sock = ws.current;
    if (!sock || sock.readyState !== WebSocket.OPEN || !opened.current) { resolve({ ok: false, error: "Not connected to the server." }); return; }
    voiceWaiter.current = resolve;
    window.setTimeout(() => { if (voiceWaiter.current === resolve) { voiceWaiter.current = null; resolve({ ok: false, error: "No answer from the server." }); } }, 15000);
    const reader = new FileReader();
    reader.onload = () => {
      const url = String(reader.result);
      if (sock.readyState === WebSocket.OPEN) sock.send(JSON.stringify({ type: "audio", mime: blob.type || "audio/webm", data: url.slice(url.indexOf(",") + 1) }));
    };
    reader.onerror = () => { if (voiceWaiter.current === resolve) { voiceWaiter.current = null; resolve({ ok: false, error: "Could not read the audio." }); } };
    reader.readAsDataURL(blob);
  });
  const voice = useVoiceInput(uploadViaSocket);     // capture, status and what was heard: shared with the desktop page
  const endVoice = voice.end;                       // stable, so `stop` below can depend on it
  const streamRef = useRef<MediaStream | null>(null);
  const facingRef = useRef<Facing>("user");
  const exerciseRef = useRef<string | null>(null); // re-sent after a reconnect (the server makes a fresh session)
  const timers = useRef<{ reconnect?: number; pump?: number; ui?: number }>({});
  const attempts = useRef(0);
  const wake = useRef<WakeLockSentinel | null>(null);
  const pumpState = useRef({ seq: 0, inflight: false, sentAt: 0, q: MAX_QUALITY as Quality, fps: 0, lastAck: 0, infer: 0, rtt: 0 });

  const stopPump = () => { window.clearTimeout(timers.current.pump); timers.current.pump = undefined; };

  const flushMsg = () => { if (msgRef.current !== shownMsg.current) { shownMsg.current = msgRef.current; setMsg(msgRef.current); } };
  const flushStats = () => {
    const p = pumpState.current, s = shownStats.current;
    const next = { fps: Math.round(p.fps * 10) / 10, rtt: Math.round(p.rtt), longSide: p.q.longSide, inferMs: Math.round(p.infer * 10) / 10 };
    if (next.fps !== s.fps || next.rtt !== s.rtt || next.longSide !== s.longSide || next.inferMs !== s.inferMs) { shownStats.current = next; setStats(next); }
  };
  const clearPose = () => poseListener.current?.(null, { w: 0, h: 0 });          // wipe the skeleton: it would look live

  /** Every UI_TICK_MS while a session is wanted: bring React up to date, and check the tunnel. A quick tunnel can die without the
   *  socket ever noticing, so the socket still reads OPEN while nothing is counted or watched: that is detected here. */
  const uiTick = () => {
    flushMsg();
    flushStats();
    if (document.hidden || !opened.current) return;      // a hidden tab sends nothing, so nothing can be missing
    if (connectionLost(performance.now(), lastRecv.current, lastSent.current)) dropRef.current?.();
  };
  const startUiTick = () => { if (timers.current.ui === undefined) timers.current.ui = window.setInterval(uiTick, UI_TICK_MS); };
  const stopUiTick = () => { window.clearInterval(timers.current.ui); timers.current.ui = undefined; };

  const startPump = useCallback(() => {
    stopPump();
    const p = pumpState.current;
    p.inflight = false;
    p.lastAck = 0;
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    const tick = async () => {
      const sock = ws.current, video = videoRef.current;
      const now = performance.now();
      if (sock && video && ctx && sock.readyState === WebSocket.OPEN && video.readyState >= 2 && video.videoWidth > 0
          && canSendFrame(now, p.sentAt, p.inflight)) {      // at most 20 fps, one frame in flight
        const { w, h } = captureSize(video.videoWidth, video.videoHeight, p.q.longSide);
        canvas.width = w; canvas.height = h;
        ctx.drawImage(video, 0, 0, w, h);          // the raw, un-mirrored image: mirroring is CSS only, or left and right swap
        const seq = ++p.seq;
        p.inflight = true; p.sentAt = now;
        const blob = await new Promise<Blob | null>((res) => canvas.toBlob(res, "image/jpeg", p.q.quality));
        if (blob && sock.readyState === WebSocket.OPEN) { sock.send(packFrame(seq, now, await blob.arrayBuffer())); lastSent.current = performance.now(); }
        else p.inflight = false;
      }
      timers.current.pump = window.setTimeout(tick, 20);
    };
    tick();
  }, []);

  const stopLocal = () => { loopRef.current?.stop(); loopRef.current = null; };

  /** On-device inference: load the model (once), then detect on every new camera frame and send the joints. If it cannot run
   *  (no WebAssembly, offline and not cached, model missing, the model throws), the server takes over with frames. */
  const startLocal = useCallback(async () => {
    stopLocal();
    const fallBack = (why: unknown) => {
      if (!want.current) return;
      setError(`On-device pose is not available (${why instanceof Error ? why.message : String(why)}); the server is doing it instead.`);
      inferenceRef.current = "server"; setInferenceState("server");     // not saved: the next visit tries on-device again
      setLocal((l) => ({ ...l, status: "idle" }));
      startPump();
    };
    setLocal((l) => ({ ...l, status: "loading" }));
    try {
      if (!landmarkerRef.current) landmarkerRef.current = await createLandmarker();
    } catch (e) {
      fallBack(e);
      return;
    }
    if (!want.current || inferenceRef.current !== "device" || !opened.current) {     // things changed while the model loaded
      setLocal((l) => ({ ...l, status: "idle" }));
      return;
    }
    const { landmarker, delegate } = landmarkerRef.current;
    setLocal((l) => ({ ...l, status: "running", delegate }));
    loopRef.current = runLocalLoop(() => videoRef.current, landmarker, {
      pose: (kp, w, h, tMs, world) => {
        const box = kp ? bboxFromKeypoints(kp) : null;
        poseListener.current?.(kp && box ? { kp, bbox: box } : null, { w, h });      // the skeleton, with no network in between
        const sock = ws.current;
        if (sock && sock.readyState === WebSocket.OPEN && opened.current && sock.bufferedAmount < 65536) {   // never build a backlog
          sock.send(keypointMessage(++kpSeq.current, tMs, w, h, kp ?? {}, kp ? world : undefined));
          lastSent.current = performance.now();
        }
      },
      stats: (fps, ms) => setLocal((l) => ({ ...l, fps, ms })),
      failed: fallBack,
    });
  }, [startPump]);

  /** (Re)start whichever source of poses is selected: this device's model, or JPEG frames for the server's. */
  const startActive = useCallback(() => {
    stopPump();
    stopLocal();
    if (inferenceRef.current === "device") void startLocal();
    else startPump();
  }, [startLocal, startPump]);

  const releaseWake = () => { void wake.current?.release().catch(() => {}); wake.current = null; };
  const holdWake = useCallback(async () => {       // keep the screen on during a session; best effort, not on every browser
    try { if (want.current && !wake.current) wake.current = (await navigator.wakeLock?.request("screen")) ?? null; } catch { /* denied */ }
  }, []);

  const connect = useCallback(() => {
    if (!want.current) return;
    opened.current = false;
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const sock = new WebSocket(`${proto}://${location.host}/ws/session`);
    sock.binaryType = "arraybuffer";
    ws.current = sock;
    // The token goes in the first message, not the address, so it never shows up in server or tunnel logs.
    sock.onopen = () => { if (token.current) sock.send(JSON.stringify({ type: "auth", token: token.current })); };
    sock.onmessage = (e) => {
      lastRecv.current = performance.now();                       // any message proves the link is alive
      if (typeof e.data !== "string") return;
      const m = JSON.parse(e.data) as ServerMsg;
      const p = pumpState.current;
      if (m.type === "hello") {
        opened.current = true; attempts.current = 0;
        lastSent.current = lastRecv.current;                      // nothing is owed yet
        setError(null); setStatus("live");
        setEngines(m.engines ?? []);
        if (exerciseRef.current) sock.send(JSON.stringify({ type: "cmd", name: "exercise", value: exerciseRef.current }));
        const queued = safetyQueue.current.splice(0);             // ack / escalate typed while the link was down
        for (const q of queued) if (performance.now() - q.at < SAFETY_MAX_AGE_MS) sock.send(JSON.stringify({ type: "cmd", name: q.name }));
        startUiTick();
        startActive();
      } else if (m.type === "t") {
        const now = performance.now();
        if (m.ack === p.seq || m.error) p.inflight = false;       // an error reply also frees the pump
        if (m.ack === p.seq) {
          const rtt = now - p.sentAt;
          p.q = nextQuality(p.q, rtt);
          if (p.lastAck) p.fps = 0.8 * p.fps + 0.2 * (1000 / Math.max(now - p.lastAck, 1));
          p.lastAck = now;
          p.rtt = rtt;
          if (m.infer_ms !== undefined) p.infer = p.infer ? 0.8 * p.infer + 0.2 * m.infer_ms : m.infer_ms;   // smoothed model time
        }
        if (m.frame) lastFrame.current = m.frame;
        // the server's pose (server inference) is drawn straight onto the canvas, like the on-device one: never through state
        if (inferenceRef.current === "server" && m.pose !== undefined) poseListener.current?.(m.pose as Pose | null, m.frame ?? lastFrame.current);
        if (m.exercise !== undefined) {                           // a bare error reply carries no telemetry
          const urgent = urgentChange(msgRef.current, m);
          msgRef.current = m;
          if (urgent) flushMsg();                                 // otherwise the next UI tick shows it
        } else if (m.error) setError(m.error);
      } else if (m.type === "cmd_result" && !m.ok) {
        if (m.name === "engine") setEngineWanted(null);
        setError(m.error ?? `${m.name} failed`);
      } else if (m.type === "voice") {
        const waiting = voiceWaiter.current;                      // the answer to the utterance the shared voice hook sent
        voiceWaiter.current = null;
        waiting?.({ ok: !!m.ok, text: m.text, intent: m.intent, error: m.error, status: m.status });
      }
    };
    /** The socket is gone: closed by either side, or given up on by the watchdog (a dead tunnel never closes it). */
    const closed = (code: number) => {
      if (ws.current !== sock) return;                             // an older socket, or stop() already took care of it
      stopPump();
      stopLocal();                                                 // restarted by the next hello
      ws.current = null;
      msgRef.current = null; shownMsg.current = null; setMsg(null);   // the reps of a dead session must not look live
      clearPose();
      if (!want.current) return;
      if (code === 4401) {                                         // token missing or wrong: retrying cannot help
        token.current = ""; saveToken("");
        stop();
        setNeedsToken(true);
        setError("The access token is missing or wrong.");
        return;
      }
      if (code === 1013) setError("The server is full or temporarily locked. Retrying…");
      setStatus("reconnecting");
      if (!opened.current && attempts.current === 0) {            // refused straight away: say why, once
        fetch("/api/health").then((r) => r.json()).then((h) => { if (h.remote_sessions === false) setError(HEALTH_HINT); }).catch(() => {});
      }
      opened.current = false;
      timers.current.reconnect = window.setTimeout(connect, reconnectDelay(attempts.current++));
    };
    sock.onclose = (ev) => closed(ev.code);
    sock.onerror = () => sock.close();
    dropRef.current = () => {                                      // the watchdog: no word from the server for too long
      sock.onclose = null; sock.onmessage = null; sock.onerror = null;
      try { sock.close(); } catch { /* already gone */ }
      closed(1006);
    };
    setStatus((s) => (s === "live" ? s : "connecting"));
  }, [startActive]);

  const attach = (s: MediaStream) => {
    streamRef.current = s;
    setStream(s);
    s.getVideoTracks()[0]?.addEventListener("ended", () => {      // unplugged, or the OS took the camera (a call, Control Center)
      if (streamRef.current !== s) return;
      stop();
      setError("The camera was interrupted. Tap Start to resume.");
    });
  };

  const stop = useCallback(() => {
    want.current = false;
    window.clearTimeout(timers.current.reconnect);
    stopPump();
    ws.current?.close(1000);
    ws.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    endVoice();
    releaseWake();
    stopUiTick();
    safetyQueue.current = [];
    msgRef.current = null; shownMsg.current = null;
    clearPose();
    setStream(null); setMsg(null); setStatus("idle");
    shownStats.current = INITIAL_STATS; setStats(INITIAL_STATS);
    setEngines([]); setEngineWanted(null);
    stopLocal();
    setLocal((l) => ({ ...l, fps: 0, ms: 0, status: "idle" }));
  }, []);

  /** `pin` is a token typed on the connect screen. */
  const start = useCallback(async (pin?: string) => {
    if (want.current) return;
    want.current = true;
    if (pin !== undefined && pin.trim()) { token.current = pin.trim(); saveToken(token.current); }
    setError(null); setStatus("starting");
    attempts.current = 0;
    pumpState.current = { seq: 0, inflight: false, sentAt: 0, q: MAX_QUALITY, fps: 0, lastAck: 0, infer: 0, rtt: 0 };
    try {
      // Ask before opening the camera: no point in a permission prompt if the server will refuse us anyway.
      const h = await fetch("/api/health").then((r) => r.json()).catch(() => null);
      if (h?.remote_auth && !token.current) {
        want.current = false;
        setNeedsToken(true); setStatus("idle"); setError("This server needs an access token.");
        return;
      }
      setNeedsToken(false);
      const media = await openMedia(facingRef.current, true);
      if (!want.current) {                                         // stopped while the permission prompt was open
        [media.stream, media.mic].forEach((m) => m?.getTracks().forEach((t) => t.stop()));
        return;
      }
      attach(media.stream);
      voice.begin(media.mic, media.micError);
      void holdWake();
      connect();
    } catch (e) {
      want.current = false;
      setStatus("error");
      setError(e instanceof DOMException && e.name === "NotAllowedError" ? "Camera permission was denied. Allow it in the browser's site settings."
        : e instanceof Error ? e.message : String(e));
    }
  }, [connect, holdWake]);   // eslint-disable-line react-hooks/exhaustive-deps -- attach/stop are stable in effect

  const switchCamera = useCallback(async () => {
    if (switching.current) return;
    switching.current = true;
    const previous = facingRef.current;
    const next: Facing = previous === "user" ? "environment" : "user";
    // Phones usually cannot open a second camera while the first is still held ("Could not start video source"):
    // release the current one FIRST, give the hardware a moment, then open the other.
    const old = streamRef.current;
    streamRef.current = null;                                   // so its tracks ending is not taken for an interruption
    old?.getTracks().forEach((t) => t.stop());
    await new Promise((r) => window.setTimeout(r, 200));
    let failure: unknown = null;
    try {
      for (const [facingTry, exact] of [[next, false], [next, true], [previous, false]] as const) {   // other camera, strictly, then back
        try {
          const { stream: s } = await openMedia(facingTry, false, exact);   // the microphone keeps running
          if (!want.current) { s.getTracks().forEach((t) => t.stop()); return; }    // stopped meanwhile
          facingRef.current = facingTry;
          setFacing(facingTry);
          attach(s);
          if (facingTry !== next) setError(`Could not switch camera (${failure instanceof Error ? failure.message : "no other camera answered"}); back on the ${previous === "user" ? "front" : "rear"} camera.`);
          return;
        } catch (e) {
          failure ??= e;
          await new Promise((r) => window.setTimeout(r, 150));
        }
      }
      // neither camera would open: the session cannot continue without one
      stop();
      setError(`The camera could not be restarted (${failure instanceof Error ? failure.message : failure}). Tap Start to try again.`);
    } finally {
      switching.current = false;
    }
  }, []);   // eslint-disable-line react-hooks/exhaustive-deps

  const send = useCallback((name: string, value?: string) => {
    if (name === "exercise" && value) exerciseRef.current = value;
    const sock = ws.current;
    if (sock?.readyState === WebSocket.OPEN && opened.current) {
      sock.send(JSON.stringify({ type: "cmd", name, value }));
      lastSent.current = performance.now();
    } else if (want.current && SAFETY_COMMANDS.includes(name)) {
      // The link is down: an answer to a fall check must not be lost with it. It is sent as soon as the next connection is up.
      const q = safetyQueue.current;
      if (!q.some((c) => c.name === name)) q.push({ name, at: performance.now() });
      if (q.length > SAFETY_QUEUE_MAX) q.shift();
      setError("No connection. Your answer will be sent as soon as it is back.");
    }
  }, []);

  /** Ask the server for another pose model. It applies between batches and loading can take seconds, so the page keeps showing
   *  the old model (with a spinner) until the server's replies name the new one. The change is server-wide. */
  const setEngine = useCallback((name: string) => {
    setEngineWanted(name);
    pumpState.current.infer = 0;                                 // a different model has a different cost: average afresh
    window.setTimeout(() => setEngineWanted((w) => (w === name ? null : w)), 60000);   // give up waiting after a minute
    send("engine", name);
  }, [send]);
  const engine = msg?.engine ?? null;
  useEffect(() => { if (engine !== null && engine === engineWanted) setEngineWanted(null); }, [engine, engineWanted]);

  /** Choose who computes the pose: this device or the server. Switching while a session runs restarts the right source at once. */
  const setInference = useCallback((mode: InferenceMode) => {
    inferenceRef.current = mode;
    setInferenceState(mode);
    saveInference(mode);
    if (want.current && opened.current) startActive();
  }, [startActive]);

  /** The model badge: "device" = this phone's own MediaPipe; anything else is one of the server's models. */
  const setModel = useCallback((name: string) => {
    if (name === ON_DEVICE) { setInference("device"); return; }
    setInference("server");
    if (name !== engine) setEngine(name);
  }, [setInference, setEngine, engine]);

  useEffect(() => {
    const onVisible = () => {
      if (document.visibilityState !== "visible") return;
      lastRecv.current = lastSent.current = performance.now();      // a hidden tab sends nothing: do not mistake the silence for a dead link
      void holdWake();                                              // the lock is dropped when hidden
    };
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      document.removeEventListener("visibilitychange", onVisible);
      stop();
      landmarkerRef.current?.landmarker.close();                     // the model is released with the page, not with each stop
      landmarkerRef.current = null;
    };
  }, [holdWake, stop]);

  const models = useMemo(() => [ON_DEVICE, ...engines], [engines]);      // what the badge cycles through
  const model = inference === "device" ? ON_DEVICE : engine;
  const modelMs = inference === "device" ? local.ms : stats.inferMs;     // the model's own time per pose
  const modelWanted = inference === "device" && local.status === "loading" ? ON_DEVICE : engineWanted;
  const fps = inference === "device" ? local.fps : stats.fps;

  const telemetry: Telemetry | null = msg
    ? ({ ...msg, fps, error: msg.error ?? null, privacy: undefined } as unknown as Telemetry)
    : null;

  return {
    videoRef, status, error, stream, facing, stats, telemetry, needsToken,
    engine, engines, engineWanted, setEngine,
    inference, local, models, model, modelMs, modelWanted, setModel, setPoseListener,
    voice,
    frame: msg?.frame ?? null,
    start, stop, switchCamera, send,
  };
}

export type RemoteSession = ReturnType<typeof useRemoteSession>;
