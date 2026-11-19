import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Telemetry } from "../useTelemetry";
import {
  MAX_QUALITY, ON_DEVICE, SAFETY_COMMANDS, SAFETY_MAX_AGE_MS, SAFETY_QUEUE_MAX, UI_TICK_MS, bboxFromKeypointsInto, canSendFrame, captureSize,
  TIER_SIZE, connectionLost, createTierMonitor, defaultInference, extractToken, initialTier, isMobileDevice, keypointMessage, nextQuality,
  packFrame, reconnectDelay, tierCeiling, tierConstraints, urgentChange, analyticsLocal, defaultEngineMode,
  type CameraTier, type EngineMode, type InferenceMode, type Keypoints, type Pose, type Quality,
} from "./logic";

const NO_JOINTS: Keypoints = Object.freeze({}) as Keypoints;     // "nobody in view": one shared empty set, never written to
import { TOKEN_KEY, apiUrl, isBundled, wsUrl } from "../net";
import { LocalEngine } from "../engine/localEngine";
import { recordEngine, type SessionRecorder } from "../store";
import { setWorkoutActive, syncQueue } from "../sync";
import { getLandmarker, runLocalLoop, type LoadedLandmarker } from "./onDevicePose";
import { MIC_CONSTRAINTS, micErrorText } from "./voice";
import { useVoiceInput, type VoiceReply } from "./useVoiceInput";

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

const FIRST_ANSWER_GRACE_MS = 25_000;
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

const ENGINE_KEY = "engine";
/** Who runs the analytics (reps, falls): this phone ("local", works offline) or the server. Default: local in the bundled APK. */
function loadEngineMode(): EngineMode {
  try {
    const saved = localStorage.getItem(ENGINE_KEY);
    if (saved === "local" || saved === "server") return saved;
  } catch { /* no storage */ }
  return defaultEngineMode(isBundled());
}
/** The server measures joint angles in 2D when backend/.env has ANGLES_3D=0 (what this project runs with: the phone's 3D world
 *  landmarks counted worse). The offline engine cannot read that file, so it follows the same choice: 2D unless
 *  localStorage "angles3d"="1" or the build has VITE_ANGLES_3D=1. */
function localEngineOptions(): { angles3d: boolean } {
  let v: string | null = null;
  try { v = localStorage.getItem("angles3d"); } catch { /* no storage */ }
  return { angles3d: (v ?? import.meta.env.VITE_ANGLES_3D ?? "0") === "1" };
}
// With the analytics on this phone, these commands also go to the server (when linked) so its session state matches: voice "pause"
// and "let's go" are interpreted there. The fall answers and pause/resume are local only.
const LOCAL_ONLY = ["ack_fall", "escalate", "test_fall", "pause", "resume"];

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
async function openMedia(facing: Facing, audio: boolean, exact = false, tier: CameraTier = "mid"): Promise<{ stream: MediaStream; mic: MediaStream | null; micError: string | null }> {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error(window.isSecureContext ? "This browser has no camera API." : "The camera needs HTTPS (or localhost). Open this page over https://.");
  }
  // `ideal` first: a laptop with one camera still starts when "rear" is asked for. `exact` is the retry for phones that
  // ignore `ideal` and hand back the same camera.
  // The size comes from the camera tier (720p on a computer, 640x480 on a phone, lower when the phone struggles), not a fixed 1280x720.
  const video = { facingMode: exact ? { exact: facing } : { ideal: facing }, ...tierConstraints(tier) };
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
  // Analytics on this phone (frontend/src/engine): the engine lives here while a local session runs; null = the server does it.
  const [engineMode, setEngineModeState] = useState<EngineMode>(loadEngineMode);
  const engineModeRef = useRef(engineMode);
  const localRef = useRef<LocalEngine | null>(null);
  const seenVersion = useRef(-1);                          // the engine's change counter at the last telemetry push
  const recorderRef = useRef<SessionRecorder | null>(null);   // writes this session's events to the device (store/), never throws
  /** The local session is over (stop, or the analytics moved to the server): one last write with the end time. */
  const endRecording = () => {
    const r = recorderRef.current;
    recorderRef.current = null;
    if (localRef.current) localRef.current.listener = null;
    setWorkoutActive(false);                                // uploads may run again
    void r?.end().then(() => syncQueue().trigger());       // the finished session is saved: send it as soon as the server can be reached
  };
  // Can the server be reached? Only matters with the analytics on this phone: then a dead link does not stop the workout, but nobody
  // can be told about a fall. null = not known yet (the first few seconds of a session).
  const [linkUp, setLinkUp] = useState<boolean | null>(null);
  const [local, setLocal] = useState({ fps: 0, ms: 0, status: "idle" as "idle" | "loading" | "running", delegate: "" });
  // Adaptive camera resolution: the tier the camera runs at, and the monitor that steps it down / up from the model's measured time.
  const mobile = useRef(isMobileDevice(navigator.userAgent, navigator.maxTouchPoints ?? 0)).current;
  const tierRef = useRef<CameraTier>(initialTier(mobile, navigator.hardwareConcurrency,
    (navigator as Navigator & { deviceMemory?: number }).deviceMemory));
  const [cameraTier, setCameraTier] = useState<CameraTier>(tierRef.current);
  const monitorRef = useRef<ReturnType<typeof createTierMonitor> | null>(null);
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
  const timers = useRef<{ reconnect?: number; pump?: number; ui?: number; link?: number }>({});
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
  /** Local analytics: build the telemetry (allocates, so only on a UI tick or when something important changed) and show it. */
  const pushLocal = () => {
    const e = localRef.current;
    if (!e) return;
    seenVersion.current = e.version;
    msgRef.current = e.telemetry() as unknown as ServerMsg;
    flushMsg();
  };

  /** Every UI_TICK_MS while a session is wanted: bring React up to date, and check the tunnel. A quick tunnel can die without the
   *  socket ever noticing, so the socket still reads OPEN while nothing is counted or watched: that is detected here. */
  const uiTick = () => {
    if (localRef.current) { msgRef.current = localRef.current.telemetry() as unknown as ServerMsg; flushMsg(); flushStats(); return; }   // nothing to watch: no server involved
    flushMsg();
    flushStats();
    if (document.hidden || !opened.current) return;      // a hidden tab sends nothing, so nothing can be missing
    // Before the first answer of this socket the server may be loading the pose model for this new session (seconds, on CPU):
    // that is not a dead link, so the first answer gets a long grace. Dropping then made each reconnect load the model again.
    const grace = pumpState.current.lastAck ? undefined : FIRST_ANSWER_GRACE_MS;
    if (connectionLost(performance.now(), lastRecv.current, lastSent.current, grace)) dropRef.current?.();
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

  /** Changes the size of the RUNNING camera track (no reopening: reopening a phone camera is what failed on camera switches). The
   *  frames carry their own width and height, the overlay maps every pose by them and the preview letterboxes (object-contain), so
   *  nothing else needs to know. If the camera refuses, it keeps its current size. */
  const applyTier = useCallback(async (tier: CameraTier) => {
    tierRef.current = tier;
    setCameraTier(tier);
    const track = streamRef.current?.getVideoTracks()[0];
    if (!track) return;
    try { await track.applyConstraints(tierConstraints(tier)); } catch { /* keeps the current size */ }
  }, []);

  /** On-device inference: load the model (once), then detect on every new camera frame and send the joints. If it cannot run
   *  (no WebAssembly, offline and not cached, model missing, the model throws), the server takes over with frames. */
  const startLocal = useCallback(async () => {
    stopLocal();
    const fallBack = (why: unknown) => {
      if (!want.current) return;
      setError(`On-device pose is not available (${why instanceof Error ? why.message : String(why)}); the server is doing it instead.`);
      inferenceRef.current = "server"; setInferenceState("server");     // not saved: the next visit tries on-device again
      endRecording();
      localRef.current = null;                                          // the analytics need the on-device pose: the server takes them over too
      setLocal((l) => ({ ...l, status: "idle" }));
      startPump();
    };
    setLocal((l) => ({ ...l, status: "loading" }));
    try {
      if (!landmarkerRef.current) landmarkerRef.current = await getLandmarker();      // already loaded and warmed up by preloadLandmarker()
    } catch (e) {
      fallBack(e);
      return;
    }
    if (!want.current || inferenceRef.current !== "device" || (!opened.current && !localRef.current)) {     // things changed while the model loaded
      setLocal((l) => ({ ...l, status: "idle" }));
      return;
    }
    const { landmarker, delegate } = landmarkerRef.current;
    setLocal((l) => ({ ...l, status: "running", delegate }));
    // Created once per loop, overwritten in place by every frame (the per-frame path must not allocate: see the cv-fitness-pipeline
    // skill). The listener draws from them synchronously and keeps no copy; the message is serialised before the callback returns.
    const bbox: [number, number, number, number] = [0, 0, 0, 0];
    const shown: Pose = { kp: NO_JOINTS, bbox };
    const frame = { w: 0, h: 0 };
    loopRef.current = runLocalLoop(() => videoRef.current, landmarker, {
      pose: (kp, w, h, tMs, world) => {
        frame.w = w; frame.h = h;
        if (kp && bboxFromKeypointsInto(bbox, kp)) { shown.kp = kp; poseListener.current?.(shown, frame); }   // the skeleton, with no network in between
        else poseListener.current?.(null, frame);
        const eng = localRef.current;
        if (eng) {                                                    // the analytics run right here: no message, no network
          eng.process(kp, world, w, h, tMs / 1000);                   // allocation-free; the same pose the server would get
          if (eng.version !== seenVersion.current) pushLocal();        // a rep, a fall change, standby <-> counting: shown at once
          return;
        }
        const sock = ws.current;
        if (sock && sock.readyState === WebSocket.OPEN && opened.current && sock.bufferedAmount < 65536) {   // never build a backlog
          sock.send(keypointMessage(++kpSeq.current, tMs, w, h, kp ?? NO_JOINTS, kp ? world : undefined));
          lastSent.current = performance.now();
        }
      },
      stats: (fps, ms) => {
        setLocal((l) => ({ ...l, fps, ms }));
        if (localRef.current) localRef.current.latencyMs = ms;
        const next = monitorRef.current?.sample(fps, ms);            // over budget for 3 s: a smaller camera image; 20 s of headroom: back up
        if (next) void applyTier(next);
      },
      failed: fallBack,
    });
  }, [startPump, applyTier]);

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
    if (localRef.current && !navigator.onLine) {              // local analytics and no network: do not even try, check again shortly
      setLinkUp(false);
      timers.current.reconnect = window.setTimeout(connect, 3000);
      return;
    }
    const sock = new WebSocket(wsUrl("/ws/session"));
    sock.binaryType = "arraybuffer";
    ws.current = sock;
    // The token goes in the first message, not the address, so it never shows up in server or tunnel logs.
    sock.onopen = () => { if (token.current) sock.send(JSON.stringify({ type: "auth", token: token.current })); };
    sock.onmessage = (e) => {
      lastRecv.current = performance.now();                       // any message proves the link is alive
      if (typeof e.data !== "string") return;
      const m = JSON.parse(e.data) as ServerMsg;
      const p = pumpState.current;
      if (m.type === "hello" && localRef.current) {
        // The analytics run on this phone; the link is only for voice commands (the server interprets them). Mirror the state the
        // server must know, so its "pause" / "let's go" make sense, and change nothing else.
        opened.current = true; attempts.current = 0;
        window.clearTimeout(timers.current.link); setLinkUp(true);
        setEngines(m.engines ?? []);
        if (exerciseRef.current) sock.send(JSON.stringify({ type: "cmd", name: "exercise", value: exerciseRef.current }));
        if (localRef.current.pipeline.counting) sock.send(JSON.stringify({ type: "cmd", name: "start_session" }));
      } else if (m.type === "hello") {
        opened.current = true; attempts.current = 0;
        lastSent.current = lastRecv.current;                      // nothing is owed yet
        setError(null); setStatus("live");
        setEngines(m.engines ?? []);
        if (exerciseRef.current) sock.send(JSON.stringify({ type: "cmd", name: "exercise", value: exerciseRef.current }));
        const queued = safetyQueue.current.splice(0);             // ack / escalate typed while the link was down
        for (const q of queued) if (performance.now() - q.at < SAFETY_MAX_AGE_MS) sock.send(JSON.stringify({ type: "cmd", name: q.name }));
        startUiTick();
        startActive();
      } else if (m.type === "t" && localRef.current) {
        // not expected (no poses are sent while the analytics are local); never let a server reply replace the local numbers
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
      } else if (m.type === "cmd_result" && !m.ok && !localRef.current) {
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
      if (localRef.current) {                                      // the analytics run here: only the voice link is down, the session goes on
        ws.current = null; opened.current = false;
        setLinkUp(false);
        if (code === 4401) {                                       // the token is missing or wrong: retrying cannot help; the next start asks for it
          token.current = ""; saveToken("");
          setError("Voice link: the access token is missing or wrong.");
          return;
        }
        if (!want.current) return;
        timers.current.reconnect = window.setTimeout(connect, reconnectDelay(attempts.current++));
        return;
      }
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
      if (code === 1008) setError("The server refused the connection (remote sessions off, or this address is not allowed).");
      setStatus("reconnecting");
      if (!opened.current && (attempts.current === 0 || attempts.current === 3)) {   // refused straight away, or still failing: say why
        fetch(apiUrl("/api/health")).then((r) => r.json()).then((h) => {
          if (h.remote_sessions === false) setError(HEALTH_HINT);
          else if (h.remote_auth && !token.current && want.current) {   // the server wants a token and we have none: ask, do not loop
            stop(); setNeedsToken(true); setError("This server needs an access token.");
          }
        }).catch(() => {});
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
    const old = ws.current;
    ws.current = null;                                      // first: its onclose then knows it is stale
    if (old) { old.onclose = null; old.onerror = null; old.onmessage = null; try { old.close(1000); } catch { /* already gone */ } }
    dropRef.current = null; opened.current = false;         // nothing of this session may reach the next one
    lastRecv.current = lastSent.current = performance.now();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    endVoice();
    releaseWake();
    stopUiTick();
    safetyQueue.current = [];
    window.clearTimeout(timers.current.link); setLinkUp(null);
    endRecording();                                         // before the engine goes: the last write reads its final counts
    localRef.current = null; seenVersion.current = -1;     // the engine (and its counts) goes with the session
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
    // "This phone" means standalone: the pose model must run here too (a desktop browser defaults to the server's), or the server
    // would still be the counter and a dead link would show "Connection lost". Not saved: the badge choice is left as it was.
    if (engineModeRef.current === "local" && inferenceRef.current !== "device") { inferenceRef.current = "device"; setInferenceState("device"); }
    const onPhone = analyticsLocal(engineModeRef.current, inferenceRef.current);     // the reps and falls are worked out on this phone
    try {
      // Ask before opening the camera: no point in a permission prompt if the server will refuse us anyway. Not needed when the
      // analytics are local: the session must start with no network at all (a dead tunnel would otherwise delay it).
      // First run with no token saved: ask the server even when the analytics are local (2.5 s cap, so a dead tunnel or no network
      // still starts the session), or the first-time token prompt never appears and the voice link is refused (4401) silently.
      const askServer = !onPhone || (!token.current && navigator.onLine);
      const h = askServer
        ? await fetch(apiUrl("/api/health"), onPhone ? { signal: AbortSignal.timeout(2500) } : undefined).then((r) => r.json()).catch(() => null)
        : null;
      if (h?.remote_auth && !token.current) {
        want.current = false;
        setNeedsToken(true); setStatus("idle"); setError("This server needs an access token.");
        return;
      }
      setNeedsToken(false);
      monitorRef.current = createTierMonitor(tierRef.current, tierCeiling(mobile));    // a tier an earlier session settled on carries over
      const media = await openMedia(facingRef.current, true, false, tierRef.current);
      if (!want.current) {                                         // stopped while the permission prompt was open
        [media.stream, media.mic].forEach((m) => m?.getTracks().forEach((t) => t.stop()));
        return;
      }
      attach(media.stream);
      voice.begin(media.mic, media.micError);
      void holdWake();
      if (onPhone) {
        localRef.current = new LocalEngine(exerciseRef.current ?? undefined, localEngineOptions());
        recorderRef.current = recordEngine(localRef.current);     // keeps the session on this device (IndexedDB)
        setWorkoutActive(true);                                    // no uploads while the pose loop runs
        setLinkUp(null);
        window.clearTimeout(timers.current.link);
        timers.current.link = window.setTimeout(() => setLinkUp((l) => (l === null ? false : l)), 4000);   // no answer in 4 s: unreachable
        seenVersion.current = -1;
        setStatus("live");
        startUiTick();
        pushLocal();
        startActive();             // the model loop, which feeds the engine
        connect();                 // optional: voice commands (and later the sync) need the server; counting does not
      } else connect();
    } catch (e) {
      want.current = false;
      setStatus("error");
      setError(e instanceof DOMException && e.name === "NotAllowedError" ? "Camera permission was denied. Allow it in the browser's site settings."
        : e instanceof Error ? e.message : String(e));
    }
  }, [connect, holdWake, startActive]);   // eslint-disable-line react-hooks/exhaustive-deps -- attach/stop are stable in effect

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
          const { stream: s } = await openMedia(facingTry, false, exact, tierRef.current);   // the microphone keeps running
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
    const eng = localRef.current;
    if (eng) {                                                    // the analytics are on this phone: apply at once, offline too
      switch (name) {
        case "exercise": if (value) eng.setExercise(value); break;
        case "reset": eng.reset(); break;
        case "ack_fall": eng.ack(); break;
        case "escalate": eng.escalate(); break;
        case "test_fall": eng.simulateFall(); break;
        case "start_session": eng.start(); break;
        case "stop_session": eng.stop(); break;
        case "pause": eng.pause(); break;
        case "resume": eng.resume(); break;
      }
      pushLocal();
      if (!LOCAL_ONLY.includes(name) && sock?.readyState === WebSocket.OPEN && opened.current) sock.send(JSON.stringify({ type: "cmd", name, value }));
      return;
    }
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
    if (mode === "server" && localRef.current) {                    // frames go to the server, so it must do the analytics too
      endRecording();
      localRef.current = null; seenVersion.current = -1;
      msgRef.current = null; shownMsg.current = null; setMsg(null);
    }
    if (want.current && opened.current) startActive();
  }, [startActive]);

  /** Who runs the analytics from the next session on (the choice is remembered). */
  const setEngineMode = useCallback((m: EngineMode) => {
    engineModeRef.current = m;
    setEngineModeState(m);
    try { localStorage.setItem(ENGINE_KEY, m); } catch { /* storage blocked */ }
  }, []);

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
      landmarkerRef.current = null;       // the shared model (onDevicePose.getLandmarker) lives with the page: a StrictMode re-mount reuses it
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
    engineMode, setEngineMode, analyticsLocal: analyticsLocal(engineMode, inference), linkUp,
    voice,
    frame: msg?.frame ?? null,
    camera: { tier: cameraTier, ...TIER_SIZE[cameraTier] },       // what the camera was last asked for (the browser picks its nearest mode)
    start, stop, switchCamera, send,
  };
}

export type RemoteSession = ReturnType<typeof useRemoteSession>;
