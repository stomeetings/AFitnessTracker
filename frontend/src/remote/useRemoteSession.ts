import { useCallback, useEffect, useRef, useState } from "react";
import type { Telemetry } from "../useTelemetry";
import { MAX_QUALITY, canSendFrame, captureSize, extractToken, nextQuality, packFrame, reconnectDelay, type Pose, type Quality } from "./logic";
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
  const [msg, setMsg] = useState<ServerMsg | null>(null);
  const [needsToken, setNeedsToken] = useState(false);   // the server wants an access token we do not have (or it was wrong)
  const token = useRef<string | null>(null);
  if (token.current === null) token.current = loadToken();
  const [stats, setStats] = useState({ fps: 0, rtt: 0, longSide: MAX_QUALITY.longSide, inferMs: 0 });
  const [engines, setEngines] = useState<string[]>([]);              // pose models the server offers to switch between
  const [engineWanted, setEngineWanted] = useState<string | null>(null);   // asked for, not yet confirmed by the server

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
  const timers = useRef<{ reconnect?: number; pump?: number }>({});
  const attempts = useRef(0);
  const wake = useRef<WakeLockSentinel | null>(null);
  const pumpState = useRef({ seq: 0, inflight: false, sentAt: 0, q: MAX_QUALITY as Quality, fps: 0, lastAck: 0, infer: 0 });

  const stopPump = () => { window.clearTimeout(timers.current.pump); timers.current.pump = undefined; };

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
        if (blob && sock.readyState === WebSocket.OPEN) sock.send(packFrame(seq, now, await blob.arrayBuffer()));
        else p.inflight = false;
      }
      timers.current.pump = window.setTimeout(tick, 20);
    };
    tick();
  }, []);

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
      if (typeof e.data !== "string") return;
      const m = JSON.parse(e.data) as ServerMsg;
      const p = pumpState.current;
      if (m.type === "hello") {
        opened.current = true; attempts.current = 0;
        setError(null); setStatus("live");
        setEngines(m.engines ?? []);
        if (exerciseRef.current) sock.send(JSON.stringify({ type: "cmd", name: "exercise", value: exerciseRef.current }));
        startPump();
      } else if (m.type === "t") {
        const now = performance.now();
        if (m.ack === p.seq || m.error) p.inflight = false;       // an error reply also frees the pump
        if (m.ack === p.seq) {
          const rtt = now - p.sentAt;
          p.q = nextQuality(p.q, rtt);
          if (p.lastAck) p.fps = 0.8 * p.fps + 0.2 * (1000 / Math.max(now - p.lastAck, 1));
          p.lastAck = now;
          if (m.infer_ms !== undefined) p.infer = p.infer ? 0.8 * p.infer + 0.2 * m.infer_ms : m.infer_ms;   // smoothed model time
          setStats({ fps: Math.round(p.fps * 10) / 10, rtt: Math.round(rtt), longSide: p.q.longSide, inferMs: Math.round(p.infer * 10) / 10 });
        }
        if (m.exercise !== undefined) setMsg(m);                  // a bare error reply carries no telemetry
        else if (m.error) setError(m.error);
      } else if (m.type === "cmd_result" && !m.ok) {
        if (m.name === "engine") setEngineWanted(null);
        setError(m.error ?? `${m.name} failed`);
      } else if (m.type === "voice") {
        const waiting = voiceWaiter.current;                      // the answer to the utterance the shared voice hook sent
        voiceWaiter.current = null;
        waiting?.({ ok: !!m.ok, text: m.text, intent: m.intent, error: m.error, status: m.status });
      }
    };
    sock.onclose = (ev) => {
      stopPump();
      if (ws.current === sock) ws.current = null;
      if (!want.current) return;
      if (ev.code === 4401) {                                      // token missing or wrong: retrying cannot help
        token.current = ""; saveToken("");
        stop();
        setNeedsToken(true);
        setError("The access token is missing or wrong.");
        return;
      }
      if (ev.code === 1013) setError("The server is full or temporarily locked. Retrying…");
      setStatus("reconnecting");
      if (!opened.current && attempts.current === 0) {            // refused straight away: say why, once
        fetch("/api/health").then((r) => r.json()).then((h) => { if (h.remote_sessions === false) setError(HEALTH_HINT); }).catch(() => {});
      }
      timers.current.reconnect = window.setTimeout(connect, reconnectDelay(attempts.current++));
    };
    sock.onerror = () => sock.close();
    setStatus((s) => (s === "live" ? s : "connecting"));
  }, [startPump]);

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
    setStream(null); setMsg(null); setStatus("idle");
    setStats({ fps: 0, rtt: 0, longSide: MAX_QUALITY.longSide, inferMs: 0 });
    setEngines([]); setEngineWanted(null);
  }, []);

  /** `pin` is a token typed on the connect screen. */
  const start = useCallback(async (pin?: string) => {
    if (want.current) return;
    want.current = true;
    if (pin !== undefined && pin.trim()) { token.current = pin.trim(); saveToken(token.current); }
    setError(null); setStatus("starting");
    attempts.current = 0;
    pumpState.current = { seq: 0, inflight: false, sentAt: 0, q: MAX_QUALITY, fps: 0, lastAck: 0, infer: 0 };
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
    if (ws.current?.readyState === WebSocket.OPEN) ws.current.send(JSON.stringify({ type: "cmd", name, value }));
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

  useEffect(() => {
    const onVisible = () => { if (document.visibilityState === "visible") void holdWake(); };   // the lock is dropped when hidden
    document.addEventListener("visibilitychange", onVisible);
    return () => { document.removeEventListener("visibilitychange", onVisible); stop(); };
  }, [holdWake, stop]);

  const telemetry: Telemetry | null = msg
    ? ({ ...msg, fps: stats.fps, error: msg.error ?? null, privacy: undefined } as unknown as Telemetry)
    : null;

  return {
    videoRef, status, error, stream, facing, stats, telemetry, needsToken,
    engine, engines, engineWanted, setEngine,
    voice,
    pose: msg?.pose ?? null, frame: msg?.frame ?? null,
    start, stop, switchCamera, send,
  };
}

export type RemoteSession = ReturnType<typeof useRemoteSession>;
