import { useEffect, useRef, useState } from "react";
import { Activity, AlertTriangle, Camera, Cpu, Footprints, Gauge, Info, Loader2, Mic, RefreshCw, RotateCcw, ShieldAlert, Timer } from "lucide-react";
import { ExerciseGuide } from "./ExerciseGuide";
import { FallCheck } from "./FallCheck";
import { RemotePanel, RemoteView } from "./remote/RemoteView";
import { calibrationNote, isLocalHost, nextEngine } from "./remote/logic";
import { uploadToServer, useVoiceInput } from "./remote/useVoiceInput";
import { useRemoteSession } from "./remote/useRemoteSession";
import { Hud } from "./remote/Hud";
import { useFullscreen } from "./remote/useFullscreen";
import { VOICE_HELP, VoiceChip } from "./remote/VoiceChip";
import { intentEffects } from "./remote/voice";
import { cue, runCountdown, unlockSpeech } from "./cues";
import { preloadLandmarker } from "./remote/onDevicePose";
import { useTelemetry, type Limb } from "./useTelemetry";

const ENGINES = [
  { id: "mediapipe", label: "MediaPipe Full" },
  { id: "mediapipe_lite", label: "MediaPipe Lite" },
  { id: "yolo", label: "YOLOv11n-Pose GPU" },
] as const;
type CamInfo = { index: number; status: "active" | "ok" | "unavailable"; backend?: string; width?: number; height?: number; reason?: string; name?: string; label?: string };
type CamScan = { active: number; available: number[]; cameras: CamInfo[]; devices: string[]; error: string | null };
const SIDE_LABEL: Record<string, string> = { L: "Left", R: "Right", B: "Bilateral" };

const post = (url: string, body?: unknown) =>
  fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: body ? JSON.stringify(body) : undefined });

function Card({ icon, label, value, sub, accent }: { icon: React.ReactNode; label: string; value: React.ReactNode; sub?: React.ReactNode; accent?: string }) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
      <div className="flex items-center gap-2 text-xs uppercase tracking-wide text-slate-400">{icon}{label}</div>
      <div className={`mt-2 text-2xl font-semibold ${accent ?? ""}`}>{value}</div>
      {sub && <div className="mt-1 text-xs text-slate-500">{sub}</div>}
    </div>
  );
}

function LimbCard({ side, limb }: { side: string; limb: Limb }) {
  const angle = limb.angle ?? 0;
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
      <div className="flex items-center justify-between text-xs uppercase tracking-wide text-slate-400">
        <span>{SIDE_LABEL[side] ?? side}</span>
        <span className={`rounded px-1.5 py-0.5 ${limb.state === "IDLE" ? "bg-slate-800" : "bg-emerald-900 text-emerald-300"}`}>{limb.state}</span>
      </div>
      <div className="mt-2 flex items-end justify-between">
        <div className="text-4xl font-bold">{limb.reps}<span className="ml-1 text-sm font-normal text-slate-400">reps</span></div>
        <div className="text-right">
          <div className="text-xl font-semibold tabular-nums">{limb.angle === null ? "–" : `${angle.toFixed(0)}°`}</div>
          <div className="text-[10px] uppercase text-slate-500">joint angle</div>
        </div>
      </div>
      <div className="mt-2 h-1.5 w-full rounded bg-slate-800">
        <div className="h-1.5 origin-left rounded bg-emerald-500 transition-transform" style={{ transform: `scaleX(${Math.min(100, Math.max(0, angle / 1.8)) / 100})` }} />
      </div>
      <div className="mt-2 flex gap-2 text-xs">
        <span className="rounded-full border border-slate-700 px-2 py-0.5">ROM {limb.last_rom.toFixed(0)}°</span>
        <span className="rounded-full border border-emerald-800 px-2 py-0.5 text-emerald-300">best {limb.best_rom.toFixed(0)}°</span>
      </div>
    </div>
  );
}

function CameraPicker({ active, backend, error }: { active: number | undefined; backend: string | null | undefined; error: string | null | undefined }) {
  const [scan, setScan] = useState<CamScan | null>(null);
  const [scanning, setScanning] = useState(false);
  const [wanted, setWanted] = useState<number | null>(null);
  const rescan = () => {
    setScanning(true);
    fetch("/api/cameras").then((r) => r.json()).then(setScan).catch(() => {}).finally(() => setScanning(false));
  };
  useEffect(rescan, []);
  useEffect(() => { if (wanted !== null && (active === wanted || error)) setWanted(null); }, [active, error, wanted]);  // switch finished, fell back, or failed
  const switching = wanted !== null;
  // only real devices: the active camera, plus anything the scan found working or Windows lists by name
  const indices = [...(active !== undefined ? [active] : []), ...(scan?.cameras.filter((c) => c.status !== "unavailable" || c.name).map((c) => c.index) ?? [])]
    .filter((v, i, a) => a.indexOf(v) === i).sort((a, b) => a - b);
  const info = (i: number) => scan?.cameras.find((c) => c.index === i);
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
      <div className="mb-2 flex items-center gap-2 text-xs uppercase tracking-wide text-slate-400">
        <Camera size={14} /> Camera source
        {switching && <Loader2 size={12} className="animate-spin text-emerald-400" />}
        <button onClick={rescan} disabled={scanning} title="Probe camera indices 0–4 again" className="ml-auto rounded p-1 hover:bg-slate-800 disabled:opacity-50">
          <RefreshCw size={12} className={scanning ? "animate-spin" : ""} />
        </button>
      </div>
      <div className="flex flex-wrap gap-1 rounded-lg bg-slate-800 p-1">
        {indices.map((i) => {
          const isActive = active === i;
          const st = info(i)?.status;
          const dead = scan !== null && (st === "unavailable" || st === undefined);
          return (
            <button key={i} disabled={switching} onClick={() => { setWanted(i); post("/api/camera", { index: i }); }}
              title={[info(i)?.name, info(i)?.reason ?? (info(i) ? `${info(i)!.backend} ${info(i)!.width}x${info(i)!.height}` : `index ${i}`)].filter(Boolean).join(" · ")}
              className={`flex-1 rounded-md px-2 py-1.5 text-xs transition disabled:opacity-60 ${isActive ? "bg-emerald-500 text-slate-950" : wanted === i ? "bg-slate-700 text-slate-200" : dead ? "text-slate-500 hover:bg-slate-700" : "text-slate-300 hover:bg-slate-700"}`}>
              {info(i)?.label ?? `Cam ${i}`} <span className="opacity-60">({i})</span>
              {!dead && scan && !isActive && <span className="ml-1 inline-block h-1.5 w-1.5 rounded-full bg-emerald-400 align-middle" />}
            </button>
          );
        })}
      </div>
      <div className={`mt-2 text-xs ${error ? "text-amber-400" : "text-slate-500"}`}>
        {switching ? `switching to camera ${wanted}…` : error ?? `active: camera ${active ?? "…"}${backend ? ` · ${backend}` : ""}`}
      </div>
      {scan && scan.devices.length > 0 && (
        <div className="mt-1 text-[10px] text-slate-600" title="DirectShow device names, in camera index order">
          devices: {scan.devices.join(" · ")}
        </div>
      )}
    </div>
  );
}

export default function App() {
  // Video source: the server's own camera (default, unchanged) or this device's camera streamed to /ws/session.
  // The laptop itself starts in "Server camera"; any other address (a phone, a tunnel) starts in "This device".
  const [mode, setMode] = useState<"local" | "device">(() => {
    try { const saved = localStorage.getItem("mode"); if (saved === "device" || saved === "local") return saved; } catch { /* no storage */ }
    return isLocalHost(location.hostname) ? "local" : "device";
  });
  const device = mode === "device";
  const chooseMode = (m: "local" | "device") => {
    setMode(m);
    try { localStorage.setItem("mode", m); } catch { /* storage blocked: session-only */ }
    if (m === "local") remote.stop();
    else localVoice.end();                  // the desktop page's microphone is for Server camera mode only
  };
  const remote = useRemoteSession();
  const local = useTelemetry(!device);
  const t = device ? remote.telemetry : local.t;
  const online = device ? remote.status === "live" : local.online;
  // The same user actions, routed to whichever source is showing.
  const act = {
    exercise: (v: string) => (device ? remote.send("exercise", v) : void post("/api/config", { exercise: v })),
    reset: () => (device ? remote.send("reset") : void post("/api/reset")),
    ack: () => (device ? remote.send("ack_fall") : void post("/api/fall/ack")),
    escalate: () => (device ? remote.send("escalate") : void post("/api/fall/escalate")),
    testFall: () => (device ? remote.send("test_fall") : void post("/api/fall/test")),
  };
  const [exercises, setExercises] = useState<string[]>([]);
  useEffect(() => { fetch("/api/exercises").then((r) => r.json()).then(setExercises).catch(() => {}); }, []);
  const [guideInfo, setGuideInfo] = useState<Record<string, { rest: number; peak: number; sides?: string[] }>>({});
  useEffect(() => { fetch("/api/exercise_info").then((r) => r.json()).then(setGuideInfo).catch(() => {}); }, []);
  const [guideOpen, setGuideOpen] = useState(false);
  const guide = t ? guideInfo[t.exercise] : undefined;

  // Standby / active: a phone session starts in standby (skeleton live, reps NOT counted) so stepping into the frame cannot
  // score; "let's go" / "start my session" / the Start button reset the counts to 0 and start counting, and close the guide.
  // Pause (or "stop session" on the desktop) goes back to standby. The voice commands do the same on the server (voicecmd.py).
  // "Ready... 3 2 1 Go!" overlay + tones (cues.ts): timers only, never touches the frame loop. Restarting replaces a running one.
  const [count, setCount] = useState<string | null>(null);
  const [paused, setPaused] = useState(false);     // paused by voice ("pause" / "hold please"): the server keeps the counts
  const stopCount = useRef<(() => void) | null>(null);
  const countdown = () => {
    stopCount.current?.();
    stopCount.current = runCountdown((s) => { setCount(s); if (s === null) stopCount.current = null; });
  };
  useEffect(() => { const off = unlockSpeech(); return () => { off(); stopCount.current?.(); }; }, []);
  // Preview (the Exercise Guide) opened/closed by the user (button or voice): says so. This is a handler, NOT an effect on
  // `guideOpen`: an effect also runs on mount (twice under React StrictMode) and when a session start or end closes the guide,
  // which spoke "Closing preview" when the app opened. Only explicit user actions call it.
  const guideNow = useRef(false);
  guideNow.current = guideOpen;
  const showGuide = (open: boolean) => {
    if (open === guideNow.current) return;                // nothing changes: nothing is said
    guideNow.current = open;
    setGuideOpen(open);
    cue(open ? "previewOpen" : "previewClose");
  };
  // Start the on-device pose model loading as soon as this device is the camera: the Start button then finds it ready.
  // (Not in Server camera mode: that visitor never needs the ~10 MB of WASM and model.)
  useEffect(() => { if (device) preloadLandmarker(); }, [device]);
  const startSession = () => { setPaused(false); countdown(); setGuideOpen(false); if (device) remote.send("start_session"); else void post("/api/session/start"); };
  const pauseSession = () => { if (device) remote.send("stop_session"); else void post("/api/session/stop"); };
  const fs = useFullscreen();               // the video wrapper goes full screen with its HUD; modal dialogs still sit on top

  const fall = t?.fall?.status ?? "NORMAL";
  const fallAlert = fall !== "NORMAL";
  const alert = t?.fall?.alert;

  // Privacy mode: the preference lives here (saved in the browser), the backend applies it. Re-sent whenever the
  // backend disagrees (after a change, a backend restart or a reconnect), not on every telemetry tick.
  const [privacyPref, setPrivacyPref] = useState(() => { try { return localStorage.getItem("privacy") === "1"; } catch { return false; } });
  const applyPrivacy = (v: boolean) => {
    setPrivacyPref(v);
    try { localStorage.setItem("privacy", v ? "1" : "0"); } catch { /* storage blocked: session-only */ }
  };
  const togglePrivacy = () => applyPrivacy(!privacyPref);

  // ---- Voice commands: ONE path for both modes ----------------------------------------------------------------
  // Microphone and upload differ (a phone's session socket, the desktop page's POST /api/voice); what a command MEANS
  // is decided by the same server code (app/voicecmd.py) and what it CHANGES here is decided by `intentEffects` only.
  const fallOpen = alert?.state === "COUNTDOWN" || alert?.state === "ALERTED";
  const localVoice = useVoiceInput(uploadToServer);
  const voice = device ? remote.voice : localVoice;
  // "Preview" = the Exercise Guide modal (the Info button's): show / display preview opens it, hide / close preview closes it.
  const heardSeq = voice.heard.seq, heardIntent = voice.heard.intent;
  const commandTimer = useRef<number | undefined>(undefined);
  useEffect(() => {
    if (!heardSeq) return;
    console.info("[voice] heard", JSON.stringify(voice.heard.text), "-> intent:", heardIntent);   // diagnostic: what the server understood
    if (!heardIntent) return;                        // talk that was not a command must not cancel one already waiting
    // Pause / resume act at once (no 700 ms read-the-chip delay): the server has already changed state when it answered.
    if (heardIntent === "pause_session") { stopCount.current?.(); setPaused(true); cue("pause", true); return; }
    if (heardIntent === "resume_session") { setPaused(false); cue("resume", true); setGuideOpen(false); return; }
    if (heardIntent === "stop_session") setPaused(false);
    // The chip shows `Heard: "..."` right away; the change itself follows a moment later so it can be read first.
    window.clearTimeout(commandTimer.current);       // a newer command replaces an older one that has not run yet
    commandTimer.current = window.setTimeout(() => {
      const fx = intentEffects(heardIntent, device);
      if (heardIntent === "start_session") countdown();
      console.info("[voice] applying", heardIntent, fx);
      // the Exercise Guide: the same state the Info button sets. Only "show/hide preview" announce it; a start, resume or stop
      // that closes the guide does so silently (the countdown or "Paused" is speaking).
      if (fx.showGuide !== undefined) (heardIntent === "show_preview" || heardIntent === "hide_preview" ? showGuide : setGuideOpen)(fx.showGuide);
      if (fx.stopDeviceSession) remote.stop();
    }, 700);
  }, [heardSeq]);   // eslint-disable-line react-hooks/exhaustive-deps -- one run per spoken sentence
  useEffect(() => () => window.clearTimeout(commandTimer.current), []);
  // A phone session that ends for any reason (stop, interrupted camera) closes the guide (a completed session).
  const lastStatus = useRef(remote.status);
  useEffect(() => {
    const was = lastStatus.current;
    lastStatus.current = remote.status;
    if (remote.status === "idle" && was !== "idle" && was !== "starting") { setGuideOpen(false); setPaused(false); }
  }, [remote.status]);
  // Server camera: the browser microphone is opt-in (a tap; remembered, and restarted by itself when the browser has
  // already granted it). While the fall dialog is open the BACKEND's microphone answers it, so the browser releases its own.
  const toggleLocalVoice = () => {
    const on = localVoice.mic === "off" || localVoice.mic === "blocked" || localVoice.mic === "unavailable";
    try { localStorage.setItem("voiceLocal", on ? "1" : "0"); } catch { /* storage blocked */ }
    if (on) void localVoice.enable(); else localVoice.end();
  };
  useEffect(() => {
    if (device) return;
    try { if (localStorage.getItem("voiceLocal") !== "1") return; } catch { return; }
    navigator.permissions?.query({ name: "microphone" as PermissionName })
      .then((p) => { if (p.state === "granted") void localVoice.enable(); }).catch(() => {});
  }, []);   // eslint-disable-line react-hooks/exhaustive-deps -- once, on load
  const resumeVoice = useRef(false);
  useEffect(() => {
    if (device) return;
    if (fallOpen && localVoice.mic !== "off") { resumeVoice.current = true; localVoice.end(); }
    else if (!fallOpen && resumeVoice.current) { resumeVoice.current = false; void localVoice.enable(); }
  }, [fallOpen, device]);   // eslint-disable-line react-hooks/exhaustive-deps

  const backendPrivacy = t?.privacy;
  useEffect(() => {
    if (!device && backendPrivacy !== undefined && backendPrivacy !== privacyPref) post("/api/config", { privacy: privacyPref });
  }, [device, backendPrivacy, privacyPref, online]);   // `online` re-checks after a reconnect
  // Never show video the backend has not confirmed is blanked (covers page load and backend restarts). With this
  // device's camera the preview is local, so hiding it is immediate.
  const hideVideo = privacyPref && (device || !(online && t?.privacy));

  return (
    <div className="min-h-screen p-3 sm:p-6">
      {fallAlert && (
        <div className={`mb-4 flex items-center gap-3 rounded-xl border px-4 py-3 text-lg font-semibold ${fall === "CONFIRMED" ? "animate-pulse border-red-500 bg-red-600 text-white" : "border-amber-500 bg-amber-500/20 text-amber-200"}`}>
          <AlertTriangle />
          {fall === "CONFIRMED" ? "FALL CONFIRMED — check on the patient now" : "Possible fall detected — confirming…"}
          <span className="ml-auto text-sm font-normal opacity-80">{t?.fall?.seconds_in_state ?? 0}s</span>
        </div>
      )}

      <header className="mb-6 flex flex-wrap items-center gap-3">
        <Activity className="text-emerald-400" />
        <h1 className="text-xl font-semibold">AI Rehab Tracker</h1>
        <div role="group" aria-label="Video source" className="flex rounded-full border border-slate-700 p-0.5 text-xs">
          {([["local", "Server camera"], ["device", "This device"]] as const).map(([m, label]) => (
            <button key={m} aria-pressed={mode === m} onClick={() => chooseMode(m)}
              className={`rounded-full px-3 py-1 ${mode === m ? "bg-emerald-500 text-slate-950" : "text-slate-300 hover:bg-slate-800"}`}>{label}</button>
          ))}
        </div>
        <button role="switch" aria-checked={privacyPref} onClick={togglePrivacy}
          title="Privacy mode: hide the camera image and stream only the skeleton on a black background"
          className="ml-auto flex items-center gap-2 rounded-full border border-slate-700 px-3 py-1 text-xs text-slate-300 hover:bg-slate-800">
          Privacy
          <span className={`relative h-4 w-7 rounded-full transition-colors ${privacyPref ? "bg-emerald-500" : "bg-slate-700"}`}>
            <span className={`absolute left-0.5 top-0.5 h-3 w-3 rounded-full bg-white transition-transform ${privacyPref ? "translate-x-3" : ""}`} />
          </span>
        </button>
        {!device && (
          <button onClick={toggleLocalVoice} aria-pressed={localVoice.mic !== "off"}
            title="Voice commands through this computer's microphone (the browser records, the server understands)"
            className={`flex items-center gap-1 rounded-full border px-3 py-1 text-xs hover:bg-slate-800 ${localVoice.mic === "off" ? "border-slate-700 text-slate-300" : "border-emerald-600 text-emerald-300"}`}>
            <Mic size={12} /> Voice
          </button>
        )}
        {/* TEMP: remove with POST /api/fall/test once the fall dialog has been tested */}
        <button onClick={act.testFall} title="Simulate a fall to test the Are-you-okay dialog (temporary)"
          className="rounded-full border border-amber-600 px-3 py-1 text-xs text-amber-300 hover:bg-amber-500/10">
          Test Fall
        </button>
        <span className="rounded-full border border-slate-700 px-3 py-1 text-xs text-slate-300">P1 · single patient</span>
      </header>

      <main className="grid gap-6 lg:grid-cols-[1fr_360px]">
        <section ref={fs.ref}
          className={fs.full ? "fixed inset-0 z-40 overflow-hidden bg-black"
            : `relative overflow-hidden rounded-2xl border bg-black ${fall === "CONFIRMED" ? "border-red-500" : "border-slate-800"}`}>
          {/* Inside the section so it is also visible in full screen (only the fullscreen element is drawn). No blur: it sits over live video. */}
          {(count || paused) && (
            <div aria-live="assertive" className="pointer-events-none absolute inset-0 z-30 flex items-center justify-center p-4">
              {count
                ? <span key={count} className="min-w-[8rem] rounded-3xl border border-white/20 bg-slate-950/70 px-10 py-6 text-center text-6xl font-extrabold tracking-tight text-emerald-300 shadow-2xl sm:text-7xl">{count}</span>
                : <span className="rounded-3xl border border-amber-300/40 bg-slate-950/70 px-8 py-5 text-center text-4xl font-extrabold tracking-tight text-amber-300 shadow-2xl sm:text-5xl">
                    Paused<span className="mt-1 block text-base font-medium text-slate-300">Say “resume” or “let’s go” to continue</span>
                  </span>}
            </div>
          )}
          {device ? (
            <RemoteView s={remote} hidden={privacyPref} level={fall === "CONFIRMED" ? "confirmed" : fall === "SUSPECTED" ? "suspected" : "ok"}
              full={fs.full} onToggleFull={fs.toggle} onStart={startSession} onPause={pauseSession} />
          ) : (
            <>
              <img src="/api/video_feed" alt="live feed"
                className={`block w-full object-contain ${fs.full ? "h-full" : "aspect-[4/3]"} ${hideVideo ? "invisible" : ""}`} />
              <Hud
                status={online ? (t?.pose_visible ? `LIVE · ${t?.posture ?? "pose"}` : "LIVE · no pose") : "OFFLINE"}
                statusTone={online ? "ok" : "bad"}
                counting={online ? t?.counting : undefined}
                reps={online ? t?.reps : undefined}
                note={online ? calibrationNote(t?.reps) : undefined}
                fall={fall}
                model={online && t?.engine ? {
                  name: t.engine, ms: Math.round((t.timings?.infer_ms ?? 0) * 10) / 10,
                  other: nextEngine(t.engine, ENGINES.map((e) => e.id)), wanted: null,
                  onSwitch: () => { const n = nextEngine(t.engine ?? null, ENGINES.map((e) => e.id)); if (n) void post("/api/config", { engine: n }); },
                } : null}
                full={fs.full} onToggleFull={fs.toggle} onStart={startSession} onPause={pauseSession}
                extra={localVoice.mic !== "off" ? <VoiceChip mic={localVoice.mic} error={localVoice.micError} heard={localVoice.heard.text} intent={localVoice.heard.intent} /> : null}
              />
            </>
          )}
        </section>

        <aside className="flex flex-col gap-4">
          {device ? <RemotePanel s={remote} /> : <CameraPicker active={t?.camera_index} backend={t?.camera_backend} error={t?.camera_error} />}
          {!device && localVoice.mic !== "off" && (
            <div className={`text-xs ${localVoice.mic === "blocked" || localVoice.mic === "unavailable" ? "text-amber-400" : "text-slate-500"}`}>
              {localVoice.micError ?? ""}
              <div className="text-[10px] text-slate-600">{VOICE_HELP}</div>
            </div>
          )}

          {!device && (<div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
            <div className="mb-2 text-xs uppercase tracking-wide text-slate-400">Pose engine</div>
            <div className="grid grid-cols-3 gap-1 rounded-lg bg-slate-800 p-1">
              {ENGINES.map((e) => (
                <button key={e.id} onClick={() => post("/api/config", { engine: e.id })}
                  className={`rounded-md px-1 py-1.5 text-xs transition ${t?.engine === e.id ? "bg-emerald-500 text-slate-950" : "text-slate-300 hover:bg-slate-700"}`}>
                  {e.label}
                </button>
              ))}
            </div>
            <div className="mt-2 text-xs text-slate-500">device: {t?.device ?? "…"}</div>
          </div>)}

          <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
            <label className="mb-2 block text-xs uppercase tracking-wide text-slate-400">Exercise</label>
            <div className="flex gap-2">
              <select value={t?.exercise ?? ""} onChange={(e) => act.exercise(e.target.value)}
                className="w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm">
                {exercises.map((x) => <option key={x}>{x}</option>)}
              </select>
              <button onClick={() => showGuide(true)} disabled={!guide} title="Exercise Guide" aria-label="Exercise Guide"
                className="flex items-center rounded-lg border border-slate-700 bg-slate-800 px-3 text-sm hover:bg-slate-700 disabled:opacity-50">
                <Info size={14} />
              </button>
              <button onClick={act.reset} title="Reset session"
                className="flex items-center gap-1 rounded-lg border border-slate-700 bg-slate-800 px-3 text-sm hover:bg-slate-700">
                <RotateCcw size={14} /> Reset
              </button>
            </div>
          </div>

          <div className="grid grid-cols-1 gap-3">
            {Object.entries(t?.reps ?? {}).map(([side, limb]) => <LimbCard key={side} side={side} limb={limb} />)}
          </div>

          <div className="grid grid-cols-2 gap-3">
            <Card icon={<Footprints size={14} />} label="Mobility" value={t?.mobility?.state ?? "–"}
              sub={`${t?.mobility?.steps ?? 0} steps · ${t?.mobility?.cadence_spm ?? 0} spm`} />
            <Card icon={<ShieldAlert size={14} />} label="Fall status" value={fall}
              sub={t?.fall?.tilt_deg != null ? `tilt ${t.fall.tilt_deg}°` : undefined}
              accent={fall === "NORMAL" ? "text-emerald-400" : "text-red-400"} />
            <Card icon={<Gauge size={14} />} label="FPS" value={t?.fps ?? "–"} />
            <Card icon={<Timer size={14} />} label="Latency" value={t ? `${t.latency_ms} ms` : "–"} />
          </div>
          {t?.timings && (
            <div className="text-[10px] tabular-nums text-slate-500" title="Smoothed time per stage. Capture is mostly waiting for the camera.">
              capture {t.timings.capture_ms.toFixed(1)} · infer {t.timings.infer_ms.toFixed(1)} · pipeline {t.timings.pipeline_ms.toFixed(1)} · draw {t.timings.draw_ms.toFixed(1)} · encode {t.timings.encode_ms.toFixed(1)} ms
            </div>
          )}

          {t?.error && (
            <div className="flex items-center gap-2 rounded-xl border border-red-900 bg-red-950/50 p-3 text-sm text-red-300">
              <Cpu size={16} /> {t.error}
            </div>
          )}
        </aside>
      </main>
      {guideOpen && guide && t && <ExerciseGuide info={guideInfo} initial={t.exercise} onClose={() => showGuide(false)} />}
      {alert && (alert.state === "COUNTDOWN" || alert.state === "ALERTED") && <FallCheck alert={alert} voice={device ? remote.voice.voiceState : t?.fall?.voice} onAck={act.ack} onEscalate={act.escalate} canListen={!device} />}
    </div>
  );
}
