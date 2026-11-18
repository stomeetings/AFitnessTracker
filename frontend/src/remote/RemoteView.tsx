import { useCallback, useEffect, useRef, useState } from "react";
import { Camera, Loader2, Play, SwitchCamera, Square } from "lucide-react";
import { Hud } from "./Hud";
import { calibrationNote, drawPose, engineLabel, fitContain, nextEngine, type AlertLevel, type Pose } from "./logic";
import type { Facing, RemoteSession as Session, RemoteStatus } from "./useRemoteSession";
import { MIC_TEXT, VOICE_HELP, VoiceChip } from "./VoiceChip";

const STATUS_TEXT: Record<RemoteStatus, string> = {
  idle: "Camera off", starting: "Starting camera…", connecting: "Connecting…", live: "LIVE", reconnecting: "Reconnecting…", error: "Camera error",
};

/** The phone's own camera preview with the skeleton drawn over it on a canvas. The preview is mirrored with CSS for the
 *  front camera (the canvas inside the same mirrored box follows it); the frames sent to the server never are. `hidden`
 *  (the Privacy switch) hides the preview image only: the server still needs the frames, so they are still sent. */
export function RemoteView({ s, hidden, level, full, onToggleFull, onStart, onPause }:
  { s: Session; hidden: boolean; level: AlertLevel; full: boolean; onToggleFull: () => void; onStart: () => void; onPause: () => void }) {
  const box = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const [pin, setPin] = useState("");
  const t = s.telemetry;
  const running = s.stream !== null;
  const otherModel = s.model ? nextEngine(s.model, s.models) : null;         // what a tap on the model badge switches to
  // The "stage": the exact rectangle the video image occupies inside the outer box (the camera's aspect ratio, centred). The video
  // AND the canvas fill the stage and nothing else, so the canvas is never wider/taller than the picture it annotates.
  const fw = s.frame?.w || 4, fh = s.frame?.h || 3;
  const fit = size.w && size.h ? fitContain(fw, fh, size.w, size.h) : { x: 0, y: 0, w: 0, h: 0 };
  const stage = { x: Math.round(fit.x), y: Math.round(fit.y), w: Math.round(fit.w), h: Math.round(fit.h) };
  const sizeRef = useRef(stage);
  sizeRef.current = stage;                                // what paint() maps onto: the stage, not the outer box
  const levelRef = useRef(level);
  levelRef.current = level;
  const dims = useRef({ w: 0, h: 0 });
  // The last pose drawn (a reference to the loop's reused buffers, not a copy): repainted after a resize. One object, mutated.
  const lastLocal = useRef<{ pose: Pose | null; frame: { w: number; h: number } }>({ pose: null, frame: { w: 0, h: 0 } });
  const ctxRef = useRef<CanvasRenderingContext2D | null>(null);          // fetched once, not per frame
  const lastSeen = useRef(0);                                            // when a pose was last drawn (performance.now())

  /** Draws a pose onto the canvas. Used for 30 poses a second from the on-device model, so it only resizes the canvas when its
   *  size really changed (resizing clears and reallocates it) and allocates nothing itself. */
  const paint = useCallback((pose: Pose | null, frame: { w: number; h: number }) => {
    const c = canvas.current;
    const ctx = ctxRef.current ?? (c ? (ctxRef.current = c.getContext("2d")) : null);
    const sz = sizeRef.current;
    if (!c || !ctx || !sz.w) return;
    // A short dropout (the model loses the person for a frame or two) must not blank the skeleton: that is the flicker. Keep the
    // last drawing (the canvas holds it) for 150 ms; after that, or when nobody is there for longer, it is cleared as before.
    const now = performance.now();
    if (pose) lastSeen.current = now;
    else if (now - lastSeen.current < 150) return;
    const dpr = window.devicePixelRatio || 1;
    const pw = Math.round(sz.w * dpr), ph = Math.round(sz.h * dpr);
    if (dims.current.w !== pw || dims.current.h !== ph) { c.width = pw; c.height = ph; dims.current = { w: pw, h: ph }; }
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, pw, ph);                            // the whole backing store, in device pixels: nothing of the last frame can stay
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);                 // then CSS-pixel coordinates for drawPose
    drawPose(ctx, pose, frame, sz, levelRef.current);
  }, []);

  useEffect(() => {                                      // every pose (the phone's model or the server's answer) is drawn directly, never via state
    const last = lastLocal.current;
    s.setPoseListener((pose, frame) => { last.pose = pose; last.frame = frame; paint(pose, frame); });   // no object per pose
    return () => s.setPoseListener(null);
  }, [s.setPoseListener, paint]);   // eslint-disable-line react-hooks/exhaustive-deps -- `s` itself changes every render

  useEffect(() => {
    const v = s.videoRef.current;
    if (v && v.srcObject !== s.stream) { v.srcObject = s.stream; void v.play().catch(() => {}); }
  }, [s.stream, s.videoRef]);

  useEffect(() => {                                      // shown again: make sure the video is actually playing
    if (!hidden) void s.videoRef.current?.play().catch(() => {});
  }, [hidden, s.videoRef]);

  useEffect(() => {                                      // keep the canvas's pixels equal to its CSS size (sharp on retina)
    const el = box.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {                                      // new poses draw themselves; this repaints the last one after a resize or an alert colour change
    paint(lastLocal.current.pose, lastLocal.current.frame);
  }, [stage.w, stage.h, level, paint]);

  const ratio = s.frame ? `${s.frame.w} / ${s.frame.h}` : "4 / 3";
  return (
    <div ref={box} className={`relative w-full overflow-hidden bg-black ${full ? "h-full" : ""}`}
      style={full ? undefined : { aspectRatio: ratio, maxHeight: "78dvh" }}>
      <div className="absolute" style={{ left: stage.x, top: stage.y, width: stage.w, height: stage.h, transform: s.facing === "user" ? "scaleX(-1)" : undefined }}>
        {/* opacity, not visibility: a hidden video can be paused by the browser and stay black when shown again */}
        <video ref={s.videoRef} playsInline muted autoPlay className={`h-full w-full object-contain ${hidden ? "opacity-0" : ""}`} />
        <canvas ref={canvas} className="pointer-events-none absolute inset-0 block h-full w-full" />
      </div>

      {/* The HUD: status, READY / ACTIVE, big rep counter, model badge, camera flip, full screen, Start / Pause, voice chip */}
      <Hud
        status={`${s.status === "live" ? (t?.pose_visible ? `LIVE · ${t.posture ?? "pose"}` : "LIVE · no pose") : STATUS_TEXT[s.status]}${hidden && running ? " · privacy" : ""}`}
        statusTone={s.status === "live" ? "ok" : s.status === "error" ? "bad" : "warn"}
        lost={running && s.status === "reconnecting"}
        counting={s.status === "live" ? t?.counting : undefined}
        reps={s.status === "live" ? t?.reps : undefined}
        note={s.status === "live" ? calibrationNote(t?.reps) : undefined}
        fall={t?.fall?.status ?? "NORMAL"}
        model={running && s.model ? {
          name: s.model, ms: s.modelMs, fps: s.inference === "device" ? s.local.fps : undefined, other: otherModel, wanted: s.modelWanted,
          onSwitch: () => { if (otherModel) s.setModel(otherModel); },
        } : null}
        onFlip={running ? () => void s.switchCamera() : undefined}
        flipLabel={s.facing === "user" ? "Use rear" : "Use front"}
        full={full} onToggleFull={onToggleFull} onStart={onStart} onPause={onPause}
        extra={running ? <VoiceChip mic={s.voice.mic} error={s.voice.micError} heard={s.voice.heard.text} intent={s.voice.heard.intent} /> : null}
      />

      {!running && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 p-4 text-center">
          {s.needsToken && (
            <input type="password" value={pin} onChange={(e) => setPin(e.target.value)} placeholder="Access token / PIN"
              aria-label="Access token" autoComplete="off" autoCapitalize="off" autoCorrect="off" spellCheck={false}
              onKeyDown={(e) => { if (e.key === "Enter" && pin.trim()) void s.start(pin); }}
              className="w-full max-w-xs rounded-xl border border-slate-600 bg-slate-900 px-4 py-3 text-center text-lg text-slate-100" />
          )}
          <button onClick={() => void s.start(s.needsToken ? pin : undefined)} disabled={s.status === "starting" || (s.needsToken && !pin.trim())}
            className="flex items-center gap-2 rounded-2xl bg-emerald-500 px-8 py-4 text-xl font-bold text-slate-950 hover:bg-emerald-400 disabled:opacity-60">
            {s.status === "starting" ? <Loader2 className="animate-spin" /> : <Play />} Start camera
          </button>
          <p className="max-w-sm text-sm text-slate-400">
            {s.inference === "device"
              ? "Uses this device's camera. The pose is worked out on this device; only joint positions go to the server, never video."
              : "Uses this device's camera. Frames go to the server for analysis; the video is not stored."}
          </p>
        </div>
      )}
      {s.error && (
        <div role="alert" className="absolute inset-x-3 bottom-3 rounded-lg border border-amber-700 bg-slate-950/90 px-3 py-2 text-sm text-amber-300">{s.error}</div>
      )}
    </div>
  );
}

/** Sidebar card for the browser camera: front/rear switch, stop, and the link's measured speed. */
export function RemotePanel({ s, onStop }: { s: Session; onStop: () => void }) {
  const running = s.stream !== null;
  const facing: Facing = s.facing;
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
      <div className="mb-2 flex items-center gap-2 text-xs uppercase tracking-wide text-slate-400"><Camera size={14} /> This device&apos;s camera</div>
      <div className="grid grid-cols-2 gap-2">
        <button onClick={() => void s.switchCamera()} disabled={!running}
          className="flex min-h-11 items-center justify-center gap-2 rounded-lg border border-slate-700 bg-slate-800 px-3 text-sm hover:bg-slate-700 disabled:opacity-50">
          <SwitchCamera size={16} /> {facing === "user" ? "Use rear" : "Use front"}
        </button>
        <button onClick={onStop} disabled={!running}
          className="flex min-h-11 items-center justify-center gap-2 rounded-lg border border-slate-700 bg-slate-800 px-3 text-sm hover:bg-slate-700 disabled:opacity-50">
          <Square size={14} /> Stop
        </button>
      </div>
      <div className="mt-2 text-xs text-slate-500">
        {s.status === "live"
          ? s.inference === "device"
            ? `On-device ${s.local.fps} fps · ${s.local.ms} ms${s.local.delegate ? ` (${s.local.delegate})` : ""}${s.local.status === "loading" ? " · loading the model…" : ""} · camera ${s.camera.w}×${s.camera.h} · only joint positions are sent`
            : `${s.stats.fps} fps · round trip ${s.stats.rtt} ms · frames ${s.stats.longSide}px${s.engine ? ` · ${engineLabel(s.engine)} ${s.stats.inferMs} ms` : ""}`
          : STATUS_TEXT[s.status]}
      </div>
      {running && (
        <div className={`mt-1 text-xs ${s.voice.mic === "blocked" || s.voice.mic === "unavailable" ? "text-amber-400" : "text-slate-500"}`}>
          Voice: {MIC_TEXT[s.voice.mic]}{s.voice.micError ? ` · ${s.voice.micError}` : ""}
          <div className="mt-0.5 text-[10px] text-slate-600">{VOICE_HELP}</div>
        </div>
      )}
    </div>
  );
}
