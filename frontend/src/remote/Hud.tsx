import type { ReactNode } from "react";
import { AlertTriangle, Cpu, Loader2, Maximize2, Minimize2, Pause, Play, SwitchCamera, WifiOff } from "lucide-react";
import type { Limb } from "../useTelemetry";
import { engineLabel } from "./logic";

export type HudModel = { name: string; ms: number; fps?: number; other: string | null; wanted: string | null; onSwitch: () => void };

export type HudProps = {
  status: string;                                       // "LIVE · standing", "Camera off" ...
  statusTone: "ok" | "warn" | "bad";
  counting: boolean | undefined;                        // false = standby (READY), true = ACTIVE, undefined = not known yet
  reps: Record<string, Limb> | undefined;
  fall: "NORMAL" | "SUSPECTED" | "CONFIRMED";
  model: HudModel | null;
  onFlip?: () => void;                                  // phone only: front / rear camera
  flipLabel?: string;
  full: boolean;
  onToggleFull: () => void;
  onStart: () => void;                                  // standby -> counting from zero
  onPause: () => void;                                  // counting -> standby
  extra?: ReactNode;                                    // the voice chip
  note?: string;                                        // e.g. "calibrated rest 92°": what the standby measured for this person
  lost?: boolean;                                       // the link to the server dropped: show the "Connection lost" banner
  offline?: boolean;                                    // reps and falls are worked out on this phone, but the server cannot be reached
};

const TONE = { ok: "bg-emerald-400", warn: "bg-amber-400", bad: "bg-red-500" } as const;
const SIDE: Record<string, string> = { L: "Left", R: "Right", B: "" };

// No backdrop-blur / blur anywhere in the HUD: it sits over a live video, so a blur would be recomputed on every video frame
// (a stutter on mid-range phone GPUs). A plain translucent fill reads just as well.
const btnBase = "pointer-events-auto flex min-h-10 items-center gap-2 rounded-full px-3 py-1 text-xs disabled:cursor-default";
const btn = `${btnBase} bg-slate-900/90 font-medium text-slate-100 hover:bg-slate-800 disabled:hover:bg-slate-900/90`;
const btnStart = `${btnBase} bg-emerald-500 font-bold text-slate-950 hover:bg-emerald-400`;

/** The top strip over the video, the same in both modes and in full screen: the status and the READY / ACTIVE badge, the rep
 *  counter in large type, the model badge (tap to switch), camera flip, full-screen toggle, and Start / Pause. It never blocks
 *  taps on the video itself: only its buttons take pointer events. */
export function Hud(p: HudProps) {
  const entries = Object.entries(p.reps ?? {});
  const standby = p.counting === false;
  return (
    <div className="pointer-events-none absolute inset-x-0 top-0 z-50 flex flex-col gap-2 bg-gradient-to-b from-black/75 via-black/40 to-transparent p-2 sm:p-3">
      {p.lost && (
        // the tunnel or Wi-Fi dropped: nothing is being counted or watched on the server until it is back (the reps are hidden)
        <div role="alert" className="flex animate-pulse flex-wrap items-center justify-center gap-x-2 rounded-lg bg-red-600 px-3 py-2 text-sm font-bold text-white">
          <WifiOff size={16} /> Connection lost — reconnecting…
          <span className="text-xs font-normal">Reps and fall monitoring are paused</span>
        </div>
      )}
      {p.offline && !p.lost && (
        // Everything still works (counting, the fall check with its beep and spoken question), but nobody is told about an alarm:
        // that is said plainly, not hidden. Solid red like the lost-link banner, without the pulse (it stays for a whole workout).
        <div role="alert" className="flex flex-wrap items-center justify-center gap-x-2 rounded-lg bg-red-700 px-3 py-2 text-sm font-bold text-white">
          <WifiOff size={16} /> Offline — caregiver will not be notified
        </div>
      )}
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="flex flex-wrap items-center gap-2">
          <span className="flex items-center gap-2 rounded-full bg-slate-900/90 px-3 py-1 text-xs">
            <span className={`h-2 w-2 rounded-full ${TONE[p.statusTone]}`} />{p.status}
          </span>
          {p.counting !== undefined && (
            <span role="status" className={`rounded-full px-3 py-1 text-xs font-bold ${standby ? "bg-amber-400 text-slate-950" : "bg-emerald-500 text-slate-950"}`}>
              {standby ? "READY — Say “Let’s go” to start" : "ACTIVE"}
            </span>
          )}
          {p.note && !standby && (
            <span className="rounded-full bg-slate-900/90 px-3 py-1 text-xs text-slate-300" title="Your starting angle, measured while you held still before you started">
              {p.note}
            </span>
          )}
          {p.fall !== "NORMAL" && (
            <span role="alert" className={`flex items-center gap-1 rounded-full px-3 py-1 text-xs font-bold ${p.fall === "CONFIRMED" ? "animate-pulse bg-red-600 text-white" : "bg-amber-500 text-slate-950"}`}>
              <AlertTriangle size={12} /> {p.fall === "CONFIRMED" ? "FALL" : "possible fall"}
            </span>
          )}
        </div>
        <div className="flex flex-wrap items-center justify-end gap-2">
          {p.model && (
            <button onClick={p.model.onSwitch} disabled={!p.model.other || p.model.wanted !== null} className={btn}
              title="Pose model in use. Tap to switch the model (the on-device one only affects this phone; a server one the whole server)">
              {p.model.wanted ? <Loader2 size={14} className="animate-spin text-emerald-400" /> : <Cpu size={14} className="text-emerald-400" />}
              <span className="font-semibold">{engineLabel(p.model.name)}</span>
              {p.model.ms > 0 && !p.model.wanted && (
                <span className="tabular-nums text-slate-400">{p.model.fps ? `${p.model.fps} fps · ` : ""}{p.model.ms} ms</span>
              )}
              {p.model.wanted && <span className="text-slate-400">→ {engineLabel(p.model.wanted)}…</span>}
            </button>
          )}
          {p.onFlip && (
            <button onClick={p.onFlip} className={btn} aria-label={p.flipLabel ?? "Switch camera"}>
              <SwitchCamera size={14} /> {p.flipLabel ?? "Flip"}
            </button>
          )}
          <button onClick={p.onToggleFull} className={`${btn} min-h-11 min-w-11 touch-manipulation`} aria-label={p.full ? "Exit full screen" : "Full screen"}>
            {p.full ? <Minimize2 size={14} /> : <Maximize2 size={14} />}{p.full ? "Exit" : "Full screen"}
          </button>
        </div>
      </div>

      <div className="flex items-end justify-between gap-3">
        <div className={`flex flex-1 items-end justify-center gap-6 text-white [text-shadow:0_1px_2px_rgb(0_0_0/0.9)] ${standby ? "opacity-60" : ""}`} aria-label="Repetitions">
          {entries.map(([side, limb]) => (
            <div key={side} className="flex items-baseline gap-2">
              <span className="text-6xl font-black leading-none tabular-nums sm:text-7xl">{limb.reps}</span>
              <span className="text-xs font-semibold uppercase tracking-wide text-slate-300">{SIDE[side] ? `${SIDE[side]} reps` : "reps"}</span>
            </div>
          ))}
        </div>
        <div className="flex flex-col items-end gap-2">
          {p.counting !== undefined && (
            standby
              ? <button onClick={p.onStart} className={btnStart}><Play size={14} /> Start</button>
              : <button onClick={p.onPause} className={btn}><Pause size={14} /> Pause</button>
          )}
          {p.extra}
        </div>
      </div>
    </div>
  );
}
