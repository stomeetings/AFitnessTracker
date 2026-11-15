import { useEffect, useRef, useState } from "react";
import { X } from "lucide-react";

/** Exercise Guide modal: animated full-body skeleton with angle arcs on the target joints, plus a simulated
 *  left/right breakdown (reps, joint angle, ROM). Left and right alternate; bilateral exercises move together.
 *  rest/peak/sides come from the backend's EXERCISES table, so the demo matches what the rep counter expects. */

const REP_MS = 3000;      // one demo rep
const FRAME = "#64748b", ACTIVE = "#34d399", BRIGHT = "#a7f3d0";   // slate-500, emerald-400, emerald-200
const JOINT: Record<string, string> = {
  "Elbow Curl": "elbow", "Shoulder Raise": "shoulder", "Knee Flexion": "knee", "Hip Flexion": "hip", "Sit-to-Stand": "knee",
};
const TIPS: Record<string, string> = {
  "Elbow Curl": "Keep your upper arm still by your side. Bend the elbow to bring your hand toward your shoulder, then lower slowly.",
  "Shoulder Raise": "Keep your arm straight and raise it out to shoulder height, then lower slowly.",
  "Knee Flexion": "Stand tall and bend one knee to bring your heel back and up, then lower slowly. Hold a support if you need one.",
  "Hip Flexion": "Stand tall and lift one knee forward, then lower slowly. Hold a support if you need one.",
  "Sit-to-Stand": "From a chair, lean slightly forward and stand up fully, then sit back down with control.",
};

type Pt = [number, number];
type Tone = 0 | 1 | 2;    // 0 = body frame, 1 = exercise limb at rest, 2 = exercise limb moving
type Side = { S: Pt; E: Pt; W: Pt; Hd: Pt; H: Pt; K: Pt; A: Pt; T: Pt; arc: [Pt, Pt, Pt] };

/** Point `len` away from `o`; deg 0 = straight down, 90 = +x, 180 = up. */
const at = ([x, y]: Pt, len: number, deg: number): Pt => {
  const r = (deg * Math.PI) / 180;
  return [x + len * Math.sin(r), y + len * Math.cos(r)];
};
const pts = (...p: Pt[]) => p.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" ");

/** Skeleton for the tracked joint angles [viewer-left side, viewer-right side] (angle definitions as in
 *  backend kinematics.py). Arm exercises are drawn front-on; leg exercises in a 3/4 view turned toward +x,
 *  because knee and hip flexion move forward/back and would be invisible from the front. */
function skeleton(name: string, angles: [number, number]) {
  const front = name === "Elbow Curl" || name === "Shoulder Raise", sit = name === "Sit-to-Stand";
  const sw = front ? 46 : 26, hw = front ? 22 : 13;                    // half shoulder / hip width
  const P: Pt = sit ? at([205, 288], 76, -angles[0]) : [180, 212];     // pelvis centre; sitting keeps the feet planted
  const N: Pt = [P[0], P[1] - 116];                                    // neck base
  const sides = [-1, 1].map((sgn, i): Side => {
    const a = angles[i];
    const S: Pt = [N[0] + sgn * sw, N[1] + 4], H: Pt = [P[0] + sgn * hw, P[1]];
    let upper = front ? sgn * 6 : 0, fore = upper, thigh = front ? sgn * 3 : 0, shank = thigh;
    if (name === "Elbow Curl") fore = upper + sgn * (180 - a);
    if (name === "Shoulder Raise") upper = fore = sgn * a;
    if (name === "Knee Flexion") shank = a - 180;
    if (name === "Hip Flexion" || sit) thigh = 180 - a;
    const E = at(S, 58, upper), W = at(E, 52, fore), Hd = at(W, 14, fore);
    const K = at(H, 76, thigh), A = at(K, 72, shank);
    const T: Pt = [A[0] + (front ? sgn * 12 : 18), A[1] + 9];          // toe
    // [joint, end of first segment, end of second segment]; shoulder and hip are measured against the upright trunk
    const arc: [Pt, Pt, Pt] = name === "Elbow Curl" ? [E, S, W] : name === "Shoulder Raise" ? [S, [S[0], S[1] + 1], E]
      : name === "Hip Flexion" ? [H, [H[0], H[1] - 1], K] : [K, H, A];
    return { S, E, W, Hd, H, K, A, T, arc };
  });
  return { P, N, sides, front, sit };
}

/** Dashed arc between the two segments at joint J, a reference tick along the first one, and the label spot. */
function angleArc([J, U, V]: [Pt, Pt, Pt], r = 26) {
  const tu = Math.atan2(U[1] - J[1], U[0] - J[0]), tv = Math.atan2(V[1] - J[1], V[0] - J[0]);
  const d = Math.atan2(Math.sin(tv - tu), Math.cos(tv - tu));         // signed shortest sweep from U to V
  const on = (t: number, rad: number): Pt => [J[0] + rad * Math.cos(t), J[1] + rad * Math.sin(t)];
  const s = on(tu, r), e = on(tv, r);
  return {
    path: `M${s[0].toFixed(1)} ${s[1].toFixed(1)} A${r} ${r} 0 0 ${d > 0 ? 1 : 0} ${e[0].toFixed(1)} ${e[1].toFixed(1)}`,
    ref: on(tu, r + 10),
    label: on(tu + d / 2, r + 19),
  };
}

function Skeleton({ name, angles, moving }: { name: string; angles: [number, number]; moving: [boolean, boolean] }) {
  const { P, N, sides, front, sit } = skeleton(name, angles);
  const bone = (key: string, tone: Tone, ...p: Pt[]) => (
    <polyline key={key} points={pts(...p)} stroke={tone ? ACTIVE : FRAME} strokeWidth={tone ? 4.5 : 4}
      opacity={tone === 1 ? 0.6 : 1} filter={tone === 2 ? "url(#guide-glow)" : undefined} />
  );
  const node = (key: string, [x, y]: Pt, tone: Tone) => (
    <g key={key}>
      <circle cx={x} cy={y} r={tone === 2 ? 11 : 8.5} fill={ACTIVE} opacity={tone === 2 ? 0.3 : 0.15} />
      <circle cx={x} cy={y} r={tone === 2 ? 5 : 4} fill={tone === 2 ? BRIGHT : ACTIVE} opacity={tone ? 1 : 0.8} />
    </g>
  );
  const tones = sides.map((_, i): [Tone, Tone] => {       // [arm tone, leg tone] per side
    const t: Tone = moving[i] ? 2 : 1;
    return front ? [t, 0] : [0, t];
  });
  return (
    <svg viewBox="0 0 360 420" role="img" aria-label={`Animated demonstration of ${name}`}
      fill="none" strokeLinecap="round" strokeLinejoin="round" className="max-h-[72vh] w-full rounded-2xl border border-slate-800 bg-slate-950">
      <defs>
        <filter id="guide-glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="3" result="blur" />
          <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
        <radialGradient id="guide-bg" cx="50%" cy="45%" r="60%">
          <stop offset="0" stopColor={ACTIVE} stopOpacity="0.10" /><stop offset="1" stopColor={ACTIVE} stopOpacity="0" />
        </radialGradient>
      </defs>
      <rect width="360" height="420" fill="url(#guide-bg)" />
      <text x="16" y="26" fill={FRAME} fontSize="10" letterSpacing="2">RIGHT</text>
      <text x="344" y="26" fill={FRAME} fontSize="10" letterSpacing="2" textAnchor="end">LEFT</text>
      <ellipse cx={(sides[0].A[0] + sides[1].A[0]) / 2 + (front ? 0 : 8)} cy="376" rx="70" ry="7" fill={ACTIVE} opacity="0.08" />
      {sit && <polyline points="78,374 78,292 158,292 158,374" stroke="#334155" strokeWidth="4" />}

      {/* body frame: head, neck + spine, collarbones, pelvis */}
      <ellipse cx={N[0]} cy={N[1] - 44} rx="19" ry="24" stroke={FRAME} strokeWidth="4" />
      {bone("spine", 0, [N[0], N[1] - 20], P)}
      {bone("collar", 0, sides[0].S, N, sides[1].S)}
      {bone("pelvis", 0, sides[0].H, [P[0], P[1] - 18], sides[1].H, sides[0].H)}

      {sides.map((s, i) => [
        bone(`arm${i}`, tones[i][0], s.S, s.E, s.W, s.Hd),
        bone(`leg${i}`, tones[i][1], s.H, s.K, s.A, s.T),
      ])}

      {/* angle arcs on the target joint of each side */}
      {sides.map((s, i) => {
        const J = s.arc[0], g = angleArc(s.arc), on = moving[i];
        return (
          <g key={`arc${i}`} opacity={on ? 1 : 0.55}>
            <circle cx={J[0]} cy={J[1]} r="18" fill={ACTIVE} opacity="0.2" />
            <line x1={J[0]} y1={J[1]} x2={g.ref[0]} y2={g.ref[1]} stroke={ACTIVE} strokeWidth="1.5" strokeDasharray="3 4" />
            <path d={g.path} stroke={ACTIVE} strokeWidth="1.8" strokeDasharray="4 4" />
            <text x={g.label[0]} y={g.label[1]} textAnchor="middle" dominantBaseline="middle" fill={on ? BRIGHT : ACTIVE}
              fontSize="14" fontWeight="600" style={{ fontVariantNumeric: "tabular-nums" }}>{angles[i].toFixed(0)}°</text>
          </g>
        );
      })}

      <g filter="url(#guide-glow)">
        {node("neck", N, 0)}
        {node("pelvis", P, 0)}
        {sides.map((s, i) => [
          node(`S${i}`, s.S, tones[i][0]), node(`E${i}`, s.E, tones[i][0]), node(`W${i}`, s.W, tones[i][0]),
          node(`H${i}`, s.H, tones[i][1]), node(`K${i}`, s.K, tones[i][1]), node(`A${i}`, s.A, tones[i][1]),
        ])}
      </g>
    </svg>
  );
}

type SideStat = { label: string; reps: number; p: number; angle: number; state: string };

function SideCard({ s, target }: { s: SideStat; target: number }) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-950/40 p-4">
      <div className="flex items-center justify-between text-xs uppercase tracking-wide text-slate-400">
        <span>{s.label}</span>
        <span className={`rounded px-1.5 py-0.5 ${s.state === "IDLE" ? "bg-slate-800" : "bg-emerald-900 text-emerald-300"}`}>{s.state}</span>
      </div>
      <div className="mt-3 flex items-end justify-between">
        <div className="text-4xl font-bold tabular-nums">{s.reps}<span className="ml-1 text-sm font-normal text-slate-400">reps</span></div>
        <div className="text-right">
          <div className="text-xl font-semibold tabular-nums">{s.angle.toFixed(0)}°</div>
          <div className="text-[10px] uppercase text-slate-500">joint angle</div>
        </div>
      </div>
      <div className="mt-3 h-1.5 w-full rounded bg-slate-800">
        <div className="h-1.5 rounded bg-emerald-500" style={{ width: `${s.p * 100}%` }} />
      </div>
      <div className="mt-2 text-xs tabular-nums text-slate-400">ROM {(target * s.p).toFixed(0)}° of {target}°</div>
    </div>
  );
}

export function ExerciseGuide({ name, rest, peak, sides, onClose }: { name: string; rest: number; peak: number; sides?: string[]; onClose: () => void }) {
  const ref = useRef<HTMLDialogElement>(null);
  const [ms, setMs] = useState(0);
  useEffect(() => {
    const d = ref.current;
    if (d && !d.open) d.showModal();              // native modal: focus trap, Esc to close, ::backdrop
    const t0 = performance.now();
    let id: number = requestAnimationFrame(function tick(now) { setMs(now - t0); id = requestAnimationFrame(tick); });
    return () => cancelAnimationFrame(id);        // animation only runs while the modal is mounted
  }, []);

  // bilateral exercise: both sides move as one (name fallback covers a backend that predates the `sides` field)
  const together = sides ? sides.includes("B") : name === "Sit-to-Stand";
  const target = Math.abs(peak - rest), joint = JOINT[name] ?? "joint";
  const stat = (label: string, t: number, reps: number): SideStat => {    // t = ms into this side's rep, or -1 if resting
    const c = t < 0 ? 0 : t / REP_MS, p = 0.5 - 0.5 * Math.cos(2 * Math.PI * c);   // 0 at rest, 1 at peak
    const state = p < 0.2 ? "IDLE" : p > 0.8 ? "PEAK" : c < 0.5 ? "FLEXING" : "RETURNING";   // same labels as live tracking
    return { label, reps, p, angle: rest + (peak - rest) * p, state };
  };
  const round = Math.floor(ms / (2 * REP_MS)), tt = ms % (2 * REP_MS), leftTurn = tt < REP_MS;
  const left = together ? stat("Left", ms % REP_MS, Math.floor(ms / REP_MS)) : stat("Left", leftTurn ? tt : -1, round + (leftTurn ? 0 : 1));
  const right = together ? { ...left, label: "Right" } : stat("Right", leftTurn ? -1 : tt - REP_MS, round);
  const lead = together || leftTurn ? left : right;
  // the figure faces the viewer, so its right side is on the viewer's left: [viewer-left, viewer-right]
  const angles: [number, number] = [right.angle, left.angle];
  const moving: [boolean, boolean] = [right.p > 0.02, left.p > 0.02];

  return (
    <dialog ref={ref} onClose={onClose} onClick={(e) => { if (e.target === e.currentTarget) e.currentTarget.close(); }}
      className="m-auto w-[96vw] max-w-5xl rounded-2xl border border-slate-700 bg-slate-900 text-slate-100 backdrop:bg-black/70">
      <div className="p-6 sm:p-8">
        <div className="mb-6 flex items-center">
          <div>
            <div className="text-xs uppercase tracking-widest text-slate-400">Exercise Guide</div>
            <h2 className="text-2xl font-semibold">{name}</h2>
          </div>
          <button onClick={() => ref.current?.close()} aria-label="Close guide" className="ml-auto rounded-lg p-2 text-slate-400 hover:bg-slate-800 hover:text-slate-100">
            <X size={20} />
          </button>
        </div>
        <div className="grid gap-8 lg:grid-cols-2">
          <Skeleton name={name} angles={angles} moving={moving} />
          <div className="flex flex-col gap-5">
            <div>
              <div className="text-7xl font-bold leading-none tabular-nums text-emerald-400">{lead.angle.toFixed(0)}°</div>
              <div className="mt-2 text-sm font-semibold uppercase tracking-wide">{together ? `Both ${joint}s` : `${lead.label} ${joint}`}</div>
              <div className="mt-1 text-sm text-slate-400">Rest <span className="tabular-nums text-slate-200">{rest}°</span> · Target <span className="tabular-nums text-emerald-300">{peak}°</span></div>
            </div>
            <p className="text-sm text-slate-300">{TIPS[name] ?? "Move smoothly from the rest position to the target position and back."}</p>
            <div className="grid grid-cols-2 gap-4">
              <SideCard s={left} target={target} />
              <SideCard s={right} target={target} />
            </div>
            <p className="text-xs text-slate-500">
              A rep counts when the {joint} moves from about {rest}° to about {peak}° and back. This is a simulation; your live left and right numbers appear on the dashboard.
            </p>
          </div>
        </div>
      </div>
    </dialog>
  );
}
