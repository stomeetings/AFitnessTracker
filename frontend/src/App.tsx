import { useEffect, useState } from "react";
import { Activity, AlertTriangle, Camera, Cpu, Footprints, Gauge, Info, Loader2, RefreshCw, RotateCcw, ShieldAlert, Timer } from "lucide-react";
import { ExerciseGuide } from "./ExerciseGuide";
import { FallCheck } from "./FallCheck";
import { useTelemetry, type Limb } from "./useTelemetry";

const ENGINES = [
  { id: "mediapipe", label: "MediaPipe Full" },
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
        <div className="h-1.5 rounded bg-emerald-500 transition-[width]" style={{ width: `${Math.min(100, Math.max(0, angle / 1.8))}%` }} />
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
  const { t, online } = useTelemetry();
  const [exercises, setExercises] = useState<string[]>([]);
  useEffect(() => { fetch("/api/exercises").then((r) => r.json()).then(setExercises).catch(() => {}); }, []);
  const [guideInfo, setGuideInfo] = useState<Record<string, { rest: number; peak: number; sides?: string[] }>>({});
  useEffect(() => { fetch("/api/exercise_info").then((r) => r.json()).then(setGuideInfo).catch(() => {}); }, []);
  const [guideOpen, setGuideOpen] = useState(false);
  const guide = t ? guideInfo[t.exercise] : undefined;

  const fall = t?.fall?.status ?? "NORMAL";
  const fallAlert = fall !== "NORMAL";
  const alert = t?.fall?.alert;

  return (
    <div className="min-h-screen p-6">
      {fallAlert && (
        <div className={`mb-4 flex items-center gap-3 rounded-xl border px-4 py-3 text-lg font-semibold ${fall === "CONFIRMED" ? "animate-pulse border-red-500 bg-red-600 text-white" : "border-amber-500 bg-amber-500/20 text-amber-200"}`}>
          <AlertTriangle />
          {fall === "CONFIRMED" ? "FALL CONFIRMED — check on the patient now" : "Possible fall detected — confirming…"}
          <span className="ml-auto text-sm font-normal opacity-80">{t?.fall?.seconds_in_state ?? 0}s</span>
        </div>
      )}

      <header className="mb-6 flex items-center gap-3">
        <Activity className="text-emerald-400" />
        <h1 className="text-xl font-semibold">AI Rehab Tracker</h1>
        {/* TEMP: remove with POST /api/fall/test once the fall dialog has been tested */}
        <button onClick={() => post("/api/fall/test")} title="Simulate a fall to test the Are-you-okay dialog (temporary)"
          className="ml-auto rounded-full border border-amber-600 px-3 py-1 text-xs text-amber-300 hover:bg-amber-500/10">
          Test Fall
        </button>
        <span className="rounded-full border border-slate-700 px-3 py-1 text-xs text-slate-300">P1 · single patient</span>
      </header>

      <main className="grid gap-6 lg:grid-cols-[1fr_360px]">
        <section className={`relative overflow-hidden rounded-2xl border bg-black ${fall === "CONFIRMED" ? "border-red-500" : "border-slate-800"}`}>
          <img src="/api/video_feed" alt="live feed" className="block aspect-[4/3] w-full object-contain" />
          <div className="absolute left-3 top-3 flex items-center gap-2 rounded-full bg-slate-900/80 px-3 py-1 text-xs backdrop-blur">
            <span className={`h-2 w-2 rounded-full ${online ? "bg-emerald-400" : "bg-red-500"}`} />
            {online ? (t?.pose_visible ? `LIVE · ${t?.posture ?? "pose"}` : "LIVE · no pose") : "OFFLINE"}
          </div>
        </section>

        <aside className="flex flex-col gap-4">
          <CameraPicker active={t?.camera_index} backend={t?.camera_backend} error={t?.camera_error} />

          <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
            <div className="mb-2 text-xs uppercase tracking-wide text-slate-400">Pose engine</div>
            <div className="grid grid-cols-2 gap-1 rounded-lg bg-slate-800 p-1">
              {ENGINES.map((e) => (
                <button key={e.id} onClick={() => post("/api/config", { engine: e.id })}
                  className={`rounded-md px-2 py-1.5 text-sm transition ${t?.engine === e.id ? "bg-emerald-500 text-slate-950" : "text-slate-300 hover:bg-slate-700"}`}>
                  {e.label}
                </button>
              ))}
            </div>
            <div className="mt-2 text-xs text-slate-500">device: {t?.device ?? "…"}</div>
          </div>

          <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
            <label className="mb-2 block text-xs uppercase tracking-wide text-slate-400">Exercise</label>
            <div className="flex gap-2">
              <select value={t?.exercise ?? ""} onChange={(e) => post("/api/config", { exercise: e.target.value })}
                className="w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm">
                {exercises.map((x) => <option key={x}>{x}</option>)}
              </select>
              <button onClick={() => setGuideOpen(true)} disabled={!guide} title="Exercise Guide" aria-label="Exercise Guide"
                className="flex items-center rounded-lg border border-slate-700 bg-slate-800 px-3 text-sm hover:bg-slate-700 disabled:opacity-50">
                <Info size={14} />
              </button>
              <button onClick={() => post("/api/reset")} title="Reset session"
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

          {t?.error && (
            <div className="flex items-center gap-2 rounded-xl border border-red-900 bg-red-950/50 p-3 text-sm text-red-300">
              <Cpu size={16} /> {t.error}
            </div>
          )}
        </aside>
      </main>
      {guideOpen && guide && t && <ExerciseGuide info={guideInfo} initial={t.exercise} onClose={() => setGuideOpen(false)} />}
      {alert && (alert.state === "COUNTDOWN" || alert.state === "ALERTED") && <FallCheck alert={alert} />}
    </div>
  );
}
