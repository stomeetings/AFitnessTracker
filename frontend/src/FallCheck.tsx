import { useEffect, useRef, useState } from "react";
import { AlertTriangle, VolumeX } from "lucide-react";
import type { FallAlert } from "./useTelemetry";

/** Full-screen "Are you okay?" check shown while the backend's fall escalation is in COUNTDOWN or ALERTED.
 *  The backend owns the timer; this only displays `seconds_left`, makes noise, and posts the acknowledgement.
 *  Mounted only while needed, so the audio context and timers exist only during a fall check. */

const RING_R = 96, RING_C = 2 * Math.PI * RING_R;

function beep(ctx: AudioContext | null, freq: number, seconds: number) {
  if (!ctx || ctx.state !== "running") return;          // browser has not allowed sound yet
  const osc = ctx.createOscillator(), gain = ctx.createGain();
  osc.type = "square";
  osc.frequency.value = freq;
  gain.gain.setValueAtTime(0.2, ctx.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + seconds);
  osc.connect(gain).connect(ctx.destination);
  osc.start();
  osc.stop(ctx.currentTime + seconds);
}

function say(text: string) {
  if (!window.speechSynthesis) return;
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.lang = "en-US";
  speechSynthesis.speak(u);
}

export function FallCheck({ alert }: { alert: FallAlert }) {
  const ref = useRef<HTMLDialogElement>(null);
  const audio = useRef<AudioContext | null>(null);
  const [blocked, setBlocked] = useState(false);        // browsers mute pages that have not been clicked yet
  const alerted = alert.state === "ALERTED";
  const total = 10;                                     // matches COUNTDOWN_S in backend escalation.py
  const second = Math.ceil(alert.seconds_left);

  useEffect(() => {
    const d = ref.current;
    if (d && !d.open) d.showModal();
    const ctx = new AudioContext();
    audio.current = ctx;
    const sync = () => setBlocked(ctx.state !== "running");
    ctx.onstatechange = sync;
    ctx.resume().catch(() => {});
    sync();
    return () => { ctx.onstatechange = null; void ctx.close(); window.speechSynthesis?.cancel(); };
  }, []);

  useEffect(() => {
    say(alerted ? "No response. Please check on the patient now." : "Are you okay? Press I'm OK, or stand up.");
  }, [alerted]);
  useEffect(() => { if (!alerted) beep(audio.current, 880, 0.15); }, [second, alerted]);    // one tick per second
  useEffect(() => {                                                                          // two-tone alarm
    if (!alerted) return;
    let high = false;
    const id = window.setInterval(() => { high = !high; beep(audio.current, high ? 960 : 720, 0.4); }, 450);
    return () => window.clearInterval(id);
  }, [alerted]);

  const ack = () => { fetch("/api/fall/ack", { method: "POST" }).catch(() => {}); };

  return (
    <dialog ref={ref} role="alertdialog" aria-labelledby="fall-check-title"
      onCancel={(e) => e.preventDefault()}                       /* Esc must not silently dismiss an alarm */
      onClick={() => { void audio.current?.resume(); }}          /* any click unlocks sound if it was blocked */
      className={`m-0 h-screen max-h-none w-screen max-w-none text-slate-100 backdrop:bg-black ${alerted ? "bg-red-950" : "bg-slate-950"}`}>
      <div className="flex h-full flex-col items-center justify-center gap-8 p-6 text-center">
        <AlertTriangle size={56} className={alerted ? "animate-pulse text-red-400" : "text-amber-400"} />
        <h2 id="fall-check-title" className="text-5xl font-bold sm:text-6xl">{alerted ? "No response" : "Are you okay?"}</h2>
        <p className="max-w-xl text-lg text-slate-300">
          {alerted
            ? "A fall was detected and nobody answered. Check on the patient now."
            : "A possible fall was detected. Press the button or stand up to cancel the alarm."}
        </p>
        {!alerted && (
          <svg viewBox="0 0 220 220" className="h-52 w-52" role="timer" aria-label={`${second} seconds left`}>
            <circle cx="110" cy="110" r={RING_R} fill="none" stroke="#1e293b" strokeWidth="12" />
            <circle cx="110" cy="110" r={RING_R} fill="none" stroke="#fbbf24" strokeWidth="12" strokeLinecap="round"
              strokeDasharray={RING_C} strokeDashoffset={RING_C * (1 - Math.min(1, alert.seconds_left / total))}
              transform="rotate(-90 110 110)" style={{ transition: "stroke-dashoffset 0.1s linear" }} />
            <text x="110" y="110" textAnchor="middle" dominantBaseline="central" fill="#f8fafc" fontSize="72" fontWeight="700">{second}</text>
          </svg>
        )}
        <button onClick={ack} autoFocus
          className="rounded-2xl bg-emerald-500 px-12 py-6 text-3xl font-bold text-slate-950 hover:bg-emerald-400 focus:outline-none focus:ring-4 focus:ring-emerald-300">
          {alerted ? "I'm OK — stop alarm" : "I'm OK"}
        </button>
        {blocked && (
          <p className="flex items-center gap-2 text-sm text-amber-300"><VolumeX size={16} /> Sound is blocked by the browser. Click anywhere to enable it.</p>
        )}
      </div>
    </dialog>
  );
}
