import { useEffect, useRef, useState } from "react";
import { AlertTriangle, VolumeX } from "lucide-react";
import { holdQuiet } from "./remote/voice";
import { apiUrl } from "./net";
import type { FallAlert, VoiceState } from "./useTelemetry";

/** Full-screen "Are you okay?" check shown while the backend's fall escalation is in COUNTDOWN or ALERTED.
 *  The backend owns the timer; this only displays `seconds_left`, makes noise, and posts the answers.
 *  Mounted only while needed, so the audio context and timers exist only during a fall check.
 *  Voice: once the spoken prompt has FINISHED it asks the backend to open the microphone (the reply is
 *  transcribed there by Groq's Whisper); what it heard comes back in telemetry. The buttons always work, voice is an extra. */

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

/** `onDone` runs when the utterance ends or fails (also if speech is unsupported). */
function say(text: string, onDone?: () => void) {
  if (!window.speechSynthesis) { onDone?.(); return; }
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.lang = "en-US";
  if (onDone) { u.onend = onDone; u.onerror = onDone; }
  speechSynthesis.speak(u);
}

const post = (url: string) => { fetch(apiUrl(url), { method: "POST" }).catch(() => {}); };

// DEBUG (temporary): the browser records no audio. Its only part in voice is this request; the microphone is opened
// by the backend (see the "voice:" lines in the backend terminal).
const startListening = () => {
  console.log("[voice] DEBUG: asking the backend to open the microphone");
  fetch(apiUrl("/api/fall/listen"), { method: "POST" })
    .then((r) => r.json())
    .then((j) => console.log("[voice] DEBUG: /api/fall/listen answered", j))
    .catch((e) => console.warn("[voice] DEBUG: /api/fall/listen failed", e));
};

/** `onAck` / `onEscalate` default to the server-camera endpoints; the browser-camera mode sends them over its socket
 *  instead. `canListen` is false there: the microphone is opened by the backend, which has none for a remote patient. */
export function FallCheck({ alert, voice, onAck = () => post("/api/fall/ack"), onEscalate = () => post("/api/fall/escalate"), canListen = true }:
  { alert: FallAlert; voice?: VoiceState; onAck?: () => void; onEscalate?: () => void; canListen?: boolean }) {
  const ref = useRef<HTMLDialogElement>(null);
  const audio = useRef<AudioContext | null>(null);
  const speakId = useRef(0);
  const [blocked, setBlocked] = useState(false);        // browsers mute pages that have not been clicked yet
  const alerted = alert.state === "ALERTED";
  const total = 10;                                     // matches COUNTDOWN_S in backend escalation.py
  const second = Math.ceil(alert.seconds_left);
  const again = !alerted && (alert.rechecks ?? 0) > 0;  // asked again after an earlier "I'm OK"

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
    // Echo protection: the prompt itself contains the words "I'm OK", so the microphone must stay closed until
    // it has finished. The id check ignores the end event of a prompt that was cancelled by a newer one.
    const id = ++speakId.current;
    let timer: number | undefined;
    const listen = () => { if (canListen && id === speakId.current) startListening(); };
    timer = window.setTimeout(() => {
      console.warn("[voice] DEBUG: the spoken prompt reported no end after 6 s, opening the microphone anyway");
      listen();
    }, 6000);            // fallback: speech blocked or never reports its end
    say(alerted ? "No response. Please check on the patient now." : "Are you okay? Press I'm OK, or stand up.", () => {
      if (id !== speakId.current) return;
      window.clearTimeout(timer);
      console.log("[voice] DEBUG: spoken prompt finished, opening the microphone in 300 ms");
      timer = window.setTimeout(listen, 300);           // short guard so the tail of the prompt is not heard
    });
    return () => { speakId.current++; window.clearTimeout(timer); };
  }, [alerted]);
  useEffect(() => {                                                       // DEBUG (temporary): what the backend reports
    console.log("[voice] DEBUG: backend voice status:", voice?.status ?? "(none)", voice?.error ?? "",
      voice?.transcript ? `heard: ${JSON.stringify(voice.transcript)} -> ${voice.intent}` : "");
  }, [voice?.status, voice?.error, voice?.transcript, voice?.intent]);       // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {                                                                          // one tick per second
    if (alerted) return;
    beep(audio.current, 880, 0.15);
    holdQuiet(400);                      // the browser microphone (device mode) must not take the tick for speech
  }, [second, alerted]);
  useEffect(() => {                                                                          // two-tone alarm
    if (!alerted) return;
    let high = false;
    const id = window.setInterval(() => { high = !high; beep(audio.current, high ? 960 : 720, 0.4); }, 450);
    return () => window.clearInterval(id);
  }, [alerted]);

  const ack = onAck;
  const escalate = onEscalate;

  return (
    <dialog ref={ref} role="alertdialog" aria-labelledby="fall-check-title"
      onCancel={(e) => e.preventDefault()}                       /* Esc must not silently dismiss an alarm */
      onClick={() => { void audio.current?.resume(); }}          /* any click unlocks sound if it was blocked */
      className={`m-0 h-screen max-h-none w-screen max-w-none overflow-y-auto text-slate-100 backdrop:bg-black ${alerted ? "bg-red-950" : "bg-slate-950"}`}>
      <div className="flex min-h-full flex-col items-center justify-center gap-8 p-6 text-center">
        <AlertTriangle size={56} className={alerted ? "animate-pulse text-red-400" : "text-amber-400"} />
        <h2 id="fall-check-title" className="text-5xl font-bold sm:text-6xl">
          {alerted ? "No response" : again ? "Still down — are you okay?" : "Are you okay?"}
        </h2>
        <div className="flex flex-col items-center gap-3">
          <p className="max-w-xl text-lg text-slate-300">
            {alerted
              ? "A fall was detected and nobody answered. Check on the patient now."
              : "A possible fall was detected. Press the button or stand up to cancel the alarm."}
          </p>
          {voice && (
            <div role="status" aria-live="polite" className="flex items-center gap-2 text-base text-slate-300">
              {voice.status === "starting" || voice.status === "listening" || voice.status === "transcribing" ? (
                <>
                  <span className="h-2.5 w-2.5 animate-pulse rounded-full bg-emerald-400" />
                  {voice.status === "starting" ? "Starting microphone…"
                    : voice.status === "transcribing" ? "Processing…"
                    : voice.transcript ? `Heard: "${voice.transcript}"` : "Listening… say “I am fine” or “help”"}
                </>
              ) : voice.status === "unavailable" || voice.status === "off" ? (
                <span className="text-slate-400" title={voice.error ?? undefined}>Voice unavailable — use the buttons</span>
              ) : null}
            </div>
          )}
          {voice?.warning && <p className="max-w-md text-sm text-amber-300">{voice.warning}</p>}
        </div>
        {!alerted && (
          <svg viewBox="0 0 220 220" className="h-52 w-52" role="timer" aria-label={`${second} seconds left`}>
            <circle cx="110" cy="110" r={RING_R} fill="none" stroke="#1e293b" strokeWidth="12" />
            <circle cx="110" cy="110" r={RING_R} fill="none" stroke="#fbbf24" strokeWidth="12" strokeLinecap="round"
              strokeDasharray={RING_C} strokeDashoffset={RING_C * (1 - Math.min(1, alert.seconds_left / total))}
              transform="rotate(-90 110 110)" style={{ transition: "stroke-dashoffset 0.1s linear" }} />
            <text x="110" y="110" textAnchor="middle" dominantBaseline="central" fill="#f8fafc" fontSize="72" fontWeight="700">{second}</text>
          </svg>
        )}
        <div className="flex flex-wrap items-center justify-center gap-4">
          <button onClick={ack} autoFocus
            className="rounded-2xl bg-emerald-500 px-12 py-6 text-3xl font-bold text-slate-950 hover:bg-emerald-400 focus:outline-none focus:ring-4 focus:ring-emerald-300">
            {alerted ? "I'm OK — stop alarm" : "I'm OK"}
          </button>
          {!alerted && (
            <button onClick={escalate}
              className="rounded-2xl border-2 border-red-500 px-8 py-6 text-2xl font-bold text-red-300 hover:bg-red-500/10 focus:outline-none focus:ring-4 focus:ring-red-400">
              Trigger Emergency
            </button>
          )}
        </div>
        {blocked && (
          <p className="flex items-center gap-2 text-sm text-amber-300"><VolumeX size={16} /> Sound is blocked by the browser. Click anywhere to enable it.</p>
        )}
      </div>
    </dialog>
  );
}
