/** Browser microphone for the "This device" mode: listens continuously, cuts speech into utterances, hands each one to
 *  the caller (which uploads it over the session socket for Groq's Whisper). Nothing is recorded to disk or kept.
 *  `pickAudioMime`, `Vad` and `rms` are pure and tested in voice.test.ts; `startVoiceCapture` needs a browser. */

/** In order of preference. Chrome / Firefox / Android: webm. iOS Safari: mp4 (AAC). The empty string at the end of
 *  pickAudioMime means "let the browser choose", which is still better than failing. */
export const AUDIO_MIME_PREFERENCE = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/aac"] as const;

export const pickAudioMime = (isSupported: (type: string) => boolean): string => AUDIO_MIME_PREFERENCE.find(isSupported) ?? "";

export function rms(samples: ArrayLike<number>): number {
  let sum = 0;
  for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
  return samples.length ? Math.sqrt(sum / samples.length) : 0;
}

export type VoiceIntent = "start_session" | "stop_session" | "show_preview" | "hide_preview" | "ok" | "help";

/** What a recognised command changes on the page: THE ONE PLACE both modes ("Server camera" and "This device") get their
 *  state transitions from, so they cannot differ.
 *  "Preview" means the EXERCISE GUIDE (the modal with the animated skeleton):
 *  - show / hide preview: open / close the guide.
 *  - start session: the server has reset the counters; the guide closes so the screen goes cleanly into tracking.
 *  - stop session: the session is over, so the guide closes too, and a phone also stops its camera (the server's own
 *    camera belongs to the server and keeps running).
 *  "ok" / "help" act on the server (they answer the fall check) and change nothing on the page. */
export function intentEffects(intent: string | null, deviceMode: boolean): { showGuide?: boolean; stopDeviceSession?: boolean } {
  switch (intent) {
    case "show_preview": return { showGuide: true };
    case "hide_preview":
    case "resume_session":      // "pause_session" changes nothing on the page but the Paused card (App)
    case "start_session": return { showGuide: false };
    case "stop_session": return { showGuide: false, stopDeviceSession: deviceMode };
    default: return {};
  }
}

/** How a recognised command is shown on the chip, so "heard" and "understood" can be told apart. */
export const intentLabel = (intent: string | null): string =>
  ({ show_preview: "show preview", hide_preview: "hide preview", start_session: "start session", stop_session: "stop session", pause_session: "pause", resume_session: "resume",
     ok: "I am fine", help: "help" } as Record<string, string>)[intent ?? ""] ?? "not a command";

/** Constraints for a speech microphone: the phone or laptop cancels what its own speaker plays. */
export const MIC_CONSTRAINTS = { echoCancellation: true, noiseSuppression: true, autoGainControl: true };
export const MIC_BLOCKED = "Microphone blocked. Allow it in the browser's site settings, then try again.";

export function micErrorText(e: unknown): string {
  const name = e instanceof DOMException ? e.name : "";
  return name === "NotAllowedError" ? MIC_BLOCKED : name === "NotFoundError" ? "No microphone found." : `Microphone unavailable (${name || "error"}).`;
}

/** Sounds the page itself makes (the countdown tick) must not be heard as speech. */
let quietUntil = 0;
export const holdQuiet = (ms: number) => { quietUntil = Math.max(quietUntil, performance.now() + ms); };
export const isQuiet = () => performance.now() < quietUntil;

export type VadEvent = "start" | "end" | "cancel" | "force" | null;

/** Energy-based speech detector. `feed` is called with the current loudness every ~50 ms:
 *  "start"  loudness rose above the threshold;
 *  "end"    it stayed below for `trailMs` after at least `minSpeechMs` of speech: one utterance is complete;
 *  "cancel" the same, but too short to be speech (a click, a cough);
 *  "force"  it ran past `maxMs`: cut here so one long sound cannot grow without bound.
 *  The threshold follows the room: 4x the running quiet level, never below `minThreshold`. */
export class Vad {
  speaking = false;
  floor = 0.004;
  private since = 0;
  private lastLoud = 0;
  constructor(private o = { minThreshold: 0.02, margin: 4, minSpeechMs: 200, trailMs: 700, maxMs: 8000 }) {}

  get threshold() { return Math.max(this.o.minThreshold, this.floor * this.o.margin); }

  reset() { this.speaking = false; }

  feed(level: number, now: number): VadEvent {
    const loud = level >= this.threshold;
    if (!this.speaking) {
      if (!loud) { this.floor = Math.min(0.02, 0.95 * this.floor + 0.05 * level); return null; }
      this.speaking = true; this.since = this.lastLoud = now;
      return "start";
    }
    if (loud) this.lastLoud = now;
    if (now - this.since >= this.o.maxMs) { this.speaking = false; return "force"; }
    if (now - this.lastLoud >= this.o.trailMs) {
      this.speaking = false;
      return this.lastLoud - this.since >= this.o.minSpeechMs ? "end" : "cancel";
    }
    return null;
  }
}

const IDLE_ROTATE_MS = 4000;      // with no speech, the running recorder is thrown away and restarted this often
const MIN_BLOB_BYTES = 1000;

export type Capture = { stop: () => void };

/** Starts listening on `stream` (audio tracks only). A MediaRecorder is always running, so the start of a sentence is
 *  already in it when the detector notices speech; when the sentence ends the recorder is stopped and its blob handed to
 *  `onUtterance`, then a fresh one starts. Returns null when the browser lacks MediaRecorder or Web Audio.
 *  `paused()` true (the page is speaking or beeping) throws away whatever is being heard. */
export function startVoiceCapture(
  stream: MediaStream,
  opts: { onUtterance: (blob: Blob) => void; paused: () => boolean; onLevel?: (level: number) => void },
): Capture | null {
  const Ctx: typeof AudioContext | undefined = window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  if (typeof MediaRecorder === "undefined" || !Ctx) return null;
  const mime = pickAudioMime((t) => MediaRecorder.isTypeSupported(t));
  const ctx = new Ctx();
  void ctx.resume().catch(() => {});
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 1024;
  ctx.createMediaStreamSource(stream).connect(analyser);        // not connected to the speakers: no feedback loop
  const buf = new Float32Array(analyser.fftSize);
  const vad = new Vad();

  type Rec = { r: MediaRecorder; parts: Blob[]; started: number };
  let cur: Rec | null = null;
  const begin = () => {
    try {
      const r = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
      const rec: Rec = { r, parts: [], started: performance.now() };
      r.ondataavailable = (e) => { if (e.data.size) rec.parts.push(e.data); };   // per recorder: a new one must not inherit these
      r.start();
      cur = rec;
    } catch { cur = null; }
  };
  const finish = (send: boolean) => {
    const rec = cur;
    cur = null;
    if (!rec || rec.r.state === "inactive") return;
    rec.r.onstop = () => {
      if (!send) return;
      const blob = new Blob(rec.parts, { type: rec.r.mimeType || mime || "audio/webm" });
      if (blob.size >= MIN_BLOB_BYTES) opts.onUtterance(blob);
    };
    rec.r.stop();
  };

  begin();
  const timer = window.setInterval(() => {
    const now = performance.now();
    if (opts.paused()) {
      if (vad.speaking) { vad.reset(); finish(false); begin(); }
      opts.onLevel?.(0);
      return;
    }
    analyser.getFloatTimeDomainData(buf);
    const level = rms(buf);
    opts.onLevel?.(level);
    const ev = vad.feed(level, now);
    if (ev === "end" || ev === "force") { finish(true); begin(); }
    else if (!vad.speaking && cur && now - (cur as Rec).started > IDLE_ROTATE_MS) { finish(false); begin(); }
    else if (!cur) begin();
  }, 50);

  return {
    stop: () => {
      window.clearInterval(timer);
      finish(false);
      void ctx.close().catch(() => {});
    },
  };
}
