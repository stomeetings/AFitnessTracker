import { Capacitor } from "@capacitor/core";
import { TextToSpeech } from "@capacitor-community/text-to-speech";

/** Spoken cues for the countdown, preview toggles and pause/resume: the native Capacitor TTS plugin inside the APK,
 *  window.speechSynthesis in a normal browser.
 *  Runs on timers and UI events only: nothing here is called from the per-frame pose loop. */

export const COUNTDOWN_STEPS = ["Ready...", "3", "2", "1", "Go!"] as const;
export const COUNTDOWN_STEP_MS = 1000;

export const PHRASES = {
  ready: "Ready", tick3: "3", tick2: "2", tick1: "1", go: "Go!",
  previewOpen: "OK, displaying preview", previewClose: "OK, closing preview",
  pause: "Paused", resume: "Resuming", end: "OK, closing session. Thank you!",
} as const;
export type CueName = keyof typeof PHRASES;

// Android Chrome / WebView drops an utterance nothing references (it is garbage collected mid-speech): live ones are held here.
const live = new Set<SpeechSynthesisUtterance>();

function utter(text: string, volume = 1, onDone?: () => void): SpeechSynthesisUtterance {
  const u = new SpeechSynthesisUtterance(text);
  u.lang = "en-US";
  u.volume = volume;
  live.add(u);
  u.onend = u.onerror = () => { live.delete(u); onDone?.(); };
  return u;
}

/** Speaks `text`. `urgent` replaces whatever is playing; otherwise it queues behind it, so "Ready" is not cut off by "3".
 *  Silent where speech is unsupported or blocked (the on-screen cue still shows). The promise resolves when the phrase has
 *  finished (or was cut off, failed, or is unsupported): never rejects. Most callers ignore it. */
export function speak(text: string, urgent = false): Promise<void> {
  return new Promise<void>((done) => {
    if (Capacitor.isNativePlatform()) {                // the APK: the phone's own TTS engine, no WebView speech support needed
      // The plugin's default queue strategy replaces what is playing, which is right for 1-second-apart countdown words.
      try {
        TextToSpeech.speak({ text, lang: "en-US", rate: 1.0, pitch: 1.0, volume: 1.0, category: "ambient" })
          .catch((e) => console.warn("[cues] native TTS failed", e))
          .finally(() => done());                      // the native call resolves when the phrase has been spoken
      } catch (e) { console.warn("[cues] native TTS failed", e); done(); }   // the UI flow never depends on speech
      return;
    }
    try {
      const s = window.speechSynthesis;
      if (!s) { done(); return; }
      const go = () => {
        afterCancel = false;
        try {
          if (s.paused) s.resume();                    // Chrome sometimes leaves the queue paused: before EVERY speech call
          s.speak(utter(text, 1, done));
        } catch (e) { console.warn("[cues] speech failed", e); done(); }
      };
      if (urgent) {
        // Windows/Chrome silently drops a speak() that follows cancel() in the same tick: let the cancel settle first.
        s.cancel(); live.clear(); afterCancel = true;
        window.setTimeout(go, 30);
      } else if (afterCancel) window.setTimeout(go, 30);   // equal timers run in order, so it still comes after the urgent one
      else go();
    } catch (e) { console.warn("[cues] speech failed", e); done(); }
  });
}
let afterCancel = false;                               // an urgent phrase is waiting out its 30 ms

export const cue = (name: CueName, urgent = false): Promise<void> => speak(PHRASES[name], urgent);

/** Waits for a cue to finish, but never longer than `maxMs` (a TTS engine that never answers must not hold the UI). */
export const cueThen = (name: CueName, maxMs: number, urgent = true): Promise<void> =>
  Promise.race([cue(name, urgent), new Promise<void>((r) => window.setTimeout(r, maxMs))]);

/** Mobile browsers and the Android WebView allow speech only after a touch, and the TTS engine starts slowly: every
 *  touch/click (capture phase, kept) sends one zero-volume utterance, which wakes the engine and unlocks speech, so a later
 *  timer or voice-command cue is heard at once. Returns a cleanup function. */
export function unlockSpeech(): () => void {
  if (Capacitor.isNativePlatform()) return () => {};   // native TTS needs no touch unlock
  let last = 0;
  const unlock = () => {
    const now = Date.now();
    if (now - last < 5000) return;                     // one wake-up is enough per few seconds, not one per tap
    last = now;
    try {
      const s = window.speechSynthesis;
      if (!s) return;
      if (s.paused) s.resume();
      s.speak(utter(" ", 0));
    } catch { /* ignore */ }
  };
  const events = ["pointerdown", "touchstart", "click", "keydown"] as const;
  for (const e of events) window.addEventListener(e, unlock, { capture: true, passive: true });
  return () => { for (const e of events) window.removeEventListener(e, unlock, true); };
}

/** Shows each step and says it one second apart, then calls `show(null)`. Returns a cancel function. */
export function runCountdown(show: (step: string | null) => void): () => void {
  let i = 0, timer = 0;
  const names: CueName[] = ["ready", "tick3", "tick2", "tick1", "go"];
  const tick = () => {
    if (i >= COUNTDOWN_STEPS.length) { show(null); return; }
    show(COUNTDOWN_STEPS[i]);
    cue(names[i], i === 0);                            // the first word clears older speech
    i++;
    timer = window.setTimeout(tick, i === COUNTDOWN_STEPS.length ? 800 : COUNTDOWN_STEP_MS);
  };
  tick();
  return () => { window.clearTimeout(timer); show(null); };
}
