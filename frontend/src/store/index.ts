import type { LocalEngine } from "../engine/localEngine";
import { RehabStore } from "./db";
import { SessionRecorder } from "./recorder";
import type { Summary } from "./types";

export { RehabStore } from "./db";
export { SessionRecorder } from "./recorder";
export type * from "./types";

/** Asks the browser not to evict this site's storage under pressure (a WebView does, with no warning). Call once at start-up;
 *  best effort, the answer is only logged. */
export async function requestPersistence(): Promise<boolean> {
  try {
    const s = navigator.storage;
    if (!s?.persist) return false;
    if (await s.persisted?.()) return true;
    const granted = await s.persist();
    console.info(`[store] persistent storage ${granted ? "granted" : "not granted (the browser may still evict it under pressure)"}`);
    return granted;
  } catch { return false; }
}

let opening: Promise<RehabStore | null> | null = null;
/** The one shared store, opened on first use. null (and one warning) when IndexedDB is unavailable: the app works without history. */
export function getStore(): Promise<RehabStore | null> {
  opening ??= RehabStore.open().catch((e) => { console.warn("[store] no local storage, sessions will not be kept", e); return null; });
  return opening;
}

/** The recorder's view of an engine: the counts, range of motion and calibration per limb. Called every few seconds, not per frame. */
export function engineSummary(eng: LocalEngine): Summary {
  const out: Summary = { exercise: eng.pipeline.exercise, limbs: {}, calibration: {} };
  const snap = eng.pipeline.reps.snapshot();
  for (const side in snap) {
    const l = snap[side];
    out.limbs[side] = { reps: l.reps, bestRom: l.best_rom, lastRom: l.last_rom };
    out.calibration[side] = { calibrated: l.calibrated, rest: l.rest, peak: l.peak };
  }
  return out;
}

/** A recorder for `eng` that writes to the shared store: begin() it, and call end() when the session stops. The engine's events
 *  feed it from here on. */
export function recordEngine(eng: LocalEngine, appVersion = (import.meta.env.VITE_APP_VERSION as string | undefined) ?? "dev"): SessionRecorder {
  const rec = new SessionRecorder(getStore, () => engineSummary(eng), { appVersion });
  eng.listener = (e) => rec.add(e.type, e.payload);
  rec.begin();
  return rec;
}
