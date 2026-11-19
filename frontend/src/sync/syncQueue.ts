/** Uploads the sessions saved on this device (store/) to the backend when it can be reached, in the background, safely.
 *
 *  - Starts on request (`trigger`): the app opening, the browser coming back online, the native network / app-resume events, and a
 *    60 s timer while anything waits (wired in sync/index.ts).
 *  - Never trusts `navigator.onLine`: every attempt first asks GET /api/health with a 2 s timeout.
 *  - Never competes with the camera: while a workout session runs it does nothing (it asks again in 5 s), and each batch waits for an
 *    idle moment (requestIdleCallback, a short timeout where that does not exist) before the heavy part.
 *  - Sends up to 20 sessions per request, those with a fall or alert first. A session is marked synced only after a 2xx answer. A
 *    network or server (5xx / 429) failure backs off exponentially with jitter (`reconnectDelay`); a wrong token waits for the
 *    next trigger; a request the server calls malformed is split, and one session it refuses alone is set aside (`syncFailed`).
 *  - Synced sessions older than 30 days are deleted. */
import { reconnectDelay } from "../remote/logic";
import type { EventRecord, SessionRecord } from "../store/types";

export const BATCH_SIZE = 20;
export const MAX_BODY_CHARS = 900_000;               // the server refuses 1,000,000 bytes: stay below, splitting a batch when needed
export const HEALTH_TIMEOUT_MS = 2000;
export const UPLOAD_TIMEOUT_MS = 30_000;
export const BUSY_RETRY_MS = 5000;
export const INTERVAL_MS = 60_000;
export const PRUNE_AFTER_MS = 30 * 24 * 3600 * 1000;

export type SyncState = "idle" | "syncing" | "offline" | "auth" | "error";
export type SyncStatus = { pending: number; state: SyncState };

/** What the queue needs from the store (RehabStore implements it). */
export interface SyncStore {
  listUnsynced(): Promise<SessionRecord[]>;
  listEvents(sessionId: string): Promise<EventRecord[]>;
  markSynced(ids: string[]): Promise<void>;
  markFailed(ids: string[]): Promise<void>;
  pruneSynced(maxAgeMs: number): Promise<number>;
  deviceId(): Promise<string>;
  setMeta(key: string, value: unknown): Promise<void>;
}

export type SyncDeps = {
  store: () => Promise<SyncStore | null>;
  fetch: (url: string, init?: RequestInit) => Promise<Response>;
  url: (path: string) => string;
  token: () => string;
  /** True while a workout session is running: no uploads then. */
  busy: () => boolean;
  /** Runs `cb` when the page has nothing better to do. */
  idle: (cb: () => void) => void;
  now?: () => number;
  random?: () => number;
};

/** "All synced" / "3 sessions waiting to sync" (with the reason when something is wrong). */
export function syncText(s: SyncStatus): string {
  if (s.pending === 0) return "All synced";
  const n = `${s.pending} session${s.pending === 1 ? "" : "s"} waiting to sync`;
  return s.state === "offline" ? `${n} · offline` : s.state === "auth" ? `${n} · access token needed` : s.state === "error" ? `${n} · will retry` : n;
}

/** Fall / alert sessions first, otherwise oldest first (the store lists oldest first and sort is stable). */
export const prioritize = (sessions: SessionRecord[]): SessionRecord[] => [...sessions].sort((a, b) => Number(!!b.flagged) - Number(!!a.flagged));

export class SyncQueue {
  private status: SyncStatus = { pending: 0, state: "idle" };
  private readonly listeners = new Set<() => void>();
  private running = false;
  private again = false;
  private attempts = 0;
  private retry: ReturnType<typeof setTimeout> | undefined;
  private interval: ReturnType<typeof setInterval> | undefined;
  private readonly now: () => number;
  private readonly random: () => number;

  constructor(private readonly deps: SyncDeps) {
    this.now = deps.now ?? Date.now;
    this.random = deps.random ?? Math.random;
  }

  // ---- status for the UI (useSyncExternalStore): a new object only when something changed -----------------------------------------
  getStatus = (): SyncStatus => this.status;
  subscribe = (fn: () => void): (() => void) => { this.listeners.add(fn); return () => { this.listeners.delete(fn); }; };

  private set(next: Partial<SyncStatus>): void {
    const merged = { ...this.status, ...next };
    if (merged.pending === this.status.pending && merged.state === this.status.state) return;
    this.status = merged;
    // the 60 s timer exists only while something waits
    if (merged.pending > 0 && !this.interval) this.interval = setInterval(() => this.trigger(), INTERVAL_MS);
    else if (merged.pending === 0 && this.interval) { clearInterval(this.interval); this.interval = undefined; }
    for (const fn of this.listeners) fn();
  }

  /** Ask for a sync. Cheap and safe to call from anywhere, any number of times: calls made during a run are folded into one more pass. */
  trigger(): void {
    clearTimeout(this.retry); this.retry = undefined;
    if (this.running) { this.again = true; return; }
    void this.run();
  }

  /** Stops the timers (tests, hot reload). */
  dispose(): void {
    clearTimeout(this.retry); clearInterval(this.interval);
    this.retry = undefined; this.interval = undefined; this.listeners.clear();
  }

  private later(ms: number): void {
    clearTimeout(this.retry);
    this.retry = setTimeout(() => this.trigger(), ms);
  }

  /** A failed pass: remember why and try again after an exponential, jittered wait. */
  private failed(state: SyncState): void {
    this.set({ state });
    this.later(reconnectDelay(this.attempts++) * (1 + this.random() * 0.5));
  }

  private async run(): Promise<void> {
    this.running = true;
    try {
      do { this.again = false; await this.cycle(); } while (this.again);
    } catch (e) {
      console.warn("[sync] stopped by an error", e);          // never leaves the queue stuck "running", never reaches the UI
      this.failed("error");
    } finally { this.running = false; }
  }

  private async cycle(): Promise<void> {
    const store = await this.deps.store();
    if (!store) return;                                       // no local storage: nothing to upload
    let pending = await store.listUnsynced();
    this.set({ pending: pending.length });
    if (!pending.length) { this.set({ state: "idle" }); this.attempts = 0; return; }
    if (this.deps.busy()) { this.later(BUSY_RETRY_MS); return; }       // a workout is running: the camera loop comes first
    if (!(await this.reachable())) { this.failed("offline"); return; }
    this.set({ state: "syncing" });
    const device = await store.deviceId();
    while (pending.length) {
      if (this.deps.busy()) { this.set({ state: "idle" }); this.later(BUSY_RETRY_MS); return; }    // a workout started meanwhile
      await new Promise<void>((resolve) => this.deps.idle(resolve));
      const outcome = await this.send(store, device, prioritize(pending).slice(0, BATCH_SIZE));
      if (outcome !== "ok") { outcome === "auth" ? this.set({ state: "auth" }) : this.failed("error"); return; }
      pending = await store.listUnsynced();
      this.set({ pending: pending.length });
    }
    try { await store.pruneSynced(PRUNE_AFTER_MS); await store.setMeta("lastSync", this.now()); } catch { /* housekeeping only */ }
    this.attempts = 0;
    this.set({ state: "idle" });
  }

  /** GET /api/health within 2 s: the real test of "can the server be reached", which `navigator.onLine` is not. */
  private async reachable(): Promise<boolean> {
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), HEALTH_TIMEOUT_MS);
    try {
      const r = await this.deps.fetch(this.deps.url("/api/health"), { signal: ctl.signal, cache: "no-store" });
      return r.ok;
    } catch { return false; } finally { clearTimeout(timer); }
  }

  /** Uploads `batch` (split while the body would be too large or the server calls it malformed). "ok" = all of it is on the server
   *  (or set aside as refused); "auth" = wrong or missing token; "retry" = try again later. */
  private async send(store: SyncStore, device: string, batch: SessionRecord[]): Promise<"ok" | "auth" | "retry"> {
    const sessions = [];
    for (const s of batch) {
      const events = await store.listEvents(s.id);
      sessions.push({
        id: s.id, exercise: s.exercise, startedAt: s.startedAt, endedAt: s.endedAt, limbs: s.limbs, calibration: s.calibration,
        workouts: s.workouts, appVersion: s.appVersion, events: events.map((e) => ({ t: e.t, type: e.type, payload: e.payload })),
      });
    }
    const body = JSON.stringify({ deviceId: device, sessions });
    if (body.length > MAX_BODY_CHARS && batch.length > 1) return this.sendHalves(store, device, batch);
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), UPLOAD_TIMEOUT_MS);
    try {
      const token = this.deps.token();
      const res = await this.deps.fetch(this.deps.url("/api/sync/sessions"), {
        method: "POST", body, signal: ctl.signal,
        headers: { "Content-Type": "application/json", ...(token ? { Authorization: `Bearer ${token}` } : {}) },
      });
      if (res.ok) { await store.markSynced(batch.map((s) => s.id)); return "ok"; }               // 2xx: the server has them (new or already known)
      if (res.status === 401 || res.status === 403) return "auth";
      if (res.status === 400 || res.status === 413 || res.status === 422) {
        if (batch.length > 1) return this.sendHalves(store, device, batch);                         // find the one it does not like
        console.warn(`[sync] the server refused session ${batch[0].id} (${res.status}); it stays on this device`);
        await store.markFailed([batch[0].id]);
        return "ok";
      }
      return "retry";                                                                              // 429, 5xx
    } catch { return "retry"; } finally { clearTimeout(timer); }
  }

  private async sendHalves(store: SyncStore, device: string, batch: SessionRecord[]): Promise<"ok" | "auth" | "retry"> {
    const mid = Math.ceil(batch.length / 2);
    for (const half of [batch.slice(0, mid), batch.slice(mid)]) {
      const r = await this.send(store, device, half);
      if (r !== "ok") return r;
    }
    return "ok";
  }
}

/** requestIdleCallback where it exists (Chrome / the Android WebView), else a short timeout. The 2 s deadline means a busy page still
 *  gets its turn eventually. */
export const whenIdle = (cb: () => void): void => {
  const w = globalThis as { requestIdleCallback?: (cb: () => void, o?: { timeout: number }) => number };
  if (typeof w.requestIdleCallback === "function") w.requestIdleCallback(cb, { timeout: 2000 });
  else setTimeout(cb, 50);
};
