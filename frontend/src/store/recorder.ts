/** Writes what happens in a workout to the local store without ever touching the workout. Events are collected in memory and written
 *  in ONE transaction every `flushMs` (5 s), when the session ends, and when the page is hidden or closed. A failed write keeps the
 *  events for the next try (capped) and is logged, never thrown: the UI and the frame loop do not know it happened. */
import type { EventRecord, EventType, SessionRecord, SessionStore, Summary, WorkoutRecord } from "./types";

export const FLUSH_MS = 5000;
export const MAX_BUFFER = 5000;                    // events kept in memory while writes keep failing: the oldest go first
const WARN_EVERY_MS = 30000;

export type RecorderOptions = {
  flushMs?: number;
  now?: () => number;                              // epoch ms (tests pass a fake clock)
  newId?: () => string;
  appVersion?: string;
};

const defaultId = (): string => globalThis.crypto?.randomUUID?.() ?? `${Date.now().toString(16)}-${Math.random().toString(16).slice(2)}`;

export class SessionRecorder {
  private session: SessionRecord | null = null;
  private workout: WorkoutRecord | null = null;      // the open Start .. Stop stretch, if any
  private buffer: EventRecord[] = [];
  private timer: ReturnType<typeof setInterval> | undefined;
  private chain: Promise<void> = Promise.resolve();  // writes happen one after the other, in order
  private lastWarn = 0;
  private unhook: (() => void) | null = null;
  private readonly flushMs: number;
  private readonly now: () => number;
  private readonly newId: () => string;
  private readonly appVersion: string;

  /** `getStore` resolves to the store, or null when there is none (storage blocked): the recorder then does nothing. `summary` is
   *  read at every write, so the record always carries the latest counts. */
  constructor(private readonly getStore: () => Promise<SessionStore | null>, private readonly summary: () => Summary, opts: RecorderOptions = {}) {
    this.flushMs = opts.flushMs ?? FLUSH_MS;
    this.now = opts.now ?? Date.now;
    this.newId = opts.newId ?? defaultId;
    this.appVersion = opts.appVersion ?? "dev";
  }

  get id(): string | null { return this.session?.id ?? null; }
  get active(): boolean { return this.session !== null; }

  /** A camera session starts. Writes its record at once (so a crash leaves it behind) and starts the 5 s timer. */
  begin(): void {
    if (this.session) return;
    const s = this.read();
    this.session = {
      id: this.newId(), exercise: s.exercise, startedAt: this.now(), endedAt: null, limbs: s.limbs, calibration: s.calibration,
      workouts: [], appVersion: this.appVersion, synced: false,
    };
    this.timer = setInterval(() => { void this.flush(); }, this.flushMs);
    this.hookPage();
    void this.flush();
  }

  /** Something happened (an engine event). Cheap: it only appends to the buffer. */
  add(type: EventType, payload: Record<string, unknown> = {}): void {
    const s = this.session;
    if (!s) return;
    const at = this.now();
    this.buffer.push({ sessionId: s.id, t: Math.round(at - s.startedAt) / 1000, type, payload });
    if (this.buffer.length > MAX_BUFFER) this.buffer.splice(0, this.buffer.length - MAX_BUFFER);
    if (type === "fall" || type === "alert") s.flagged = true;        // safety first: such a session is uploaded before the others
    if (type === "start") {
      this.closeWorkout(at);
      this.workout = { startedAt: at, endedAt: null, exercise: this.read().exercise, limbs: {} };
      s.workouts.push(this.workout);
    } else if (type === "stop") this.closeWorkout(at);
  }

  private closeWorkout(at: number): void {
    if (!this.workout) return;
    this.workout.limbs = this.read().limbs;
    this.workout.endedAt = at;
    this.workout = null;
  }

  private read(): Summary {
    try { return this.summary(); } catch { return { exercise: this.session?.exercise ?? "", limbs: {}, calibration: {} }; }
  }

  /** Writes the buffered events and the session's current state in one transaction. Never rejects. */
  flush(): Promise<void> {
    const s = this.session;
    if (!s) return this.chain;
    const now = this.read();
    s.exercise = now.exercise; s.limbs = now.limbs; s.calibration = now.calibration;
    if (this.workout) this.workout.limbs = now.limbs;
    const record = JSON.parse(JSON.stringify(s)) as SessionRecord;       // a snapshot: the write happens later
    const events = this.buffer.splice(0);
    this.chain = this.chain.then(() => this.write(record, events));
    return this.chain;
  }

  private async write(record: SessionRecord, events: EventRecord[]): Promise<void> {
    try {
      const store = await this.getStore();
      if (!store) return;                                                // no storage: nothing is kept, nothing breaks
      await store.commit(record, events);
    } catch (e) {
      this.buffer = events.concat(this.buffer).slice(-MAX_BUFFER);       // keep them for the next write
      const at = this.now();
      if (at - this.lastWarn > WARN_EVERY_MS) { this.lastWarn = at; console.warn("[store] could not save the session, will retry", e); }
    }
  }

  /** The session ends (Stop, the camera went away, the page is closing): one last write, with `endedAt`. */
  end(): Promise<void> {
    const s = this.session;
    if (!s) return this.chain;
    this.add("end", {});
    this.closeWorkout(this.now());
    s.endedAt = this.now();
    const done = this.flush();
    this.session = null;
    clearInterval(this.timer); this.timer = undefined;
    this.unhook?.(); this.unhook = null;
    return done;
  }

  /** Hidden tab / closing page: the browser may not give us another chance, so write now. */
  private hookPage(): void {
    const doc = (globalThis as { document?: Document }).document;
    const onHide = () => { void this.flush(); };
    const onVisibility = () => { if (doc?.visibilityState === "hidden") onHide(); };
    doc?.addEventListener?.("visibilitychange", onVisibility);
    globalThis.addEventListener?.("pagehide", onHide);
    this.unhook = () => { doc?.removeEventListener?.("visibilitychange", onVisibility); globalThis.removeEventListener?.("pagehide", onHide); };
  }
}
