/** What is kept on the device (IndexedDB, see db.ts). Plain JSON-able data so a later sync can upload it as it is. */

export type EventType = "start" | "stop" | "pause" | "resume" | "exercise" | "rep" | "fall" | "alert" | "end";

/** One thing that happened. `t` = seconds since the session began. */
export type EventRecord = { id?: number; sessionId: string; t: number; type: EventType; payload: Record<string, unknown> };

export type LimbSummary = { reps: number; bestRom: number; lastRom: number };
export type Calibration = { calibrated: boolean; rest: number; peak: number };

/** One Start .. Stop stretch of counting inside a session (Start resets the counts, so each one is kept on its own). */
export type WorkoutRecord = { startedAt: number; endedAt: number | null; exercise: string; limbs: Record<string, LimbSummary> };

/** One camera session on this device: from Start camera to Stop. Times are epoch milliseconds. */
export type SessionRecord = {
  id: string;                                   // uuid: the primary key, also what makes a later upload idempotent
  exercise: string;                             // the exercise at the last write
  startedAt: number;
  endedAt: number | null;                       // null while running (or if the app died: the events up to then are still there)
  limbs: Record<string, LimbSummary>;           // the counts at the last write: reps and range of motion per limb
  calibration: Record<string, Calibration>;     // the rest/peak angles in use per limb, and whether they were measured
  workouts: WorkoutRecord[];
  appVersion: string;
  synced: boolean;                              // false until the server has it (sync/syncQueue.ts)
  flagged?: boolean;                            // a fall or alert happened in it: uploaded before everything else
  syncFailed?: boolean;                         // the server refused this one as malformed (never retried; kept on the device)
};

/** The numbers the recorder reads from the engine whenever it writes. */
export type Summary = { exercise: string; limbs: Record<string, LimbSummary>; calibration: Record<string, Calibration> };

/** What the recorder needs from storage: one atomic write of the session and its new events. */
export interface SessionStore {
  commit(session: SessionRecord, events: EventRecord[]): Promise<void>;
}
