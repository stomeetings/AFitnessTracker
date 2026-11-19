import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { LocalEngine, type EngineEvent } from "../engine/localEngine";
import { SessionRecorder } from "./recorder";
import { engineSummary } from "./index";
import type { EventRecord, SessionRecord, SessionStore, Summary } from "./types";

/** An in-memory store: every commit is kept, and one can be made to fail. */
function fakeStore() {
  const commits: Array<{ session: SessionRecord; events: EventRecord[] }> = [];
  const store: SessionStore & { fail: number; commits: typeof commits } = {
    fail: 0, commits,
    async commit(session, events) {
      if (this.fail > 0) { this.fail--; throw new Error("disk full"); }
      commits.push({ session, events });
    },
  };
  return store;
}

const summary = (reps = 0): Summary => ({
  exercise: "Elbow Curl", limbs: { L: { reps, bestRom: 100, lastRom: 90 }, R: { reps, bestRom: 95, lastRom: 90 } },
  calibration: { L: { calibrated: false, rest: 160, peak: 50 }, R: { calibrated: false, rest: 160, peak: 50 } },
});

let clock = 1_000_000;
beforeEach(() => { vi.useFakeTimers(); clock = 1_000_000; });
afterEach(() => { vi.useRealTimers(); vi.restoreAllMocks(); });
const make = (store: ReturnType<typeof fakeStore> | null, sum: () => Summary = () => summary()) =>
  new SessionRecorder(async () => store, sum, { now: () => clock, newId: () => "s1", appVersion: "9.9" });

describe("SessionRecorder", () => {
  it("writes the session at once, then the buffered events together every 5 seconds", async () => {
    const store = fakeStore(), rec = make(store);
    rec.begin();
    await vi.advanceTimersByTimeAsync(0);
    expect(store.commits.length).toBe(1);
    expect(store.commits[0].session).toMatchObject({ id: "s1", startedAt: 1_000_000, endedAt: null, synced: false, appVersion: "9.9", exercise: "Elbow Curl" });
    clock += 1500; rec.add("start", { calibrated: false });
    clock += 2000; rec.add("rep", { total: 1 });
    clock += 500; rec.add("fall", { status: "SUSPECTED" });
    expect(store.commits.length).toBe(1);                              // nothing is written per event
    await vi.advanceTimersByTimeAsync(5000);
    expect(store.commits.length).toBe(2);                              // one transaction with all three
    expect(store.commits[1].events.map((e) => [e.type, e.t])).toEqual([["start", 1.5], ["rep", 3.5], ["fall", 4]]);
    expect(store.commits[1].events.every((e) => e.sessionId === "s1")).toBe(true);
    await vi.advanceTimersByTimeAsync(5000);
    expect(store.commits.length).toBe(3);                              // the session record is refreshed even with no events
    expect(store.commits[2].events).toEqual([]);
    await rec.end();
  });

  it("writes on demand (a hidden tab) and at the end, with the end time and the final counts", async () => {
    const store = fakeStore();
    let reps = 0;
    const rec = make(store, () => summary(reps));
    rec.begin();
    clock += 1000; rec.add("start");
    reps = 4; clock += 9000; rec.add("rep", { total: 4 });
    await rec.flush();
    expect(store.commits.at(-1)?.session.limbs.L.reps).toBe(4);
    clock += 1000; rec.add("stop");
    await rec.end();
    const last = store.commits.at(-1)!;
    expect(last.session.endedAt).toBe(clock);
    expect(last.events.map((e) => e.type)).toEqual(["stop", "end"]);
    expect(last.session.workouts).toEqual([{ startedAt: 1_001_000, endedAt: clock, exercise: "Elbow Curl", limbs: summary(4).limbs }]);
    expect(rec.active).toBe(false);
  });

  it("keeps each Start..Stop stretch on its own, because Start resets the counts", async () => {
    const store = fakeStore();
    let reps = 0;
    const rec = make(store, () => summary(reps));
    rec.begin();
    clock += 1000; rec.add("start"); reps = 3;
    clock += 1000; rec.add("stop");
    clock += 1000; reps = 0; rec.add("start"); reps = 2;
    await rec.end();
    const w = store.commits.at(-1)!.session.workouts;
    expect(w.map((x) => x.limbs.L.reps)).toEqual([3, 2]);
    expect(w.every((x) => x.endedAt !== null)).toBe(true);
  });

  it("a failed write is logged, never thrown, and its events are written with the next one", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const store = fakeStore(), rec = make(store);
    store.fail = 1;
    rec.begin();                                                       // this first write fails
    await vi.advanceTimersByTimeAsync(0);
    expect(warn).toHaveBeenCalledTimes(1);
    clock += 1000; rec.add("rep", { total: 1 });
    await expect(rec.flush()).resolves.toBeUndefined();
    expect(store.commits.length).toBe(1);
    expect(store.commits[0].events.map((e) => e.type)).toEqual(["rep"]);
    store.fail = 2;
    rec.add("rep", { total: 2 });
    await rec.flush();                                                  // fails again: kept
    rec.add("rep", { total: 3 });
    await rec.flush();                                                  // fails again: kept
    await rec.flush();
    expect(store.commits.at(-1)!.events.map((e) => e.payload.total)).toEqual([2, 3]);
    expect(warn.mock.calls.length).toBeLessThanOrEqual(2);              // rate limited: not one line per failure
  });

  it("does nothing, and says nothing, when there is no storage or the summary throws", async () => {
    const rec = make(null, () => { throw new Error("boom"); });
    rec.begin();
    rec.add("rep");
    await expect(rec.end()).resolves.toBeUndefined();
  });

  it("ignores events before begin() and after end(), and a second begin()", async () => {
    const store = fakeStore(), rec = make(store);
    rec.add("rep");
    rec.begin(); rec.begin();
    await rec.end();
    rec.add("rep");
    await rec.flush();
    expect(store.commits.length).toBe(2);                              // the first write, and the final one
    expect(store.commits.flatMap((c) => c.events).map((e) => e.type)).toEqual(["end"]);
  });
});

describe("the engine feeds the recorder", () => {
  const standing: Record<string, [number, number, number]> = {
    nose: [320, 130, 0.9], l_shoulder: [290, 160, 0.9], r_shoulder: [350, 160, 0.9], l_hip: [300, 260, 0.9], r_hip: [340, 260, 0.9],
    l_knee: [300, 350, 0.9], r_knee: [340, 350, 0.9], l_ankle: [300, 440, 0.9], r_ankle: [340, 440, 0.9],
    l_elbow: [290, 220, 0.9], r_elbow: [350, 220, 0.9], l_wrist: [290, 275, 0.9], r_wrist: [350, 275, 0.9],
  };

  it("reports commands, a fall and its alert, and nothing for an ordinary frame", () => {
    const eng = new LocalEngine("Elbow Curl"), seen: EngineEvent[] = [];
    eng.listener = (e) => seen.push(e);
    for (let i = 0; i < 30; i++) eng.process(standing, undefined, 640, 480, i / 30);
    expect(seen).toEqual([]);                                          // frames alone are silent
    eng.start(); eng.pause(); eng.resume(); eng.stop();
    expect(seen.map((e) => e.type)).toEqual(["start", "pause", "resume", "stop"]);
    seen.length = 0;
    eng.simulateFall();
    eng.process(standing, undefined, 640, 480, 2);
    expect(seen.map((e) => [e.type, e.payload.state])).toEqual([["alert", "COUNTDOWN"]]);
    eng.ack();
    eng.process(standing, undefined, 640, 480, 2.1);
    expect(seen.map((e) => e.payload.state)).toEqual(["COUNTDOWN", "ACKNOWLEDGED"]);
  });

  it("a listener that throws cannot disturb the workout", () => {
    const eng = new LocalEngine("Elbow Curl");
    eng.listener = () => { throw new Error("recorder bug"); };
    expect(() => { eng.start(); eng.pause(); eng.resume(); }).not.toThrow();
    expect(eng.pipeline.counting).toBe(true);
  });

  it("the summary carries reps, range of motion and calibration per limb", () => {
    const eng = new LocalEngine("Elbow Curl");
    const s = engineSummary(eng);
    expect(s.exercise).toBe("Elbow Curl");
    expect(Object.keys(s.limbs)).toEqual(["L", "R"]);
    expect(s.calibration.L).toEqual({ calibrated: false, rest: 160, peak: 50 });
  });
});
