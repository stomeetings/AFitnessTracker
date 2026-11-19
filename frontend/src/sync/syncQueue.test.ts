import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { EventRecord, SessionRecord } from "../store/types";
import { BATCH_SIZE, INTERVAL_MS, PRUNE_AFTER_MS, SyncQueue, prioritize, syncText, type SyncStore } from "./syncQueue";

const session = (n: number, over: Partial<SessionRecord> = {}): SessionRecord => ({
  id: `sess-${String(n).padStart(4, "0")}-test`, exercise: "Elbow Curl", startedAt: 1000 + n, endedAt: 2000 + n, limbs: {}, calibration: {}, workouts: [],
  appVersion: "1", synced: false, ...over,
});

type FakeStore = SyncStore & { sessions: SessionRecord[]; meta: Record<string, unknown>; pruned: number[] };
function fakeStore(sessions: SessionRecord[], eventPayload: Record<string, unknown> = {}): FakeStore {
  const meta: Record<string, unknown> = {}, pruned: number[] = [];
  return {
    sessions, meta, pruned,
    async listUnsynced(): Promise<SessionRecord[]> { return sessions.filter((x) => !x.synced && !x.syncFailed); },
    async listEvents(id: string): Promise<EventRecord[]> { return [{ sessionId: id, t: 1, type: "start", payload: eventPayload }]; },
    async markSynced(ids: string[]): Promise<void> { sessions.forEach((x) => { if (ids.includes(x.id)) x.synced = true; }); },
    async markFailed(ids: string[]): Promise<void> { sessions.forEach((x) => { if (ids.includes(x.id)) x.syncFailed = true; }); },
    async pruneSynced(ms: number): Promise<number> { pruned.push(ms); return 0; },
    async deviceId(): Promise<string> { return "dev-1"; },
    async setMeta(k: string, v: unknown): Promise<void> { meta[k] = v; },
  };
}

type Call = { url: string; init?: RequestInit; ids: string[] };
type Net = {
  calls: Call[];
  post: (ids: string[], n: number) => number;                   // the status of an upload; n = how many uploads so far, this one included
  health: (init?: RequestInit) => Promise<Response>;
  fetch: (url: string, init?: RequestInit) => Promise<Response>;
  posts: () => Call[];
};
function fakeNet(post: Net["post"] = () => 200, health: Net["health"] = async () => new Response("{}")): Net {
  const calls: Call[] = [];
  const net: Net = {
    calls, post, health,
    async fetch(url, init) {
      if (url.endsWith("/api/health")) { calls.push({ url, init, ids: [] }); return net.health(init); }
      const ids = (JSON.parse(String(init?.body)).sessions as Array<{ id: string }>).map((x) => x.id);
      calls.push({ url, init, ids });
      return new Response("{}", { status: net.post(ids, calls.filter((c) => c.ids.length).length) });
    },
    posts: () => calls.filter((c) => c.ids.length),
  };
  return net;
}

let busy = false;
const make = (store: FakeStore, net: Net) =>
  new SyncQueue({ store: async () => store, fetch: (u, i) => net.fetch(u, i), url: (p) => p, token: () => "tok", busy: () => busy, idle: (cb) => cb(), random: () => 0 });
/** Lets every promise chain finish (each await is a microtask hop; no clock time passes). */
const drain = async () => { for (let i = 0; i < 40; i++) await vi.advanceTimersByTimeAsync(0); };
/** Moves the clock, then lets what the timers started finish. */
const tick = async (ms: number) => { await vi.advanceTimersByTimeAsync(ms); await drain(); };

beforeEach(() => { vi.useFakeTimers(); busy = false; });
afterEach(() => { vi.useRealTimers(); vi.restoreAllMocks(); });

describe("SyncQueue", () => {
  it("does nothing, and asks nobody, when everything is already synced", async () => {
    const store = fakeStore([session(1, { synced: true })]), net = fakeNet(), q = make(store, net);
    const seen = vi.fn();
    q.subscribe(seen);
    q.trigger(); await drain();
    expect(net.calls).toEqual([]);
    expect(q.getStatus()).toEqual({ pending: 0, state: "idle" });
    expect(seen).not.toHaveBeenCalled();                                // the status did not change, so nobody is told
    expect(syncText(q.getStatus())).toBe("All synced");
  });

  it("asks the server first (2 s), goes offline when it does not answer, and uploads after a backed-off retry", async () => {
    const store = fakeStore([session(1)]);
    const net = fakeNet(() => 200, (init) => new Promise((_, reject) => init?.signal?.addEventListener("abort", () => reject(new Error("aborted")))));
    const q = make(store, net);
    q.trigger(); await drain();
    expect(net.calls.map((c) => c.url)).toEqual(["/api/health"]);       // only the health check, nothing uploaded
    expect(q.getStatus()).toEqual({ pending: 1, state: "idle" });
    await tick(2000);                                                   // the timeout
    expect(q.getStatus()).toEqual({ pending: 1, state: "offline" });
    expect(syncText(q.getStatus())).toBe("1 session waiting to sync · offline");
    net.health = async () => new Response("{}");                        // the connection is back
    await tick(500);                                                    // the first backoff step (jitter 0)
    expect(net.posts().length).toBe(1);
    expect(store.sessions[0].synced).toBe(true);
    expect(q.getStatus()).toEqual({ pending: 0, state: "idle" });
  });

  it("waits while a workout is running: no health check, no upload; then goes ahead", async () => {
    const store = fakeStore([session(1)]), net = fakeNet(), q = make(store, net);
    busy = true;
    q.trigger(); await drain();
    expect(net.calls).toEqual([]);
    expect(q.getStatus().pending).toBe(1);
    await tick(5000);
    expect(net.calls).toEqual([]);                                      // still running: asks again later
    busy = false;
    await tick(5000);
    expect(net.posts().length).toBe(1);
    expect(store.sessions[0].synced).toBe(true);
  });

  it("uploads in batches of 20, sessions with a fall or alert first, and marks them synced only after a 2xx", async () => {
    const all = Array.from({ length: 45 }, (_, i) => session(i + 1, i >= 42 ? { flagged: true } : {}));
    const store = fakeStore(all), net = fakeNet(), q = make(store, net);
    expect(prioritize(all).slice(0, 4).map((s) => s.id)).toEqual([all[42].id, all[43].id, all[44].id, all[0].id]);
    q.trigger(); await drain();
    const posts = net.posts();
    expect(posts.map((p) => p.ids.length)).toEqual([BATCH_SIZE, BATCH_SIZE, 5]);
    expect(posts[0].ids.slice(0, 3)).toEqual([all[42].id, all[43].id, all[44].id]);      // the flagged ones lead the first batch
    expect(posts[0].ids[3]).toBe(all[0].id);
    expect(all.every((s) => s.synced)).toBe(true);
    expect((posts[0].init?.headers as Record<string, string>).Authorization).toBe("Bearer tok");
    const body = JSON.parse(String(posts[0].init?.body));
    expect(body.deviceId).toBe("dev-1");
    expect(body.sessions[0].events).toEqual([{ t: 1, type: "start", payload: {} }]);
    expect(store.pruned).toEqual([PRUNE_AFTER_MS]);                      // and old synced ones are pruned
    expect(typeof store.meta.lastSync).toBe("number");
  });

  it("keeps everything when the server fails, backs off, and succeeds on the retry (never marked early)", async () => {
    const store = fakeStore([session(1), session(2)]), net = fakeNet((_, n) => (n === 1 ? 500 : 200)), q = make(store, net);
    q.trigger(); await drain();
    expect(store.sessions.some((s) => s.synced)).toBe(false);
    expect(q.getStatus()).toEqual({ pending: 2, state: "error" });
    await tick(500);
    expect(store.sessions.every((s) => s.synced)).toBe(true);
    expect(net.posts().length).toBe(2);
  });

  it("a wrong token is not hammered: it waits for the next trigger", async () => {
    const store = fakeStore([session(1)]), net = fakeNet(() => 401), q = make(store, net);
    q.trigger(); await drain();
    expect(q.getStatus()).toEqual({ pending: 1, state: "auth" });
    expect(syncText(q.getStatus())).toContain("access token");
    await tick(5000);
    expect(net.posts().length).toBe(1);                                  // no retry loop
    net.post = () => 200;                                                // the token was typed meanwhile
    q.trigger(); await drain();
    expect(q.getStatus()).toEqual({ pending: 0, state: "idle" });
  });

  it("splits a batch the server calls malformed and sets aside only the session it refuses", async () => {
    const bad = session(2);
    const store = fakeStore([session(1), bad, session(3)]);
    const net = fakeNet((ids) => (ids.includes(bad.id) ? 400 : 200)), q = make(store, net);
    vi.spyOn(console, "warn").mockImplementation(() => {});
    q.trigger(); await drain();
    expect(store.sessions.map((s) => [s.synced, !!s.syncFailed])).toEqual([[true, false], [false, true], [true, false]]);
    expect(q.getStatus()).toEqual({ pending: 0, state: "idle" });        // the refused one is not retried for ever
  });

  it("splits a batch whose body would be too large for the server", async () => {
    const store = fakeStore([session(1), session(2)], { pad: "x".repeat(500_000) }), net = fakeNet(), q = make(store, net);
    q.trigger(); await drain();
    expect(net.posts().map((p) => p.ids.length)).toEqual([1, 1]);
    expect(store.sessions.every((s) => s.synced)).toBe(true);
  });

  it("folds triggers made during a run into one more pass, not one upload each", async () => {
    const store = fakeStore([session(1)]), net = fakeNet(), q = make(store, net);
    q.trigger(); q.trigger(); q.trigger(); await drain();
    expect(net.posts().length).toBe(1);
  });

  it("keeps trying while something waits, and stops all traffic and timers once everything is synced", async () => {
    const store = fakeStore([session(1)]), net = fakeNet(() => 503), q = make(store, net);
    q.trigger(); await drain();
    const first = net.calls.length;
    await tick(INTERVAL_MS);
    expect(net.calls.length).toBeGreaterThan(first);                     // the backoff and the 60 s timer kept trying
    net.post = () => 200;
    await tick(INTERVAL_MS);
    expect(q.getStatus().pending).toBe(0);
    const after = net.calls.length;
    await tick(INTERVAL_MS * 3);
    expect(net.calls.length).toBe(after);                                // nothing waits: no timer, no traffic
  });

  it("tells subscribers only when the status really changes", async () => {
    const store = fakeStore([session(1)]), net = fakeNet(), q = make(store, net);
    const seen = vi.fn();
    q.subscribe(seen);
    q.trigger(); await drain();
    const told = seen.mock.calls.length;
    expect(told).toBeGreaterThan(0);
    q.trigger(); await drain();
    expect(seen.mock.calls.length).toBe(told);
  });
});

describe("status text", () => {
  it("reads like a sentence", () => {
    expect(syncText({ pending: 0, state: "idle" })).toBe("All synced");
    expect(syncText({ pending: 3, state: "idle" })).toBe("3 sessions waiting to sync");
    expect(syncText({ pending: 1, state: "error" })).toBe("1 session waiting to sync · will retry");
  });
});
