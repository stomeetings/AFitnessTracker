/** A small IndexedDB wrapper (no library): the same code in a browser and in the Capacitor WebView, no native plugin.
 *  Database "rehab": `sessions` (key: id), `events` (auto id, index: sessionId), `meta` (key: key). Every write is one transaction. */
import type { EventRecord, SessionRecord, SessionStore } from "./types";

export const DB_NAME = "rehab";
export const DB_VERSION = 1;

export function openDb(factory: IDBFactory = indexedDB): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = factory.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains("sessions")) db.createObjectStore("sessions", { keyPath: "id" }).createIndex("startedAt", "startedAt");
      if (!db.objectStoreNames.contains("events")) db.createObjectStore("events", { keyPath: "id", autoIncrement: true }).createIndex("sessionId", "sessionId");
      if (!db.objectStoreNames.contains("meta")) db.createObjectStore("meta", { keyPath: "key" });
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error ?? new Error("could not open the local database"));
    req.onblocked = () => reject(new Error("the local database is blocked by another tab"));
  });
}

const request = <T>(r: IDBRequest<T>): Promise<T> => new Promise((resolve, reject) => { r.onsuccess = () => resolve(r.result); r.onerror = () => reject(r.error); });
const finished = (tx: IDBTransaction): Promise<void> => new Promise((resolve, reject) => {
  tx.oncomplete = () => resolve();
  tx.onerror = () => reject(tx.error ?? new Error("transaction failed"));
  tx.onabort = () => reject(tx.error ?? new Error("transaction aborted"));
});

const newId = (): string => (globalThis.crypto?.randomUUID?.() ?? `${Date.now().toString(16)}-${Math.random().toString(16).slice(2)}`);

export class RehabStore implements SessionStore {
  private db: IDBDatabase | null = null;

  private constructor(db: IDBDatabase, private readonly factory: IDBFactory) { this.attach(db); }

  static async open(factory: IDBFactory = indexedDB): Promise<RehabStore> {
    return new RehabStore(await openDb(factory), factory);
  }

  private attach(db: IDBDatabase): void {
    this.db = db;
    db.onversionchange = () => { db.close(); if (this.db === db) this.db = null; };
    db.onclose = () => { if (this.db === db) this.db = null; };
  }

  /** The connection; opened again if the browser closed it (a WebView under memory pressure does that). */
  private async conn(): Promise<IDBDatabase> {
    if (!this.db) this.attach(await openDb(this.factory));
    return this.db!;
  }

  /** Runs `fn` on the stores in ONE transaction and waits for it to commit. A failure drops the connection so the next call reopens. */
  private async run<T>(stores: string[], mode: IDBTransactionMode, fn: (tx: IDBTransaction) => Promise<T> | T): Promise<T> {
    try {
      const db = await this.conn();
      const tx = db.transaction(stores, mode);
      const done = finished(tx);
      done.catch(() => {});                          // reported below through `await done`; no unhandled-rejection noise if fn fails first
      const result = await fn(tx);
      await done;
      return result;
    } catch (e) {
      this.db = null;
      throw e;
    }
  }

  /** The session record and its new events, together or not at all. */
  commit(session: SessionRecord, events: EventRecord[]): Promise<void> {
    return this.run(["sessions", "events"], "readwrite", (tx) => {
      tx.objectStore("sessions").put(session);
      const store = tx.objectStore("events");
      for (const e of events) store.add(e);
    });
  }

  getSession(id: string): Promise<SessionRecord | undefined> {
    return this.run(["sessions"], "readonly", (tx) => request<SessionRecord | undefined>(tx.objectStore("sessions").get(id)));
  }
  /** Newest first. */
  async listSessions(): Promise<SessionRecord[]> {
    const all = await this.run(["sessions"], "readonly", (tx) => request<SessionRecord[]>(tx.objectStore("sessions").getAll()));
    return all.sort((a, b) => b.startedAt - a.startedAt);
  }
  listEvents(sessionId: string): Promise<EventRecord[]> {
    return this.run(["events"], "readonly", (tx) => request<EventRecord[]>(tx.objectStore("events").index("sessionId").getAll(sessionId)));
  }
  /** The finished sessions the server does not have yet, oldest first (booleans are not valid IndexedDB keys, so this filters).
   *  A session the server refused as malformed (`syncFailed`) is left out: retrying it would never end. A session still open from
   *  a crash (no `endedAt`) is taken as finished once it is `staleAfterMs` old. */
  async listUnsynced(staleAfterMs = 24 * 3600 * 1000, now = Date.now()): Promise<SessionRecord[]> {
    return (await this.listSessions())
      .filter((s) => !s.synced && !s.syncFailed && (s.endedAt !== null || now - s.startedAt > staleAfterMs))
      .reverse();
  }
  private patch(ids: string[], change: Partial<SessionRecord>): Promise<void> {
    return this.run(["sessions"], "readwrite", async (tx) => {
      const store = tx.objectStore("sessions");
      for (const id of ids) {
        const s = await request<SessionRecord | undefined>(store.get(id));
        if (s) store.put({ ...s, ...change });
      }
    });
  }
  markSynced(ids: string[]): Promise<void> { return this.patch(ids, { synced: true }); }
  /** The server answered 4xx for this session alone: keep it, stop retrying it. */
  markFailed(ids: string[]): Promise<void> { return this.patch(ids, { syncFailed: true }); }
  /** Deletes sessions that were uploaded more than `maxAgeMs` ago, with their events. Returns how many. */
  async pruneSynced(maxAgeMs: number, now = Date.now()): Promise<number> {
    const old = (await this.listSessions()).filter((s) => s.synced && now - (s.endedAt ?? s.startedAt) > maxAgeMs).map((s) => s.id);
    if (!old.length) return 0;
    await this.run(["sessions", "events"], "readwrite", async (tx) => {
      const sessions = tx.objectStore("sessions"), index = tx.objectStore("events").index("sessionId");
      for (const id of old) {
        sessions.delete(id);
        const keys = await request<IDBValidKey[]>(index.getAllKeys(id));
        for (const k of keys) tx.objectStore("events").delete(k);
      }
    });
    return old.length;
  }

  async getMeta<T = unknown>(key: string): Promise<T | undefined> {
    const row = await this.run(["meta"], "readonly", (tx) => request<{ key: string; value: T } | undefined>(tx.objectStore("meta").get(key)));
    return row?.value;
  }
  setMeta(key: string, value: unknown): Promise<void> {
    return this.run(["meta"], "readwrite", (tx) => { tx.objectStore("meta").put({ key, value }); });
  }
  /** This device's id: made once, kept for good (what the server tells devices apart by). */
  async deviceId(): Promise<string> {
    const saved = await this.getMeta<string>("deviceId");
    if (saved) return saved;
    const id = newId();
    await this.setMeta("deviceId", id);
    return id;
  }
}
