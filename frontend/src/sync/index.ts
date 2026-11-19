import { useSyncExternalStore } from "react";
import { Capacitor } from "@capacitor/core";
import { apiUrl, savedToken } from "../net";
import { getStore } from "../store";
import { SyncQueue, whenIdle, type SyncStatus } from "./syncQueue";

export { syncText, type SyncStatus } from "./syncQueue";

let workoutActive = false;
/** The workout screen tells the queue whether a session is running: uploads wait for it to end. */
export const setWorkoutActive = (v: boolean): void => { workoutActive = v; };

let queue: SyncQueue | null = null;
export function syncQueue(): SyncQueue {
  queue ??= new SyncQueue({
    store: async () => (await getStore()),
    fetch: (url, init) => fetch(url, init),
    url: apiUrl,
    token: savedToken,
    busy: () => workoutActive,
    idle: whenIdle,
  });
  return queue;
}

/** Starts the triggers: now (app start), the browser's `online` event, the native network and app-resume events. The 60 s timer is the
 *  queue's own, and only runs while something waits. Call once; returns a cleanup. */
export function startSync(): () => void {
  const q = syncQueue();
  const onOnline = () => q.trigger();
  window.addEventListener("online", onOnline);
  const handles: Array<Promise<{ remove: () => Promise<void> }>> = [];
  if (Capacitor.isNativePlatform()) {
    // the native plugins know about the connection (and the app coming back to the front) better than the WebView does
    void import("@capacitor/network").then(({ Network }) => handles.push(Network.addListener("networkStatusChange", (s) => { if (s.connected) q.trigger(); })));
    void import("@capacitor/app").then(({ App }) => handles.push(App.addListener("resume", () => q.trigger())));
  }
  q.trigger();
  return () => {
    window.removeEventListener("online", onOnline);
    for (const h of handles) void h.then((x) => x.remove()).catch(() => {});
  };
}

/** "N sessions waiting to sync" / "All synced": re-renders only when the queue's status changes. */
export function useSyncStatus(): SyncStatus {
  const q = syncQueue();
  return useSyncExternalStore(q.subscribe, q.getStatus);
}
