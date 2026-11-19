/** Where the backend is. In the browser the page and the API share one origin, so the base is "" and every path stays relative
 *  (exactly as before). In the bundled Android app the page comes from the APK (https://localhost), so the API lives elsewhere:
 *  the base comes from `localStorage.apiBase` (changeable without a rebuild) or the build-time `VITE_API_BASE`. */

/** `base` + `path`; a trailing slash on the base is dropped. */
export const joinApi = (base: string, path: string): string => base.replace(/\/+$/, "") + path;

/** The WebSocket address for `path`: on the API base when there is one (http -> ws, https -> wss), else on this page's own host. */
export function toWsUrl(base: string, path: string, loc: { protocol: string; host: string }): string {
  if (base) return joinApi(base, path).replace(/^http/i, "ws");
  return `${loc.protocol === "https:" ? "wss" : "ws"}://${loc.host}${path}`;
}

/** Only an http(s) address is accepted as a base: anything else is ignored. */
export const cleanBase = (v: string | null | undefined): string => {
  const s = (v ?? "").trim().replace(/\/+$/, "");
  return /^https?:\/\/[^\s?#]+$/i.test(s) ? s : "";
};

export function apiBase(): string {
  try {
    const saved = cleanBase(localStorage.getItem("apiBase"));
    if (saved) return saved;
  } catch { /* storage blocked */ }
  return cleanBase(import.meta.env.VITE_API_BASE as string | undefined);
}

/** Where the access token is kept (typed once on the connect screen, or from `?token=` in the address). */
export const TOKEN_KEY = "remoteToken";
export function savedToken(): string {
  try { return localStorage.getItem(TOKEN_KEY) ?? ""; } catch { return ""; }
}

export const apiUrl = (path: string): string => joinApi(apiBase(), path);
export const wsUrl = (path: string): string => toWsUrl(apiBase(), path, location);
/** True in the Android app built without a server URL (the page is local, so it must work without the network). */
export const isBundled = (): boolean => import.meta.env.VITE_BUNDLED === "1";
