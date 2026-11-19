import { useCallback, useEffect, useRef, useState } from "react";

/** Full screen for one element (the video wrapper), so everything inside it (the HUD, the buttons, the voice chip) stays visible.
 *  Uses the browser's Fullscreen API where it exists (Android Chrome, desktop, iPad). iPhone Safari has none for elements, so
 *  the fallback is the same element pinned over the whole page (`fixed inset-0`), which `full` reports too: the caller adds
 *  the `fixed inset-0 z-40` classes whenever `full` is true (a native full-screen element is already fixed, so it is harmless).
 *  Modal dialogs (the Exercise Guide, the fall check) live in the browser's top layer and still appear above either kind. */
export function useFullscreen() {
  const ref = useRef<HTMLElement | null>(null);
  const [native, setNative] = useState(false);
  const [fallback, setFallback] = useState(false);

  useEffect(() => {
    const onChange = () => setNative(document.fullscreenElement !== null && document.fullscreenElement === ref.current);
    document.addEventListener("fullscreenchange", onChange);
    return () => document.removeEventListener("fullscreenchange", onChange);
  }, []);

  const full = native || fallback;

  useEffect(() => {                         // while the page-covering fallback is on: no scrolling behind it, Esc leaves it
    if (!fallback) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setFallback(false); };
    window.addEventListener("keydown", onKey);
    return () => { document.body.style.overflow = prev; window.removeEventListener("keydown", onKey); };
  }, [fallback]);

  const enter = useCallback(async () => {
    const el = ref.current;
    try {
      if (el?.requestFullscreen) { await el.requestFullscreen(); return; }
    } catch { /* refused (no user gesture, or not allowed): use the fallback */ }
    setFallback(true);
  }, []);

  const exit = useCallback(async () => {
    setFallback(false);
    setNative(false);       // reset our state even when the WebView never fires `fullscreenchange` or exitFullscreen throws
    try { if (document.fullscreenElement) await document.exitFullscreen(); } catch { /* already out, or refused: state is reset above */ }
  }, []);

  const toggle = useCallback(() => { void (full ? exit() : enter()); }, [full, enter, exit]);
  return { ref, full, toggle, exit };
}
