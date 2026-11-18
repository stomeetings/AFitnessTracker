---
name: react-capacitor-mobile
description: UI performance rules for the React + Tailwind frontend when it runs in a phone browser or inside the Capacitor Android WebView (no React state for per-frame tracking data, canvas overlay painted from refs, GPU-only animations, dvh / safe-area / touch handling). Use whenever writing or changing components, styles, the HUD, overlays, modals or layout in frontend/src.
---

# React + Capacitor mobile UI rules

The app runs in Chrome / Android System WebView on mid-range phones, over a live camera video. Every avoidable re-render,
layout or repaint competes with the camera and the pose model for the same few milliseconds. These rules are strict; where
the repo does not yet follow them, see "Audit" at the end (raise gaps, do not fix them as a side effect of another task).

Related skill: `cv-fitness-pipeline` (frame throttling, allocation-free hot loops, smoothing, memory release).

## 1. Zero re-renders on tracking loops

- **FORBIDDEN: `useState` / `useReducer` / context for anything that changes per frame or per message from the pose loop:**
  keypoints, landmarks, joint angles, frame size, timestamps, fps/ms counters, "latest pose", server telemetry copied straight
  into state. A `setState` re-renders the component and its whole subtree 15-30 times a second, which tanks mobile FPS.
- **Mutable tracking data lives in `useRef`** (or module-level/plain objects owned by the hook), written by the loop and read by the
  painter. Refs never trigger a render.
- **Paint straight to a `<canvas>` from the loop, not through React.** The pattern already in the repo: the loop calls a
  listener held in a ref (`setPoseListener` in `useRemoteSession.ts`), the listener calls `drawPose(ctx, ...)` in
  `RemoteView.tsx`. Keep it: one overlay canvas, context fetched once, `requestAnimationFrame` or the new-pose callback drives
  the drawing, nothing is created per frame (see `cv-fitness-pipeline`).
- **Numbers the user reads (reps, state, fps) are not per-frame.** Update them at most 2-4 times a second, and only when the
  displayed (rounded) value changed: compare to the previous value in a ref before calling `setState`.
- **Imperative text for fast counters:** a value that must update faster than ~4 Hz (a live angle, a progress bar) is written
  to a DOM node held in a ref (`el.textContent = ...`, `el.style.transform = ...`), not rendered through React.
- **Contain the render blast radius.** Components that read changing telemetry are small leaf components, wrapped in
  `React.memo` with primitive props; the page shell, the video and the canvas must not re-render when telemetry changes.
  Never create a new object/array/function prop inline for a memoised child (`useMemo` / `useCallback` or hoist it).
- **`useEffect` hygiene:** one effect per concern, narrow dependency array, always return the cleanup (listeners, rAF handle,
  timers, streams). A dependency like `s` (a whole hook result recreated every render) silently re-subscribes every render.
- Do not `JSON.parse` + `setState` inside a socket `onmessage` for every message; coalesce (latest wins) and apply on a timer
  or rAF.

## 2. Hardware-accelerated rendering

- **Animate only `transform` and `opacity`.** They run on the compositor thread and skip layout and paint. Never animate or
  transition `width`, `height`, `top`, `left`, `right`, `bottom`, `margin`, `padding`, `border-width`, `font-size`.
  - progress bar: `transform: scaleX(p)` on a full-width element with `transform-origin: left` (`will-change: transform` only on
    the one element that really animates), not `style={{ width: p% }}`;
  - toggle knob / slide-in panel: `translateX/Y`, not `left`/`top`.
- **Tailwind:** use `transition-transform`, `transition-opacity`, `translate-*`, `scale-*`. Do not use `transition-all` and do not
  use `transition-[width]` / `transition-[height]` / `transition-[left]`. Spinners (`animate-spin`) and pulses (`animate-pulse`) are
  already compositor-friendly.
- **Overlays over the live video are the most expensive pixels on screen** (the video repaints 30 times a second under them).
  Avoid `backdrop-blur`, `blur-*`, large `box-shadow`/`drop-shadow` and `mix-blend-*` on anything that sits over the video; use a
  plain translucent background (`bg-black/60`). Keep the HUD as few, small, opaque-ish layers.
- **Promote deliberately, not everywhere:** `will-change` and `translateZ(0)` cost GPU memory per layer; apply to the video,
  the overlay canvas and genuinely animating elements only, and remove it when the animation ends.
- Size the overlay canvas to the displayed video once (and on resize), with `devicePixelRatio` capped at 2; never re-set
  `canvas.width/height` per frame (it clears and reallocates the backing store).
- Respect `prefers-reduced-motion`. No continuous decorative animation while a session is running.
- Verify in Chrome DevTools on the phone: Performance > Frames (no purple "Layout" bars during a session), Rendering > Paint
  flashing and Layer borders (only the canvas and the animating element repaint).

## 3. Mobile viewport and touch

- **Viewport meta:** `width=device-width, initial-scale=1, viewport-fit=cover` (already set). Do not add `user-scalable=no`
  (accessibility); `touch-action` below handles the tap delay.
- **Dynamic viewport units.** `100vh` / `h-screen` / `min-h-screen` ignore the mobile browser bars and the Capacitor system bars,
  so content is cut off or jumps. Use `dvh` (`h-dvh`, `min-h-dvh`, `max-h-[78dvh]`) and `dvw`/`%` for width, with an `svh` or `vh`
  fallback declared first for older WebViews. Full-screen layers: `fixed inset-0` (not `h-screen`).
- **Safe-area insets.** With `viewport-fit=cover`, anything pinned to an edge (HUD top bar, bottom controls, the full-screen
  fallback, modals) pads with `env(safe-area-inset-top|right|bottom|left)`, e.g. `pt-[max(0.5rem,env(safe-area-inset-top))]`.
  Notches, punch-holes, rounded corners and the gesture bar would otherwise cover controls. In Capacitor keep the status bar
  overlay setting and the inset handling consistent.
- **No tap delay, no accidental zoom.** `touch-action: manipulation` on `html`/`body` and on every button/interactive element
  (Tailwind `touch-manipulation`) removes the double-tap-zoom wait and disables double-tap zoom. Use `onPointerDown` only for
  controls where instant action matters more than cancelling (e.g. Start/Pause during an exercise); otherwise `onClick`.
- **Instant press feedback (under 100 ms, no layout shift):** every tappable element has an `active:` style that changes only
  `opacity`/`transform` (`active:scale-95 active:opacity-80`), plus `-webkit-tap-highlight-color: transparent` so the native
  grey flash does not double up with it. Never rely on `:hover` for state on touch.
- **Target size:** at least 44x44 CSS px (48 dp is Android's guideline) with at least 8 px between neighbours; small icons get
  padding, not a smaller button (`min-h-11 min-w-11`). Primary exercise controls larger still: they are used with a sweaty hand
  at arm's length.
- **No accidental scrolling or selection during exercise:** the live view uses `overscroll-behavior: none` and
  `touch-action: none` (on the video/canvas only), `user-select: none` on controls, and no scroll chaining from the HUD.
- **Text:** at least 16 px for form inputs (smaller makes iOS Safari zoom in); body text never below 12 px.
- **Capacitor/WebView:** keep `Permissions` and the manifest as in `NATIVE_DEMO_PLAN.md`; the WebView has no pull-to-refresh or
  back-forward gestures you can rely on, so give every modal an explicit close button and handle the Android back button.

## Checklist before finishing a UI change

1. Does anything per-frame or per-message reach `useState`? (Must be a ref + canvas / DOM write.)
2. Is every animation `transform`/`opacity` only? No `transition-all`, no width/left/top transitions, no blur over the video.
3. `dvh` not `vh`; edge-pinned UI uses `env(safe-area-inset-*)`.
4. Every control: `touch-manipulation`, an `active:` transform/opacity state, at least 44 px.
5. `npx tsc --noEmit`, `npm test`, `npm run build`; a phone check in DevTools (frames and paint).

## Audit of the current code (2026-11-18, read-only; nothing was changed)

Already compliant:
- Skeleton painted outside React: `setPoseListener` ref -> `drawPose` on the overlay canvas (`useRemoteSession.ts`, `RemoteView.tsx`).
- On-device fps/ms stats at 2 Hz; `viewport-fit=cover` is set in `index.html`; HUD buttons are `min-h-10`; `animate-spin` and
  `animate-pulse` are compositor-friendly; the guide dialog animates `opacity` + `transform` and honours reduced motion.

Gaps (priority order):
1. **Server telemetry goes through React state.** `useRemoteSession.ts:243` `setMsg(m)` on every telemetry reply (up to ~10 per
   second in keypoint mode, every frame in server-camera mode) and `:241` `setStats(...)` on every ack (a fresh object each time, so
   it always re-renders). `useRemoteSession` is one hook holding ~15 `useState`s whose result `s` changes every render, and only 2
   `memo`/`useMemo` exist in the whole `src`: each reply re-renders `RemoteView`, `Hud` and children. Desktop: `useTelemetry.ts:53`
   `ws.onmessage = (e) => setT(JSON.parse(e.data))` re-renders `App` per message. Fix: coalesce to 2-4 Hz and skip unchanged values,
   memoised leaf components for HUD/reps/angles, refs + DOM writes for the live angle.
2. **Layout-property animations.** `App.tsx:53` angle bar `transition-[width]` + inline `width` (updates with telemetry, so a layout
   per message); `ExerciseGuide.tsx:238` progress bar via inline `width: p%`; `App.tsx:249` toggle `transition-all` + `left-3.5/left-0.5`.
   Fix: `scaleX` / `translateX` with `transition-transform`.
3. **`backdrop-blur` on the HUD over live video:** `Hud.tsx` lines 28, 42, 46, 51 (the button base and status pills). The video repaints
   continuously under them; blur is among the costliest effects on mid-range GPUs. Fix: `bg-black/60` without blur.
4. **Viewport units:** `App.tsx:226` `min-h-screen`, `FallCheck.tsx:114` `h-screen w-screen`, `RemoteView.tsx:78` `maxHeight: "78vh"`,
   `ExerciseGuide.tsx:167` `max-h-[68vh]`, `:306` `w-[96vw]`. All should be `dvh`/`dvw` (or `inset-0`), with fallbacks.
5. **No safe-area handling anywhere** (no `env(safe-area-inset-*)` in `src`), although `viewport-fit=cover` is on: in the Capacitor
   WebView and full-screen mode the HUD top bar and bottom controls can sit under the camera cut-out or gesture bar. Fix: pad
   `Hud.tsx` and the full-screen container with the insets.
6. **Touch:** no `touch-action: manipulation` / `touch-manipulation` and no tap-highlight reset anywhere; `active:` styles exist in only
   4 places in `App.tsx` and none in the HUD/remote components (the Start/Pause/Flip/Full-screen buttons give no instant press
   feedback). HUD buttons are 40 px (`min-h-10`), below the 44-48 px target; HUD pills use 12-14 px icons.
7. **Not found / to verify:** Android back-button handling for modals in Capacitor; `overscroll-behavior` on the live view; whether
   `prefers-reduced-motion` is respected outside the guide dialog (the pulse/spin animations are not guarded).
