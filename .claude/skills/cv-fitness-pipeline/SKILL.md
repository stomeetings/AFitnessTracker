---
name: cv-fitness-pipeline
description: Performance guardrails for the computer-vision / pose-estimation code of the AI Fitness Tracker (frame throttling, allocation-free hot loops, cheap smoothing, memory cleanup on Android phones). Use whenever writing or changing the camera loop, on-device MediaPipe, landmark handling, overlay drawing, smoothing filters, or anything that runs per frame, in frontend/src/remote/ or backend/app/pipeline/.
---

# CV fitness pipeline: performance guardrails

These rules protect mid-range Android phones (thermal throttling, battery, RAM). They apply to code that runs **per frame**.
Where this repo already follows a rule, the file is named so you reuse it instead of re-inventing it. Where it does not yet,
the gap is listed under "Current state" and is for the user to schedule: do not change pose logic as a side effect of an
unrelated task.

## 1. Frame throttling and budget

- **Never process every camera frame.** The camera may deliver 30 or 60 frames a second; the model runs on a budgeted subset.
  Target a **stable 15-24 fps**, not the maximum. A steady 20 fps beats a 30 fps that alternates with stalls.
- **Throttle on a clock, not on a counter.** Compare `performance.now()` with the last run time (`canRunInference` in
  `frontend/src/remote/logic.ts`). Skipping "every 2nd frame" gives 30 fps on one phone and 15 on another.
- **Only look at NEW frames.** Compare `video.currentTime` with the last one (see `runLocalLoop` in `onDevicePose.ts`), or use
  `requestVideoFrameCallback` where available. A 60 Hz `requestAnimationFrame` on a 30 fps camera sees every frame twice.
- **Never queue.** Latest frame wins: if inference is still running, the next tick looks at the newest frame and the older one is
  dropped. No backlog of pending frames, ever (same rule as `InferenceService` on the server).
- **Budget by measurement.** Track an exponential average of inference ms and fps. If inference exceeds the frame interval, lower
  the rate or the input size (adaptive tiers, roadmap B3), and show the tier. Never raise the rate above the cap to "catch up".
- **Pause when nobody looks.** No loop in a hidden tab (rAF pauses by itself; do not defeat it with timers) and none while
  standing by if the UI does not need the skeleton.
- **Small input.** Ask the camera for what the model uses (MediaPipe works on a ~256 px crop): 640x480 is plenty on a phone.
  More pixels cost texture upload and conversion for nothing.
- **Network is a budget too.** Send keypoints (about 1 kB), not video, in on-device mode. Round numbers (see `landmarksToKeypoints`)
  and keep telemetry replies rate-limited.
- Smoothing and counting math must be **time-based** (`dt`), not frame-based, so a throttled or slow phone behaves like a fast one
  (`EMA(tau=...)` in `backend/app/pipeline/kinematics.py`).

## 2. Zero-allocation hot loop

The hot path is everything inside the rAF tick, the pose callback and the drawing code. In it:

- **No `new`, no `[]`/`{}` literals, no spread, no `.map/.filter/.slice/.concat`, no string building, no closures created per tick.**
  Each one is garbage; garbage means GC pauses; pauses mean visible stutter on a phone.
- **Preallocate once, outside the loop, and overwrite in place:** one keypoint buffer (`Float32Array` of joints x 3), one world
  buffer, one result object, one reusable `Path2D`/point list for drawing. Index by constant (joint index), not by string key.
- **Hoist everything stable:** canvas context, `getBoundingClientRect` results, constant tables (`LANDMARK_JOINTS`), bound
  callbacks. Never call `getContext`, `getImageData`, `toDataURL` or measure layout inside the loop.
- **Never put per-frame data in React state.** The skeleton is drawn imperatively through `setPoseListener` (not `setState`);
  React gets stats at 2 Hz at most. A `setState` per frame re-renders the tree 20-30 times a second.
- **Draw only what changed**, and only once per new pose. Clear and redraw one canvas; avoid shadows, gradients and per-frame
  `ctx.save/restore` pairs where a plain stroke does.
- **MediaPipe results are borrowed.** The landmarks arrays inside a result are reused by the library on the next call: copy the
  numbers you need into your own buffers in the same tick, do not keep references, and call `result.close()` (or the
  `detectForVideo` callback form) when the API version in use exposes it.
- **Measure, do not guess.** Verify with Chrome DevTools > Performance on the phone (remote debugging): the heap graph must be
  flat in steady state (a sawtooth means per-frame allocation) and there must be no long GC bars.
- JSON for the socket is the one place a per-message string is unavoidable: build it once per sent frame, never per rendered frame.

## 3. Jitter reduction at O(1) cost

- **Use a filter with constant time and constant memory per joint:** a time-based EMA (`alpha = 1 - exp(-dt / tau)`; already in
  `EMA`) or a **One-Euro filter** (adaptive cutoff: heavy smoothing when still, light when moving, so reps do not lag).
  Per sample it is a handful of multiplications and one stored value (two for One-Euro).
- **No windows, no history arrays, no median over recent frames in the per-frame path.** (The one-off standby median in
  `Pipeline.calibrate` runs once per start, not per frame: that is fine.) No `Array.sort`, no FFT, no Kalman matrices per frame.
- **Filter once, in one place.** Smooth the **angles** that the rep counter uses (server `Pipeline`), and if the overlay needs
  smoother landmarks, filter those in the same buffers in place. Never smooth the same signal twice: lag adds up and peaks get
  flattened below the rep threshold.
- **Keep filter state in plain preallocated numbers** (arrays indexed by joint), reset on exercise change, camera switch and
  after a gap (a long `dt` should snap, not blend with a stale value).
- **Never block the UI thread:** if a filter or model ever costs more than about 2 ms per frame, it belongs in a Web Worker or
  on the server, not in the tick.
- **Tune with the person, not on paper.** Expose `tau` / One-Euro `min_cutoff` and `beta` as constants; test seated and standing
  and fast and slow reps. Smoothing that hides a real rep peak is a bug.

## 4. Memory hygiene

Release everything you created, **when the view unmounts, the session stops, or the page goes to the background** (`pagehide`,
`visibilitychange` to hidden). A phone browser that holds a camera, a WASM model and a GPU context in the background is killed
by the OS or leaks until it crashes.

- **Media streams:** `stream.getTracks().forEach(t => t.stop())`, then `video.srcObject = null`. On a camera switch, stop the old
  stream and wait before opening the new one (this repo already does; keep it). Never leave a track running "for later".
- **Models:** call `landmarker.close()` (MediaPipe) on stop, unmount and delegate/model change. Do not create a second landmarker
  before the first is closed. If TF.js or ONNX tensors are ever used again: `tensor.dispose()` / `tf.tidy()` around every
  inference, and `session.release()` at the end. Count `tf.memory().numTensors` in a test; it must not grow.
- **Canvases:** drop references to offscreen canvases and contexts, set `canvas.width = canvas.height = 0` to free the backing
  store (important on iOS Safari), and `bitmap.close()` for every `ImageBitmap` you create.
- **Loops, listeners and timers:** cancel the rAF handle (`cancelAnimationFrame`), `clearInterval`/`clearTimeout`, remove event
  listeners, and unregister the pose listener. A `useEffect` that starts any of these must return the cleanup that stops it.
- **Background:** on `visibilitychange` to hidden, stop the loop and (if the session is not meant to continue) release the camera
  and the wake lock; restart on return. Browsers cannot monitor in the background; do not pretend they can.
- **Object URLs and buffers:** `URL.revokeObjectURL` after use; do not keep recorded chunks (MediaRecorder `ondataavailable`
  blobs) longer than the upload.
- **Verify:** start and stop a session 10 times in a row and compare the heap (DevTools Memory > heap snapshot). The count of
  detached DOM nodes, canvases, MediaStream and landmarker objects must return to the baseline.

## Current state in this repo (check before relying on it)

Follows the rules: time-based throttling and new-frame check (`runLocalLoop`), latest-frame-wins (client and `InferenceService`),
skeleton outside React (`setPoseListener`), stats at 2 Hz, 3-decimal and 1-decimal rounding on the wire, time-based EMA,
stream release on camera switch, landmarker created once.

Known gaps (not fixed by this skill; raise them, do not slip them into unrelated changes):
- The throttle cap is `MAX_LOCAL_FPS = 30`, above the 15-24 target here. Changing it is a product decision (rep-counting accuracy
  needs about 12-15 fps minimum with the time-based filter); an adaptive tier (roadmap B3) is the planned answer.
- Per-frame allocations remain in `landmarksToKeypoints` / `worldToJoints` (a new object and arrays per joint, every frame) and in
  `keypointMessage` (`JSON.stringify`). Moving to preallocated buffers is a refactor with its own tests.
- Camera is requested at 1280x720 in on-device mode (roadmap B1: 640x480).
- Check that `landmarker.close()` and the stream stop run on unmount and on `visibilitychange` before adding new resources.

## Checklist before finishing a change that touches a per-frame path

1. Does anything in the tick allocate (literal, `new`, array method, spread, template string)?
2. Is the rate clock-based, capped, and "latest frame wins"?
3. Is the filter O(1) per joint, applied once, with `dt`?
4. Does every resource you opened have a matching close on unmount, stop and hidden?
5. Did you run `npx tsc --noEmit`, `npm test` and the Python suite, and avoid adding per-frame React state?
