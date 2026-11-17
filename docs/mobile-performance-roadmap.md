# Mobile performance & seated/bed tracking roadmap

*Written 2026-11-17. A strategy document: no application code has been changed for it yet.*

## Context

The phone runs MediaPipe Pose Landmarker (Lite) in the browser and sends joint positions to the server, where the existing
Python pipeline counts reps and watches for falls (`InferenceService.submit_pose`, `app/pipeline/`). Two problems were observed:
some mid-range phones feel sluggish, and reps are missed when exercising seated on a bed or the floor. Reading the code shows the
second problem is mostly **not** a model problem, and that the cheapest, highest-impact fixes are in the pipeline, not the GPU.

## 1. Verdict first

| Question | Recommendation |
|---|---|
| Biggest cause of missed seated/bed reps | The rep thresholds assume a standing body, and phone poses arrive as 2D only. Fix these before touching the model. |
| Lower MediaPipe's detection/tracking confidences for seated use? | No. They control whether a person is tracked, not which joints count. Use per-exercise joint visibility rules instead. |
| A separate upper-body model? | Does not exist in the current MediaPipe Tasks API. Use exercise-specific joint choices and framing guidance. |
| Phone performance | Smaller camera input on phones, frame-rate-independent smoothing, fewer React re-renders, then an adaptive tier. Web Worker later. |
| Capacitor native app | Worth it for background fall monitoring and a higher ceiling, not as the fix for sluggishness: a WebView runs the same JavaScript at the same speed. The gain only comes from a native camera + inference plugin. |
| Safety issue found | Lying in bed looks like "down on the floor" to the fall detector and would raise an alarm after 5 s. Must be designed before any lying exercises are offered. |

## 2. Why seated / bed tracking fails (findings from the code)

1. **Thresholds are calibrated for standing.** `rep_counter.EXERCISES` fixes `rest`/`peak` angles per exercise. Seated, the
   rest posture is different: e.g. Hip Flexion has rest 170 deg (standing straight) but a seated hip rests near 90 deg, so the
   progress value never returns to "rest" and no rep completes. Same class of problem for Knee Flexion on a bed with legs out.
2. **Phone poses lost their 3D.** `joint_angles` (`app/pipeline/kinematics.py`) uses MediaPipe's metric 3D world landmarks when all
   three joints have them, else 2D image angles. The server's MediaPipe engine supplies them, but the phone path does not:
   `landmarksToKeypoints` (`frontend/src/remote/logic.ts`) sends only x, y, visibility, and `parse_keypoints` (`backend/app/remote.py`)
   builds 2D keypoints. 2D angles change with the viewing angle; a knee bending toward the camera, typical when the phone faces a
   person sitting on a bed, barely changes in 2D. This alone explains many missed seated reps.
3. **One hidden joint switches an angle off.** `joint_angles` drops an angle when any of its three joints is below confidence 0.5.
   The shoulder angle is hip-shoulder-elbow: with hips hidden by bedding or the frame edge, arm exercises stop counting too.
   Knee angles need the ankle, which is the first joint lost when legs are under a blanket or out of frame.
4. **Why MediaPipe is weaker seated:** the model was trained mostly on upright people; folded legs overlap the torso, the lower
   body is often outside the frame, and the person's height in the frame is small, so ankle/knee visibility drops and MediaPipe
   extrapolates them (low `visibility`, unreliable positions).

## 3. Phase A — seated/bed robustness (pipeline; do this first)

Ordered by value per effort. All server-side except A1.

- **A1. Send 3D world landmarks from the phone.** Add MediaPipe's `worldLandmarks` (3 numbers per joint) to the `kp` message, and
  accept them in `parse_keypoints` as `Keypoint.world`. `joint_angles` then uses view-independent 3D angles automatically. About
  1 kB per message instead of 0.6 kB. Highest value, smallest change.
- **A2. Calibrate "rest" during standby.** The READY phase (`Pipeline.counting = False`) is exactly when the person holds the
  starting posture. Record the median angle of the exercise's joint over the last ~1.5 s of standby and use it as `rest`; keep
  `peak` as `rest +/- the exercise's range` (direction from the table). A seated user then gets a seated baseline automatically.
  Show it in the HUD ("calibrated: rest 92 deg"). Fall back to the table value if the person was not visible.
- **A3. Posture profiles per exercise.** Extend `EXERCISES` with `posture: standing | chair | bed_long_sit | supine` and, per
  profile, which joints are required and which may be missing. The Exercise picker and guide show the posture; seated variants of
  standing exercises become explicit entries (as "Seated Knee Extension" already is) instead of hoping one set of angles fits.
- **A4. Joint fallbacks for upper-body exercises.** If the hip is missing, measure arm elevation against the torso line from the
  shoulder midpoint, or against the image vertical (phone held upright). Per-exercise confidence for the joints that matter
  (e.g. 0.3 for wrists in arm exercises) with a short presence hysteresis, instead of one global 0.5.
- **A5. Frame-rate-independent smoothing.** `EMA(alpha=0.35)` smooths per frame, so it lags 3x more in time on a 10 fps phone than
  on a 30 fps one, and fast reps can fall under the `PEAK_IN` threshold. Use a time constant (`alpha = 1 - exp(-dt / tau)`).
- **A6. Fall detection in bed (safety; needs a decision).** The "flat body" cue needs no standing baseline, so a person lying for a
  supine exercise is "down" after 5 s. Proposal: for exercises whose profile is `supine`/`bed_long_sit`, while the session is
  ACTIVE, require a real transition (fast drop or a fall from sitting) to suspect a fall, and never disable detection outside an
  active session. This changes safety behaviour, so it needs explicit sign-off and new tests from recorded clips.
- **Not recommended:** lowering `minPosePresenceConfidence` / `minTrackingConfidence` for seated use. They decide whether MediaPipe
  keeps tracking a person at all; lowering them adds jitter and ghost poses without making hidden joints reliable.

## 4. Phase B — browser-side performance and adaptive tiers

- **B1. Smaller camera input in on-device mode.** `openMedia` asks for 1280x720 for every mode; MediaPipe only uses a 256x256 crop,
  so the extra pixels only cost texture upload and conversion. Ask for 640x480 when the phone runs the model.
- **B2. Fewer React re-renders.** The skeleton already bypasses React (`setPoseListener`), and the on-device stats update at 2 Hz.
  What remains is the server telemetry answer (~10 Hz) re-rendering the whole `App`. Lower the answer rate for keypoint sessions
  to ~4 Hz (`KP_REPLY_GAP_S`) and memoise the side panel and HUD.
- **B3. Adaptive tier (measured, automatic, visible).** Tier from the measured `ms` and fps after warm-up:
  `high` 30 fps cap; `mid` 640x480 + 20 fps; `low` 480x360 + 15 fps (process every 2nd frame). Step down after a sustained miss,
  step up after a sustained recovery, show the tier on the badge. Rep counting stays reliable above roughly 12-15 fps with A5.
- **B4. Inference off the main thread (later).** Run the landmarker in a Web Worker with `createImageBitmap(video)` transfers.
  It keeps the UI and canvas smooth while inference runs; the WebGL/GPU path inside a worker needs `OffscreenCanvas` (current
  Chrome on Android; check Safari). Do after B1-B3, only if the main thread is still the bottleneck.
- **B5. Measure first.** Log `ms`, fps, delegate (GPU/CPU), tier and phone model on the server per session, so tiers are tuned
  from real devices instead of guesses.

## 5. Phase C — Capacitor native app (long-term ceiling)

- **What changes, and what does not.** A Capacitor app shows the same page in Android System WebView, which is Chrome's engine: the
  in-page MediaPipe runs at the same speed as in the browser. Wrapping alone does not fix sluggishness.
- **Where the gain comes from:** a native plugin that owns the camera (CameraX) and runs MediaPipe Tasks Pose Landmarker for
  Android on the GPU, and sends only keypoints to the web layer. Frames must stay native: passing frames over the bridge is
  slower than the browser. Using MediaPipe natively keeps the same model and joint set as the web version (no second set of
  angle thresholds). ONNX Runtime Mobile is an option only for other models; Google has deprecated NNAPI (Android 15), so rely on
  the GPU delegate rather than NNAPI.
- **The real reason to go native:** background operation. A browser stops the camera when the screen is off or the app is in the
  background, so fall monitoring cannot run continuously on the web. A native foreground service can, together with reliable
  alarms and notifications. Native also allows the Full or Heavy model on capable phones and better thermal control.
- **Cost:** Android-specific code (iOS would need its own plugin), app store releases, device testing, and keeping web and native
  keypoints identical. Recommended only when continuous fall monitoring or the Full model on phones is required.

## Status

- **A1, A2 and A5 implemented 2026-11-17, awaiting a phone test** (see `docs/activity_log.md`). A3, A4, A6 and all of Phases B and C
  are still open.
- A2 note: the calibrated `peak` keeps the table's RANGE (e.g. 60 degrees for Hip Flexion) from the measured rest. For exercises
  whose seated range of motion is smaller than the table's, the peak may not be reachable; that is what the posture profiles (A3)
  and a physiotherapist's seated angles are for.

## 6. Recommended order

1. A1 (3D world landmarks), A2 (standby calibration), A5 (time-based smoothing): small changes, directly targets missed seated reps.
2. B1 (640x480 on-device), B2 (fewer re-renders), B5 (measurement).
3. A3/A4 (posture profiles and fallbacks), with a physiotherapist reviewing angles and seated variants.
4. A6 (fall logic in bed): only with sign-off and recorded test clips.
5. B3 (adaptive tiers), then B4 (worker) if still needed.
6. C (native) when background fall monitoring becomes a product requirement.

## Verification (when implemented)

- Recorded clips (standing, chair, bed long-sit, supine; front and side camera) replayed through the pipeline in tests: reps
  counted per clip before/after A1/A2/A5, and no fall alarm for supine exercise clips once A6 lands.
- Phone matrix (one low, one mid, one high Android; one iPhone): fps, ms, delegate and tier from B5 logs; HUD updates stay smooth.
- Existing Python and vitest suites stay green.
