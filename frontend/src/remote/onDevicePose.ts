import type { PoseLandmarker } from "@mediapipe/tasks-vision";
import { isBundled } from "../net";
import { canRunInference, landmarksToKeypointsInto, worldToJointsInto, type Keypoints, type World } from "./logic";

/** MediaPipe Pose Landmarker running in this browser (WebAssembly, on the GPU where the phone allows it): the pose is computed
 *  where the camera is, so no video crosses the network, and the skeleton is drawn from the same pass with no round trip.
 *  The library is imported on demand, so desktop visitors who never use it do not download it. */

// Must be the same version as the installed package (package.json pins it): the JavaScript and the WebAssembly belong together.
export const TASKS_VISION_VERSION = "0.10.14";
// The offline (bundled) APK ships the WASM inside itself (scripts/copy-assets.mjs); every other build uses the CDN as before.
const WASM_BASE = isBundled() ? "/mediapipe/wasm" : `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VISION_VERSION}/wasm`;
/** Tried in order: this server (loaded once, then cached by the browser) and then Google's host as a fallback. */
export const MODEL_URLS = [
  "/models/pose_landmarker_lite.task",
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
];

export type LoadedLandmarker = { landmarker: PoseLandmarker; delegate: "GPU" | "CPU" };

/** Lite model (complexity 0), VIDEO mode, one person, confidences 0.5. GPU delegate first; if the phone refuses it, the CPU one.
 *  Rejects with a readable message when nothing works (no WebAssembly, offline, model missing). */
export async function createLandmarker(): Promise<LoadedLandmarker> {
  const { FilesetResolver, PoseLandmarker: Landmarker } = await import("@mediapipe/tasks-vision");
  const fileset = await FilesetResolver.forVisionTasks(WASM_BASE);
  let last: unknown = null;
  for (const url of MODEL_URLS) {
    for (const delegate of ["GPU", "CPU"] as const) {
      try {
        const landmarker = await Landmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: url, delegate },
          runningMode: "VIDEO",
          numPoses: 1,
          minPoseDetectionConfidence: 0.5,
          minPosePresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });
        return { landmarker, delegate };
      } catch (e) {
        last = e;
      }
    }
  }
  throw new Error(`could not load the pose model (${last instanceof Error ? last.message : String(last)})`);
}

// ---- ModelManager: one shared landmarker, loaded in the background --------------------------------------------------------
let shared: Promise<LoadedLandmarker> | null = null;

/** One dummy inference on a blank frame, so the GPU shader/graph compilation (the slow first detection) happens now, in the
 *  background, and not when the user presses Start. Failure is harmless: the first real frame just pays the cost. The timestamp
 *  1 ms stays below every later frame's (performance.now()), as VIDEO mode requires increasing times. */
function warmUp(loaded: LoadedLandmarker): LoadedLandmarker {
  try {
    const c = document.createElement("canvas");
    c.width = c.height = 256;
    loaded.landmarker.detectForVideo(c, 1);
  } catch { /* ignore */ }
  return loaded;
}

/** The landmarker, created at most once per page: the WASM, the model and the shader compile run on the first call (made by
 *  `preloadLandmarker` when the app opens), every later call gets the same instance at once. A failed load is not cached, so the
 *  next call retries. The instance lives as long as the page (React StrictMode re-mounts must not close it). */
export function getLandmarker(): Promise<LoadedLandmarker> {
  if (!shared) shared = createLandmarker().then(warmUp).catch((e) => { shared = null; throw e; });
  return shared;
}

/** Fire and forget: start loading now; errors surface later, when Start asks for the model and it is retried. */
export function preloadLandmarker(): void {
  void getLandmarker().catch(() => {});
}

export type LoopHooks = {
  /** One result per processed video frame: the joints in pixels of the w x h frame (null = nobody found), its time, and the
   *  metric 3D world joints when the model gave them (the server then measures angles independent of the view).
   *  `kp` and `world` are the loop's own REUSED buffers, overwritten by the next frame: use them inside this call (draw, serialise)
   *  and never keep a reference; copy what must outlive the call. */
  pose: (kp: Keypoints | null, w: number, h: number, tMs: number, world?: World) => void;
  /** About twice a second: frames per second achieved and the smoothed time one detection takes, in ms. */
  stats: (fps: number, ms: number) => void;
  /** The model threw (the loop stops). */
  failed: (e: unknown) => void;
};

/** Runs detection on the live video, as fast as the camera delivers new frames up to 30 a second. It never queues: a frame that
 *  arrives while the model is busy is simply the next one looked at. */
export function runLocalLoop(getVideo: () => HTMLVideoElement | null, landmarker: PoseLandmarker, on: LoopHooks): { stop: () => void } {
  let stopped = false, handle = 0;
  let lastRun = 0, lastTs = 0, lastVideoTime = -1, lastStats = 0, fps = 0, ms = 0;
  const round1 = (n: number) => Math.round(n * 10) / 10;
  const kpBuf: Keypoints = {}, worldBuf: World = {};        // allocated once per loop; every frame overwrites them in place

  const tick = () => {
    if (stopped) return;
    const video = getVideo();
    const now = performance.now();
    // a NEW video frame (currentTime moved), not too soon after the last detection
    if (video && video.readyState >= 2 && video.videoWidth > 0 && video.currentTime !== lastVideoTime && canRunInference(now, lastRun)) {
      lastVideoTime = video.currentTime;
      const ts = Math.max(now, lastTs + 1);                 // VIDEO mode needs strictly increasing timestamps
      lastTs = ts;
      try {
        const began = performance.now();
        const result = landmarker.detectForVideo(video, ts);
        const took = performance.now() - began;
        ms = ms ? 0.8 * ms + 0.2 * took : took;
        if (lastRun) fps = fps ? 0.8 * fps + 0.2 * (1000 / (now - lastRun)) : 1000 / (now - lastRun);
        lastRun = now;
        const lm = result.landmarks[0];
        const wl = result.worldLandmarks?.[0];
        on.pose(lm ? landmarksToKeypointsInto(kpBuf, lm, video.videoWidth, video.videoHeight) : null, video.videoWidth, video.videoHeight, now,
          lm && wl ? worldToJointsInto(worldBuf, wl) : undefined);
      } catch (e) {
        stopped = true;
        on.failed(e);
        return;
      }
      if (now - lastStats > 500) { lastStats = now; on.stats(round1(fps), round1(ms)); }
    }
    handle = requestAnimationFrame(tick);                   // a hidden tab pauses this, which also spares the battery
  };
  handle = requestAnimationFrame(tick);
  return { stop: () => { stopped = true; cancelAnimationFrame(handle); } };
}
